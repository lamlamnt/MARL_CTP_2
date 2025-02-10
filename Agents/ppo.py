from functools import partial
import jax.numpy as jnp
import jax
import sys
import optax
from flax.training.train_state import TrainState

sys.path.append("..")
import flax.linen as nn
from typing import Sequence, NamedTuple, Any
from Environment.CTP_environment import MA_CTP_General
from Environment import CTP_generator
from Utils.augmented_belief_state import get_augmented_optimistic_pessimistic_belief
from Utils.optimal_combination import get_optimal_combination_and_cost


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    critic_value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    belief_state: jnp.ndarray
    shortest_path: jnp.ndarray


class PPO:
    def __init__(
        self,
        model: nn.Module,
        environment: MA_CTP_General,
        discount_factor: float,
        gae_lambda: float,
        clip_eps: float,
        vf_coeff: float,
        ent_coeff: float,
        batch_size: int,
        num_minibatches: int,
        horizon_length: int,
        reward_exceed_horizon: float,
        num_loops: int,
        anneal_ent_coeff: bool,
        deterministic_inference_policy: bool,
        ent_coeff_schedule: str,
        sigmoid_beginning_offset_num: int,
        sigmoid_total_nums_all: int,
        num_agents: int,
    ) -> None:
        self.model = model
        self.environment = environment
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coeff = vf_coeff
        self.ent_coeff = ent_coeff
        self.batch_size = batch_size
        self.num_minibatches = num_minibatches
        self.horizon_length = horizon_length
        self.reward_exceed_horizon = jnp.float16(reward_exceed_horizon)
        self.num_loops = num_loops
        self.anneal_ent_coeff = anneal_ent_coeff
        self.deterministic_inference_policy = deterministic_inference_policy
        self.ent_coeff_schedule = ent_coeff_schedule
        self.sigmoid_beginning_offset_num = sigmoid_beginning_offset_num
        self.sigmoid_total_nums_all = sigmoid_total_nums_all
        self.num_agents = num_agents

    def _ent_coeff_schedule(self, loop_count):
        # linear or sigmoid or plateau schedule
        frac = jax.lax.cond(
            self.ent_coeff_schedule == "sigmoid",
            lambda _: 1 / (1 + jnp.exp(10 * (loop_count / self.num_loops - 0.5))),
            lambda _: 1
            / (
                1
                + jnp.exp(
                    10
                    * (
                        (
                            (loop_count + self.sigmoid_beginning_offset_num)
                            / self.sigmoid_total_nums_all
                        )
                        - 0.5
                    )
                )
            ),
            operand=None,
        )
        return self.ent_coeff * frac

    # return the actions for all agents
    @partial(jax.jit, static_argnums=(0,))
    def act(
        self, key, params, belief_states, unused
    ) -> tuple[jnp.array, jax.random.PRNGKey]:
        augmented_belief_states = get_augmented_optimistic_pessimistic_belief(
            belief_states
        )

        def _choose_action(belief_state):
            pi, _ = self.model.apply(params, belief_state)
            random_action = pi.sample(
                seed=key
            )  # use the same key for all agents (maybe not good)
            mode_action = pi.mode()
            return random_action, mode_action

        random_actions, mode_actions = jax.vmap(_choose_action, in_axes=0)(
            augmented_belief_states
        )
        actions = jax.lax.cond(
            self.deterministic_inference_policy,
            lambda _: mode_actions,
            lambda _: random_actions,
            operand=None,
        )
        old_key, new_key = jax.random.split(key)
        return actions, new_key

    @partial(jax.jit, static_argnums=(0,))
    def env_step(self, runner_state, unused):
        # Collect trajectories
        (
            train_state,
            current_env_state,
            current_belief_states,
            key,
            timestep_in_episode,
            loop_count,
            previous_episode_dones,
        ) = runner_state
        action_key, env_key = jax.random.split(key, 2)

        # Agent acts
        augmented_belief_states = get_augmented_optimistic_pessimistic_belief(
            current_belief_states
        )

        def _choose_action(belief_state):
            pi, critic_value = self.model.apply(train_state.params, belief_state)
            action = pi.sample(
                seed=action_key
            )  # use the same key for all agents (maybe not good)
            log_prob = pi.log_prob(action)
            return action, critic_value, log_prob

        actions, critic_values, log_probs = jax.vmap(_choose_action, in_axes=0)(
            augmented_belief_states
        )
        new_env_state, new_belief_states, rewards, dones, env_key = (
            self.environment.step(
                env_key, current_env_state, current_belief_states, actions
            )
        )
        episode_done = jnp.all(dones)

        # Stop the episode and reset if exceed horizon length
        env_key, reset_key = jax.random.split(env_key)
        # Reset timestep if finish episode
        timestep_in_episode = jax.lax.cond(
            episode_done, lambda _: 0, lambda _: timestep_in_episode, operand=None
        )

        # Reset if exceed horizon length. Otherwise, increment
        new_env_state, new_belief_states, rewards, timestep_in_episode, dones = (
            jax.lax.cond(
                timestep_in_episode >= self.horizon_length,
                lambda _: (
                    *self.environment.reset(reset_key),
                    jnp.full(
                        self.num_agents, self.reward_exceed_horizon, dtype=jnp.float16
                    ),
                    0,
                    jnp.full(self.num_agents, True, dtype=bool),
                ),
                lambda _: (
                    new_env_state,
                    new_belief_states,
                    rewards,
                    timestep_in_episode + 1,
                    dones,
                ),
                operand=None,
            )
        )

        # Calculate shortest total cost at the beginning of the episode. But we don't need this for training
        def _calculate_optimal_cost(env_state):
            _, goals = jax.lax.top_k(
                jnp.diag(env_state[3, self.num_agents :, :]), self.num_agents
            )
            origins = jnp.argmax(env_state[0, : self.num_agents, :], axis=1)
            _, optimal_cost = get_optimal_combination_and_cost(
                env_state[1, self.num_agents :, :],
                env_state[0, self.num_agents :, :],
                origins,
                goals,
                self.num_agents,
            )
            return jnp.array(optimal_cost, dtype=jnp.float16)

        # This adds to the compilation time a lot
        shortest_path = jax.lax.cond(
            previous_episode_dones,
            _calculate_optimal_cost,
            lambda _: jnp.array(0.0, dtype=jnp.float16),
            operand=current_env_state,
        )

        runner_state = (
            train_state,
            new_env_state,
            new_belief_states,
            env_key,
            timestep_in_episode,
            loop_count,
            episode_done,
        )

        # Keep all agents' transitions in for now
        # Only add to the transition if the agent is not done done (already serviced the goal)
        # agents_that_should_be_added = jnp.logical_or(previous_episode_dones, dones)

        # Add each agent's experience as a separate transition
        # Return a stacked array of Transitions with varying lengths

        transition = Transition(
            dones,
            actions,
            critic_values,
            rewards,
            log_probs,
            current_belief_states,
            shortest_path,
        )
        return runner_state, transition

    # This is currently the same as single agent
    @partial(jax.jit, static_argnums=(0,))
    def calculate_gae(self, traj_batch, last_critic_val):
        def _get_advantages(gae_and_next_value, transition: Transition):
            gae, next_value = gae_and_next_value
            done, critic_value, reward = (
                transition.done,
                transition.critic_value,
                transition.reward,
            )
            team_reward = jnp.sum(reward, axis=0)
            print(team_reward.shape)
            delta = (
                reward + self.discount_factor * next_value * (1 - done) - critic_value
            )
            gae = delta + self.discount_factor * self.gae_lambda * (1 - done) * gae
            return (gae, critic_value), gae

        # Apply get_advantage to each element in traj_batch
        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_critic_val), last_critic_val),
            traj_batch,
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + traj_batch.critic_value

    def _loss_fn(self, params, traj_batch: Transition, gae, targets, ent_coeff):
        # RERUN NETWORK
        traj_batch_augmented_belief_state = jax.vmap(
            get_augmented_optimistic_pessimistic_belief
        )(traj_batch.belief_state)
        # vmap over agents and over timesteps
        pi, value = jax.vmap(self.model.apply, in_axes=(None, 0))(
            params, traj_batch_augmented_belief_state
        )

    @partial(jax.jit, static_argnums=(0,))
    def _update_epoch(self, update_state, unused):
        pass
