from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from Utils.augmented_belief_state import get_augmented_optimistic_pessimistic_belief
from Environment.CTP_environment import MA_CTP_General
import flax.linen as nn


class PPO_agent_collect_belief_states:
    def __init__(
        self,
        model: nn.Module,
        environment: MA_CTP_General,
        horizon_length: int,
        reward_exceed_horizon: float,
        num_agents: int,
    ) -> None:
        self.model = model
        self.environment = environment
        self.horizon_length = horizon_length
        self.reward_exceed_horizon = jnp.float16(reward_exceed_horizon)
        self.num_agents = num_agents

    # agent-like for autoencoder
    @partial(jax.jit, static_argnums=(0,))
    def act(self, key, params, belief_states) -> tuple[jnp.array, jax.random.PRNGKey]:
        augmented_belief_states = get_augmented_optimistic_pessimistic_belief(
            belief_states
        )

        def _choose_action(belief_state):
            pi, _ = self.model.apply(params, belief_state)
            random_action = pi.sample(
                seed=key
            )  # use the same key for all agents (maybe not good)
            return random_action

        # Because we want diverse states, so will use non-deterministic inference policy
        actions = jax.vmap(_choose_action, in_axes=0)(augmented_belief_states)
        old_key, new_key = jax.random.split(key)
        return actions, new_key

    @partial(jax.jit, static_argnums=(0,))
    def env_step(self, runner_state, unused):
        # Collect trajectories
        (
            ppo_train_state,
            autoencoder_train_state,
            current_env_state,
            current_belief_states,
            key,
            timestep_in_episode,
            previous_episode_dones,
        ) = runner_state
        action_key, env_key = jax.random.split(key, 2)

        # Agent acts
        augmented_belief_states = get_augmented_optimistic_pessimistic_belief(
            current_belief_states
        )

        def _choose_action(belief_state):
            pi, critic_value = self.model.apply(ppo_train_state.params, belief_state)
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
        reward_agent_exceed_horizon = jnp.where(
            dones, jnp.float16(0), self.reward_exceed_horizon
        )

        # Reset if exceed horizon length. Otherwise, increment
        new_env_state, new_belief_states, rewards, timestep_in_episode, dones = (
            jax.lax.cond(
                timestep_in_episode >= self.horizon_length,
                lambda _: (
                    *self.environment.reset(reset_key),
                    reward_agent_exceed_horizon,
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
        episode_done = jnp.all(dones)
        runner_state = (
            ppo_train_state,
            autoencoder_train_state,
            new_env_state,
            new_belief_states,
            env_key,
            timestep_in_episode,
            episode_done,
        )
        return runner_state, current_belief_states
