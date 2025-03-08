from functools import partial
import sys

sys.path.append("..")
from Utils.invalid_action_masking import decide_validity_of_action_space
import jax.numpy as jnp
import jax
from Environment.CTP_environment import MA_CTP_General
from Utils.augmented_belief_state import get_augmented_optimistic_pessimistic_belief


# Choose a random action from the valid actions for one agent
# We want to simulate the beginning of training as much as possible, so we don't mask out servicing the same goals
def random_action(
    current_belief_state: jnp.ndarray, key: jax.random.PRNGKey
) -> jnp.array:
    valid_mask = decide_validity_of_action_space(
        current_belief_state
    )  # size num_nodes+1 with 1 if valid and -jnp.inf otherwise
    valid_mask = jnp.where(valid_mask == -jnp.inf, 0, valid_mask)
    action = jax.random.choice(key, jnp.arange(len(valid_mask)), p=valid_mask)
    return action


# Collect random trajectories for multi_agent to ensure that the autoencoder sees diverse states
class Random_Agent:
    def __init__(
        self,
        environment: MA_CTP_General,
        horizon_length: int,
        reward_exceed_horizon: float,
        num_agents: int,
    ) -> None:
        self.environment = environment
        self.horizon_length = horizon_length
        self.reward_exceed_horizon = jnp.float16(reward_exceed_horizon)
        self.num_agents = num_agents

    @partial(jax.jit, static_argnums=(0,))
    def env_step(self, runner_state, unused):
        # Collect trajectories
        (
            current_env_state,
            current_belief_states,
            key,
            timestep_in_episode,
        ) = runner_state
        action_key, env_key = jax.random.split(key, 2)
        action_keys = jax.random.split(action_key, self.num_agents)
        augmented_belief_states = get_augmented_optimistic_pessimistic_belief(
            current_belief_states
        )

        # Agent acts
        actions = jax.vmap(random_action, in_axes=(0, 0))(
            current_belief_states, action_keys
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
            new_env_state,
            new_belief_states,
            env_key,
            timestep_in_episode,
        )
        return runner_state, augmented_belief_states
