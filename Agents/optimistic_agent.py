import jax
import jax.numpy as jnp
from functools import partial
import sys

sys.path.append("..")
from Environment import CTP_environment
from Environment import CTP_generator


class Optimistic_Agent:
    def __init__(self, num_agents: int, num_nodes: int):
        self.num_agents = num_agents
        self.num_nodes = num_nodes

    @partial(jax.jit, static_argnums=(0))
    def act(self, belief_states: jnp.ndarray) -> jnp.array:
        # returns an array of actions - size equal to num_agents
        # Assume all unknown stochastic edges are not blocked
        belief_state = belief_state.at[:, 0, self.num_agents :, :].set(
            jnp.where(
                belief_state[:, 0, 1:, :] == CTP_generator.UNKNOWN,
                CTP_generator.UNBLOCKED,
                belief_state[:, 0, 1:, :],
            )
        )
        # # dijkstra expects env_state. Change blocking_prob of known blocked edges to 1.
        belief_state = belief_state.at[:, 1, 1:, :].set(
            jnp.where(
                belief_state[:, 0, 1:, :] == CTP_generator.BLOCKED,
                1,
                belief_state[:, 1, 1:, :],
            )
        )

        # If all except one agent is done, then don't need to calculate best combination
        # Only applicable for full communication!
        done_agents = jnp.where(
            jnp.sum(belief_state[0, 3, : self.num_agents, :], axis=1) > 0,
            jnp.bool_(True),
            jnp.bool_(False),
        )  # size num_agents
        # Eliminate the goals of the done agents

        # If at the goal and correspond to allocated goal, then choose service action
