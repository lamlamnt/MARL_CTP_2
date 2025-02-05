import jax
import jax.numpy as jnp
from functools import partial
import sys

sys.path.append("..")
from Environment import CTP_environment
from Environment import CTP_generator
from Utils.optimal_combination import (
    get_optimal_combination_and_cost,
    dijkstra_with_path,
)


class Optimistic_Agent:
    def __init__(self, num_agents: int, num_nodes: int):
        self.num_agents = num_agents
        self.num_nodes = num_nodes

    @partial(jax.jit, static_argnums=(0))
    def allocate_goals(self, belief_state: jnp.ndarray) -> jnp.array:
        # use the first agent's belief state to allocate goals
        # allocate goals at the beginning of the episode
        # Assume all unknown stochastic edges are not blocked
        belief_state = belief_state.at[0, self.num_agents :, :].set(
            jnp.where(
                belief_state[0, self.num_agents :, :] == CTP_generator.UNKNOWN,
                CTP_generator.UNBLOCKED,
                belief_state[0, self.num_agents :, :],
            )
        )
        # # dijkstra expects env_state. Change blocking_prob of known blocked edges to 1.
        belief_state = belief_state.at[1, self.num_agents :, :].set(
            jnp.where(
                belief_state[0, self.num_agents :, :] == CTP_generator.BLOCKED,
                1,
                belief_state[1, self.num_agents :, :],
            )
        )
        _, goals = jax.lax.top_k(
            jnp.diag(belief_state[3, self.num_agents :, :]), self.num_agents
        )
        one_origin = jnp.argmax(belief_state[0, 0, :])
        other_origins = jnp.argmax(belief_state[1, self.num_agents :, :], axis=1)
        origins = other_origins.at[0].set(one_origin)
        allocated_goals = get_optimal_combination_and_cost(
            belief_state[1, self.num_agents :, :],
            belief_state[0, self.num_agents :, :],
            origins,
            goals,
            self.num_agents,
        )  # size num_agents
        return allocated_goals

    # for one agent at a time (though if done together, would be slightly more efficient)
    @partial(jax.jit, static_argnums=(0))
    def act(
        self,
        belief_state: jnp.ndarray,
        pre_allocated_goals: jnp.array,
        agent_index: int,
    ) -> int:
        # returns an array of actions - size equal to num_agents
        # Assume all unknown stochastic edges are not blocked
        # agent_index = jnp.argmin(jnp.sum(belief_state[1, self.num_agents :, :], axis=1))
        belief_state = belief_state.at[0, self.num_agents :, :].set(
            jnp.where(
                belief_state[0, self.num_agents :, :] == CTP_generator.UNKNOWN,
                CTP_generator.UNBLOCKED,
                belief_state[0, self.num_agents :, :],
            )
        )
        # # dijkstra expects env_state. Change blocking_prob of known blocked edges to 1.
        belief_state = belief_state.at[1, self.num_agents :, :].set(
            jnp.where(
                belief_state[0, self.num_agents :, :] == CTP_generator.BLOCKED,
                1,
                belief_state[1, self.num_agents :, :],
            )
        )
        done = jax.lax.cond(
            jnp.sum(belief_state[3, agent_index, :]) > 0,
            lambda x: jnp.bool_(True),
            lambda x: jnp.bool_(False),
            None,
        )
        current_node = jnp.argmax(belief_state[0, agent_index, :])
        # If agent is done or at the goal corresponding to the allocated goal, then choose service action
        # Else, Dijkstra with path
        pre_allocated_goals = jnp.asarray(pre_allocated_goals)
        action = jax.lax.cond(
            jnp.logical_or(
                done, jnp.equal(current_node, pre_allocated_goals[agent_index])
            ),
            lambda x: self.num_nodes,
            lambda x: dijkstra_with_path(
                belief_state, current_node, pre_allocated_goals[agent_index]
            ),
            None,
        )
        return action
