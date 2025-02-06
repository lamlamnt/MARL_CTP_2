import jax
import jax.numpy as jnp
import os
import sys

sys.path.append("..")
from Utils.optimal_combination import dijkstra_shortest_path
from Environment import CTP_generator


@jax.jit
def _get_optimistic_belief(belief_state: jnp.ndarray) -> jnp.ndarray:
    # input a belief state of one agent
    # Assume all unknown stochastic edges are not blocked
    num_agents = belief_state.shape[1] - belief_state.shape[2]
    num_nodes = belief_state.shape[2]
    optimistic_belief_state = belief_state.at[0, num_agents:, :].set(
        jnp.where(
            belief_state[0, num_agents:, :] == CTP_generator.UNKNOWN,
            CTP_generator.UNBLOCKED,
            belief_state[0, num_agents:, :],
        )
    )
    # dijkstra expects env_state. Change blocking_prob of known blocked edges to 1.
    optimistic_belief_state = optimistic_belief_state.at[1, num_agents:, :].set(
        jnp.where(
            belief_state[0, num_agents:, :] == CTP_generator.BLOCKED,
            1,
            belief_state[1, num_agents:, :],
        )
    )
    shortest_paths = jax.vmap(dijkstra_shortest_path, in_axes=(None, None, None, 0))(
        optimistic_belief_state[1, num_agents:, :],
        optimistic_belief_state[0, num_agents:, :],
        jnp.arange(num_nodes),
        jnp.arange(num_nodes),
    )

    # Replace inf with -1. Possible that a node is not reachable from another node
    shortest_paths = jnp.where(jnp.isinf(shortest_paths), -1, shortest_paths)

    empty = jnp.zeros((num_agents, num_nodes), dtype=jnp.float16)
    shortest_paths = jnp.concatenate((empty, shortest_paths), axis=0)
    return shortest_paths


@jax.jit
def _get_pessimistic_belief(belief_state: jnp.ndarray):
    num_nodes = belief_state.shape[2]
    num_agents = belief_state.shape[1] - belief_state.shape[2]
    # Assume all unknown edges are blocked
    pessimistic_belief_state = belief_state.at[0, num_agents:, :].set(
        jnp.where(
            belief_state[0, num_agents:, :] == CTP_generator.UNKNOWN,
            CTP_generator.BLOCKED,
            belief_state[0, num_agents:, :],
        )
    )
    # dijkstra expects env_state. Change blocking_prob of known blocked edges to 1.
    pessimistic_belief_state = pessimistic_belief_state.at[1, num_agents:, :].set(
        jnp.where(
            belief_state[0, num_agents:, :] == CTP_generator.BLOCKED,
            1,
            belief_state[1, num_agents:, :],
        )
    )
    pessimistic_path_lengths = jax.vmap(
        dijkstra_shortest_path, in_axes=(None, None, None, 0)
    )(
        pessimistic_belief_state[1, num_agents:, :],
        pessimistic_belief_state[0, num_agents:, :],
        jnp.arange(num_nodes),
        jnp.arange(num_nodes),
    )

    # Replace inf with -1. Possible that a node is not reachable from another node
    pessimistic_path_lengths = jnp.where(
        jnp.isinf(pessimistic_path_lengths), -1, pessimistic_path_lengths
    )
    empty = jnp.zeros((num_agents, num_nodes), dtype=jnp.float16)
    pessimistic_path_lengths = jnp.concatenate(
        (empty, pessimistic_path_lengths), axis=0
    )
    return pessimistic_path_lengths


@jax.jit
def get_augmented_optimistic_pessimistic_belief(
    belief_states: jnp.ndarray,
) -> jnp.ndarray:
    # returns the full augmented belief states for all agents
    optimistic_paths = _get_optimistic_belief(belief_states[0])
    pessimistic_paths = _get_pessimistic_belief(belief_states[0])

    def _stack(belief_state):
        return jnp.vstack(
            (
                belief_state,
                jnp.expand_dims(optimistic_paths, axis=0),
                jnp.expand_dims(pessimistic_paths, axis=0),
            ),
            dtype=jnp.float16,
        )

    augmented_belief_states = jax.vmap(_stack, in_axes=0)(belief_states)
    return augmented_belief_states
