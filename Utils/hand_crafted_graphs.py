import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment.CTP_environment import MA_CTP_General
from Environment import CTP_generator


# Need the graphs in loaded_graph format
def get_sacrifice_in_choosing_goals_graph() -> tuple[int, jnp.ndarray]:
    weights = jnp.array(
        [
            [
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                0.1,
                0.3,
                CTP_generator.NOT_CONNECTED,
            ],
            [
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                0.5,
                1.0,
                CTP_generator.NOT_CONNECTED,
            ],
            [
                0.1,
                0.5,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
            ],
            [
                0.3,
                1.0,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
            ],
            [
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
            ],
        ],
        dtype=jnp.float16,
    )
    blocking_prob = jnp.array(
        [
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=jnp.float16,
    )
    goals = jnp.array([2, 3])
    origins = jnp.array([0, 1])
    n_nodes = 5
    stored_graph = jnp.zeros((1, 3, n_nodes, n_nodes), dtype=jnp.float16)
    stored_graph = stored_graph.at[0, 0, :, :].set(weights)
    stored_graph = stored_graph.at[0, 1, :, :].set(blocking_prob)
    stored_graph = stored_graph.at[0, 2, 0, :].set(
        jnp.zeros(n_nodes, dtype=int).at[origins].set(1)
    )
    stored_graph = stored_graph.at[0, 2, 1, :].set(
        jnp.zeros(n_nodes, dtype=int).at[goals].set(1)
    )
    return (n_nodes, stored_graph)


def get_sacrifice_in_exploring_graph() -> tuple[int, jnp.ndarray]:
    weights = jnp.array(
        [
            [
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                0.05,
                CTP_generator.NOT_CONNECTED,
                0.1,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
            ],
            [
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                0.05,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
            ],
            [
                0.05,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                0.4,
                CTP_generator.NOT_CONNECTED,
                0.1,
                CTP_generator.NOT_CONNECTED,
            ],
            [
                CTP_generator.NOT_CONNECTED,
                0.05,
                0.4,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                1.0,
                CTP_generator.NOT_CONNECTED,
            ],
            [
                0.1,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
            ],
            [
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                0.1,
                1.0,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
            ],
            [
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
            ],
        ],
        dtype=jnp.float16,
    )
    blocking_prob = jnp.array(
        [
            [1, 1, 0, 1, 0, 1, 1],
            [1, 1, 1, 0, 1, 1, 1],
            [0, 1, 1, 0, 1, 0.9, 1],
            [1, 0, 0, 1, 1, 0, 1],
            [0, 1, 1, 1, 1, 1, 1],
            [1, 1, 0.9, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=jnp.float16,
    )
    goals = jnp.array([4, 5])
    origins = jnp.array([0, 1])
    n_nodes = 7
    stored_graph = jnp.zeros((1, 3, n_nodes, n_nodes), dtype=jnp.float16)
    stored_graph = stored_graph.at[0, 0, :, :].set(weights)
    stored_graph = stored_graph.at[0, 1, :, :].set(blocking_prob)
    stored_graph = stored_graph.at[0, 2, 0, :].set(
        jnp.zeros(n_nodes, dtype=int).at[origins].set(1)
    )
    stored_graph = stored_graph.at[0, 2, 1, :].set(
        jnp.zeros(n_nodes, dtype=int).at[goals].set(1)
    )
    return (n_nodes, stored_graph)


def get_smaller_index_agent_behaves_differently_graph() -> tuple[int, dict]:
    weights = jnp.array(
        [
            [
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                0.1,
                0.1,
                CTP_generator.NOT_CONNECTED,
            ],
            [
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                0.1,
                0.1,
                CTP_generator.NOT_CONNECTED,
            ],
            [
                0.1,
                0.1,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
            ],
            [
                0.1,
                0.1,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
            ],
            [
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
            ],
        ],
        dtype=jnp.float16,
    )
    blocking_prob = jnp.array(
        [
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=jnp.float16,
    )
    goals = jnp.array([2, 3])
    origins = jnp.array([0, 1])
    n_nodes = 5
    stored_graph = jnp.zeros((1, 3, n_nodes, n_nodes), dtype=jnp.float16)
    stored_graph = stored_graph.at[0, 0, :, :].set(weights)
    stored_graph = stored_graph.at[0, 1, :, :].set(blocking_prob)
    stored_graph = stored_graph.at[0, 2, 0, :].set(
        jnp.zeros(n_nodes, dtype=int).at[origins].set(1)
    )
    stored_graph = stored_graph.at[0, 2, 1, :].set(
        jnp.zeros(n_nodes, dtype=int).at[goals].set(1)
    )
    return (n_nodes, stored_graph)


def get_go_past_goal_without_servicing_graph() -> tuple[int, dict]:
    weights = jnp.array(
        [
            [
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                0.1,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
            ],
            [
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                0.1,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
            ],
            [0.1, 0.1, CTP_generator.NOT_CONNECTED, 0.1, CTP_generator.NOT_CONNECTED],
            [
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                0.1,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
            ],
            [
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
            ],
        ],
        dtype=jnp.float16,
    )
    blocking_prob = jnp.array(
        [
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [0, 0, 1, 0, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=jnp.float16,
    )
    goals = jnp.array([2, 3])
    origins = jnp.array([0, 1])
    n_nodes = 5
    stored_graph = jnp.zeros((1, 3, n_nodes, n_nodes), dtype=jnp.float16)
    stored_graph = stored_graph.at[0, 0, :, :].set(weights)
    stored_graph = stored_graph.at[0, 1, :, :].set(blocking_prob)
    stored_graph = stored_graph.at[0, 2, 0, :].set(
        jnp.zeros(n_nodes, dtype=int).at[origins].set(1)
    )
    stored_graph = stored_graph.at[0, 2, 1, :].set(
        jnp.zeros(n_nodes, dtype=int).at[goals].set(1)
    )
    return (n_nodes, stored_graph)
