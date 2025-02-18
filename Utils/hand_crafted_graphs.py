import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment.CTP_environment import MA_CTP_General
from Environment import CTP_generator


def get_sacrifice_in_choosing_goals_graph() -> tuple[int, dict]:
    senders = jnp.array([0, 0, 0, 2])
    receivers = jnp.array([1, 2, 3, 3])
    weights = jnp.array(
        [
            [CTP_generator.NOT_CONNECTED, 0.1, 0.1, 0.2, CTP_generator.NOT_CONNECTED],
            [
                0.1,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
            ],
            [
                0.1,
                CTP_generator.NOT_CONNECTED,
                CTP_generator.NOT_CONNECTED,
                0.5,
                CTP_generator.NOT_CONNECTED,
            ],
            [
                0.2,
                CTP_generator.NOT_CONNECTED,
                0.5,
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
        ]
    )
    node_pos = jnp.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 1]])
    blocking_prob = jnp.array(
        [
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 0, 1],
            [0, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    goals = jnp.array([2, 3])
    origins = jnp.array([0, 1])
    n_nodes = 5
    n_edges = 4
    defined_graph = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "senders": senders,
        "receivers": receivers,
        "weights": weights,
        "node_pos": node_pos,
        "blocking_prob": blocking_prob,
        "goals": goals,
        "origins": origins,
    }
    return (n_nodes, defined_graph)


def get_sacrifice_in_exploring_graph() -> tuple[int, dict]:
    senders = jnp.array([0, 0, 1, 2, 3])
    receivers = jnp.array([2, 4, 3, 3, 5])
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
        ]
    )
    node_pos = jnp.array([[0, 0], [2, -1], [1, 1], [2, 0], [0, 2], [2, 2], [0, -1]])
    blocking_prob = jnp.array(
        [
            [1, 1, 0, 1, 0, 1, 1],
            [1, 1, 1, 0, 1, 1, 1],
            [0, 1, 1, 0, 1, 0.5, 1],
            [1, 0, 0, 1, 1, 0, 1],
            [0, 1, 1, 1, 1, 1, 1],
            [1, 1, 0.5, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ]
    )
    goals = jnp.array([4, 5])
    origins = jnp.array([0, 1])
    n_nodes = 7
    n_edges = 6
    defined_graph = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "senders": senders,
        "receivers": receivers,
        "weights": weights,
        "node_pos": node_pos,
        "blocking_prob": blocking_prob,
        "goals": goals,
        "origins": origins,
    }
    return (n_nodes, defined_graph)


def get_smaller_index_agent_behaves_differently_graph() -> tuple[int, dict]:
    senders = jnp.array([0, 0, 1, 1])
    receivers = jnp.array([2, 3, 2, 3])
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
        ]
    )
    node_pos = jnp.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 1]])
    blocking_prob = jnp.array(
        [
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    goals = jnp.array([2, 3])
    origins = jnp.array([0, 1])
    n_nodes = 5
    n_edges = 4
    defined_graph = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "senders": senders,
        "receivers": receivers,
        "weights": weights,
        "node_pos": node_pos,
        "blocking_prob": blocking_prob,
        "goals": goals,
        "origins": origins,
    }
    return (n_nodes, defined_graph)


def get_go_past_goal_without_servicing_graph() -> tuple[int, dict]:
    senders = jnp.array([0, 1, 2])
    receivers = jnp.array([2, 2, 3])
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
        ]
    )
    node_pos = jnp.array([[0, 0], [2, 0], [1, 1], [1, 2], [0, 2]])
    blocking_prob = jnp.array(
        [
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [0, 0, 1, 0, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    goals = jnp.array([2, 3])
    origins = jnp.array([0, 1])
    n_nodes = 5
    n_edges = 3
    defined_graph = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "senders": senders,
        "receivers": receivers,
        "weights": weights,
        "node_pos": node_pos,
        "blocking_prob": blocking_prob,
        "goals": goals,
        "origins": origins,
    }
    return (n_nodes, defined_graph)
