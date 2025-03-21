import argparse
import jax.numpy as jnp
import jax
import sys

sys.path.append("..")
from Environment import CTP_generator
import os
import numpy as np


def get_Sioux_Falls_Network(
    original_key, prop_stoch: int, num_graphs
) -> tuple[int, dict]:
    n_nodes = 20
    goals = jnp.array([17, 18])
    origins = jnp.array([0, 1])
    nc = CTP_generator.NOT_CONNECTED

    weights = jnp.ones((n_nodes, n_nodes), dtype=jnp.float16) * nc
    # 27 edges -> 54 lines
    weights = weights.at[0, 2].set(63)
    weights = weights.at[2, 0].set(63)
    weights = weights.at[0, 3].set(65.5)
    weights = weights.at[3, 0].set(65.5)
    weights = weights.at[0, 4].set(60)
    weights = weights.at[4, 0].set(60)
    weights = weights.at[0, 9].set(58)
    weights = weights.at[9, 0].set(58)
    weights = weights.at[3, 5].set(56.5)
    weights = weights.at[5, 3].set(56.5)
    weights = weights.at[2, 3].set(67)
    weights = weights.at[3, 2].set(67)
    weights = weights.at[4, 5].set(51)
    weights = weights.at[5, 4].set(51)
    weights = weights.at[5, 10].set(35.5)
    weights = weights.at[10, 5].set(35.5)
    weights = weights.at[1, 2].set(70.5)
    weights = weights.at[2, 1].set(70.5)
    weights = weights.at[1, 6].set(74.5)
    weights = weights.at[6, 1].set(74.5)
    weights = weights.at[6, 7].set(56.5)
    weights = weights.at[7, 6].set(56.5)
    weights = weights.at[7, 14].set(37)
    weights = weights.at[14, 7].set(37)
    weights = weights.at[14, 15].set(20.5)
    weights = weights.at[15, 14].set(20.5)
    weights = weights.at[18, 16].set(10)
    weights = weights.at[16, 18].set(10)
    weights = weights.at[16, 17].set(13.5)
    weights = weights.at[17, 16].set(13.5)
    weights = weights.at[4, 8].set(55.5)
    weights = weights.at[8, 4].set(55.5)
    weights = weights.at[8, 11].set(50.5)
    weights = weights.at[11, 8].set(50.5)
    weights = weights.at[10, 11].set(38.5)
    weights = weights.at[11, 10].set(38.5)
    weights = weights.at[8, 10].set(40.5)
    weights = weights.at[10, 8].set(40.5)
    weights = weights.at[10, 13].set(25.5)
    weights = weights.at[13, 10].set(25.5)
    weights = weights.at[12, 13].set(22.5)
    weights = weights.at[13, 12].set(22.5)
    weights = weights.at[11, 12].set(18.5)
    weights = weights.at[12, 11].set(18.5)
    weights = weights.at[9, 11].set(52.5)
    weights = weights.at[11, 9].set(52.5)
    weights = weights.at[9, 15].set(36)
    weights = weights.at[15, 9].set(36)
    weights = weights.at[12, 15].set(18.5)
    weights = weights.at[15, 12].set(18.5)
    weights = weights.at[13, 16].set(18)
    weights = weights.at[16, 13].set(18)
    weights = weights.at[12, 17].set(17.5)
    weights = weights.at[17, 12].set(17.5)

    # Check symmetry
    assert jnp.all(weights == weights.T)
    senders = jnp.array(
        [
            0,
            0,
            0,
            0,
            1,
            1,
            2,
            3,
            4,
            4,
            5,
            6,
            7,
            8,
            8,
            9,
            9,
            10,
            10,
            11,
            12,
            12,
            12,
            13,
            14,
            16,
            16,
        ]
    )
    receivers = jnp.array(
        [
            2,
            3,
            4,
            9,
            2,
            6,
            3,
            5,
            5,
            8,
            10,
            7,
            14,
            10,
            11,
            11,
            15,
            11,
            13,
            12,
            13,
            15,
            17,
            16,
            15,
            17,
            18,
        ]
    )
    # assert senders and receivers are correct
    for i in range(len(senders)):
        if weights[senders[i], receivers[i]] == nc:
            print(senders[i], receivers[i])
        assert weights[senders[i], receivers[i]] != nc
    n_edges = 27
    assert n_edges == len(senders)
    assert n_edges == len(receivers)

    # move import inside function to prevent circular import
    from Utils.normalize_add_expensive_edge import get_expected_optimal_total_cost

    def get_one_stored_graph(key) -> tuple[jnp.ndarray, jnp.ndarray]:
        # get blocking prob matrix
        def __assign_prob_edge(subkey, is_stochastic_edge):
            prob = jax.random.uniform(subkey, minval=0.0, maxval=1.0)
            prob = jnp.round(prob, 2)  # Round to 2 decimal places
            return jax.lax.cond(is_stochastic_edge, lambda _: prob, lambda _: 0.0, prob)

        # Assign blocking probability to each edge
        num_stoch_edges = jnp.round(prop_stoch * n_edges).astype(int)

        stoch_edge_idx = jax.random.choice(
            key, n_edges, shape=(num_stoch_edges,), replace=False
        )
        edge_indices = jnp.arange(n_edges)
        keys = jax.random.split(key, num=n_edges)
        is_stochastic_edges = jnp.isin(edge_indices, stoch_edge_idx)
        edge_probs = jax.vmap(__assign_prob_edge, in_axes=(0, 0))(
            keys, is_stochastic_edges
        )
        edge_probs = jnp.asarray(edge_probs, dtype=jnp.float16)

        blocking_prob_matrix = jnp.full((n_nodes, n_nodes), 1.0, dtype=jnp.float16)
        for i in range(n_edges):
            blocking_prob_matrix = blocking_prob_matrix.at[
                senders[i], receivers[i]
            ].set(edge_probs[i])
            blocking_prob_matrix = blocking_prob_matrix.at[
                receivers[i], senders[i]
            ].set(edge_probs[i])

        # turn into graph_realisation
        node_positions = jnp.ones(
            (n_nodes, 2), dtype=jnp.float16
        )  # just dummy because we don't need node positions
        key = jax.random.PRNGKey(30)
        handcrafted_graph = dict()
        handcrafted_graph["senders"] = senders
        handcrafted_graph["receivers"] = receivers
        handcrafted_graph["node_pos"] = node_positions
        handcrafted_graph["weights"] = weights
        handcrafted_graph["blocking_prob"] = blocking_prob_matrix
        handcrafted_graph["n_edges"] = n_edges
        handcrafted_graph["origin"] = origins
        handcrafted_graph["goal"] = goals
        # doesn't matter what key we put
        graph_realisation = CTP_generator.CTPGraph_Realisation(
            key, n_nodes, handcrafted_graph=handcrafted_graph
        )

        # Normalize the weights
        normalizing_factor = get_expected_optimal_total_cost(graph_realisation, key)
        normalized_weights = jnp.where(
            weights != CTP_generator.NOT_CONNECTED,
            weights / normalizing_factor,
            CTP_generator.NOT_CONNECTED,
        )

        one_stored_graph = jnp.zeros((3, n_nodes, n_nodes), dtype=jnp.float16)
        one_stored_graph = one_stored_graph.at[0, :, :].set(normalized_weights)
        one_stored_graph = one_stored_graph.at[1, :, :].set(blocking_prob_matrix)
        one_stored_graph = one_stored_graph.at[2, 0, :].set(
            jnp.zeros(n_nodes, dtype=int).at[origins].set(1)
        )
        one_stored_graph = one_stored_graph.at[2, 1, :].set(
            jnp.zeros(n_nodes, dtype=int).at[goals].set(1)
        )
        return one_stored_graph

    # Vmap for 2000 graphs
    stored_graphs = jax.vmap(get_one_stored_graph)(
        jax.random.split(original_key, num_graphs)
    )
    return stored_graphs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed_training", type=int, default=30, help="Seed for random number"
    )
    parser.add_argument(
        "--seed_inference", type=int, default=31, help="Seed for random number"
    )
    parser.add_argument(
        "--num_training_graphs",
        type=int,
        default=2000,
        help="Number of graphs to generate",
    )
    parser.add_argument(
        "--num_inference_graphs",
        type=int,
        default=667,
        help="Number of graphs to generate",
    )
    parser.add_argument(
        "--folder_name",
        type=str,
        default="Sioux_Falls_1",
        help="Name of the folder to save the generated graphs",
    )
    args = parser.parse_args()
    key_training = jax.random.PRNGKey(args.seed_training)
    training_graphs = get_Sioux_Falls_Network(
        key_training, 0.8, args.num_training_graphs
    )
    parent_directory = os.path.dirname(os.getcwd())
    directory = os.path.join(parent_directory, "Generated_graphs", args.folder_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    training_graph_npy_file = os.path.join(directory, "training_graphs.npy")
    np.save(training_graph_npy_file, np.array(training_graphs))

    key_testing = jax.random.PRNGKey(args.seed_inference)
    inference_graphs = get_Sioux_Falls_Network(
        key_testing, 0.8, args.num_inference_graphs
    )
    inference_graph_npy_file = os.path.join(directory, "inference_graphs.npy")
    np.save(inference_graph_npy_file, np.array(inference_graphs))
