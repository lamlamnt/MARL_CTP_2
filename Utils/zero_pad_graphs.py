from functools import partial
import jax.numpy as jnp
import numpy as np
import sys

sys.path.append("..")
from Utils.hand_crafted_graphs import (
    get_dynamic_choose_goal_graph,
    get_go_past_goal_without_servicing_graph,
    get_sacrifice_in_choosing_goals_graph,
    get_smaller_index_agent_behaves_differently_graph,
    get_go_past_goal_without_servicing_graph,
    get_sacrifice_in_exploring_graph,
)
from Environment import CTP_generator
import os
from Utils.normalize_add_expensive_edge import get_expected_optimal_total_cost
import jax
import warnings


class StoredGraphs:
    def __init__(self, n_nodes, desired_num_nodes, num_agents, normalizing_factor):
        self.n_nodes = n_nodes
        self.desired_num_nodes = desired_num_nodes
        self.num_agents = num_agents
        self.normalizing_factor = normalizing_factor

    # @partial(jax.jit, static_argnums=(0,))
    def zero_pad_graph(
        self,
        stored_graph: jnp.ndarray,
    ) -> jnp.ndarray:
        # assume origins are the first n_nodes and goals are the last n_nodes-1
        # For delauney graphs, normalizing factor should be 1
        # takes as input 1 stored graph
        weights = CTP_generator.NOT_CONNECTED * jnp.ones(
            (self.desired_num_nodes, self.desired_num_nodes), dtype=jnp.float16
        )
        blocking_prob = 1 * jnp.ones(
            (self.desired_num_nodes, self.desired_num_nodes), dtype=jnp.float16
        )

        # Make the goal nodes and special node the last 3 nodes and the buffer nodes before these
        weights = weights.at[
            : self.n_nodes - self.num_agents - 1, : self.n_nodes - self.num_agents - 1
        ].set(
            stored_graph[
                0,
                : self.n_nodes - self.num_agents - 1,
                : self.n_nodes - self.num_agents - 1,
            ]
        )
        last_bit = self.num_agents + 1
        weights = weights.at[-last_bit:, -last_bit:].set(
            stored_graph[0, -last_bit:, -last_bit:]
        )

        # Copy the edges between the rest of the graph and the goal nodes
        weights = weights.at[: self.n_nodes - self.num_agents - 1, -last_bit:].set(
            stored_graph[0, : self.n_nodes - self.num_agents - 1, -last_bit:]
        )
        weights = weights.at[-last_bit:, : self.n_nodes - self.num_agents - 1].set(
            stored_graph[0, -last_bit:, : self.n_nodes - self.num_agents - 1]
        )

        blocking_prob = blocking_prob.at[
            : self.n_nodes - self.num_agents - 1, : self.n_nodes - self.num_agents - 1
        ].set(
            stored_graph[
                1,
                : self.n_nodes - self.num_agents - 1,
                : self.n_nodes - self.num_agents - 1,
            ]
        )
        blocking_prob = blocking_prob.at[-last_bit:, -last_bit:].set(
            stored_graph[1, -last_bit:, -last_bit:]
        )
        blocking_prob = blocking_prob.at[
            : self.n_nodes - self.num_agents - 1, -last_bit:
        ].set(stored_graph[1, : self.n_nodes - self.num_agents - 1, -last_bit:])
        blocking_prob = blocking_prob.at[
            -last_bit:, : self.n_nodes - self.num_agents - 1
        ].set(stored_graph[1, -last_bit:, : self.n_nodes - self.num_agents - 1])

        # Normalize the weights. Need to turn into graph_realisation to get the normalizing factor
        weights = jnp.where(
            weights != CTP_generator.NOT_CONNECTED,
            weights / self.normalizing_factor,
            CTP_generator.NOT_CONNECTED,
        )
        goals = jnp.arange(
            self.desired_num_nodes - 2, self.desired_num_nodes - 2 - self.num_agents, -1
        )

        # Put back into stored_graph format
        stored_graph = jnp.zeros(
            (3, self.desired_num_nodes, self.desired_num_nodes), dtype=jnp.float16
        )
        stored_graph = stored_graph.at[0, :, :].set(weights)
        stored_graph = stored_graph.at[1, :, :].set(blocking_prob)
        stored_graph = stored_graph.at[2, 0, :].set(
            jnp.zeros(self.desired_num_nodes, dtype=int)
            .at[jnp.arange(self.num_agents)]
            .set(1)
        )
        stored_graph = stored_graph.at[2, 1, :].set(
            jnp.zeros(self.desired_num_nodes, dtype=int).at[goals].set(1)
        )
        return stored_graph


if __name__ == "__main__":
    # Get all handcrafted graphs -> Zero pad to make them 10 nodes -> Save in the same format as inference graphs. Separate file for each handcrafted graph
    # Delauney graphs with 5,7,9 nodes and zero pad to be 10 ->  Save in the same format as inference graphs
    # Final: separate folders with training_graphs taken from previous and new inference_graphs. Use pre-trained network (from prop 0.4) and just perform inference.
    """
    # Zero pad handcrafted graphs
    n_node, stored_graph = get_sacrifice_in_choosing_goals_graph()
    # vmapping over scalar arguments, even when their in_axis is None, is problematic -> that's why we have to do the class and self workaround
    stored_graph_maker = StoredGraphs(n_node, 10, 2, 0.8)
    zero_padded_graph = jax.vmap(stored_graph_maker.zero_pad_graph)(stored_graph)
    parent_directory = os.path.dirname(os.getcwd())
    directory = os.path.join(
        parent_directory, "Generated_graphs", "Handcrafted_sacrifice_choose_goals"
    )
    if not os.path.exists(directory):
        os.makedirs(directory)
    inference_graph_npy_file = os.path.join(directory, "inference_graphs.npy")
    np.save(inference_graph_npy_file, np.array(zero_padded_graph))
    # save a dummy training_graph file
    training_graph_npy_file = os.path.join(directory, "training_graphs.npy")
    np.save(training_graph_npy_file, np.array(zero_padded_graph))

    n_node, stored_graph = get_sacrifice_in_exploring_graph()
    stored_graph_maker = StoredGraphs(n_node, 10, 2, 10.15)
    zero_padded_graph = jax.vmap(stored_graph_maker.zero_pad_graph)(stored_graph)
    directory = os.path.join(
        parent_directory, "Generated_graphs", "Handcrafted_sacrifice_explore"
    )
    if not os.path.exists(directory):
        os.makedirs(directory)
    inference_graph_npy_file = os.path.join(directory, "inference_graphs.npy")
    np.save(inference_graph_npy_file, np.array(zero_padded_graph))
    training_graph_npy_file = os.path.join(directory, "training_graphs.npy")
    np.save(training_graph_npy_file, np.array(zero_padded_graph))

    n_node, stored_graph = get_smaller_index_agent_behaves_differently_graph()
    stored_graph_maker = StoredGraphs(n_node, 10, 2, 0.2)
    zero_padded_graph = jax.vmap(stored_graph_maker.zero_pad_graph)(stored_graph)
    directory = os.path.join(
        parent_directory,
        "Generated_graphs",
        "Handcrafted_smaller_index_behaves_differently",
    )
    if not os.path.exists(directory):
        os.makedirs(directory)
    inference_graph_npy_file = os.path.join(directory, "inference_graphs.npy")
    np.save(inference_graph_npy_file, np.array(zero_padded_graph))
    training_graph_npy_file = os.path.join(directory, "training_graphs.npy")
    np.save(training_graph_npy_file, np.array(zero_padded_graph))

    n_node, stored_graph = get_dynamic_choose_goal_graph()
    stored_graph_maker = StoredGraphs(n_node, 10, 2, 0.8)
    zero_padded_graph = jax.vmap(stored_graph_maker.zero_pad_graph)(stored_graph)
    directory = os.path.join(
        parent_directory, "Generated_graphs", "Handcrafted_dynamic_choose_goal"
    )
    if not os.path.exists(directory):
        os.makedirs(directory)
    inference_graph_npy_file = os.path.join(directory, "inference_graphs.npy")
    np.save(inference_graph_npy_file, np.array(zero_padded_graph))
    training_graph_npy_file = os.path.join(directory, "training_graphs.npy")
    np.save(training_graph_npy_file, np.array(zero_padded_graph))

    n_node, stored_graph = get_go_past_goal_without_servicing_graph()
    stored_graph_maker = StoredGraphs(n_node, 10, 2, 0.3)
    zero_padded_graph = jax.vmap(stored_graph_maker.zero_pad_graph)(stored_graph)
    directory = os.path.join(
        parent_directory,
        "Generated_graphs",
        "Handcrafted_go_past_goal_without_servicing",
    )
    if not os.path.exists(directory):
        os.makedirs(directory)
    inference_graph_npy_file = os.path.join(directory, "inference_graphs.npy")
    np.save(inference_graph_npy_file, np.array(zero_padded_graph))
    training_graph_npy_file = os.path.join(directory, "training_graphs.npy")
    np.save(training_graph_npy_file, np.array(zero_padded_graph))
    """

    # Zero pad delauney graphs
    # Read from a file. Need to create these inference graphs first (prop 0.1 or 0.2 for 5 nodes)
    parent_directory = os.path.dirname(os.getcwd())
    n_node = 9
    directory = os.path.join(
        parent_directory, "Generated_graphs", "9_nodes_2_agents_prop_0.1"
    )
    stored_graphs = np.load(os.path.join(directory, "inference_graphs.npy"))
    stored_graph_maker = StoredGraphs(n_node, 10, 2, 1)
    zero_padded_graph = jax.vmap(stored_graph_maker.zero_pad_graph)(stored_graphs)
    directory = os.path.join(
        parent_directory,
        "Generated_graphs",
        "Zero_padded_delauney_original_9_nodes_2_agents",
    )
    if not os.path.exists(directory):
        os.makedirs(directory)
    inference_graph_npy_file = os.path.join(directory, "inference_graphs.npy")
    np.save(inference_graph_npy_file, np.array(zero_padded_graph))
    training_graph_npy_file = os.path.join(directory, "training_graphs.npy")
    np.save(training_graph_npy_file, np.array(zero_padded_graph))
