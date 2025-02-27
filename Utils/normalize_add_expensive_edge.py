from functools import partial
import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_generator
from Utils import graph_functions
from Utils.optimal_combination import get_optimal_combination_and_cost

NUM_SAMPLES_FACTOR = 10


# Check solvability and add expensive edge in one go
@jax.jit
def add_expensive_edge(
    weights: jnp.ndarray,
    blocking_prob: jnp.ndarray,
    blocking_status: jnp.ndarray,
    origins: jnp.array,
    goals: jnp.array,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    # Returns new weights, blocking_prob, blocking_status
    num_agents = origins.shape[0]
    num_nodes = weights.shape[1]
    solvability_matrix = get_solvability_matrix(
        weights, blocking_status, origins, goals
    )
    # Divides by 2 because split into 2 edges
    upper_bound = jnp.max(weights) * (num_nodes - 2) / 2

    def _for_one_pair(graph, i, j):
        (weights, blocking_prob, blocking_status) = graph
        origin = origins[i]
        goal = goals[j]
        weights = weights.at[origin, -1].set(upper_bound)
        weights = weights.at[-1, origin].set(upper_bound)
        weights = weights.at[goal, -1].set(upper_bound)
        weights = weights.at[-1, goal].set(upper_bound)
        blocking_prob = blocking_prob.at[origin, -1].set(0)
        blocking_prob = blocking_prob.at[-1, origin].set(0)
        blocking_prob = blocking_prob.at[goal, -1].set(0)
        blocking_prob = blocking_prob.at[-1, goal].set(0)
        blocking_status = blocking_status.at[origin, -1].set(CTP_generator.UNBLOCKED)
        blocking_status = blocking_status.at[-1, origin].set(CTP_generator.UNBLOCKED)
        blocking_status = blocking_status.at[goal, -1].set(CTP_generator.UNBLOCKED)
        blocking_status = blocking_status.at[-1, goal].set(CTP_generator.UNBLOCKED)
        return weights, blocking_prob, blocking_status

    """
    for i in range(num_agents):
        for j in range(num_agents):
            weights, blocking_prob, blocking_status = jax.lax.cond(
                solvability_matrix[i, j],
                lambda graph: graph,
                lambda graph: _for_one_pair(graph, i, j),
                operand=(weights, blocking_prob, blocking_status),
            )
    """

    indices = jnp.array(
        [(i, j) for i in range(num_agents) for j in range(num_agents)]
    )  # Shape: (num_agents^2, 2)

    def _process_one_pair(carry, index):
        weights, blocking_prob, blocking_status = carry
        i, j = index
        updated_graph = jax.lax.cond(
            solvability_matrix[i, j],
            lambda graph: graph,  # If True, don't modify
            lambda graph: _for_one_pair(graph, i, j),  # If False, add expensive edge
            operand=(weights, blocking_prob, blocking_status),
        )
        return updated_graph, None

    initial_carry = (weights, blocking_prob, blocking_status)
    final_graph, _ = jax.lax.scan(_process_one_pair, initial_carry, indices)
    (weights, blocking_prob, blocking_status) = final_graph

    return weights, blocking_prob, blocking_status


@jax.jit
def _origin_goal_pair_is_solvable(
    weights: jnp.ndarray,
    blocking_status: jnp.ndarray,
    origin_array: jnp.ndarray,
    goal_array: jnp.ndarray,
    origin_index: int,
) -> bool:
    # origin_index is the index in the origin array, not the actual index of the node

    num_nodes = weights.shape[1]

    # Modified version of dijkstra_shortest_path
    graph = weights
    # Change the weights element where blocking_prob is 1 to -1
    graph = jnp.where(
        blocking_status == CTP_generator.BLOCKED,
        CTP_generator.NOT_CONNECTED,
        graph,
    )
    # Change all -1 elements to infinity
    graph = jnp.where(graph == CTP_generator.NOT_CONNECTED, jnp.inf, graph)

    # Initialize distances with "infinity" and visited nodes
    distances = jnp.inf * jnp.ones(num_nodes)
    distances = distances.at[origin_array[origin_index]].set(0)
    visited = jnp.zeros(num_nodes, dtype=bool)

    def body_fun(i, carry):
        distances, visited = carry

        # Find the node with the minimum distance that hasn't been visited yet
        unvisited_distances = jnp.where(visited, jnp.inf, distances)
        current_node = jnp.argmin(unvisited_distances)
        current_distance = distances[current_node]

        # Mark this node as visited
        visited = visited.at[current_node].set(True)

        # Update distances to neighboring nodes
        neighbors = graph[current_node, :]
        new_distances = jnp.where(
            (neighbors < jnp.inf) & (~visited),
            jnp.minimum(distances, current_distance + neighbors),
            distances,
        )
        return new_distances, visited

    distances, visited = jax.lax.fori_loop(0, num_nodes, body_fun, (distances, visited))

    def _check_solvability(index):
        solvable = jax.lax.cond(
            distances[goal_array[index]] == jnp.inf,
            lambda _: jnp.bool_(False),
            lambda _: jnp.bool_(True),
            operand=None,
        )
        return solvable

    solvable_array = jax.vmap(_check_solvability)(jnp.arange(goal_array.shape[0]))
    return solvable_array


@jax.jit
def get_solvability_matrix(
    weights: jnp.ndarray,
    blocking_status: jnp.ndarray,
    origins: jnp.array,
    goals: jnp.array,
) -> jnp.ndarray:
    num_agents = origins.shape[0]
    solvability_matrix = jax.vmap(
        _origin_goal_pair_is_solvable, in_axes=(None, None, None, None, 0)
    )(weights, blocking_status, origins, goals, jnp.arange(num_agents))
    return solvability_matrix


def get_expected_optimal_total_cost(
    graphRealisation: CTP_generator.CTPGraph_Realisation, key: jax.random.PRNGKey
) -> float:
    # Sample blocking status for several times and get the average path length
    num_agents = graphRealisation.graph.num_agents
    num_samples = NUM_SAMPLES_FACTOR * graphRealisation.graph.n_nodes
    many_keys = jax.random.split(key, num=num_samples)

    def get_sampled_optimal_cost_for_one_instance(key):
        blocking_status = graphRealisation.sample_blocking_status(key)
        graph_weights, graph_blocking_prob, blocking_status = add_expensive_edge(
            graphRealisation.graph.weights,
            graphRealisation.graph.blocking_prob,
            blocking_status,
            graphRealisation.graph.origin,
            graphRealisation.graph.goal,
        )
        _, one_normalizing_factor = get_optimal_combination_and_cost(
            graph_weights,
            blocking_status,
            graphRealisation.graph.origin,
            graphRealisation.graph.goal,
            num_agents,
        )
        return one_normalizing_factor

    normalizing_factor = jax.vmap(get_sampled_optimal_cost_for_one_instance)(many_keys)
    return jnp.mean(normalizing_factor, dtype=jnp.float16)
