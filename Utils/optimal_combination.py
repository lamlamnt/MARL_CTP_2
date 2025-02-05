import jax
import jax.numpy as jnp
from functools import partial
from jax import jit, lax, random, value_and_grad, vmap
import sys

sys.path.append("..")
from Environment import CTP_generator
from Environment import CTP_environment
import itertools


# Given the env_state, return the allocated-goal for each agent and the total optimal cost
@partial(jit, static_argnums=(4,))
def get_optimal_combination_and_cost(
    weights: jnp.ndarray,
    blocking_status: jnp.ndarray,
    origin_array: jnp.array,
    goals: jnp.array,
    num_agents: int,
) -> tuple[jnp.array, float]:
    # Origin_array and goals have size num_agents
    # Get basic info
    # goals has size num_agents

    # Get the distance from each origin to each goal - vmap across origins. (n^2)
    distances = jax.vmap(_dijkstra_shortest_path, in_axes=(None, None, None, 0))(
        weights,
        blocking_status,
        origin_array,
        jnp.arange(num_agents),
    )  # size num_agents x num_nodes

    distances = distances[
        :, goals
    ]  # size num_agents x num_agents with row corresponding to agent and column corresponding to goal

    # Get the best combination (n!)
    # Get permutation
    permutations = jnp.array(list(itertools.permutations(jnp.arange(num_agents))))

    # Sum up the distances for each permutation
    selected_distances = jnp.take_along_axis(distances, permutations, axis=1)
    total_distances = jnp.sum(selected_distances, axis=0)
    best_combination_index = jnp.argmin(total_distances)
    best_combination_cost = jnp.min(total_distances)
    return permutations[best_combination_index], best_combination_cost


# Return distance from an origin to all other nodes
def _dijkstra_shortest_path(
    weights: jnp.ndarray,
    blocking_status: jnp.ndarray,
    origin_array: jnp.ndarray,
    origin_index: int,
) -> jnp.ndarray:
    graph = weights
    num_nodes = graph.shape[1]
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
    return distances


@jax.jit
def dijkstra_with_path(
    env_state: jnp.ndarray, origin: int, goal: int
) -> tuple[int, int]:
    # returns shortest path and next node for a single origin and goal
    num_nodes = env_state.shape[2]
    num_agents = env_state.shape[1] - num_nodes
    graph = env_state[1, num_agents:, :]
    # Change the weights element where blocking_prob is 1 to -1
    graph = jnp.where(
        env_state[0, num_agents:, :] == CTP_generator.BLOCKED,
        CTP_generator.NOT_CONNECTED,
        graph,
    )
    # Change all -1 elements to infinity
    graph = jnp.where(graph == CTP_generator.NOT_CONNECTED, jnp.inf, graph)

    # Initialize distances with "infinity" and visited nodes
    distances = jnp.inf * jnp.ones(num_nodes, dtype=jnp.float16)
    distances = distances.at[origin].set(0)
    visited = jnp.zeros(num_nodes, dtype=bool)
    predecessors = -jnp.ones(num_nodes, dtype=int)  # -1 means no predecessor

    def body_fun(i, carry):
        distances, visited, predecessors = carry

        # Find the node with the minimum distance that hasn't been visited yet
        unvisited_distances = jnp.where(visited, jnp.inf, distances)
        current_node = jnp.argmin(unvisited_distances)
        current_distance = distances[current_node]

        # Mark this node as visited
        visited = visited.at[current_node].set(True)

        # Update distances and predecessors for neighboring nodes
        neighbors = graph[current_node, :]
        updates = (neighbors < jnp.inf) & (~visited)  # Unvisited neighbors
        new_distances = jnp.where(
            updates, jnp.minimum(distances, current_distance + neighbors), distances
        )
        predecessors = jnp.where(
            updates & (new_distances < distances),
            current_node,
            predecessors,
        )
        return new_distances, visited, predecessors

    # Run the loop with `jax.lax.fori_loop`
    distances, visited, predecessors = jax.lax.fori_loop(
        0, num_nodes, body_fun, (distances, visited, predecessors)
    )

    # Reconstruct the path from the goal to the origin and get the next node
    def get_next_node(goal_node, preds):
        def body_fn(carry):
            path, current, index = carry
            path = path.at[index].set(current)  # Set current node in the path
            current = preds[current]  # Move to the predecessor
            return path, current, index + 1

        def cond_fn(carry):
            _, current, index = carry
            return current != -1  # Continue while the current node is not -1

        # Initialize the path and fill with a sentinel value (-1)
        path = jnp.full(preds.shape[0], -1, dtype=preds.dtype)
        path, current_node, index = jax.lax.while_loop(
            cond_fn, body_fn, (path, goal_node, 0)
        )

        # If want to get the full path,
        # path = jax.lax.slice(path,(0,),(index,))
        # return path[::-1]
        return path[index - 2]

    next_node = get_next_node(goal, predecessors)
    return distances[goal], next_node
