import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_generator
import networkx as nx


# Plot the realised graph given env state
# Get senders and receivers and node pos and create graph realisation using hand-crafted graph
# Not jax-jittable
# Each time this is called, the plotted graph will look slightly different - unless we set fixed seed
# for networkx.
# Very difficult to visualize because of the positions of the nodes
# The last goal node is not highlighted green because the special node gets put on top of it - but it's ok
def plot_realised_graph_from_env_state(env_state: jnp.ndarray, log_directory: str):
    num_nodes = env_state.shape[2]
    num_agents = env_state.shape[1] - env_state.shape[2]
    # include blocked edges as well
    senders, receivers = jnp.where(
        env_state[2, num_agents:, :] < 1
    )  # blocking probability is less than 1
    # Need to sort them so that the sender is a smaller number than the receiver
    edges = jnp.stack([senders, receivers], axis=1)
    senders = jnp.minimum(edges[:, 0], edges[:, 1])
    receivers = jnp.maximum(edges[:, 0], edges[:, 1])
    edges = jnp.stack([senders, receivers], axis=1)
    unique_edges = jnp.unique(edges, axis=0)
    unique_senders = unique_edges[:, 0]
    unique_receivers = unique_edges[:, 1]
    G = nx.DiGraph()
    G.add_edges_from(zip(unique_senders.tolist(), unique_receivers.tolist()))
    # add an extra edge for the last node (for expensive edge) so that it doesn't get the same position as node 8
    # G.add_edges_from([(num_nodes - 2, num_nodes - 1)])
    # G.add_edges_from([(num_nodes - 3, num_nodes - 1)])
    # G.add_edges_from([(0, num_nodes - 1)])
    node_positions = nx.spring_layout(G)
    node_positions = jnp.vstack(list(node_positions.values()))
    # node positions in the range -1 and 1 -> scale to grid size?
    key = jax.random.PRNGKey(30)
    handcrafted_graph = dict()
    handcrafted_graph["senders"] = unique_senders
    handcrafted_graph["receivers"] = unique_receivers
    handcrafted_graph["node_pos"] = node_positions
    handcrafted_graph["weights"] = env_state[1, num_agents:, :]
    handcrafted_graph["blocking_prob"] = env_state[2, num_agents:, :]
    handcrafted_graph["n_edges"] = len(unique_senders)
    origins = jnp.argmax(env_state[0, :num_agents, :], axis=1)
    handcrafted_graph["origin"] = origins
    _, goals = jax.lax.top_k(jnp.diag(env_state[3, num_agents:, :]), num_agents)
    handcrafted_graph["goal"] = goals.ravel()
    # doesn't matter what key we put
    graph_realisation = CTP_generator.CTPGraph_Realisation(
        key, num_nodes, handcrafted_graph=handcrafted_graph
    )
    graph_realisation.plot_realised_graph(
        env_state[0, num_agents:, :], log_directory, file_name="from_env_state"
    )
