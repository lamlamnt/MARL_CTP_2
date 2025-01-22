import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import Delaunay
from functools import partial
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
from typing import Literal, Final

Status = Literal[1, -1, 0]
UNBLOCKED: Status = 0
BLOCKED: Status = 1
UNKNOWN: Status = -1

NOT_CONNECTED: Final[int] = -1


class CTPGraph:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        n_nodes: int,
        grid_size=None,
        prop_stoch=None,
        k_edges=None,
        num_agents=2,
        handcrafted_graph=None,
    ):
        """
        List of properties:
        n_nodes: Number of nodes in the graph
        n_edges: Number of edges in the graph
        weights: Adjacency matrix of the graph
        senders: Senders of the edges (is redundant but make some things easier) - don't have to be ordered
        receivers: Receivers of the edges (is redundant but make some things easier)
        node_pos: Position of the nodes in the grid
        blocking_prob: Blocking probability of the edges in the graph
        goal: Array of goal nodes in the graph (with length equal to n_agents)
        origin: Array of origin nodes in the graph (with length equal to n_agents)
        num_agents: Number of agents, also equal to the number of goals and origins
        """
        self.n_nodes = n_nodes
        self.num_agents = num_agents
        if handcrafted_graph is not None:
            self.weights = handcrafted_graph["weights"]
            self.senders = handcrafted_graph["senders"]
            self.receivers = handcrafted_graph["receivers"]
            self.node_pos = handcrafted_graph["node_pos"]
            self.blocking_prob = handcrafted_graph["blocking_prob"]
            self.origin = jnp.array([handcrafted_graph["origin"]])
            self.goal = jnp.array([handcrafted_graph["goal"]])
            self.n_edges = handcrafted_graph["n_edges"]
        else:
            if grid_size == None:
                grid_size = n_nodes
            if (prop_stoch is None and k_edges is None) or (
                prop_stoch is not None and k_edges is not None
            ):
                raise ValueError(
                    "Either prop_stoch or k_edges (but not both) must be specified"
                )
            # Technically we don't need to store senders and receivers but storing it to make some operations, such as setting blocking prob easier
            self.weights, self.n_edges, self.senders, self.receivers, self.node_pos = (
                self.__generate_connectivity_weight(key, grid_size)
            )
            key, subkey = jax.random.split(key)
            self.blocking_prob = self.set_blocking_prob(
                subkey,
                prop_stoch=prop_stoch,
                k_edges=k_edges,
            )

            # Set origin and goal nodes
            self.origin = jnp.arange(0, self.num_agents)
            self.goal = jnp.arange(self.n_nodes - self.num_agents - 1, self.n_nodes - 1)

        # Initial normalization
        max_weight = jnp.max(self.weights)
        self.weights = jnp.where(
            self.weights != NOT_CONNECTED, self.weights / max_weight, NOT_CONNECTED
        )

    # Returns the weight adjacency matrix and n_edges
    def __generate_connectivity_weight(
        self, key, grid_size
    ) -> tuple[jnp.ndarray, int, jnp.array, jnp.array, jnp.ndarray]:
        def __convert_to_grid(i, ymax):
            return (i // (ymax + 1), i % (ymax + 1))

        def __on_same_line(grid_nodes):
            # Extract x and y coordinates
            x_coords, y_coords = grid_nodes
            # Check if all x-coordinates or all y-coordinates are the same
            all_same_x = jnp.all(x_coords == x_coords[0])
            all_same_y = jnp.all(y_coords == y_coords[0])
            return jnp.logical_or(all_same_x, all_same_y)

        def __resample(key):
            subkey1, subkey2 = jax.random.split(key)
            node_pos = jax.random.choice(
                subkey2, grid_size * grid_size, (self.n_nodes,), replace=False
            )
            grid_nodes = jax.vmap(__convert_to_grid, in_axes=(0, None))(
                node_pos, grid_size
            )
            return grid_nodes

        def __extract_edges(simplex):
            edges = jnp.array(
                [
                    [
                        jnp.minimum(simplex[0], simplex[1]),
                        jnp.maximum(simplex[0], simplex[1]),
                    ],
                    [
                        jnp.minimum(simplex[1], simplex[2]),
                        jnp.maximum(simplex[1], simplex[2]),
                    ],
                    [
                        jnp.minimum(simplex[0], simplex[2]),
                        jnp.maximum(simplex[0], simplex[2]),
                    ],
                ]
            )
            return edges

        xmax = grid_size
        ymax = grid_size
        # Generate random points in the grid
        node_pos = jax.random.choice(key, xmax * ymax, (self.n_nodes,), replace=False)
        grid_nodes = jax.vmap(__convert_to_grid, in_axes=(0, None))(node_pos, ymax)
        # Check that not all points are on the same line (have the same x or y coordinate)
        grid_nodes = jax.lax.cond(
            __on_same_line(grid_nodes),
            lambda key: __resample(key),
            lambda _: grid_nodes,
            key,
        )
        grid_nodes_jax = jnp.array(grid_nodes, dtype=jnp.float16).T

        # 2 nodes that are furthest apart from each other
        far_1, far_2 = self.__find_single_goal_and_origin(grid_nodes_jax)

        # Sort nodes (including the origin and goal) by euclidean distance
        grid_nodes_jax = self.__sort_nodes_by_euclidean_distance(
            grid_nodes_jax, far_1, far_2
        )

        # far_1 is first origin. Far_2 -1 is the first goal. Far_2 is the special node

        # Apply Delauney triangulation to get edges
        delaunay = Delaunay(grid_nodes_jax[:-1])
        simplices = delaunay.simplices
        simplices_jax = jnp.array(simplices)

        # Extract edges from the simplices
        # All_edges are not unique but that's okay
        all_edges = jnp.concatenate(jax.vmap(__extract_edges)(simplices_jax))
        unique_edges = jnp.unique(all_edges, axis=0)
        senders = unique_edges[:, 0]
        receivers = unique_edges[:, 1]
        weights = jnp.full(
            (self.n_nodes, self.n_nodes), NOT_CONNECTED, dtype=jnp.float16
        )

        # Ideally use vmap or fori_loop here
        for sender, receiver in all_edges:
            euclidean_distance = jnp.linalg.norm(
                grid_nodes_jax[sender] - grid_nodes_jax[receiver]
            )
            weights = weights.at[sender, receiver].set(euclidean_distance)
            weights = weights.at[receiver, sender].set(euclidean_distance)

        # n_edges = int(jnp.sum(weights > 0) / 2)
        n_edges = len(senders)

        return weights, n_edges, senders, receivers, grid_nodes_jax

    # Find 2 nodes furthest apart
    @partial(jax.jit, static_argnums=(0,))
    def __find_single_goal_and_origin(self, grid_nodes) -> tuple[int, int]:
        def __distance(a, b):
            return jnp.sqrt(jnp.sum((a - b) ** 2))

        distances_from_origin = jax.vmap(lambda x: __distance(grid_nodes[0], x))(
            grid_nodes
        )
        goal = jnp.argmax(distances_from_origin)
        distances_from_goal = jax.vmap(lambda x: __distance(grid_nodes[goal], x))(
            grid_nodes
        )
        origin = jnp.argmax(distances_from_goal)
        return goal, origin

    def __sort_nodes_by_euclidean_distance(
        self, grid_nodes, origin, goal
    ) -> jnp.ndarray:
        def __distance(a, b):
            return jnp.sqrt(jnp.sum((a - b) ** 2))

        # Origin is the first node and goal is the last node
        origin_pos = grid_nodes[origin]
        goal_pos = grid_nodes[goal]
        grid_nodes_not_origin_goal = jnp.delete(
            grid_nodes, np.array([origin, goal]), axis=0
        )

        distances_from_goal = jax.vmap(lambda x: __distance(grid_nodes[goal], x))(
            grid_nodes_not_origin_goal
        )
        sorted_indices = jnp.argsort(-distances_from_goal)
        grid_nodes = jnp.vstack(
            [origin_pos, grid_nodes_not_origin_goal[sorted_indices], goal_pos]
        )
        return grid_nodes

    def set_blocking_prob(
        self,
        key,
        prop_stoch: float,
        k_edges: float,
    ) -> jnp.ndarray:

        def __assign_prob_edge(subkey, is_stochastic_edge):
            prob = jax.random.uniform(subkey, minval=0.0, maxval=1.0)
            prob = jnp.round(prob, 2)  # Round to 2 decimal places
            return jax.lax.cond(is_stochastic_edge, lambda _: prob, lambda _: 0.0, prob)

        # Assign blocking probability to each edge
        if prop_stoch is None:
            num_stoch_edges = k_edges
        else:
            num_stoch_edges = jnp.round(prop_stoch * self.n_edges).astype(int)

        stoch_edge_idx = jax.random.choice(
            key, self.n_edges, shape=(num_stoch_edges,), replace=False
        )
        edge_indices = jnp.arange(self.n_edges)
        keys = jax.random.split(key, num=self.n_edges)
        is_stochastic_edges = jnp.isin(edge_indices, stoch_edge_idx)
        edge_probs = jax.vmap(__assign_prob_edge, in_axes=(0, 0))(
            keys, is_stochastic_edges
        )
        edge_probs = jnp.asarray(edge_probs, dtype=jnp.float16)

        blocking_prob_matrix = jnp.full(
            (self.n_nodes, self.n_nodes), 1.0, dtype=jnp.float16
        )
        for i in range(self.n_edges):
            blocking_prob_matrix = blocking_prob_matrix.at[
                self.senders[i], self.receivers[i]
            ].set(edge_probs[i])
            blocking_prob_matrix = blocking_prob_matrix.at[
                self.receivers[i], self.senders[i]
            ].set(edge_probs[i])
        return blocking_prob_matrix

    def _convert_to_networkx(self) -> nx.Graph:
        graph_NX = nx.Graph()
        for i in range(self.n_nodes):
            graph_NX.add_node(i, pos=tuple(self.node_pos[i].tolist()))
        for i in range(self.n_edges):
            graph_NX.add_edge(self.senders[i].item(), self.receivers[i].item())

        # Should be of the format {(sender,receiver):weight}
        weight_edge_dict = {
            (s, r): w
            for s, r, w in zip(
                self.senders.tolist(),
                self.receivers.tolist(),
                self.weights[self.senders, self.receivers].tolist(),
            )
        }
        # Only add to the blocking_prob attribute of the networkx graph if the blocking probability is greater than 0 (stochastic edge)
        blocking_prob_dict = {
            (s, r): w
            for s, r, w in zip(
                self.senders.tolist(),
                self.receivers.tolist(),
                self.blocking_prob[self.senders, self.receivers].tolist(),
            )
            if w > 0
        }
        nx.set_edge_attributes(graph_NX, values=weight_edge_dict, name="weight")
        nx.set_edge_attributes(graph_NX, values=blocking_prob_dict, name="blocked_prob")
        return graph_NX

    def plot_nx_graph(self, directory, file_name="unrealised_graph.png"):
        G = self._convert_to_networkx()
        node_colour = []
        for node in G.nodes:
            c = "white"
            if node in self.goal:
                c = "#2ca02c"  # green
            elif node in self.origin:
                c = "#ff7f0e"  # orange
            node_colour.append(c)
        edge_labels = []
        probs = nx.get_edge_attributes(G, "blocked_prob")
        weights = nx.get_edge_attributes(G, "weight")
        edge_labels = {
            e: (
                f"{np.round(w,3)}\np: {np.round(probs[e],2)}"
                if e in probs
                else f"{np.round(w,3)}"
            )
            for e, w in weights.items()
        }
        edge_style = ["dashed" if edge in probs.keys() else "solid" for edge in G.edges]
        pos = nx.get_node_attributes(G, "pos")
        nx.draw(
            G,
            with_labels=True,
            node_size=500,
            node_color=node_colour,
            edgecolors="black",
            pos=pos,
            style=edge_style,
        )
        nx.draw_networkx_edge_labels(
            G,
            pos={p: (v[0], v[1]) for p, v in pos.items()},
            edge_labels=edge_labels,
            bbox={"boxstyle": "square", "pad": 0, "color": "white"},
            rotate=False,
            font_size=8,
            verticalalignment="baseline",
            clip_on=False,
        )
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        plt.savefig(os.path.join(directory, file_name))
        plt.close()


class CTPGraph_Realisation:
    def __init__(
        self,
        key: int,
        n_nodes: int,
        grid_size=None,
        prop_stoch=None,
        k_edges=None,
        num_agents=2,
        handcrafted_graph=None,
    ):
        """
        List of properties:
        graph: CTPGraph object
        """
        if handcrafted_graph is not None:
            self.graph = CTPGraph(key, n_nodes, handcrafted_graph=handcrafted_graph)
        else:
            self.graph = CTPGraph(
                key, n_nodes, grid_size, prop_stoch, k_edges, num_agents
            )

    # CTP_environment actually doesn't use this function - it uses the one in Utils
    def sample_blocking_status(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        keys = jax.random.split(key, num=self.graph.n_edges)
        blocking_status = jnp.full(
            (self.graph.n_nodes, self.graph.n_nodes), BLOCKED, dtype=jnp.float16
        )
        # 0 means not blocked, 1 means blocked
        for i in range(self.graph.n_edges):
            element_blocking_status = jax.random.bernoulli(
                keys[i],
                p=self.graph.blocking_prob[
                    self.graph.senders[i], self.graph.receivers[i]
                ],
            )
            blocking_status = blocking_status.at[
                self.graph.senders[i], self.graph.receivers[i]
            ].set(element_blocking_status)
            blocking_status = blocking_status.at[
                self.graph.receivers[i], self.graph.senders[i]
            ].set(element_blocking_status)
        return blocking_status

    def plot_realised_graph(
        self, blocking_status: jnp.ndarray, directory, file_name="realised_graph.png"
    ):
        G = self.graph._convert_to_networkx()
        node_colour = []
        for node in G.nodes:
            c = "white"
            if node in self.graph.goal:
                c = "#2ca02c"  # green
            elif node in self.graph.origin:
                c = "#ff7f0e"  # orange
            node_colour.append(c)
        edge_labels = []
        probs = nx.get_edge_attributes(G, "blocked_prob")
        weights = nx.get_edge_attributes(G, "weight")
        blocked_edges = [
            (s, r) for (s, r), v in probs.items() if blocking_status[s, r] == BLOCKED
        ]
        edge_labels = {
            e: (
                f"{np.round(w,3)}\np: {np.round(probs[e],2)}"
                if e in probs
                else f"{np.round(w,3)}"
            )
            for e, w in weights.items()
        }
        edge_style = [
            (
                ":"
                if edge in blocked_edges
                else "dashed" if edge in probs.keys() else "solid"
            )
            for edge in G.edges
        ]
        pos = nx.get_node_attributes(G, "pos")
        nx.draw(
            G,
            with_labels=True,
            node_size=500,
            node_color=node_colour,
            edgecolors="black",
            pos=pos,
            style=edge_style,
        )
        nx.draw_networkx_edge_labels(
            G,
            pos={p: (v[0], v[1]) for p, v in pos.items()},
            edge_labels=edge_labels,
            bbox={"boxstyle": "square", "pad": 0, "color": "white"},
            rotate=False,
            font_size=8,
            verticalalignment="baseline",
            clip_on=False,
        )
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        legend_elements = [
            Line2D([0], [0], linestyle=":", color="black", label="Blocked"),
            Line2D([0], [0], linestyle="dashed", color="black", label="Not Blocked"),
            Line2D([0], [0], linestyle="solid", color="black", label="Deterministic"),
        ]
        plt.legend(handles=legend_elements, loc="upper right")
        plt.savefig(os.path.join(directory, file_name))
        plt.close()
