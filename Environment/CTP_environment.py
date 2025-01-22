from functools import partial
import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments import spaces
import chex
from chex import dataclass
from Environment import CTP_generator
from typing import TypeAlias
import sys
from tqdm import tqdm

sys.path.append("..")
from Utils import graph_functions
from Utils.normalize_add_expensive_edge import add_expensive_edge

# size (4,num_agents+num_nodes,num_nodes)
Belief_State: TypeAlias = jnp.ndarray
EnvState: TypeAlias = (
    jnp.ndarray
)  # Only contains blocking status (to save on memory usage)


class MA_CTP_General(MultiAgentEnv):
    def __init__(
        self,
        num_agents: int,
        num_nodes: int,
        key: chex.PRNGKey,
        prop_stoch=None,
        k_edges=None,
        grid_size=None,
        reward_for_invalid_action=-200.0,
        reward_for_goal=0,
        handcrafted_graph=None,
        num_stored_graphs=10,
        loaded_graphs=None,
    ):
        """
        List of attributes:
        num_agents: int
        num_nodes: int
        reward_for_invalid_action: int
        reward_for_goal:int
        action_spaces
        graph_realisation: CTPGraph_Realisation
        num_stored_graphs: int # Number of stored graphs to choose from
        """
        super().__init__(num_agents=num_agents)
        self.num_agents = num_agents
        self.reward_for_invalid_action = jnp.float16(reward_for_invalid_action)
        self.reward_for_goal = jnp.float16(reward_for_goal)
        self.num_nodes = num_nodes
        self.num_stored_graphs = num_stored_graphs

        actions = [num_nodes + 1 for _ in range(num_agents)]
        self.action_spaces = spaces.MultiDiscrete(actions)

        # Generate graphs
        if loaded_graphs is not None:
            self.stored_graphs = loaded_graphs
        else:
            key, subkey = jax.random.split(key)
            self.stored_graphs = jnp.zeros(
                (num_stored_graphs, 3, num_nodes, num_nodes), dtype=jnp.float16
            )
            for i in tqdm(range(num_stored_graphs)):
                key, subkey = jax.random.split(subkey)
                # Create a graph realisation but technically only need the graph
                graph_realisation = CTP_generator.CTPGraph_Realisation(
                    subkey,
                    self.num_nodes,
                    grid_size=grid_size,
                    prop_stoch=prop_stoch,
                    k_edges=k_edges,
                    num_agents=num_agents,
                    handcrafted_graph=None,
                )

                # Plot this for debugging purposes

                # Normalize the weights using the expected optimal path length under full observability

                # Store the matrix of weights, blocking probs, and origin/goal
                self.stored_graphs = self.stored_graphs.at[i, 0, :, :].set(
                    graph_realisation.graph.weights
                )
                self.stored_graphs = self.stored_graphs.at[i, 1, :, :].set(
                    graph_realisation.graph.blocking_prob
                )

                # For the third dimension, the first row is for the origin. If a node is the origin, the value is 1, else 0
                # The second row is for the goal. If a node is the goal, the value is 1, else 0
                # This is only really necessary if we have randomized origins and goals
                self.stored_graphs = self.stored_graphs.at[i, 2, 0, :].set(
                    jnp.zeros(self.num_nodes, dtype=int)
                    .at[graph_realisation.graph.origin]
                    .set(1)
                )
                self.stored_graphs = self.stored_graphs.at[i, 2, 1, :].set(
                    jnp.zeros(self.num_nodes, dtype=int)
                    .at[graph_realisation.graph.goal]
                    .set(1)
                )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> tuple[EnvState, Belief_State]:
        key, subkey = jax.random.split(key)

        # Sample from list of stored graph realisations
        index = jax.random.randint(
            subkey, shape=(), minval=0, maxval=self.num_stored_graphs - 1
        )
        graph_weights = self.stored_graphs[index, 0, :, :]
        graph_blocking_prob = self.stored_graphs[index, 1, :, :]
        graph_origins = self.stored_graphs[index, 2, 0, :]
        graph_goals = self.stored_graphs[index, 2, 1, :]

        # Get solvable realisation. Add expensive edges as necessary
        blocking_status = graph_functions.sample_blocking_status(
            subkey, graph_blocking_prob
        )

        blocking_status, graph_weights, graph_blocking_prob = add_expensive_edge(
            graph_weights,
            graph_blocking_prob,
            blocking_status,
            graph_origins,
            graph_goals,
        )
        # If don't normalize using full observability, can normalize after adding expensive edge

        env_state = blocking_status
        # return the initial belief state
        original_observation = self.get_obs(env_state)

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, env_state: EnvState) -> jnp.ndarray:
        # Returns observation for each agent
        pass
