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
import os

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

        # +1 because service goal action
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

        # origins and goals are one-hot encoded -> convert to int16 array of size num_nodes
        _, graph_origins = jax.lax.top_k(
            self.stored_graphs[index, 2, 0, :], self.num_agents
        )
        _, graph_goals = jax.lax.top_k(
            self.stored_graphs[index, 2, 1, :], self.num_agents
        )
        graph_origins = jnp.array(graph_origins, dtype=jnp.int16)
        graph_goals = jnp.array(graph_goals, dtype=jnp.int16)

        # Get solvable realisation. Add expensive edges as necessary
        blocking_status = graph_functions.sample_blocking_status(
            subkey, graph_blocking_prob
        )

        graph_weights, graph_blocking_prob, blocking_status = add_expensive_edge(
            graph_weights,
            graph_blocking_prob,
            blocking_status,
            graph_origins,
            graph_goals,
        )
        # If don't normalize using full observability, can normalize after adding expensive edge
        # The operations performed for normalization using full observability = same as optimistic baseline

        env_state = self.__convert_graph_realisation_to_state(
            graph_origins,
            graph_goals,
            blocking_status,
            graph_weights,
            graph_blocking_prob,
        )

        # Get the initial belief states for all agents (first dimension corresponds to agent id)
        initial_beliefs = jax.vmap(
            lambda agent_id: self.get_initial_belief(env_state, agent_id)
        )(jnp.arange(self.num_agents))

        return env_state, initial_beliefs

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: jax.random.PRNGKey,
        current_env_state: EnvState,
        current_belief_state: Belief_State,
        actions: jnp.ndarray,
    ) -> tuple[EnvState, Belief_State, int, bool]:
        # Argument current_belief_state includes the belief state of all agents (first dimension corresponds to agent id)
        # return the new environment state, next belief state, reward, and whether the episode is done
        weights = current_env_state[1, self.num_agents :, :]
        blocking_status = current_env_state[0, self.num_agents :, :]
        # Which agent is servicing which goal
        goal_service_agent = self._service_goal(
            actions, current_env_state
        )  # size num_nodes

        # Get whether each agent is done -> will not incur negative rewards or move but belief state will change?
        done_agents = jnp.where(
            jnp.sum(current_env_state[3, : self.num_agents, :], axis=1) > 0,
            jnp.bool_(True),
            jnp.bool_(False),
        )  # size num_agents

        # Use environment state and action to determine if the action is valid for one agent
        def _is_invalid_action(
            action: int, current_env_state: jnp.array, agent_id: int
        ) -> bool:
            current_node_num = jnp.argmax(current_env_state[0, agent_id, :])
            # Invalid if action is service goal but agent not listed for that goal (already includes the case where agent is not at goal)
            is_invalid = jax.lax.cond(
                action == self.num_nodes,
                lambda _: jnp.logical_not(
                    goal_service_agent[current_node_num] == agent_id
                ),
                lambda _: jnp.logical_or(
                    action == current_node_num,
                    jnp.logical_or(
                        weights[current_node_num, action]
                        == CTP_generator.NOT_CONNECTED,
                        blocking_status[current_node_num, action]
                        == CTP_generator.BLOCKED,
                    ),
                ),
                None,
            )
            return is_invalid

        def _step_invalid_action(args) -> tuple[jnp.array, jnp.array, int, bool]:
            # returns one row for agent pos, one row for goal servicing, reward, and whether that agent is done
            current_env_state, actions, agent_id = args
            reward = self.reward_for_invalid_action
            agent_done = jnp.bool_(False)
            return (
                current_env_state[0, agent_id, :],
                current_env_state[3, agent_id, :],
                reward,
                agent_done,
            )

        def _service_goal(args) -> tuple[jnp.array, jnp.array, int, bool]:
            # Returns agent_pos, goal service status, reward, and whether the agent is done
            # agent already at goal -> don't need to update position
            current_env_state, actions, agent_id = args
            current_node_num = jnp.argmax(current_env_state[0, agent_id, :])
            reward = self.reward_for_goal
            # Update goal status
            new_env_state = current_env_state.at[3, agent_id, current_node_num].add(1)
            agent_done = jnp.bool_(True)
            return (
                current_env_state[0, agent_id, :],
                new_env_state[3, agent_id, :],
                reward,
                agent_done,
            )

        # Function that gets called if valid action and not servicing goal -> move to new node
        def _move_to_new_node(args) -> tuple[jnp.array, jnp.array, int, bool]:
            current_env_state, actions, agent_id = args
            current_node = jnp.argmax(current_env_state[0, agent_id, :])
            reward = -(weights[current_node, actions[agent_id]])
            # update agent position
            new_env_state = current_env_state.at[0, agent_id, current_node].set(0)
            new_env_state = new_env_state.at[0, agent_id, actions[agent_id]].set(1)
            agent_done = jnp.bool_(False)
            return (
                new_env_state[0, agent_id, :],
                current_env_state[3, agent_id, :],
                reward,
                agent_done,
            )

        def _for_done_agents(args) -> tuple[jnp.array, jnp.array, int, bool]:
            current_env_state, actions, agent_id = args
            reward = jnp.float16(0)
            agent_done = jnp.bool_(True)
            return (
                current_env_state[0, agent_id, :],
                current_env_state[3, agent_id, :],
                reward,
                agent_done,
            )

        def _update_per_agent(
            current_env_state, actions, agent_id
        ) -> tuple[jnp.array, int, bool]:
            # Update agent pos, goal service status, reward, done
            current_node = jnp.argmax(current_env_state[0, agent_id, :])
            agent_pos, goal_service, reward, done = jax.lax.cond(
                done_agents[agent_id] == jnp.bool_(True),
                lambda args: _for_done_agents(args),
                lambda args: jax.lax.cond(
                    _is_invalid_action(actions[agent_id], current_env_state, agent_id),
                    lambda args: _step_invalid_action(args),
                    lambda args: jax.lax.cond(
                        goal_service_agent[current_node] == agent_id,
                        lambda args: _service_goal(args),
                        lambda args: _move_to_new_node(args),
                        args,
                    ),
                    args,
                ),
                (current_env_state, actions, agent_id),
            )
            return agent_pos, goal_service, reward, done

        all_agents_pos, all_goals_service, rewards, all_agents_done = jax.vmap(
            _update_per_agent, in_axes=(None, None, 0)
        )(current_env_state, actions, jnp.arange(self.num_agents))

        new_env_state = current_env_state.at[0, : self.num_agents, :].set(
            all_agents_pos
        )
        new_env_state = new_env_state.at[3, : self.num_agents, :].set(all_goals_service)

        # Get belief state for each agent based on new observations, including the stationary agents that are done
        # Also update the agents' positions and goal service status in the belief state
        next_belief_states = jax.vmap(
            self._get_belief_state_per_agent, in_axes=(None, 0, 0)
        )(new_env_state, jnp.arange(self.num_agents), current_belief_state)

        # Get updated belief state from communication, including the stationary agents that are done
        next_belief_states = jax.vmap(
            self._update_belief_state_due_to_full_communication, in_axes=(None, 0)
        )(next_belief_states, jnp.arange(self.num_agents))

        # Add up rewards
        total_reward = jnp.sum(rewards)

        # Terminating the episode - if all agents are done
        terminate = jnp.all(all_agents_done)
        new_env_state, next_belief_states = jax.lax.cond(
            terminate,
            lambda x: self.reset(x),
            lambda x: (new_env_state, next_belief_states),
            key,
        )
        key, subkey = jax.random.split(key)
        return new_env_state, next_belief_states, total_reward, terminate, subkey

    # Get belief state based on observation. Belief state also has correct new agents' positions and goal status
    @partial(jax.jit, static_argnums=(0,))
    def _get_belief_state_per_agent(
        self,
        current_env_state: EnvState,
        agent_id: int,
        old_belief_state: Belief_State,
    ) -> Belief_State:
        agent_pos = current_env_state[0, agent_id, :]
        blocking_status = current_env_state[0, self.num_agents :, :]
        # Get edges connected to agent's current position
        obs_blocking_status = jnp.full(
            (
                self.num_nodes,
                self.num_nodes,
            ),
            CTP_generator.UNKNOWN,
            dtype=jnp.float16,
        )
        # replace 1 row and column corresponding to agent's position. Observation
        obs_blocking_status = obs_blocking_status.at[jnp.argmax(agent_pos), :].set(
            blocking_status[jnp.argmax(agent_pos), :]
        )
        obs_blocking_status = obs_blocking_status.at[:, jnp.argmax(agent_pos)].set(
            blocking_status[jnp.argmax(agent_pos), :]
        )
        # Combine current_blocking_status with new_observation
        new_blocking_knowledge = jnp.where(
            old_belief_state[0, self.num_agents :, :] == CTP_generator.UNKNOWN,
            obs_blocking_status[:, :],
            old_belief_state[0, self.num_agents :, :],
        )

        new_belief_state = old_belief_state.at[0, self.num_agents :, :].set(
            new_blocking_knowledge
        )
        # Adjust other agents' positions
        new_belief_state = new_belief_state.at[0, agent_id, :].set(agent_pos)
        new_belief_state = new_belief_state.at[1, : self.num_agents, :].set(
            current_env_state[0, : self.num_agents, :]
        )
        new_belief_state = new_belief_state.at[1, agent_id, :].set(0)

        # Adjust goal service status
        new_belief_state = new_belief_state.at[3, : self.num_agents, :].set(
            current_env_state[3, : self.num_agents, :]
        )
        return new_belief_state

    @partial(jax.jit, static_argnums=(0,))
    def _update_belief_state_due_to_full_communication(
        self, all_belief_states: jnp.ndarray, agent_id: int
    ) -> Belief_State:
        blocked = jnp.any(
            all_belief_states[:, 0, self.num_agents :, :] == CTP_generator.BLOCKED,
            axis=0,
            keepdims=False,
        )  # If any agent says BLOCKED, it's BLOCKED
        unblocked = jnp.any(
            all_belief_states[:, 0, self.num_agents :, :] == CTP_generator.UNBLOCKED,
            axis=0,
            keepdims=False,
        )  # If any agent says UNBLOCKED, it's UNBLOCKED
        combined = jnp.where(
            blocked,
            CTP_generator.BLOCKED,
            jnp.where(unblocked, CTP_generator.UNBLOCKED, CTP_generator.UNKNOWN),
        )  # UNKNOWN if UNKNOWN for all the agents
        new_agent_belief_state = all_belief_states[agent_id]
        new_agent_belief_state = new_agent_belief_state.at[0, self.num_agents :, :].set(
            combined
        )
        return new_agent_belief_state

    # Returns an array of size num_nodes. If the element is inf, that means it's not a goal.
    # If -1, then no valid agent servicing that goal. If not -1 or inf then it's the index of the valid agent servicing the goal
    @partial(jax.jit, static_argnums=(0,))
    def _service_goal(
        self, actions: jnp.ndarray, current_env_state: EnvState
    ) -> jnp.array:
        _, goals = jax.lax.top_k(
            jnp.diag(current_env_state[3, self.num_agents :, :]), self.num_agents
        )

        # For each goal: check if any agent's pos is at the goal, the action is to service the goal, and the goal has not been serviced yet
        def _find_agent_servicing_goal(index):
            goal = goals[index]
            not_serviced = (current_env_state[3, : self.num_agents, goal] == 0).all()
            agents_servicing = actions == self.num_nodes
            agents_at_goal = current_env_state[
                0, : self.num_agents, goal
            ]  # 1 if at goal. 0 otherwise
            valid_agents = jnp.where(
                jnp.logical_and(agents_servicing, agents_at_goal),
                1,
                jnp.iinfo(jnp.int16).max,
            )
            smallest_agent = jnp.int16(jnp.argmin(valid_agents))
            smallest_agent_value = jnp.min(valid_agents)

            final_smallest_agent = jnp.where(
                jnp.logical_and(
                    not_serviced, smallest_agent_value != jnp.iinfo(jnp.int16).max
                ),
                smallest_agent,
                jnp.int16(-1),
            )
            return final_smallest_agent

        goal_service_agent = jax.vmap(_find_agent_servicing_goal)(
            jnp.arange(self.num_agents)
        )
        full_goal_service_agent = jnp.full((self.num_nodes,), -1, dtype=jnp.int16)
        full_goal_service_agent = full_goal_service_agent.at[goals].set(
            goal_service_agent
        )
        return full_goal_service_agent

    @partial(jax.jit, static_argnums=(0,))
    def __convert_graph_realisation_to_state(
        self,
        origins: jnp.array,
        goals: int,
        blocking_status: jnp.ndarray,
        graph_weights: jnp.ndarray,
        blocking_prob: jnp.ndarray,
    ) -> EnvState:
        agents_pos = jax.nn.one_hot(origins, self.num_nodes)
        empty = jnp.zeros((self.num_agents, self.num_nodes), dtype=jnp.float16)
        edge_weights = jnp.concatenate((empty, graph_weights), axis=0)
        edge_probs = jnp.concatenate((empty, blocking_prob), axis=0)
        pos_and_blocking_status = jnp.concatenate((agents_pos, blocking_status), axis=0)

        # Top part is each agent's service history. Bottom part is number of times each goal needs to
        # be serviced
        goal_matrix = jnp.zeros((self.num_nodes, self.num_nodes), dtype=jnp.float16)
        goal_matrix = jax.vmap(lambda goal: goal_matrix.at[goal, goal].set(1))(goals)
        goal_matrix = jnp.sum(goal_matrix, axis=0, keepdims=False)
        goal_matrix = jnp.concatenate((empty, goal_matrix), axis=0)

        return jnp.stack(
            (pos_and_blocking_status, edge_weights, edge_probs, goal_matrix),
            axis=0,
            dtype=jnp.float16,
        )

    # Returns the belief state for one agent
    @partial(jax.jit, static_argnums=(0,))
    def get_initial_belief(self, env_state: EnvState, agent_id) -> jnp.ndarray:
        agent_pos = env_state[0, agent_id, :]
        blocking_status = env_state[0, self.num_agents :, :]
        # Get edges connected to agent's current position
        obs_blocking_status = jnp.full(
            (
                self.num_nodes,
                self.num_nodes,
            ),
            CTP_generator.UNKNOWN,
            dtype=jnp.float16,
        )
        # replace 1 row and column corresponding to agent's position
        obs_blocking_status = obs_blocking_status.at[jnp.argmax(agent_pos), :].set(
            blocking_status[jnp.argmax(agent_pos), :]
        )
        obs_blocking_status = obs_blocking_status.at[:, jnp.argmax(agent_pos)].set(
            blocking_status[jnp.argmax(agent_pos), :]
        )

        # Incorporate info that non-existent edges are blocked and deterministic edges are not blocked
        blocking_status_knowledge = jnp.where(
            jnp.logical_or(
                env_state[2, self.num_agents :, :] == 0,
                env_state[2, self.num_agents :, :] == 1,
            ),
            env_state[0, self.num_agents :, :],
            obs_blocking_status[:, :],
        )

        # For the first matrix (blocking status)
        all_agents_pos = jnp.zeros((self.num_agents, self.num_nodes), dtype=jnp.float16)
        all_agents_pos = all_agents_pos.at[agent_id, :].set(agent_pos)
        # Concatenate the agent's position and the blocking status knowledge
        full_blocking_status_knowledge = jnp.concatenate(
            (all_agents_pos, blocking_status_knowledge), axis=0
        )

        # Concatenate the other agents' positions and the blocking probabilities
        other_agents_pos = env_state[0, : self.num_agents, :].at[agent_id, :].set(0)
        full_blocking_prob = jnp.concatenate(
            (other_agents_pos, env_state[1, self.num_agents :, :]), axis=0
        )

        # Stack everything
        initial_belief_state = jnp.stack(
            (
                full_blocking_status_knowledge,
                full_blocking_prob,
                env_state[2, :, :],
                env_state[3, :, :],
            ),
            axis=0,
            dtype=jnp.float16,
        )

        return initial_belief_state
