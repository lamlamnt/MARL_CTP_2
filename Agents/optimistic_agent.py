import jax
import jax.numpy as jnp
from functools import partial
import sys

sys.path.append("..")
from Environment import CTP_environment
from Environment import CTP_generator
from Utils.optimal_combination import (
    get_optimal_combination_and_cost,
    dijkstra_with_path,
)


class Optimistic_Agent:
    def __init__(self, num_agents: int, num_nodes: int):
        self.num_agents = num_agents
        self.num_nodes = num_nodes

    @partial(jax.jit, static_argnums=(0))
    def allocate_goals(self, belief_state: jnp.ndarray) -> jnp.array:
        # use the first agent's belief state to allocate goals
        # allocate goals at the beginning of the episode
        # returns the goals node number, not the index for the goals array
        # Assume all unknown stochastic edges are not blocked
        belief_state = belief_state.at[0, self.num_agents :, :].set(
            jnp.where(
                belief_state[0, self.num_agents :, :] == CTP_generator.UNKNOWN,
                CTP_generator.UNBLOCKED,
                belief_state[0, self.num_agents :, :],
            )
        )
        # # dijkstra expects env_state. Change blocking_prob of known blocked edges to 1.
        belief_state = belief_state.at[1, self.num_agents :, :].set(
            jnp.where(
                belief_state[0, self.num_agents :, :] == CTP_generator.BLOCKED,
                1,
                belief_state[1, self.num_agents :, :],
            )
        )
        _, goals = jax.lax.top_k(
            jnp.diag(belief_state[3, self.num_agents :, :]), self.num_agents
        )
        one_origin = jnp.argmax(belief_state[0, 0, :])
        other_origins = jnp.argmax(belief_state[1, self.num_agents :, :], axis=1)
        origins = other_origins.at[0].set(one_origin)
        allocated_goals_indices, total_cost = get_optimal_combination_and_cost(
            belief_state[1, self.num_agents :, :],
            belief_state[0, self.num_agents :, :],
            origins,
            goals,
            self.num_agents,
        )  # size num_agents -> returns the index for the goal
        _, goals = jax.lax.top_k(
            jnp.diag(belief_state[3, self.num_agents :, :]), self.num_agents
        )
        return goals[allocated_goals_indices]

    # for one agent at a time (though if done together, would be slightly more efficient)
    @partial(jax.jit, static_argnums=(0))
    def act(
        self,
        belief_state: jnp.ndarray,
        pre_allocated_goals: jnp.array,
        agent_index: int,
    ) -> int:
        # returns an array of actions - size equal to num_agents
        # Assume all unknown stochastic edges are not blocked
        # agent_index = jnp.argmin(jnp.sum(belief_state[1, self.num_agents :, :], axis=1))
        belief_state = belief_state.at[0, self.num_agents :, :].set(
            jnp.where(
                belief_state[0, self.num_agents :, :] == CTP_generator.UNKNOWN,
                CTP_generator.UNBLOCKED,
                belief_state[0, self.num_agents :, :],
            )
        )
        # # dijkstra expects env_state. Change blocking_prob of known blocked edges to 1.
        belief_state = belief_state.at[1, self.num_agents :, :].set(
            jnp.where(
                belief_state[0, self.num_agents :, :] == CTP_generator.BLOCKED,
                1,
                belief_state[1, self.num_agents :, :],
            )
        )
        done = jax.lax.cond(
            jnp.sum(belief_state[3, agent_index, :]) > 0,
            lambda x: jnp.bool_(True),
            lambda x: jnp.bool_(False),
            None,
        )
        current_node = jnp.argmax(belief_state[0, agent_index, :])
        # If agent is done or at the goal corresponding to the allocated goal, then choose service action
        # Else, Dijkstra with path
        action = jax.lax.cond(
            jnp.logical_or(
                done, jnp.equal(current_node, pre_allocated_goals[agent_index])
            ),
            lambda x: self.num_nodes,
            lambda x: dijkstra_with_path(
                belief_state, current_node, pre_allocated_goals[agent_index]
            )[1],
            None,
        )
        return action

    @partial(jax.jit, static_argnums=(0, 1))
    def get_total_cost(
        self,
        environment: CTP_environment.MA_CTP_General,
        initial_belief_states: jnp.ndarray,
        initial_env_state: jnp.ndarray,
        env_key: jax.random.PRNGKey,
    ):
        pre_allocated_goals = self.allocate_goals(initial_belief_states[0])

        def cond_fn(carry):
            _, _, _, episode_done, _ = carry
            return ~episode_done  # Continue while not done

        # returns a positive float
        def body_fn(carry):
            env_state, belief_states, total_cost, episode_done, env_key = carry
            actions = jax.vmap(self.act, in_axes=(0, None, 0))(
                belief_states, pre_allocated_goals, jnp.arange(self.num_agents)
            )
            env_state, belief_states, rewards, dones, env_key = environment.step(
                env_key, env_state, belief_states, actions
            )
            total_cost += -jnp.sum(rewards)
            episode_done = jnp.all(dones)
            return env_state, belief_states, total_cost, episode_done, env_key

        total_cost = 0.0
        carry = (
            initial_env_state,
            initial_belief_states,
            total_cost,
            False,
            env_key,
        )
        final_carry = jax.lax.while_loop(cond_fn, body_fn, carry)
        total_cost = final_carry[2]
        return total_cost
