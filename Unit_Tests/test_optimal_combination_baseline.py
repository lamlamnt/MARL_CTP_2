import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_environment
from Environment import CTP_generator
from Utils.optimal_combination import get_optimal_combination_and_cost
import pytest
import pytest_print as pp
import os
import warnings
from Agents.optimistic_agent import Optimistic_Agent
from Utils.plot_graph_env_state import plot_realised_graph_from_env_state


@pytest.fixture
def environment():
    key = jax.random.PRNGKey(30)
    environment = CTP_environment.MA_CTP_General(
        2, 5, key, prop_stoch=0.8, grid_size=10, num_stored_graphs=1
    )
    return environment


def test_get_optimal_combination_and_cost(
    printer, environment: CTP_environment.MA_CTP_General
):
    key = jax.random.PRNGKey(30)
    initial_env_state, initial_belief_states = environment.reset(key)
    _, goals = jax.lax.top_k(jnp.diag(initial_env_state[3, 2:, :]), 2)
    best_combination, best_combination_cost = get_optimal_combination_and_cost(
        initial_env_state[1, 2:, :],
        initial_env_state[0, 2:, :],
        jnp.array([0, 1]),
        jnp.array([2, 3]),
        2,
    )
    assert best_combination.shape == (2,)
    assert jnp.isclose(best_combination_cost, 1, rtol=1e-2)
    assert jnp.array_equal(best_combination, jnp.array([0, 1]))


def test_optimistic_agent(printer, environment):
    key = jax.random.PRNGKey(30)
    initial_env_state, initial_belief_states = environment.reset(key)
    optimistic_agent = Optimistic_Agent(2, 5)
    pre_allocated_goals = optimistic_agent.allocate_goals(initial_belief_states[0])
    assert jnp.array_equal(pre_allocated_goals, jnp.array([2, 3]))
    actions = jax.vmap(optimistic_agent.act, in_axes=(0, None, 0))(
        initial_belief_states, pre_allocated_goals, jnp.arange(2)
    )
    assert jnp.array_equal(actions, jnp.array([2, 3]))


def test_optimistic_agent_2(printer):
    key = jax.random.PRNGKey(50)
    subkey1, subkey2 = jax.random.split(key)
    environment = CTP_environment.MA_CTP_General(
        2, 5, key, prop_stoch=0.8, grid_size=10, num_stored_graphs=1
    )
    initial_env_state, initial_belief_states = environment.reset(subkey1)
    optimistic_agent = Optimistic_Agent(2, 5)
    pre_allocated_goals = optimistic_agent.allocate_goals(initial_belief_states[0])
    actions = jax.vmap(optimistic_agent.act, in_axes=(0, None, 0))(
        initial_belief_states, pre_allocated_goals, jnp.arange(2)
    )
    assert jnp.array_equal(pre_allocated_goals, jnp.array([3, 2]))
    assert jnp.array_equal(actions, jnp.array([3, 2]))


# This example does not demonstrate the behaviour where an agent benefits from blocking status knowledge gathered by another agent
def test_optimistic_agent_2(printer):
    key = jax.random.PRNGKey(50)
    subkey1, subkey = jax.random.split(key)
    environment = CTP_environment.MA_CTP_General(
        2, 10, key, prop_stoch=0.8, grid_size=10, num_stored_graphs=1
    )
    initial_env_state, initial_belief_states = environment.reset(subkey)
    optimistic_agent = Optimistic_Agent(2, 10)
    pre_allocated_goals = optimistic_agent.allocate_goals(initial_belief_states[0])
    assert jnp.array_equal(pre_allocated_goals, jnp.array([7, 8]))
    actions = jax.vmap(optimistic_agent.act, in_axes=(0, None, 0))(
        initial_belief_states, pre_allocated_goals, jnp.arange(2)
    )
    assert jnp.array_equal(actions, jnp.array([7, 5]))
    env_state_1, belief_state_1, rewards_1, done_1, subkey = environment.step(
        subkey, initial_env_state, initial_belief_states, actions
    )
    actions_2 = jax.vmap(optimistic_agent.act, in_axes=(0, None, 0))(
        belief_state_1, pre_allocated_goals, jnp.arange(2)
    )
    env_state_2, belief_state_2, rewards_2, done_2, subkey = environment.step(
        subkey, env_state_1, belief_state_1, actions_2
    )
    assert jnp.array_equal(actions_2, jnp.array([10, 7]))
    actions_3 = jax.vmap(optimistic_agent.act, in_axes=(0, None, 0))(
        belief_state_2, pre_allocated_goals, jnp.arange(2)
    )
    env_state_3, belief_state_3, rewards_3, done_3, subkey = environment.step(
        subkey, env_state_2, belief_state_2, actions_3
    )
    assert jnp.array_equal(actions_3, jnp.array([10, 8]))
    actions_4 = jax.vmap(optimistic_agent.act, in_axes=(0, None, 0))(
        belief_state_3, pre_allocated_goals, jnp.arange(2)
    )
    env_state_4, belief_state_4, rewards_4, done_4, subkey = environment.step(
        subkey, env_state_3, belief_state_3, actions_4
    )
    assert jnp.array_equal(actions_4, jnp.array([10, 10]))
    total_episodic_reward = rewards_1 + rewards_2 + rewards_3 + rewards_4

    def _calculate_optimal_cost(env_state):
        _, goals = jax.lax.top_k(jnp.diag(env_state[3, 2:, :]), 2)
        origins = jnp.argmax(env_state[0, :2, :], axis=1)
        _, optimal_cost = get_optimal_combination_and_cost(
            env_state[1, 2:, :],
            env_state[0, 2:, :],
            origins,
            goals,
            2,
        )
        # minus self.reward_service goal because this is a positive number
        optimal_cost_including_service_goal_costs = optimal_cost - (-0.1) * 2
        return jnp.array(optimal_cost_including_service_goal_costs, dtype=jnp.float16)

    shortest_path = _calculate_optimal_cost(initial_env_state)
    current_directory = os.getcwd()
    parent_dir = os.path.dirname(current_directory)
    log_directory = os.path.join(parent_dir, "Logs/Unit_Tests")
    plot_realised_graph_from_env_state(initial_env_state, log_directory)
    assert jnp.isclose(-total_episodic_reward.sum(), shortest_path, atol=1e-2)


def test_optimistic_agent_full_episode(
    printer, environment: CTP_environment.MA_CTP_General
):
    key = jax.random.PRNGKey(30)
    initial_env_state, initial_belief_states = environment.reset(key)
    optimistic_agent = Optimistic_Agent(2, 5)
    total_cost = optimistic_agent.get_total_cost(
        environment, initial_belief_states, initial_env_state, key
    )
    assert jnp.isclose(total_cost, 1.2, rtol=1e-2)


# test get_optimal_combination_and_cost for 3 agents
def test_get_optimal_combination_and_cost_3_agents(printer):
    key = jax.random.PRNGKey(30)
    environment = CTP_environment.MA_CTP_General(
        3, 7, key, prop_stoch=0.8, grid_size=10, num_stored_graphs=1
    )
    initial_env_state, initial_belief_states = environment.reset(key)
    _, goals = jax.lax.top_k(jnp.diag(initial_env_state[3, 3:, :]), 3)
    best_combination, best_combination_cost = get_optimal_combination_and_cost(
        initial_env_state[1, 3:, :],
        initial_env_state[0, 3:, :],
        jnp.array([0, 1, 2]),
        jnp.array([3, 4, 5]),
        3,
    )
    assert best_combination.shape == (3,)
    assert jnp.isclose(best_combination_cost, 1, rtol=1e-2)
    assert jnp.array_equal(best_combination, jnp.array([0, 1, 2]))
