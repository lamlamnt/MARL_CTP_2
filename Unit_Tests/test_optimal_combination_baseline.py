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
    printer(pre_allocated_goals)
    printer(type(pre_allocated_goals))
    """
    actions = jax.vmap(optimistic_agent.act, in_axes=(0, None, 0))(
        initial_belief_states, pre_allocated_goals, jnp.arange(2)
    )
    printer(actions)
    """
