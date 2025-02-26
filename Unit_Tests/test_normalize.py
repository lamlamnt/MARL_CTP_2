import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_environment
from Environment import CTP_generator
import pytest
import pytest_print as pp
import os
import warnings
from Utils.normalize_add_expensive_edge import get_expected_optimal_total_cost


@pytest.fixture
def environment():
    key = jax.random.PRNGKey(30)
    environment = CTP_environment.MA_CTP_General(
        2, 5, key, prop_stoch=0.8, grid_size=10, num_stored_graphs=1
    )
    return environment


def test_normalize(printer):
    key = jax.random.PRNGKey(101)
    graphRealisation = CTP_generator.CTPGraph_Realisation(
        key, 5, prop_stoch=0.4, num_agents=2
    )
    normalizing_factor = get_expected_optimal_total_cost(graphRealisation, key)
    assert jnp.isclose(normalizing_factor, 1.728, atol=1e-2)


# test that when reset - stll have the same weight
def test_reset_normalize(printer, environment):
    key = jax.random.PRNGKey(101)
    subkey1, subkey2 = jax.random.split(key)
    initial_env_state, initial_belief_states = environment.reset(subkey1)
    next_env_state, next_belief_states = environment.reset(subkey2)
    assert jnp.array_equal(initial_env_state[1, 2:, :], next_env_state[1, 2:, :])
