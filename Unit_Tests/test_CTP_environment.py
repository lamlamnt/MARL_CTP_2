import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_environment
from Environment import CTP_generator
import pytest
import pytest_print as pp
import os


# test for 1 agent and 2 agents
@pytest.fixture
def environment():
    key = jax.random.PRNGKey(30)
    environment = CTP_environment.MA_CTP_General(
        2, 5, key, prop_stoch=0.8, grid_size=10, num_stored_graphs=1
    )
    return environment


def test_reset(printer, environment: CTP_environment.MA_CTP_General):
    key = jax.random.PRNGKey(31)
    env_state, initial_beliefs = environment.reset(key)
    # test compatibility between env_state and initial_beliefs
    assert jnp.array_equal(env_state[1:, 2:, :], initial_beliefs[0, 1:, 2:, :])
    assert jnp.array_equal(env_state[1:, 2:, :], initial_beliefs[1, 1:, 2:, :])
    assert not jnp.array_equal(env_state[0, :, :], initial_beliefs[0, 0, :, :])
    assert not jnp.array_equal(env_state[0, :, :], initial_beliefs[1, 0, :, :])
    assert not jnp.array_equal(env_state[0, :2, :], initial_beliefs[0, 0, :2, :])
    assert not jnp.array_equal(env_state[0, :2, :], initial_beliefs[1, 0, :2, :])
    assert jnp.array_equal(env_state[0, 0, :], initial_beliefs[0, 0, 0, :])
    assert jnp.array_equal(env_state[0, 1, :], initial_beliefs[1, 0, 1, :])

    # test that origins and goals are in the sorted expected order
    assert jnp.argmax(env_state[0, 0, :]) == 0
    assert jnp.argmax(env_state[0, 1, :]) == 1
    assert env_state[3, 1 + 3, 2] == 1
    assert env_state[3, 1 + 4, 3] == 1

    printer(env_state)
    printer(initial_beliefs)


def test_step(printer, environment: CTP_environment.MA_CTP_General):
    pass
    # take invalid action (same node)

    # take invalid action (not connected)

    # take valid action (service goal while not at goal)

    # one agent at goal and does not service the goal

    # one agent at goal and services the goal

    # both agents at goal and service the goal

    # both agents done
