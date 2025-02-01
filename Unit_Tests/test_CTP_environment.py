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


# def test_resample(environment: CTP_environment.MA_CTP_General):
#    pass


def test_step(printer, environment: CTP_environment.MA_CTP_General):
    # if __name__ == "__main__":
    """
    key = jax.random.PRNGKey(30)
    environment = CTP_environment.MA_CTP_General(
        2, 5, key, prop_stoch=0.8, grid_size=10, num_stored_graphs=1
    )
    """

    key = jax.random.PRNGKey(31)
    key, subkey = jax.random.split(key)
    initial_env_state, initial_belief_states = environment.reset(key)
    # take invalid action (same node)
    env_state_1, belief_state_1, reward_1, terminate_1, subkey = environment.step(
        subkey, initial_env_state, initial_belief_states, jnp.array([0, 1])
    )

    # take invalid action (not connected)
    env_state_2, belief_state_2, reward_2, terminate_2, subkey = environment.step(
        subkey, initial_env_state, initial_belief_states, jnp.array([4, 4])
    )

    # take invalid action (service goal while not at goal)
    env_state_3, belief_state_3, reward_3, terminate_3, subkey = environment.step(
        subkey, env_state_1, belief_state_1, jnp.array([5, 5])
    )
    assert reward_1 == reward_2 == reward_3 < -300
    # printer(env_state_1)
    # printer("Belief state 1")
    # printer(belief_state_1)
    # printer("Reward 1: ", reward_1)

    # Both agents at different goals
    env_state_4, belief_state_4, reward_4, terminate_4, subkey = environment.step(
        subkey, env_state_1, belief_state_1, jnp.array([2, 3])
    )
    assert jnp.isclose(reward_4, -1.123, rtol=1e-3)

    # one agent services the goal. the other agent at the same goal
    env_state_5, belief_state_5, reward_5, terminate_5, subkey = environment.step(
        subkey, env_state_4, belief_state_4, jnp.array([5, 2])
    )
    assert jnp.isclose(reward_5, -1, rtol=1e-3)

    # One agent done. The other agent goes to the goal
    env_state_6, belief_state_6, reward_6, terminate_6, subkey = environment.step(
        subkey, env_state_5, belief_state_5, jnp.array([0, 3])
    )
    assert jnp.isclose(reward_6, -1, rtol=1e-3)

    assert jnp.all(
        jnp.array(
            [
                terminate_1,
                terminate_2,
                terminate_3,
                terminate_4,
                terminate_5,
                terminate_6,
            ]
        )
        == jnp.bool_(False)
    )

    # both agents done
    env_state_7, belief_state_7, reward_7, terminate_7, subkey = environment.step(
        subkey, env_state_6, belief_state_6, jnp.array([0, 5])
    )
    assert jnp.isclose(reward_7, 0, rtol=1e-3)
    assert terminate_7 == jnp.bool_(True)
