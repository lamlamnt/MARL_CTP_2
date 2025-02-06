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


def test_step(printer, environment: CTP_environment.MA_CTP_General):
    key = jax.random.PRNGKey(31)
    key, subkey = jax.random.split(key)
    initial_env_state, initial_belief_states = environment.reset(key)
    # take invalid action (same node)
    env_state_1, belief_state_1, rewards_1, done_1, subkey = environment.step(
        subkey, initial_env_state, initial_belief_states, jnp.array([0, 1])
    )

    # take invalid action (not connected)
    env_state_2, belief_state_2, rewards_2, done_2, subkey = environment.step(
        subkey, env_state_1, belief_state_1, jnp.array([4, 4])
    )

    # take invalid action (service goal while not at goal)
    env_state_3, belief_state_3, rewards_3, done_3, subkey = environment.step(
        subkey, env_state_2, belief_state_2, jnp.array([5, 5])
    )
    assert jnp.array_equal(rewards_1, jnp.array([-200, -200]))
    assert jnp.array_equal(rewards_2, jnp.array([-200, -200]))
    assert jnp.array_equal(rewards_3, jnp.array([-200, -200]))
    assert jnp.array_equal(done_3, jnp.array([False, False]))
    assert jnp.array_equal(env_state_1, env_state_2)
    assert jnp.array_equal(env_state_2, env_state_3)
    assert jnp.array_equal(belief_state_1, belief_state_2)
    assert jnp.array_equal(belief_state_2, belief_state_3)

    # Both agents at different goals
    env_state_4, belief_state_4, rewards_4, done_4, subkey = environment.step(
        subkey, initial_env_state, initial_belief_states, jnp.array([2, 3])
    )
    assert jnp.isclose(rewards_4, jnp.array([-0.1464, -0.854]), rtol=1e-3).all()

    # make sure positions are correct
    assert jnp.sum(env_state_4[0, :2, :]) == 2
    assert jnp.sum(belief_state_4[0, 0, :2, :]) == 1
    assert jnp.sum(belief_state_4[1, 0, :2, :]) == 1
    assert jnp.sum(belief_state_4[0, 1, :2, :]) == 1
    assert jnp.sum(belief_state_4[1, 1, :2, :]) == 1

    # one agent services the goal. the other agent at the same goal
    env_state_5, belief_state_5, rewards_5, done_5, subkey = environment.step(
        subkey, env_state_4, belief_state_4, jnp.array([5, 2])
    )
    assert jnp.array_equal(done_5, jnp.array([True, False]))
    assert jnp.isclose(rewards_5, jnp.array([-0.1, -0.89]), rtol=1e-2).all()
    assert not jnp.array_equal(env_state_5[3, :, :], env_state_4[3, :, :])
    assert jnp.array_equal(env_state_5[3, :, :], belief_state_5[0, 3, :, :])
    assert jnp.array_equal(env_state_5[3, :, :], belief_state_5[1, 3, :, :])

    # invalid action - servicing a goal that has already been serviced
    (
        env_state_invalid,
        belief_state_invalid,
        rewards_invalid,
        done_invalid,
        subkey,
    ) = environment.step(subkey, env_state_5, belief_state_5, jnp.array([5, 5]))
    assert jnp.array_equal(rewards_invalid, jnp.array([0, -200]))
    assert jnp.array_equal(done_invalid, jnp.array([True, False]))

    # One agent done. The other agent goes to the goal
    env_state_6, belief_state_6, rewards_6, done_6, subkey = environment.step(
        subkey, env_state_5, belief_state_5, jnp.array([0, 3])
    )
    assert jnp.isclose(rewards_6, jnp.array([0, -0.89]), rtol=1e-2).all()
    assert jnp.array_equal(done_6, jnp.array([True, False]))

    # check symmetrical
    assert jnp.all(
        env_state_6[:, 2:, :] == jnp.transpose(env_state_6[:, 2:, :], (0, 2, 1))
    )
    assert jnp.all(
        belief_state_6[0, :, 2:, :]
        == jnp.transpose(belief_state_6[0, :, 2:, :], (0, 2, 1))
    )
    assert jnp.all(
        belief_state_6[1, :, 2:, :]
        == jnp.transpose(belief_state_6[1, :, 2:, :], (0, 2, 1))
    )

    # check no more unknown blocking status in belief states
    assert jnp.all(belief_state_6[:, 0, :, :] != -1)

    # both agents done
    env_state_7, belief_state_7, rewards_7, done_7, subkey = environment.step(
        subkey, env_state_6, belief_state_6, jnp.array([5, 5])
    )
    assert jnp.isclose(rewards_7, jnp.array([0, -0.1], dtype=jnp.float16)).all()
    assert jnp.array_equal(done_7, jnp.array([True, True]))

    # check reset
    assert jnp.array_equal(env_state_7[3, :, :], initial_env_state[3, :, :])


# when both agents service the same goal -> smaller index agent gets to service the goal. Invalid action for the other agent
def test_same_goal(printer, environment: CTP_environment.MA_CTP_General):
    key = jax.random.PRNGKey(31)
    key, subkey = jax.random.split(key)
    initial_env_state, initial_belief_states = environment.reset(key)
    env_state_1, belief_state_1, rewards_1, done_1, subkey = environment.step(
        subkey, initial_env_state, initial_belief_states, jnp.array([0, 0])
    )
    assert jnp.isclose(rewards_1, jnp.array([-200, -0.207]), rtol=1e-3).all()
    env_state_2, belief_state_2, rewards_2, done_2, subkey = environment.step(
        subkey, env_state_1, belief_state_1, jnp.array([2, 2])
    )
    assert jnp.array_equal(
        env_state_2[0, :2, :], jnp.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0]])
    )
    env_state_3, belief_state_3, rewards_3, done_3, subkey = environment.step(
        subkey, env_state_2, belief_state_2, jnp.array([5, 5])
    )
    assert jnp.isclose(
        rewards_3, jnp.array([-0.1, -0.3], dtype=jnp.float16), rtol=1e-2
    ).all()
    assert jnp.array_equal(done_3, jnp.array([True, False]))
    assert jnp.array_equal(
        env_state_3[0, :2, :], jnp.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0]])
    )


# test single agent
def test_single_agent_working(printer):
    key = jax.random.PRNGKey(30)
    key, subkey = jax.random.split(key)
    single_agent_environment = CTP_environment.MA_CTP_General(
        1, 5, key, prop_stoch=0.8, grid_size=10, num_stored_graphs=1
    )
    initial_env_state, initial_belief_states = single_agent_environment.reset(key)
    env_state_1, belief_state_1, reward_1, terminate_1, subkey = (
        single_agent_environment.step(
            subkey, initial_env_state, initial_belief_states, jnp.array([1])
        )
    )
    env_state_2, belief_state_2, reward_2, terminate_2, subkey = (
        single_agent_environment.step(
            subkey, env_state_1, belief_state_1, jnp.array([2])
        )
    )

    env_state_3, belief_state_3, reward_3, terminate_3, subkey = (
        single_agent_environment.step(
            subkey, env_state_2, belief_state_2, jnp.array([3])
        )
    )
    # service the goal
    env_state_4, belief_state_4, reward_4, terminate_4, subkey = (
        single_agent_environment.step(
            subkey, env_state_3, belief_state_3, jnp.array([5])
        )
    )
    assert terminate_4 == jnp.bool_(True)
    assert jnp.isclose(reward_4[0], -0.1, rtol=1e-3)
    assert jnp.isclose(
        reward_1[0] + reward_2[0] + reward_3[0] + reward_4[0], -1.1, rtol=1e-2
    )


# test generalizing environment
def test_generalizing_environment():
    key = jax.random.PRNGKey(30)
    subkey1, subkey2 = jax.random.split(key)
    environment = CTP_environment.MA_CTP_General(
        2, 5, key, prop_stoch=0.8, grid_size=10, num_stored_graphs=10
    )
    initial_env_state, initial_belief_states = environment.reset(key)

    # test resample
    initial_env_state_2, initial_belief_states_2 = environment.reset(subkey1)
    assert not jnp.array_equal(initial_env_state, initial_env_state_2)
