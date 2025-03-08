import pytest
import pytest_print
import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment.CTP_environment import MA_CTP_General
from Auto_encoder_related.random_walk import random_action, Random_Agent


@pytest.fixture
def environment():
    key = jax.random.PRNGKey(30)
    environment = MA_CTP_General(
        2, 5, key, prop_stoch=0.8, grid_size=10, num_stored_graphs=5
    )
    return environment


# test random action
def test_random_action(printer, environment: MA_CTP_General):
    key = jax.random.PRNGKey(31)
    key, subkey = jax.random.split(key)
    initial_env_state, initial_belief_states = environment.reset(key)
    actions = jax.vmap(random_action, in_axes=(0, 0))(
        initial_belief_states, jax.random.split(subkey, 2)
    )
    assert actions[0] == 1
    assert jnp.logical_or(
        actions[1] == 0, jnp.logical_or(actions[1] == 2, actions[1] == 3)
    )
    actions_2 = jax.vmap(random_action, in_axes=(0, 0))(
        initial_belief_states, jax.random.split(key, 2)
    )
    assert actions_2[0] == 1
    assert jnp.logical_or(
        actions_2[1] == 0, jnp.logical_or(actions_2[1] == 2, actions_2[1] == 3)
    )
    assert actions_2[1] != actions[1]


# test random env step
def test_random_env_step(printer, environment: MA_CTP_General):
    key = jax.random.PRNGKey(31)
    key, subkey = jax.random.split(key)
    initial_env_state, initial_belief_states = environment.reset(key)
    agent = Random_Agent(environment, 10, -1.5, 2)
    runner_state = (initial_env_state, initial_belief_states, key, jnp.int32(0))
    runner_state, traj_batch = agent.env_step(runner_state, 0)
    assert traj_batch.shape == (2, 6, 7, 5)
