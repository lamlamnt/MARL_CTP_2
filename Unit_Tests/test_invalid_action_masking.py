import pytest
import pytest_print
import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_environment
from Utils.invalid_action_masking import decide_validity_of_action_space


@pytest.fixture
def environment():
    key = jax.random.PRNGKey(30)
    environment = CTP_environment.MA_CTP_General(
        2, 5, key, prop_stoch=0.8, grid_size=10, num_stored_graphs=1
    )
    return environment


def test_invalid_action_masking(printer, environment: CTP_environment.MA_CTP_General):
    key = jax.random.PRNGKey(31)
    key, subkey = jax.random.split(key)
    initial_env_state, initial_belief_states = environment.reset(key)
    action_mask = decide_validity_of_action_space(initial_belief_states[0])
    assert jnp.array_equal(
        action_mask, jnp.array([-jnp.inf, 1.0, 1.0, -jnp.inf, -jnp.inf, -jnp.inf])
    )
    action_mask = decide_validity_of_action_space(initial_belief_states[1])
    assert jnp.array_equal(
        action_mask, jnp.array([1.0, -jnp.inf, -jnp.inf, 1.0, -jnp.inf, -jnp.inf])
    )
    env_state_1, belief_state_1, rewards_1, done_1, subkey = environment.step(
        subkey, initial_env_state, initial_belief_states, jnp.array([2, 0])
    )
    action_mask = decide_validity_of_action_space(belief_state_1[0])
    assert jnp.array_equal(
        action_mask, jnp.array([1.0, -jnp.inf, -jnp.inf, 1.0, -jnp.inf, 1.0])
    )
    action_mask = decide_validity_of_action_space(belief_state_1[1])
    assert jnp.array_equal(
        action_mask, jnp.array([-jnp.inf, 1.0, 1.0, -jnp.inf, -jnp.inf, -jnp.inf])
    )
    env_state_2, belief_state_2, rewards_2, done_2, subkey = environment.step(
        subkey, env_state_1, belief_state_1, jnp.array([2, 2])
    )
    action_mask = decide_validity_of_action_space(belief_state_2[0])
    assert jnp.array_equal(
        action_mask, jnp.array([1.0, -jnp.inf, -jnp.inf, 1.0, -jnp.inf, 1.0])
    )
    action_mask = decide_validity_of_action_space(belief_state_2[1])
    assert jnp.array_equal(
        action_mask, jnp.array([1.0, -jnp.inf, -jnp.inf, 1.0, -jnp.inf, 1.0])
    )
    env_state_3, belief_state_3, rewards_3, done_3, subkey = environment.step(
        subkey, env_state_2, belief_state_2, jnp.array([5, 5])
    )
    action_mask = decide_validity_of_action_space(belief_state_3[1])
    assert jnp.array_equal(
        action_mask, jnp.array([1.0, -jnp.inf, -jnp.inf, 1.0, -jnp.inf, -jnp.inf])
    )
