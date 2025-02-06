import pytest
import pytest_print
import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_environment
from Utils.augmented_belief_state import get_augmented_optimistic_pessimistic_belief


@pytest.fixture
def environment():
    key = jax.random.PRNGKey(30)
    environment = CTP_environment.MA_CTP_General(
        2, 5, key, prop_stoch=0.8, grid_size=10, num_stored_graphs=1
    )
    return environment


def test_augmented_belief_state(printer, environment: CTP_environment.MA_CTP_General):
    key = jax.random.PRNGKey(31)
    key, subkey = jax.random.split(key)
    initial_env_state, initial_belief_states = environment.reset(key)
    augmented_belief_states = get_augmented_optimistic_pessimistic_belief(
        initial_belief_states
    )
    assert augmented_belief_states.shape == (2, 6, 7, 5)
    # assert symmetrical matrices with no inf values
    assert jnp.array_equal(
        augmented_belief_states[0, 4:, :, :], augmented_belief_states[1, 4:, :, :]
    )
    assert jnp.all(
        augmented_belief_states[:, :, 2:, :]
        == augmented_belief_states[:, :, 2:, :].transpose(0, 1, 3, 2)
    )
    assert jnp.all(augmented_belief_states != jnp.inf)
    assert jnp.all(
        augmented_belief_states[0, 4, 2:, :] <= augmented_belief_states[0, 5, 2:, :]
    )
    assert not jnp.array_equal(
        augmented_belief_states[0, 4, :, :], augmented_belief_states[0, 5, :, :]
    )
