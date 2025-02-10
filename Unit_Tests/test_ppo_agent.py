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


@pytest.fixture
def environment():
    key = jax.random.PRNGKey(30)
    environment = CTP_environment.MA_CTP_General(
        2, 10, key, prop_stoch=0.4, grid_size=10, num_stored_graphs=1
    )
    return environment


def test_env_step(environment: CTP_environment.MA_CTP_General):
    pass
