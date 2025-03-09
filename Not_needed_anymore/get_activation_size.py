import jax
import jax.numpy as jnp
import pytest
import pytest_print
import haiku as hk
import sys

sys.path.append("..")
from Networks.densenet import DenseNet_ActorCritic_Same

if __name__ == "__main__":
    model = DenseNet_ActorCritic_Same(30, growth_rate=40)
    key = jax.random.PRNGKey(100)
    params = model.init(key, jnp.ones((6, 34, 30)))
