import jax
import jax.numpy as jnp
import pytest
import pytest_print
import haiku as hk
import sys

sys.path.append("..")
from Networks.densenet import DenseNet_ActorCritic_Same
from Networks.autoencoder import Autoencoder

if __name__ == "__main__":
    # model = DenseNet_ActorCritic_Same(30, growth_rate=40)
    key = jax.random.PRNGKey(100)
    # params = model.init(key, jnp.ones((6, 34, 30)))

    model = Autoencoder(170, 32, (6, 12, 10), 3, 2)
    params = model.init(key, jnp.ones((1, 6, 12, 10)))
