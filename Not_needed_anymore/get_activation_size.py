import jax
import jax.numpy as jnp
import pytest
import pytest_print
import haiku as hk
import sys

sys.path.append("..")
from Networks.densenet import DenseNet_ActorCritic_Same
from Networks.autoencoder import Autoencoder
from Networks.densenet_after_autoencoder import Densenet_1D

if __name__ == "__main__":
    # model = DenseNet_ActorCritic_Same(30, growth_rate=40)
    key = jax.random.PRNGKey(100)
    # params = model.init(key, jnp.ones((6, 34, 30)))

    # model = Autoencoder(170, 96, (6, 12, 10), 3, 2)
    # params = model.init(key, jnp.ones((1, 6, 12, 10)))

    action_mask = jnp.ones(11)
    model = Densenet_1D(10)
    params = model.init(key, jnp.ones((1, 170)), action_mask)
