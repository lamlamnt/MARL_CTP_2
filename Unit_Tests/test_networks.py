import jax
import jax.numpy as jnp
import pytest
import pytest_print
import haiku as hk
import sys

sys.path.append("..")
from Networks.densenet import DenseNet_ActorCritic, DenseNet_ActorCritic_Same


def test_densenet(printer):
    model = DenseNet_ActorCritic(10)
    key = jax.random.PRNGKey(100)
    params = model.init(key, jnp.ones((6, 12, 10)))
    action_values, critic = model.apply(params, jnp.ones((6, 12, 10)))
    assert action_values.probs.shape == (11,)
    assert critic.shape == ()
    x = jnp.ones((2, 6, 12, 10))
    pi, value = jax.vmap(model.apply, in_axes=(None, 0))(params, x)
    assert pi.probs.shape == (2, 11)
    assert value.shape == (2,)
