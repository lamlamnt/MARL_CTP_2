import jax
import jax.numpy as jnp
import pytest
import pytest_print
import sys

sys.path.append("..")
from Networks.autoencoder import Encoder, Decoder, Autoencoder
from Networks.densenet_after_autoencoder import Densenet_1D
from flax.training.train_state import TrainState
import optax
import os
import flax


def test_autoencoder(printer):
    n_nodes = 10
    n_agents = 2
    output_size = (6, 12, 10)
    # reduced from 720 to 170
    autoencoder_model = Autoencoder(170, 48, output_size)
    key = jax.random.PRNGKey(100)
    params = autoencoder_model.init(key, jnp.ones((1, 6, 12, 10)))
    x = jnp.ones((5, 6, 12, 10))
    x_latent, x_recon = autoencoder_model.apply(params, x)
    assert x_recon.shape == (5, 6, 12, 10)
    assert x_latent.shape == (5, 170)

    # vmap over agents
    x_2_agents = jnp.ones((5, 2, 6, 12, 10))
    x_swapped = jnp.swapaxes(x_2_agents, 0, 1)
    x_latent, x_recon = jax.vmap(autoencoder_model.apply, in_axes=(None, 0))(
        params, x_swapped
    )
    x_latent_swapped = jnp.swapaxes(x_latent, 0, 1)
    assert x_latent_swapped.shape == (5, 2, 170)


def test_densenet_after_autoencoder(printer):
    input = jnp.ones(170)
    model = Densenet_1D(num_classes=10, num_layers=(2, 2, 2), growth_rate=12, bn_size=2)
    key = jax.random.PRNGKey(100)
    action_mask = jnp.ones(11)
    params = model.init(key, input, action_mask)
    pi, value = model.apply(params, input, action_mask)
    assert pi.probs.shape == (11,)
    assert value.shape == ()
