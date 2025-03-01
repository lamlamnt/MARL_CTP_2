import jax
import jax.numpy as jnp
import pytest
import pytest_print
import sys

sys.path.append("..")
from Networks.autoencoder import Encoder, Decoder, Autoencoder
from flax.training.train_state import TrainState
import optax
import os
import flax


def extract_params(params, prefix=""):
    """Recursively extract layer names and weights from a parameter dictionary."""
    for name, value in params.items():
        full_name = f"{prefix}/{name}" if prefix else name
        if isinstance(value, dict):  # If the value is a nested dictionary, recurse
            yield from extract_params(value, prefix=full_name)
        else:  # If the value is a weight array, yield it
            yield full_name, value


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
