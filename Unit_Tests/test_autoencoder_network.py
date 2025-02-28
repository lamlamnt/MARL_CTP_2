import jax
import jax.numpy as jnp
import pytest
import pytest_print
import sys

sys.path.append("..")
from Networks.autoencoder import Encoder, Decoder, Autoencoder


def test_autoencoder():
    n_nodes = 10
    output_size = (6, 12, 10)
    autoencoder_model = Autoencoder(128, 64, output_size)
    key = jax.random.PRNGKey(100)
    params = autoencoder_model.init(key, jnp.ones((6, 12, 10)))
    x = jnp.ones((6, 12, 10))
    x_recon = autoencoder_model.apply(params, x)
    assert x_recon.shape == (6, 12, 10)


# def test_encoder():
