import jax
import jax.numpy as jnp
import pytest
import pytest_print
import sys

sys.path.append("..")
from Networks.autoencoder import Encoder, Decoder, Autoencoder
from flax.training.train_state import TrainState
import optax

# def test_autoencoder(printer):
if __name__ == "__main__":
    n_nodes = 10
    n_agents = 2
    output_size = (6, 12, 10)
    # reduced from 720 to 170
    autoencoder_model = Autoencoder(170, 48, output_size)
    key = jax.random.PRNGKey(100)
    params = autoencoder_model.init(key, jnp.ones((1, 6, 12, 10)))
    x = jnp.ones((5, 6, 12, 10))
    x_recon = autoencoder_model.apply(params, x)
    assert x_recon.shape == (5, 6, 12, 10)

    # test apply encoder
    optimizer = optax.chain(
        optax.adamw(learning_rate=0.001, eps=1e-5, weight_decay=0.0001),
    )
    autoencoder_train_state = TrainState.create(
        apply_fn=autoencoder_model.apply,
        params=params,
        tx=optimizer,
    )
    latent_representations = autoencoder_model.apply(
        autoencoder_train_state.params,
        x,
        method=autoencoder_model.encoder,
    )
    print(latent_representations.shape)
    assert latent_representations.shape == (5, 170)
