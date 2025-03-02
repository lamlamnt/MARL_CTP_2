from functools import partial
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import sys

sys.path.append("..")
from Utils.invalid_action_masking import decide_validity_of_action_space


# Loss function (e.g., Mean Squared Error for reconstruction)
def loss_fn(model, params, x):
    x_latent, x_recon = model.apply(params, x)
    return jnp.mean((x - x_recon) ** 2)


# Use minibatch 1 -> maybe change to multiple minibatches
@partial(jax.jit, static_argnums=(0,))
def train_step(model, train_state, batch):
    def compute_loss(params):
        return loss_fn(model, params, batch)

    # Combine the agent dimension of the batch
    batch = jnp.reshape(batch, (-1,) + batch.shape[2:])

    # Compute loss and gradients
    loss = compute_loss(train_state.params)
    grad_fn = jax.grad(compute_loss)
    grads = grad_fn(train_state.params)

    # Update the train state
    train_state = train_state.apply_gradients(grads=grads)

    # Return the updated train state and the loss
    return train_state, loss


# for the training loop in the main.py file
def get_last_critic_val_autoencoder_version(
    autoencoder_model,
    autoencoder_params,
    augmented_state,
    densenet_model,
    densenet_train_state,
):
    augmented_state = autoencoder_model.apply(autoencoder_params, augmented_state)
    action_mask = decide_validity_of_action_space(augmented_state)
    _, last_critic_val = jax.vmap(densenet_model.apply, in_axes=(None, 0))(
        densenet_train_state.params, augmented_state, action_mask
    )
    return last_critic_val


def get_last_critic_val_normal_version(
    densenet_model, densenet_train_state, augmented_state
):
    _, last_critic_val = jax.vmap(densenet_model.apply, in_axes=(None, 0))(
        densenet_train_state.params, augmented_state
    )
    return last_critic_val
