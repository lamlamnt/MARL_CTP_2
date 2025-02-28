from functools import partial
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn


# Loss function (e.g., Mean Squared Error for reconstruction)
def loss_fn(model, params, x):
    x_recon = model.apply(params, x)
    return jnp.mean((x - x_recon) ** 2)


# Use minibatch 1 -> maybe change to multiple minibatches
@jax.jit
def train_step(train_state, batch):
    def compute_loss(params):
        return loss_fn(params, batch)

    # Compute loss and gradients
    loss = compute_loss(train_state.params)
    grad_fn = jax.grad(compute_loss)
    grads = grad_fn(train_state.params)

    # Update the train state
    train_state = train_state.apply_gradients(grads=grads)

    # Return the updated train state and the loss
    return train_state, loss
