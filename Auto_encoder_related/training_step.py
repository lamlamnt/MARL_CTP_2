from functools import partial
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn


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
