import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from functools import reduce
import operator
import sys

sys.path.append("..")
from Networks.densenet import (
    DenseNet_ActorCritic_Same,
)
from Networks.autoencoder import Autoencoder


def get_model_memory_usage(model, input_shape, batch_size):
    """
    Estimate theoretical memory required for training a Flax model in bytes (32-bit precision).
    """
    dummy_input = jnp.ones(input_shape)
    variables = model.init(jax.random.PRNGKey(0), dummy_input)

    param_sizes = [
        reduce(operator.mul, p.shape, 1)
        for p in jax.tree_util.tree_leaves(variables["params"])
    ]
    total_params = sum(param_sizes)
    print(total_params)
    param_memory = total_params * 4  # FP32 (4 bytes per parameter)

    grad_memory = param_memory  # Same size as parameters
    optimizer_memory = (
        2 * param_memory
    )  # Adam optimizer has two extra states per parameter

    def forward_and_capture(x):
        activations = []

        def capture_intermediates(layer, x):
            y = layer(x)
            activations.append(y)
            return y

        # Re-run the model manually, capturing activations
        y = model.apply(variables, x, method=capture_intermediates)
        return y, activations

    _, activations = forward_and_capture(dummy_input)
    print(activations)
    activation_sizes = [
        reduce(operator.mul, a.shape, 1) for a in jax.tree_util.tree_leaves(activations)
    ]
    print(activation_sizes)
    total_activations = sum(activation_sizes) * batch_size
    print(total_activations)
    activation_memory = total_activations * 4  # FP32 (4 bytes per activation)

    input_memory = reduce(operator.mul, dummy_input.shape, 1) * 4 * batch_size

    total_memory = (
        param_memory + grad_memory + optimizer_memory + activation_memory + input_memory
    )

    # Convert to GB
    return {
        "input_memory": input_memory / 1073741824,
        "param_memory": param_memory / 1073741824,
        "grad_memory": grad_memory / 1073741824,
        "optimizer_memory": optimizer_memory / 1073741824,
        "activation_memory": activation_memory / 1073741824,
        "total_memory": total_memory / 1073741824,
    }


if __name__ == "__main__":
    model = DenseNet_ActorCritic_Same(
        10, num_layers=(4, 4, 4), bn_size=4, growth_rate=32
    )
    input_shape = (6, 12, 10)
    batch_size = 4600
    memory = get_model_memory_usage(model, input_shape, batch_size)
    print(memory)
