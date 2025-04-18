import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from functools import reduce
import operator
import sys
import os

sys.path.append("..")
from Networks.densenet import (
    DenseNet_ActorCritic_Same,
)
from Networks.autoencoder import Autoencoder
from Networks.densenet_after_autoencoder import Densenet_1D


def get_model_memory_usage(
    model, input_shape, batch_size, activation_size, action_mask=None
):
    """
    Estimate theoretical memory required for training a Flax model in bytes (32-bit precision).
    """
    dummy_input = jnp.ones(input_shape)
    if action_mask is None:
        variables = model.init(jax.random.PRNGKey(0), dummy_input)
    else:
        variables = model.init(jax.random.PRNGKey(0), dummy_input, action_mask)

    param_sizes = [
        reduce(operator.mul, p.shape, 1)
        for p in jax.tree_util.tree_leaves(variables["params"])
    ]
    total_params = sum(param_sizes)
    param_memory = total_params * 4  # FP32 (4 bytes per parameter)

    grad_memory = param_memory  # Same size as parameters
    optimizer_memory = (
        2 * param_memory
    )  # Adam optimizer has two extra states per parameter
    param_grad_optimizer_memory = param_memory + grad_memory + optimizer_memory

    total_activations = activation_size * batch_size
    activation_memory = total_activations * 4  # FP32 (4 bytes per activation)

    input_memory = reduce(operator.mul, dummy_input.shape, 1) * 4 * batch_size

    total_memory = param_grad_optimizer_memory + activation_memory + input_memory

    # Convert to GB
    return {
        "input_memory": input_memory / 1073741824,
        "param_grad_optimizer_memory": param_grad_optimizer_memory / 1073741824,
        "activation_memory": activation_memory / 1073741824,
        "total_memory": total_memory / 1073741824,
    }


# parse in from .txt file
def get_total_output_size(file_name):
    directory = "C:\\Users\\shala\\Documents\\Oxford Undergrad\\4th Year\\4YP\\For Report Writing"
    total = 0
    with open(os.path.join(directory, file_name), "r") as file:
        for line in file:
            line = line.strip("()\n ")
            numbers = [int(num) for num in line.split(",") if num.strip()]
            product = 1
            for num in numbers:
                product *= num  # Multiply all numbers

            total += product
    return total


def non_autoencoder_get_memory():
    total_10_nodes_2_agents = get_total_output_size("Output_size_10_nodes_2_agents.txt")
    total_20_nodes_2_agents = get_total_output_size("Output_size_20_nodes_2_agents.txt")
    total_30_nodes_2_agents = get_total_output_size("Output_size_30_nodes_2_agents.txt")
    total_20_nodes_4_agents = get_total_output_size("Output_size_20_nodes_4_agents.txt")
    total_30_nodes_4_agents = get_total_output_size("Output_size_30_nodes_4_agents.txt")
    model = DenseNet_ActorCritic_Same(
        10,
        num_layers=(4, 4, 4),
        bn_size=4,
        growth_rate=32,
    )
    input_shape = (6, 12, 10)
    batch_size = 4600
    node_10_agent_2_memory = get_model_memory_usage(
        model, input_shape, batch_size, total_10_nodes_2_agents
    )
    print(total_10_nodes_2_agents)
    print(node_10_agent_2_memory)

    model = DenseNet_ActorCritic_Same(
        20,
        num_layers=(4, 4, 4),
        bn_size=4,
        growth_rate=40,
    )
    input_shape = (6, 22, 20)
    batch_size = 11000
    node_20_agent_2_memory = get_model_memory_usage(
        model, input_shape, batch_size, total_20_nodes_2_agents
    )
    print(total_20_nodes_2_agents)
    print(node_20_agent_2_memory)

    model = DenseNet_ActorCritic_Same(
        20,
        num_layers=(4, 4, 4),
        bn_size=4,
        growth_rate=40,
    )
    input_shape = (6, 24, 20)
    batch_size = 12000
    node_20_agent_4_memory = get_model_memory_usage(
        model, input_shape, batch_size, total_20_nodes_4_agents
    )
    print(total_20_nodes_4_agents)
    print(node_20_agent_4_memory)

    model = DenseNet_ActorCritic_Same(
        30,
        num_layers=(4, 4, 4),
        bn_size=4,
        growth_rate=40,
    )
    input_shape = (6, 32, 30)
    batch_size = 6000
    node_30_agent_2_memory = get_model_memory_usage(
        model, input_shape, batch_size, total_30_nodes_2_agents
    )
    print(total_30_nodes_2_agents)
    print(node_30_agent_2_memory)

    model = DenseNet_ActorCritic_Same(
        30,
        num_layers=(4, 4, 4),
        bn_size=4,
        growth_rate=40,
    )
    input_shape = (6, 34, 30)
    batch_size = 6000
    node_30_agent_4_memory = get_model_memory_usage(
        model, input_shape, batch_size, total_30_nodes_4_agents
    )
    print(total_30_nodes_4_agents)
    print(node_30_agent_4_memory)


def autoencoder_get_memory():
    autoencoder_activation_files = [
        "Output_size_autoencoder_170.txt",
        "Output_size_autoencoder_135.txt",
        "Output_size_autoencoder_100.txt",
        "Output_size_autoencoder_65.txt",
    ]
    policy_network_full_files = [
        "Output_size_densenet1d_170.txt",
        "Output_size_densenet1d_135.txt",
        "Output_size_densenet1d_100.txt",
    ]
    policy_network_small_files = [
        "Output_size_densenet1d_170_growth_rate_10_bn_size_2.txt",
        "Output_size_densenet1d_135_growth_rate_12_bn_size_2.txt",
        "Output_size_densenet1d_100_growth_rate_12_bn_size_2.txt",
    ]
    latent_sizes = [170, 135, 100, 65]
    growth_rate = [10, 12, 12]
    for i, file in enumerate(autoencoder_activation_files):
        autoencoder_activation = get_total_output_size(file)
        print(autoencoder_activation)
        model = Autoencoder(latent_sizes[i], 96, (6, 12, 10), 3, 2)
        batch_size = 1000
        input_shape = (1, 6, 12, 10)
        autoencoder_memory = get_model_memory_usage(
            model, input_shape, batch_size, autoencoder_activation
        )
        print(autoencoder_memory)
        if i < 3:
            policy_network_activation_full = get_total_output_size(
                policy_network_full_files[i]
            )
            print(policy_network_activation_full)
            batch_size = 4600
            input_shape = (1, latent_sizes[i])
            action_mask = jnp.ones(11)
            model = Densenet_1D(10)
            policy_network_memory_full = get_model_memory_usage(
                model,
                input_shape,
                batch_size,
                policy_network_activation_full,
                action_mask,
            )
            print(policy_network_memory_full)

            model = Densenet_1D(10, growth_rate=growth_rate[i], bn_size=2)
            policy_network_activation_small = get_total_output_size(
                policy_network_small_files[i]
            )
            print(policy_network_activation_small)
            policy_network_memory_full = get_model_memory_usage(
                model,
                input_shape,
                batch_size,
                policy_network_activation_small,
                action_mask,
            )
            print(policy_network_memory_full)


if __name__ == "__main__":
    # non_autoencoder_get_memory()
    autoencoder_get_memory()
