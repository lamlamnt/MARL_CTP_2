from functools import partial
import jax
import jax.numpy as jnp
import sys
import os
import argparse
import optax
from distutils.util import strtobool
import wandb
import yaml
import flax

sys.path.append("..")
from Environment.CTP_environment import MA_CTP_General, CTP_generator
from Utils.load_store_graphs import load_graphs
from Utils.augmented_belief_state import get_augmented_optimistic_pessimistic_belief


@partial(jax.jit, static_argnums=(0,))
def act(model, key, params, belief_states) -> tuple[jnp.array, jax.random.PRNGKey]:
    augmented_belief_states = get_augmented_optimistic_pessimistic_belief(belief_states)

    def _choose_action(belief_state):
        pi, _ = model.apply(params, belief_state)
        random_action = pi.sample(
            seed=key
        )  # use the same key for all agents (maybe not good)
        return random_action

    # Because we want diverse states, so will use non-deterministic inference policy
    actions = jax.vmap(_choose_action, in_axes=0)(augmented_belief_states)
    old_key, new_key = jax.random.split(key)
    return actions, new_key


def main():
    current_directory = os.getcwd()
    log_directory = os.path.join(current_directory, "Logs", args.log_directory)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Create 2 environments (training and testing)
    # Determine belief state shape
    key = jax.random.PRNGKey(args.random_seed_for_training)
    subkeys = jax.random.split(key, num=2)
    inference_key, environment_key = subkeys
    environment = MA_CTP_General(
        args.n_agent,
        args.n_node,
        environment_key,
        prop_stoch=args.prop_stoch,
        k_edges=args.k_edges,
        grid_size=args.n_node,
        reward_for_invalid_action=args.reward_for_invalid_action,
        reward_service_goal=args.reward_service_goal,
        reward_fail_to_service_goal_larger_index=args.reward_fail_to_service_goal_larger_index,
        num_stored_graphs=args.num_stored_graphs,
        loaded_graphs=training_graphs,
    )
    testing_environment = MA_CTP_General(
        args.n_agent,
        args.n_node,
        inference_key,
        prop_stoch=args.prop_stoch,
        k_edges=args.k_edges,
        grid_size=args.n_node,
        reward_for_invalid_action=args.reward_for_invalid_action,
        reward_service_goal=args.reward_service_goal,
        reward_fail_to_service_goal_larger_index=args.reward_fail_to_service_goal_larger_index,
        num_stored_graphs=args.num_stored_graphs,
        loaded_graphs=inference_graphs,
    )

    # Load in trained network. Don't need PPO agent because we just need the act function
    network_file_path = os.path.join(
        current_directory, "Logs", args.load_network_directory, "weights.flax"
    )
    with open(network_file_path, "rb") as f:
        init_params = flax.serialization.from_bytes(init_params, f.read())
    optimizer = optax.chain(
        optax.adam(learning_rate=args.learning_rate, eps=1e-5),
    )

    print("Start training ...")

    # Want random action but with invalid action masked out (add later)
    # Collect new trajectories every epoch?
    # Evaluate results - plot loss and store final loss in json file. Testing set
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse command-line arguments for this unit test"
    )
    # Hyperparameters relating to the environment
    parser.add_argument(
        "--n_node",
        type=int,
        help="Number of nodes in the graph",
        required=True,
    )
    parser.add_argument(
        "--n_agent",
        type=int,
        help="Number of agents in the environment",
        required=False,
        default=1,
    )
    parser.add_argument(
        "--reward_for_invalid_action", type=float, required=False, default=-200.0
    )
    parser.add_argument(
        "--reward_service_goal",
        type=int,
        help="Should be 0 or positive",
        required=False,
        default=-0.1,
    )
    parser.add_argument(
        "--reward_fail_to_service_goal_larger_index",
        type=float,
        required=False,
        default=-0.1,
    )
    parser.add_argument(
        "--reward_exceed_horizon",
        type=float,
        help="Should be equal to or more negative than -1",
        required=False,
        default=-1.5,
    )
    parser.add_argument(
        "--horizon_length_factor",
        type=int,
        help="Factor to multiply with number of nodes to get the maximum horizon length",
        required=False,
        default=2,
    )
    parser.add_argument(
        "--prop_stoch",
        type=float,
        help="Proportion of edges that are stochastic. Only specify either prop_stoch or k_edges.",
        required=False,
        default=0.4,
    )
    parser.add_argument(
        "--k_edges",
        type=int,
        help="Number of stochastic edges. Only specify either prop_stoch or k_edges",
        required=False,
        default=None,
    )
    parser.add_argument("--random_seed", type=int, required=False, default=100)

    # Hyperparameters relating to the loaded network
    parser.add_argument(
        "--network_type",
        type=str,
        required=False,
        help="Options: Densenet,Densenet_Same",
        default="Densenet_Same",
    )
    parser.add_argument("--densenet_bn_size", type=int, required=False, default=4)
    parser.add_argument("--densenet_growth_rate", type=int, required=False, default=32)
    parser.add_argument(
        "--densenet_num_layers",
        type=str,
        required=False,
        help="Num group of layers for each dense block in string format",
        default="4,4,4",
    )
    parser.add_argument(
        "--graph_identifier",
        type=str,
        required=False,
        default="node_10_agent_2_prop_0.4",
    )
    parser.add_argument(
        "--load_network_directory",
        type=str,
        default=None,
        help="Directory to load trained network weights from",
    )

    # Hyperparameters relating to training the autoencoder
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs to train the autoencoder",
        required=False,
        default=200,
    )
    parser.add_argument("--learning_rate", type=float, required=False, default=0.001)

    # Args related to running/managing experiments
    parser.add_argument(
        "--log_directory", type=str, help="Directory to store logs", required=True
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        help="offline/online/disabled",
        required=False,
        default="disabled",
    )
    parser.add_argument(
        "--wandb_project_name", type=str, required=False, default="no_name"
    )
    parser.add_argument(
        "--yaml_file", type=str, required=False, default="sweep_config_node_10.yaml"
    )
    parser.add_argument(
        "--wandb_sweep",
        type=lambda x: bool(strtobool(x)),
        default=False,
        required=False,
        help="Whether to use yaml file to do hyperparameter sweep (Bayesian optimization)",
    )
    parser.add_argument("--sweep_run_count", type=int, required=False, default=3)
    args = parser.parse_args()

    print("Checking validity and loading graphs ...")
    # Check args match and load graphs
    training_graphs, inference_graphs, num_training_graphs, num_inference_graphs = (
        load_graphs(args)
    )
    if args.wandb_sweep == False:
        # Initialize wandb project
        wandb.init(
            project=args.wandb_project_name,
            name=args.log_directory,
            config=vars(args),
            mode=args.wandb_mode,
        )
        main(args)
        wandb.finish()
    else:
        # Hyperparameter sweep
        print("Running hyperparameter sweep ...")
        if args.wandb_mode != "online":
            raise ValueError("Wandb mode must be online for hyperparameter sweep")
        with open(args.yaml_file, "r") as file:
            sweep_config = yaml.safe_load(file)
        sweep_id = wandb.sweep(
            sweep_config,
            project=args.wandb_project_name,
            entity="lam-lam-university-of-oxford",
        )

        def wrapper_function():
            with wandb.init() as run:
                config = run.config
                # Don't need to name the run using config values (run.name = ...) because it will be very long
                # Modify args based on config
                for key in config:
                    setattr(args, key, config[key])
                # Instead of using run.id, can concatenate parameters
                log_directory = os.path.join(
                    os.getcwd(), "Logs", args.wandb_project_name, run.name
                )
                args.log_directory = log_directory
                main(args)

        wandb.agent(sweep_id, function=wrapper_function, count=args.sweep_run_count)
