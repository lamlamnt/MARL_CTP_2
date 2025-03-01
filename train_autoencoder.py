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
from Auto_encoder_related.ppo_agent import PPO_agent_collect_belief_states
from Networks.densenet import (
    DenseNet_ActorCritic,
    DenseNet_ActorCritic_Same,
    DenseNet_ActorCritic_2_Critic_Values,
    DenseNet_ActorCritic_Same_2_Critic_Values,
)
from Networks.autoencoder import Autoencoder
import flax.linen as nn
from jax_tqdm import scan_tqdm
from flax.training.train_state import TrainState
import time
from Auto_encoder_related.training_step import train_step, loss_fn
from Auto_encoder_related.plot_evaluate_autoencoder import (
    plot_store_results_autoencoder,
)

NUM_CHANNELS_IN_BELIEF_STATE = 6


def main(args):
    current_directory = os.getcwd()
    log_directory = os.path.join(current_directory, "Logs", args.log_directory)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    state_shape = (
        NUM_CHANNELS_IN_BELIEF_STATE,
        args.n_agent + args.n_node,
        args.n_node,
    )

    # Create 2 environments (training and testing)
    # Determine belief state shape
    key = jax.random.PRNGKey(args.random_seed)
    subkeys = jax.random.split(key, num=3)
    inference_key, environment_key, online_key = subkeys
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
        num_stored_graphs=num_training_graphs,
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
        num_stored_graphs=num_inference_graphs,
        loaded_graphs=inference_graphs,
    )

    # Load in trained network.
    if args.network_type == "Densenet" and args.num_critic_values == 1:
        ppo_model = DenseNet_ActorCritic(
            args.n_node,
            act_fn=nn.leaky_relu,
            densenet_kernel_init=nn.initializers.kaiming_normal(),
            bn_size=args.densenet_bn_size,
            growth_rate=args.densenet_growth_rate,
            num_layers=tuple(map(int, (args.densenet_num_layers).split(","))),
        )
    elif args.network_type == "Densenet_Same" and args.num_critic_values == 1:
        ppo_model = DenseNet_ActorCritic_Same(
            args.n_node,
            act_fn=nn.leaky_relu,
            densenet_kernel_init=nn.initializers.kaiming_normal(),
            bn_size=args.densenet_bn_size,
            growth_rate=args.densenet_growth_rate,
            num_layers=tuple(map(int, (args.densenet_num_layers).split(","))),
        )
    elif args.network_type == "Densenet" and args.num_critic_values == 2:
        ppo_model = DenseNet_ActorCritic_2_Critic_Values(
            args.n_node,
            act_fn=nn.leaky_relu,
            densenet_kernel_init=nn.initializers.kaiming_normal(),
            bn_size=args.densenet_bn_size,
            growth_rate=args.densenet_growth_rate,
            num_layers=tuple(map(int, (args.densenet_num_layers).split(","))),
        )
    else:
        ppo_model = DenseNet_ActorCritic_Same_2_Critic_Values(
            args.n_node,
            act_fn=nn.leaky_relu,
            densenet_kernel_init=nn.initializers.kaiming_normal(),
            bn_size=args.densenet_bn_size,
            growth_rate=args.densenet_growth_rate,
            num_layers=tuple(map(int, (args.densenet_num_layers).split(","))),
        )
    network_file_path = os.path.join(
        current_directory, "Logs", args.load_network_directory, "weights.flax"
    )
    with open(network_file_path, "rb") as f:
        ppo_init_params = ppo_model.init(
            jax.random.PRNGKey(0), jax.random.normal(online_key, state_shape)
        )
        ppo_init_params = flax.serialization.from_bytes(ppo_init_params, f.read())

    optimizer = optax.chain(
        optax.adamw(
            learning_rate=args.learning_rate, eps=1e-5, weight_decay=args.weight_decay
        ),
    )

    ppo_agent = PPO_agent_collect_belief_states(
        ppo_model,
        environment,
        args.horizon_length_factor * args.n_node,
        args.reward_exceed_horizon,
        args.n_agent,
    )
    testing_ppo_agent = PPO_agent_collect_belief_states(
        ppo_model,
        testing_environment,
        args.horizon_length_factor * args.n_node,
        args.reward_exceed_horizon,
        args.n_agent,
    )
    ppo_train_state = TrainState.create(
        apply_fn=ppo_model.apply,
        params=ppo_init_params,
        tx=optimizer,
    )

    # Initialize autoencoder
    latent_size = 18 * args.n_node - 30 + args.n_node * args.n_agent
    autoencoder_model = Autoencoder(
        hidden_size=args.hidden_size,
        latent_size=latent_size,
        output_size=state_shape,
    )
    autoencoder_init_params = autoencoder_model.init(
        jax.random.PRNGKey(0),
        jax.random.normal(online_key, (1,) + state_shape),
    )
    autoencoder_train_state = TrainState.create(
        apply_fn=autoencoder_model.apply,
        params=autoencoder_init_params,
        tx=optimizer,
    )
    init_key, env_action_key, evaluate_key = jax.vmap(jax.random.PRNGKey)(
        jnp.arange(3) + args.random_seed
    )
    new_env_state, new_belief_states = environment.reset(init_key)
    timestep_in_episode = jnp.int32(0)
    runner_state = (
        ppo_train_state,
        autoencoder_train_state,
        new_env_state,
        new_belief_states,
        env_action_key,
        timestep_in_episode,
    )

    # Use the testing environment to collect a validation set (used for all epochs)
    _, validation_set = jax.lax.scan(
        testing_ppo_agent.env_step, runner_state, None, args.validation_set_size
    )
    validation_set = jnp.reshape(validation_set, (-1,) + validation_set.shape[2:])

    print("Start training ...")

    # Want random action but with invalid action masked out (add later)
    # Collect new trajectories every epoch because not enough memory to store so much.
    @scan_tqdm(args.num_epochs)
    def _update_step(runner_state, unused):
        # Collect trajectories.
        runner_state, traj_batch = jax.lax.scan(
            ppo_agent.env_step, runner_state, None, args.num_steps_to_collect
        )
        (
            ppo_train_state,
            autoencoder_train_state,
            new_env_state,
            new_belief_states,
            env_key,
            timestep_in_episode,
        ) = runner_state
        # Update encoder
        autoencoder_train_state, training_loss = train_step(
            autoencoder_model, autoencoder_train_state, traj_batch
        )

        runner_state = (
            ppo_train_state,
            autoencoder_train_state,
            new_env_state,
            new_belief_states,
            env_key,
            timestep_in_episode,
        )

        # Perform inference on validation set to plot learning curve
        validation_loss = loss_fn(
            autoencoder_model, autoencoder_train_state.params, validation_set
        )
        metrics = {"training_loss": training_loss, "validation_loss": validation_loss}
        return runner_state, metrics

    start_training_time = time.time()
    autoencoder_train_state, metrics = jax.lax.scan(
        _update_step, runner_state, jnp.arange(args.num_epochs)
    )
    out = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,)), metrics)

    print("Start evaluation of trained autoencoder ...")
    # Evaluate results using testing set - plot loss and store final loss and args in json file.
    # Store autoencoder weights in a file
    plot_store_results_autoencoder(
        log_directory, start_training_time, autoencoder_train_state.params, out, args
    )


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
        default=-0.5,
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
    parser.add_argument(
        "--num_critic_values",
        type=int,
        required=False,
        default=1,
        help="Options: only 1 or 2. 2 means it tries to estimate both the individual and team reward",
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
        default="normalized_node_10_agent_2_prop_0.8",
    )
    parser.add_argument(
        "--load_network_directory",
        type=str,
        default=None,
        required=True,
        help="Directory to load trained network weights for ppo from",
    )
    parser.add_argument(
        "--autoencoder_weights_file", type=str, required=True, default=None
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
    parser.add_argument(
        "--num_steps_to_collect",
        type=int,
        help="Number of environment steps to collect before updating the encoder",
        required=False,
        default=2000,
    )
    parser.add_argument("--weight_decay", type=float, required=False, default=0.0001)
    parser.add_argument(
        "--hidden_size",
        type=int,
        required=False,
        default=48,
        help="Number of channels of the first convolutional layer in the encoder",
    )

    # Args related to evaluation
    parser.add_argument(
        "--validation_set_size",
        type=int,
        help="Number of different belief states to use for validation",
        required=False,
        default=1000,
    )

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
