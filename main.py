import os
import jax
import jax.numpy as jnp
from Environment.CTP_environment import MA_CTP_General
from Environment import CTP_generator
from Networks.densenet import (
    DenseNet_ActorCritic,
    DenseNet_ActorCritic_Same,
    DenseNet_ActorCritic_Same_2_Critic_Values,
    DenseNet_ActorCritic_2_Critic_Values,
)
from Networks.densenet_after_autoencoder import Densenet_1D
from Agents.ppo import PPO
from Agents.ppo_combine_individual_team import PPO_2_Critic_Values
from Agents.ppo_autoencoder import PPO_Autoencoder
import argparse
import optax
from flax.training.train_state import TrainState
from typing import Sequence, NamedTuple, Any
import flax
import time
from datetime import datetime
import json
import numpy as np
import wandb
from distutils.util import strtobool
from jax_tqdm import scan_tqdm
import warnings
import flax.linen as nn
import sys
import yaml
from flax.core.frozen_dict import FrozenDict
from Utils.augmented_belief_state import get_augmented_optimistic_pessimistic_belief
from Evaluation.inference import plotting_inference
from Evaluation.inference_during_training import get_average_testing_stats
from Utils.load_store_graphs import load_graphs, store_graphs
from Utils.hand_crafted_graphs import (
    get_go_past_goal_without_servicing_graph,
    get_smaller_index_agent_behaves_differently_graph,
    get_sacrifice_in_exploring_graph,
    get_sacrifice_in_choosing_goals_graph,
    get_dynamic_choose_goal_graph,
)
from Networks.autoencoder import Autoencoder
import re
from Utils.invalid_action_masking import decide_validity_of_action_space

NUM_CHANNELS_IN_BELIEF_STATE = 6


def decide_hand_crafted_graph(args):
    if args.hand_crafted_graph == "sacrifice_in_choosing_goals":
        n_node, defined_graph = get_sacrifice_in_choosing_goals_graph()
    elif args.hand_crafted_graph == "sacrifice_in_exploring":
        n_node, defined_graph = get_sacrifice_in_exploring_graph()
    elif args.hand_crafted_graph == "smaller_index_agent_behaves_differently":
        n_node, defined_graph = get_smaller_index_agent_behaves_differently_graph()
    elif args.hand_crafted_graph == "go_past_goal_without_servicing":
        n_node, defined_graph = get_go_past_goal_without_servicing_graph()
    elif args.hand_crafted_graph == "dynamic_choose_goals":
        n_node, defined_graph = get_dynamic_choose_goal_graph()
    else:
        raise ValueError("Invalid hand_crafted_graph")
    return n_node, defined_graph


def main(args):
    current_directory = os.getcwd()
    log_directory = os.path.join(current_directory, "Logs", args.log_directory)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    num_loops = args.time_steps // args.num_steps_before_update
    if args.ent_coeff_schedule == "sigmoid_checkpoint":
        assert num_loops < args.sigmoid_total_nums_all // args.num_steps_before_update
        assert args.sigmoid_beginning_offset_num < args.sigmoid_total_nums_all
    n_node = args.n_node
    if args.num_critic_values != 1 and args.num_critic_values != 2:
        raise ValueError("num_critic_values must be 1 or 2")

    # Initialize and setting things up
    print("Setting up the environment ...")
    # Determine belief state shape
    state_shape = (
        NUM_CHANNELS_IN_BELIEF_STATE,
        args.n_agent + n_node,
        n_node,
    )
    key = jax.random.PRNGKey(args.random_seed_for_training)
    subkeys = jax.random.split(key, num=2)
    online_key, environment_key = subkeys

    # Hand crafted graphs
    if args.hand_crafted_graph != "None":
        assert args.num_stored_graphs == 1
        assert args.graph_mode == "generate"
        n_node, defined_graph = decide_hand_crafted_graph(args)
        environment = MA_CTP_General(
            args.n_agent,
            n_node,
            environment_key,
            prop_stoch=args.prop_stoch,
            k_edges=args.k_edges,
            grid_size=n_node,
            reward_for_invalid_action=args.reward_for_invalid_action,
            reward_service_goal=args.reward_service_goal,
            reward_fail_to_service_goal_larger_index=args.reward_fail_to_service_goal_larger_index,
            num_stored_graphs=args.num_stored_graphs,
            loaded_graphs=defined_graph,
        )
    else:
        # Create the training environment
        environment = MA_CTP_General(
            args.n_agent,
            n_node,
            environment_key,
            prop_stoch=args.prop_stoch,
            k_edges=args.k_edges,
            grid_size=n_node,
            reward_for_invalid_action=args.reward_for_invalid_action,
            reward_service_goal=args.reward_service_goal,
            reward_fail_to_service_goal_larger_index=args.reward_fail_to_service_goal_larger_index,
            num_stored_graphs=num_training_graphs,
            loaded_graphs=training_graphs,
        )

    # Create the testing environment
    if args.num_stored_graphs == 1:
        print(
            "Test environment is the same as training environment. Including graph and reward for failing to service a goal due to smaller index agent"
        )
        testing_environment = environment
    else:
        print("Test graphs are different from training graphs")
        inference_key = jax.random.PRNGKey(args.random_seed_for_inference)
        testing_environment = MA_CTP_General(
            args.n_agent,
            n_node,
            inference_key,
            prop_stoch=args.prop_stoch,
            k_edges=args.k_edges,
            grid_size=n_node,
            reward_for_invalid_action=args.reward_for_invalid_action,
            reward_service_goal=args.reward_service_goal,
            reward_fail_to_service_goal_larger_index=args.reward_service_goal,
            num_stored_graphs=num_inference_graphs,
            loaded_graphs=inference_graphs,
        )
    if args.autoencoder_weights:
        model = Densenet_1D(
            n_node,
            act_fn=nn.leaky_relu,
            densenet_kernel_init=nn.initializers.kaiming_normal(),
            bn_size=args.densenet_bn_size,
            growth_rate=args.densenet_growth_rate,
            num_layers=tuple(map(int, (args.densenet_num_layers).split(","))),
        )
    elif args.network_type == "Densenet" and args.num_critic_values == 1:
        model = DenseNet_ActorCritic(
            n_node,
            act_fn=nn.leaky_relu,
            densenet_kernel_init=nn.initializers.kaiming_normal(),
            bn_size=args.densenet_bn_size,
            growth_rate=args.densenet_growth_rate,
            num_layers=tuple(map(int, (args.densenet_num_layers).split(","))),
        )
    elif args.network_type == "Densenet_Same" and args.num_critic_values == 1:
        model = DenseNet_ActorCritic_Same(
            n_node,
            act_fn=nn.leaky_relu,
            densenet_kernel_init=nn.initializers.kaiming_normal(),
            bn_size=args.densenet_bn_size,
            growth_rate=args.densenet_growth_rate,
            num_layers=tuple(map(int, (args.densenet_num_layers).split(","))),
        )
    elif args.network_type == "Densenet" and args.num_critic_values == 2:
        model = DenseNet_ActorCritic_2_Critic_Values(
            n_node,
            act_fn=nn.leaky_relu,
            densenet_kernel_init=nn.initializers.kaiming_normal(),
            bn_size=args.densenet_bn_size,
            growth_rate=args.densenet_growth_rate,
            num_layers=tuple(map(int, (args.densenet_num_layers).split(","))),
        )
    else:
        model = DenseNet_ActorCritic_Same_2_Critic_Values(
            n_node,
            act_fn=nn.leaky_relu,
            densenet_kernel_init=nn.initializers.kaiming_normal(),
            bn_size=args.densenet_bn_size,
            growth_rate=args.densenet_growth_rate,
            num_layers=tuple(map(int, (args.densenet_num_layers).split(","))),
        )

    # Load autoencoder weights if using autoencoder
    if args.autoencoder_weights:
        if args.num_critic_values > 1:
            raise ValueError("Autoencoder for 2 critic values not implemented yet")
        if args.network_type != "Densenet_Autoencoder":
            raise ValueError("Autoencoder must be used with Densenet_Autoencoder")
        # Load autoencoder's properties from file
        with open(
            os.path.join(
                os.getcwd(),
                "Trained_encoder_logs",
                args.autoencoder_weights,
                "Hyperparameters_Results.json",
            ),
            "r",
        ) as f:
            content = f.read()  # Read the full file as text
            # Extract everything between the first { and }
        match = re.search(r"\{.*?\}", content, re.DOTALL)
        first_json_str = match.group(0)  # Extract the JSON substring
        first_json_dict = json.loads(first_json_str)  # Convert to dictionary
        latent_size = first_json_dict["latent_size"]
        autoencoder_model = Autoencoder(
            hidden_size=first_json_dict["hidden_size"],
            latent_size=latent_size,
            output_size=state_shape,
        )
        autoencoder_weights_path = os.path.join(
            current_directory,
            "Trained_encoder_logs",
            args.autoencoder_weights,
            "weights.flax",
        )
        with open(autoencoder_weights_path, "rb") as f:
            autoencoder_params = autoencoder_model.init(
                jax.random.PRNGKey(0), jax.random.normal(online_key, (1,) + state_shape)
            )
            autoencoder_params = flax.serialization.from_bytes(
                autoencoder_params, f.read()
            )
        init_params = model.init(
            jax.random.PRNGKey(0),
            jax.random.normal(online_key, (latent_size,)),
            jnp.ones(n_node + 1),
        )
    else:
        init_params = model.init(
            jax.random.PRNGKey(0),
            jax.random.normal(online_key, state_shape),
        )
    # Load in pre-trained network weights
    if args.load_network_directory is not None:
        network_file_path = os.path.join(
            current_directory, "Logs", args.load_network_directory, "weights.flax"
        )
        with open(network_file_path, "rb") as f:
            init_params = flax.serialization.from_bytes(init_params, f.read())
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.optimizer_norm_clip),
        optax.adam(learning_rate=args.learning_rate, eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=init_params,
        tx=optimizer,
    )
    init_key, env_action_key = jax.vmap(jax.random.PRNGKey)(
        jnp.arange(2) + args.random_seed_for_training
    )

    if args.autoencoder_weights:
        agent = PPO_Autoencoder(
            model,
            environment,
            args.discount_factor,
            args.gae_lambda,
            args.clip_eps,
            args.vf_coeff,
            args.ent_coeff,
            batch_size=args.num_steps_before_update,
            num_minibatches=args.num_minibatches,
            horizon_length=args.horizon_length_factor * n_node,
            reward_exceed_horizon=args.reward_exceed_horizon,
            num_loops=num_loops,
            anneal_ent_coeff=args.anneal_ent_coeff,
            deterministic_inference_policy=args.deterministic_inference_policy,
            ent_coeff_schedule=args.ent_coeff_schedule,
            sigmoid_beginning_offset_num=args.sigmoid_beginning_offset_num
            // args.num_steps_before_update,
            sigmoid_total_nums_all=args.sigmoid_total_nums_all
            // args.num_steps_before_update,
            num_agents=args.n_agent,
            reward_service_goal=args.reward_service_goal,
            individual_reward_weight=args.individual_reward_weight,
            individual_reward_weight_schedule=args.anneal_individual_reward_weight,
            autoencoder_model=autoencoder_model,
            autoencoder_params=autoencoder_params,
        )
    elif args.num_critic_values == 1:
        agent = PPO(
            model,
            environment,
            args.discount_factor,
            args.gae_lambda,
            args.clip_eps,
            args.vf_coeff,
            args.ent_coeff,
            batch_size=args.num_steps_before_update,
            num_minibatches=args.num_minibatches,
            horizon_length=args.horizon_length_factor * n_node,
            reward_exceed_horizon=args.reward_exceed_horizon,
            num_loops=num_loops,
            anneal_ent_coeff=args.anneal_ent_coeff,
            deterministic_inference_policy=args.deterministic_inference_policy,
            ent_coeff_schedule=args.ent_coeff_schedule,
            sigmoid_beginning_offset_num=args.sigmoid_beginning_offset_num
            // args.num_steps_before_update,
            sigmoid_total_nums_all=args.sigmoid_total_nums_all
            // args.num_steps_before_update,
            num_agents=args.n_agent,
            reward_service_goal=args.reward_service_goal,
            individual_reward_weight=args.individual_reward_weight,
            individual_reward_weight_schedule=args.anneal_individual_reward_weight,
        )
    else:
        agent = PPO_2_Critic_Values(
            model,
            environment,
            args.discount_factor,
            args.gae_lambda,
            args.clip_eps,
            args.vf_coeff,
            args.ent_coeff,
            batch_size=args.num_steps_before_update,
            num_minibatches=args.num_minibatches,
            horizon_length=args.horizon_length_factor * n_node,
            reward_exceed_horizon=args.reward_exceed_horizon,
            num_loops=num_loops,
            anneal_ent_coeff=args.anneal_ent_coeff,
            deterministic_inference_policy=args.deterministic_inference_policy,
            ent_coeff_schedule=args.ent_coeff_schedule,
            sigmoid_beginning_offset_num=args.sigmoid_beginning_offset_num
            // args.num_steps_before_update,
            sigmoid_total_nums_all=args.sigmoid_total_nums_all
            // args.num_steps_before_update,
            num_agents=args.n_agent,
            reward_service_goal=args.reward_service_goal,
            individual_reward_weight=args.individual_reward_weight,
            individual_reward_weight_schedule=args.anneal_individual_reward_weight,
        )

    # For the purpose of plotting the learning curve
    arguments = FrozenDict(
        {
            "factor_testing_timesteps": args.factor_testing_timesteps,
            "n_node": args.n_node,
            "reward_exceed_horizon": args.reward_exceed_horizon,
            "horizon_length_factor": args.horizon_length_factor,
            "random_seed_for_inference": args.random_seed_for_inference,
            "n_agent": args.n_agent,
            "reward_service_goal": args.reward_service_goal,
        }
    )

    print("Start training ...")

    @scan_tqdm(num_loops)
    def _update_step(runner_state, unused):
        # Collect trajectories
        runner_state, traj_batch = jax.lax.scan(
            agent.env_step, runner_state, None, args.num_steps_before_update
        )
        # Calculate advantages
        # timestep_in_episode is unused here
        (
            train_state,
            new_env_state,
            current_belief_states,
            key,
            timestep_in_episode,
            loop_count,
            previous_episode_done,
        ) = runner_state
        augmented_state = get_augmented_optimistic_pessimistic_belief(
            current_belief_states
        )

        _, last_critic_val = jax.vmap(model.apply, in_axes=(None, 0))(
            train_state.params, augmented_state
        )

        advantages, targets = agent.calculate_gae(
            traj_batch, last_critic_val, loop_count
        )
        # advantages and targets are of shape (num_steps_before_update,num_agents)

        # Update the network
        update_state = (train_state, traj_batch, advantages, targets, key, loop_count)
        update_state, total_loss = jax.lax.scan(
            agent._update_epoch, update_state, None, args.num_update_epochs
        )
        train_state = update_state[0]
        rng = update_state[-2]

        loop_count += 1

        runner_state = (
            train_state,
            new_env_state,
            current_belief_states,
            rng,
            timestep_in_episode,
            loop_count,
            previous_episode_done,
        )

        # Perform inference (using testing environment) (if loop_count divisible by 50 for example - tunable)
        # Get average and store in metrics, just like loss
        (
            testing_average_competitive_ratio,
            testing_average_competitive_ratio_exclude,
            testing_failure_rate,
        ) = jax.lax.cond(
            loop_count % args.frequency_testing == 0,
            lambda _: get_average_testing_stats(
                testing_environment, agent, train_state.params, arguments
            ),
            lambda _: (jnp.float16(0.0), jnp.float16(0, 0), jnp.float16(0, 0)),
            None,
        )

        # Collect metrics
        metrics = {
            "losses": total_loss,
            "all_total_rewards": jnp.sum(
                traj_batch.reward, axis=1
            ),  # doing this results in all_rewards with shape (num_timesteps,)
            "all_episode_done": jnp.all(traj_batch.done, axis=1),
            "all_optimal_costs": traj_batch.shortest_path,
            "testing_average_competitive_ratio": testing_average_competitive_ratio,
            "testing_average_competitive_ratio_exclude": testing_average_competitive_ratio_exclude,
            "testing_failure_rate": testing_failure_rate,
        }
        return runner_state, metrics

    @scan_tqdm(num_loops)
    def _update_step_autoencoder(runner_state, unused):
        # Collect trajectories
        runner_state, traj_batch = jax.lax.scan(
            agent.env_step, runner_state, None, args.num_steps_before_update
        )
        # Calculate advantages
        # timestep_in_episode is unused here
        (
            train_state,
            new_env_state,
            current_belief_states,
            key,
            timestep_in_episode,
            loop_count,
            previous_episode_done,
        ) = runner_state
        augmented_state = get_augmented_optimistic_pessimistic_belief(
            current_belief_states
        )

        action_mask = jax.vmap(decide_validity_of_action_space)(augmented_state)
        latent_state, _ = autoencoder_model.apply(autoencoder_params, augmented_state)
        _, last_critic_val = jax.vmap(model.apply, in_axes=(None, 0, 0))(
            train_state.params, latent_state, action_mask
        )

        advantages, targets = agent.calculate_gae(
            traj_batch, last_critic_val, loop_count
        )
        # advantages and targets are of shape (num_steps_before_update,num_agents)

        # Update the network
        update_state = (train_state, traj_batch, advantages, targets, key, loop_count)
        update_state, total_loss = jax.lax.scan(
            agent._update_epoch, update_state, None, args.num_update_epochs
        )
        train_state = update_state[0]
        rng = update_state[-2]

        loop_count += 1

        runner_state = (
            train_state,
            new_env_state,
            current_belief_states,
            rng,
            timestep_in_episode,
            loop_count,
            previous_episode_done,
        )

        # Perform inference (using testing environment) (if loop_count divisible by 50 for example - tunable)
        # Get average and store in metrics, just like loss
        (
            testing_average_competitive_ratio,
            testing_average_competitive_ratio_exclude,
            testing_failure_rate,
        ) = jax.lax.cond(
            loop_count % args.frequency_testing == 0,
            lambda _: get_average_testing_stats(
                testing_environment, agent, train_state.params, arguments
            ),
            lambda _: (jnp.float16(0.0), jnp.float16(0, 0), jnp.float16(0, 0)),
            None,
        )

        # Collect metrics
        metrics = {
            "losses": total_loss,
            "all_total_rewards": jnp.sum(
                traj_batch.reward, axis=1
            ),  # doing this results in all_rewards with shape (num_timesteps,)
            "all_episode_done": jnp.all(traj_batch.done, axis=1),
            "all_optimal_costs": traj_batch.shortest_path,
            "testing_average_competitive_ratio": testing_average_competitive_ratio,
            "testing_average_competitive_ratio_exclude": testing_average_competitive_ratio_exclude,
            "testing_failure_rate": testing_failure_rate,
        }
        return runner_state, metrics

    start_training_time = time.time()
    new_env_state, new_belief_states = environment.reset(init_key)
    timestep_in_episode = jnp.int32(0)
    loop_count = jnp.int32(0)
    runner_state = (
        train_state,
        new_env_state,
        new_belief_states,
        env_action_key,
        timestep_in_episode,
        loop_count,
        jnp.bool_(True),
    )
    if not args.autoencoder_weights:
        print(
            "Using normal adjacency matrix representation as input to the actor and critic networks"
        )
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(num_loops)
        )
    else:
        print("Using encoder before actor and critic networks")
        runner_state, metrics = jax.lax.scan(
            _update_step_autoencoder, runner_state, jnp.arange(num_loops)
        )
    train_state = runner_state[0]
    # Metrics will be stacked. Get episode done from all_done and total rewards from all_rewards
    out = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,)), metrics)

    plotting_inference(
        log_directory,
        start_training_time,
        train_state.params,
        out,
        testing_environment,
        agent,
        args,
        n_node,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse command-line arguments for this unit test"
    )
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
        "--time_steps",
        type=int,
        help="Probably around num_episodes you want * num_nodes* 2",
        required=False,
        default=1000000,
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
        default=-3.0,
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
    parser.add_argument(
        "--random_seed_for_training", type=int, required=False, default=100
    )
    parser.add_argument(
        "--random_seed_for_inference", type=int, required=False, default=101
    )
    parser.add_argument("--discount_factor", type=float, required=False, default=1.0)
    parser.add_argument("--learning_rate", type=float, required=False, default=0.001)
    parser.add_argument(
        "--num_update_epochs",
        type=int,
        help="After collecting trajectories, how many times each minibatch is updated.",
        required=False,
        default=6,
    )
    parser.add_argument(
        "--network_type",
        type=str,
        required=False,
        help="Options: Densenet,Densenet_Same, Densenet_Autoencoder",
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
        "--optimizer_norm_clip",
        type=float,
        required=False,
        help="optimizer.clip_by_global_norm(value)",
        default=2,
    )

    # Args related to running/managing experiments
    parser.add_argument(
        "--log_directory", type=str, help="Directory to store logs", required=True
    )
    parser.add_argument(
        "--hand_crafted_graph",
        type=str,
        help="Options: None, dynamic_choose_goals, sacrifice_in_choosing_goals, sacrifice_in_exploring, smaller_index_agent_behaves_differently, go_past_goal_without_servicing. If anything other than None is specified, all other args relating to environment such as num of nodes are ignored.",
        required=False,
        default="None",
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
    parser.add_argument(
        "--wandb_sweep_id",
        type=str,
        required=False,
        default=None,
        help="ID of a sweep in progress - to resume a sweep",
    )
    parser.add_argument("--sweep_run_count", type=int, required=False, default=3)
    parser.add_argument(
        "--factor_inference_timesteps",
        type=int,
        required=False,
        default=2000,
        help="Number to multiply with the number of nodes to get the total number of inference timesteps",
    )
    parser.add_argument(
        "--graph_mode",
        type=str,
        default="load",
        required=False,
        help="Options: generate,store,load",
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
    parser.add_argument(
        "--factor_testing_timesteps",
        type=int,
        required=False,
        default=50,
        help="Factor to multiple with number of nodes to get the number of timesteps to perform testing on during training (in order to plot the learning curve)",
    )
    parser.add_argument(
        "--frequency_testing",
        type=int,
        required=False,
        default=20,
        help="How many updates before performing testing during training to plot the learning curve",
    )
    parser.add_argument(
        "--learning_curve_average_window",
        type=int,
        default=5,
        help="Number of points to average over for the smoothened learning curve plot",
    )

    # Args specific to PPO:
    parser.add_argument(
        "--num_steps_before_update",
        type=int,
        help="How many timesteps to collect before updating the network",
        required=False,
        default=1000,
    )
    parser.add_argument(
        "--gae_lambda",
        help="Control the trade-off between bias and variance in advantage estimates. High = Low bias, High variance as it depends on longer trajectories",
        type=float,
        required=False,
        default=0.95,
    )
    parser.add_argument(
        "--clip_eps",
        help="Related to how big of an update can be made",
        type=float,
        required=False,
        default=0.14,
    )
    parser.add_argument(
        "--vf_coeff",
        help="Contribution of the value loss to the total loss",
        type=float,
        required=False,
        default=0.128,
    )
    parser.add_argument(
        "--ent_coeff",
        help="Exploration coefficient",
        type=float,
        required=False,
        default=0.174,
    )
    parser.add_argument(
        "--anneal_ent_coeff",
        type=lambda x: bool(strtobool(x)),
        required=False,
        default=True,
        help="Whether to anneal the entropy (exploration) coefficient",
    )
    parser.add_argument(
        "--ent_coeff_schedule",
        type=str,
        required=False,
        help="Options: sigmoid, sigmoid_checkpoint (for checkpoint training)",
        default="sigmoid",
    )
    parser.add_argument(
        "--num_minibatches",
        help="Related to how the trajectory batch is split up for performing updating of the network",
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument(
        "--deterministic_inference_policy",
        type=lambda x: bool(strtobool(x)),
        default=True,
        required=False,
        help="Whether to choose the action with the highest probability instead of sampling from the distribution",
    )
    parser.add_argument(
        "--num_stored_graphs",
        type=int,
        required=False,
        help="How many different graphs will be seen by the agent",
        default=1,
    )
    parser.add_argument(
        "--sigmoid_beginning_offset_num",
        type=int,
        required=False,
        default=0,
        help="For sigmoid ent coeff schedule checkpoint training. Unit: in number of timesteps. In the script, it will be divided by num_steps_before_update to convert to num_loops unit",
    )
    parser.add_argument(
        "--sigmoid_total_nums_all",
        type=int,
        required=False,
        default=10000,
        help="For sigmoid ent coeff schedule checkpoint training. Unit: in number of timesteps. In the script, it will be divided by num_steps_before_update to convert to num_loops unit",
    )

    # Hyperparameters specific to multi-agent
    parser.add_argument(
        "--num_critic_values",
        type=int,
        required=False,
        default=1,
        help="Options: only 1 or 2. 2 means it tries to estimate both the individual and team reward",
    )
    parser.add_argument(
        "--individual_reward_weight",
        type=float,
        required=False,
        default=1.0,
        help="1.0 means only use individual reward. 0 means only use team reward. Related to how GAE is calculated",
    )
    parser.add_argument(
        "--anneal_individual_reward_weight",
        type=str,
        default="linear",
        required=False,
        help="Options: constant, linear. The arg individual_reward_weight is the starting weight. Constant or linear annealing to 0.0 over time during training",
    )

    # Hyperparameters specific to autoencoder
    parser.add_argument(
        "--autoencoder_weights",
        type=str,
        required=False,
        default=None,
        help="Path to trained autoencoder weights",
    )

    args = parser.parse_args()
    if args.graph_mode == "store":
        print("Generating graphs for storage ...")
        store_graphs(args)
        sys.exit(0)
    elif args.graph_mode == "generate":
        training_graphs = None
        inference_graphs = None
    else:
        # Load
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
        if args.wandb_sweep_id is None:
            sweep_id = wandb.sweep(
                sweep_config,
                project=args.wandb_project_name,
                entity="lam-lam-university-of-oxford",
            )
        else:
            sweep_id = "lam-lam-university-of-oxford/" + args.wandb_sweep_id

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
