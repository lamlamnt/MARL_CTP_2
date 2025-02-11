import time
import flax
import os
import sys
import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime

sys.path.append("..")
from Environment.CTP_environment import MA_CTP_General
from Agents.optimistic_agent import Optimistic_Agent
from Utils.optimal_combination import get_optimal_combination_and_cost
from Evaluation.plotting import save_data_and_plotting, plot_learning_curve


# Get layer name and weights from FLAX params
def extract_params(params, prefix=""):
    """Recursively extract layer names and weights from a parameter dictionary."""
    for name, value in params.items():
        full_name = f"{prefix}/{name}" if prefix else name
        if isinstance(value, dict):  # If the value is a nested dictionary, recurse
            yield from extract_params(value, prefix=full_name)
        else:  # If the value is a weight array, yield it
            yield full_name, value


def plotting_inference(
    log_directory,
    start_time,
    model_params,
    out,
    environment: MA_CTP_General,
    agent,
    args,
    n_node,
):
    print("Start plotting and storing weights ...")
    # Store weights in a file (for loading in the future)
    # File can have any ending
    with open(os.path.join(log_directory, "weights.flax"), "wb") as f:
        f.write(flax.serialization.to_bytes(model_params))
    # Put here to ensure timing is correct (plotting time is negligible)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Plot training results
    training_result_dict = save_data_and_plotting(
        out["all_episode_done"],
        out["all_total_rewards"],
        out["all_optimal_costs"],
        log_directory,
        reward_exceed_horizon=args.reward_exceed_horizon,
        num_agents=args.n_agent,
        training=True,
    )

    # Plot learning curve
    plot_learning_curve(out["testing_average_competitive_ratio"], log_directory, args)

    # Evaluate the model
    print("Start evaluating ...")
    num_steps_for_inference = n_node * args.factor_inference_timesteps
    init_key, action_key, env_key = jax.vmap(jax.random.PRNGKey)(
        jnp.arange(3) + args.random_seed_for_inference
    )
    new_env_state, new_belief_states = environment.reset(init_key)
    optimistic_agent = Optimistic_Agent(args.n_agent, n_node)

    @scan_tqdm(num_steps_for_inference)
    def _one_step_inference(runner_state, unused):
        (
            current_env_state,
            current_belief_states,
            key,
            timestep_in_episode,
            previous_episode_done,
        ) = runner_state
        action_key, env_key = jax.random.split(key, 2)
        # Agent acts
        actions, action_key = agent.act(action_key, model_params, current_belief_states)
        new_env_state, new_belief_states, rewards, dones, env_key = environment.step(
            env_key, current_env_state, current_belief_states, actions
        )
        episode_done = jnp.all(dones)

        # Stop the episode and reset if exceed horizon length
        env_key, reset_key = jax.random.split(env_key)
        # Reset timestep if finish episode
        timestep_in_episode = jax.lax.cond(
            episode_done, lambda _: 0, lambda _: timestep_in_episode, operand=None
        )

        # Reset if exceed horizon length. Otherwise, increment
        new_env_state, new_belief_states, rewards, timestep_in_episode, dones = (
            jax.lax.cond(
                timestep_in_episode >= (args.horizon_length_factor * n_node),
                lambda _: (
                    *environment.reset(reset_key),
                    jnp.full(
                        args.n_agent, args.reward_exceed_horizon, dtype=jnp.float16
                    ),
                    0,
                    jnp.full(args.n_agent, True, dtype=bool),
                ),
                lambda _: (
                    new_env_state,
                    new_belief_states,
                    rewards,
                    timestep_in_episode + 1,
                    dones,
                ),
                operand=None,
            )
        )

        # Calculate shortest total cost at the beginning of the episode. But we don't need this for training
        def _calculate_optimal_cost(env_state):
            _, goals = jax.lax.top_k(
                jnp.diag(env_state[3, args.n_agent :, :]), args.n_agent
            )
            origins = jnp.argmax(env_state[0, : args.n_agent, :], axis=1)
            _, optimal_cost = get_optimal_combination_and_cost(
                env_state[1, args.n_agent :, :],
                env_state[0, args.n_agent :, :],
                origins,
                goals,
                args.n_agent,
            )
            # minus self.reward_service goal because this is a positive number
            optimal_cost_including_service_goal_costs = (
                optimal_cost - args.reward_service_goal * args.n_agent
            )
            return jnp.array(
                optimal_cost_including_service_goal_costs, dtype=jnp.float16
            )

        # This adds to the compilation time a lot
        shortest_path = jax.lax.cond(
            previous_episode_done,
            _calculate_optimal_cost,
            lambda _: jnp.array(0.0, dtype=jnp.float16),
            operand=current_env_state,
        )
        optimistic_baseline = jax.lax.cond(
            previous_episode_done,
            lambda _: optimistic_agent.get_total_cost(
                environment, current_belief_states, current_env_state, env_key
            ),
            lambda _: jnp.array(0.0, dtype=jnp.float16),
            operand=None,
        )
        runner_state = (
            new_env_state,
            new_belief_states,
            env_key,
            timestep_in_episode,
            episode_done,
        )
        transition = (
            episode_done,
            jnp.sum(rewards),
            shortest_path,
            optimistic_baseline,
        )
        return runner_state, transition

    runner_state = (
        new_env_state,
        new_belief_states,
        env_key,
        jnp.int32(0),
        jnp.bool_(True),
    )
    runner_state, inference_traj_batch = jax.lax.scan(
        _one_step_inference, runner_state, jnp.arange(num_steps_for_inference)
    )
    test_all_episode_done = inference_traj_batch[0]
    test_all_total_rewards = inference_traj_batch[1]
    test_all_optimal_cost = inference_traj_batch[2]
    test_optimistic_baseline = inference_traj_batch[3]

    # Plot testing results
    testing_result_dict = save_data_and_plotting(
        test_all_episode_done,
        test_all_total_rewards,
        test_all_optimal_cost,
        log_directory,
        reward_exceed_horizon=args.reward_exceed_horizon,
        num_agents=args.n_agent,
        all_optimistic_baseline=test_optimistic_baseline,
        training=False,
    )

    # Plot PPO loss
    total_loss = out["losses"][0]
    value_loss = out["losses"][1][0]
    loss_actor = out["losses"][1][1]
    entropy_loss = out["losses"][1][2]
    plt.figure(figsize=(10, 6))
    plt.plot(total_loss, linestyle="-", color="red", label="Total Weighted Loss")
    plt.plot(
        args.vf_coeff * value_loss,
        linestyle="-",
        color="blue",
        label="Weighted Value Loss",
    )
    plt.plot(loss_actor, linestyle="-", color="green", label="Actor Loss")
    if args.anneal_ent_coeff:
        ent_coeff_values = np.array(
            [
                agent._ent_coeff_schedule(i)
                for i in range(args.time_steps // args.num_steps_before_update)
            ]
        )
        ent_coeff_values = np.repeat(
            ent_coeff_values, args.num_update_epochs * args.num_minibatches
        )
    else:
        ent_coeff_values = np.full(entropy_loss.shape, args.ent_coeff)
    weighted_entropy_loss = ent_coeff_values * entropy_loss
    plt.plot(
        weighted_entropy_loss,
        linestyle="-",
        color="orange",
        label="Weighted Entropy Loss",
    )
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Plot")
    plt.legend()
    plt.savefig(os.path.join(log_directory, "PPO_Loss.png"))

    # Write to JSON file
    # Record hyperparameters and results in JSON file
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_time = {"current_datetime": current_datetime}
    dict_args = vars(args)
    args_path = os.path.join(log_directory, "Hyperparameters_Results" + ".json")
    with open(args_path, "w") as fh:
        json.dump(dict_args, fh)
        fh.write("\n")
        json.dump(date_time, fh, indent=4)
        fh.write("\n")
        json.dump({"Total training time in seconds": elapsed_time}, fh)
        fh.write("\n")
        fh.write("Testing results: \n")
        json.dump(testing_result_dict, fh, indent=4)
        fh.write("\n")
        # Log the network architecture
        fh.write("\nNetwork architecture: \n")
        for layer_name, weights in extract_params(model_params):
            fh.write(f"{layer_name}: {weights.shape}\n")
        total_num_params = sum(p.size for p in jax.tree_util.tree_leaves(model_params))
        fh.write("Total number of parameters in the network: " + str(total_num_params))
    print("All done!")
