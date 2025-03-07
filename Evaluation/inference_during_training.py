from functools import partial
import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment.CTP_environment import MA_CTP_General
from Agents.optimistic_agent import Optimistic_Agent
from Utils.optimal_combination import get_optimal_combination_and_cost


# For the purpose of plotting the learning curve
# Deterministic inference
@partial(jax.jit, static_argnums=(0, 1, 3))
def get_average_testing_stats(
    environment: MA_CTP_General, agent, model_params, arguments
) -> float:
    # The last argument is a Frozen Dictionary of relevant hyperparameters
    init_key, env_key = jax.vmap(jax.random.PRNGKey)(
        jnp.arange(2) + arguments["random_seed_for_inference"]
    )
    new_env_state, new_belief_states = environment.reset(init_key)
    num_testing_timesteps = arguments["factor_testing_timesteps"] * arguments["n_node"]

    def _one_step_inference(runner_state, unused):
        (
            current_env_state,
            current_belief_states,
            key,
            timestep_in_episode,
            previous_episode_done,
            failure_counter,
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
        reward_agent_exceed_horizon = jnp.where(
            dones, jnp.float16(0), arguments["reward_exceed_horizon"]
        )

        # Reset if exceed horizon length. Otherwise, increment
        (
            new_env_state,
            new_belief_states,
            rewards,
            timestep_in_episode,
            dones,
            failure_counter,
        ) = jax.lax.cond(
            timestep_in_episode
            >= (arguments["horizon_length_factor"] * arguments["n_node"]),
            lambda _: (
                *environment.reset(reset_key),
                reward_agent_exceed_horizon,
                0,
                jnp.full(arguments["n_agent"], True, dtype=bool),
                failure_counter + 1,
            ),
            lambda _: (
                new_env_state,
                new_belief_states,
                rewards,
                timestep_in_episode + 1,
                dones,
                failure_counter,
            ),
            operand=None,
        )

        # Calculate shortest total cost at the beginning of the episode. But we don't need this for training
        def _calculate_optimal_cost(env_state):
            _, goals = jax.lax.top_k(
                jnp.diag(env_state[3, arguments["n_agent"] :, :]), arguments["n_agent"]
            )
            origins = jnp.argmax(env_state[0, : arguments["n_agent"], :], axis=1)
            _, optimal_cost = get_optimal_combination_and_cost(
                env_state[1, arguments["n_agent"] :, :],
                env_state[0, arguments["n_agent"] :, :],
                origins,
                goals,
                arguments["n_agent"],
            )
            # minus self.reward_service goal because this is a positive number
            optimal_cost_including_service_goal_costs = (
                optimal_cost - arguments["reward_service_goal"] * arguments["n_agent"]
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
        runner_state = (
            new_env_state,
            new_belief_states,
            env_key,
            timestep_in_episode,
            episode_done,
            failure_counter,
        )
        transition = (
            episode_done,
            jnp.sum(rewards),
            shortest_path,
        )
        return runner_state, transition

    runner_state = (
        new_env_state,
        new_belief_states,
        env_key,
        jnp.int32(0),
        jnp.bool_(True),
        jnp.int32(0),
    )
    runner_state, inference_traj_batch = jax.lax.scan(
        _one_step_inference, runner_state, jnp.arange(num_testing_timesteps)
    )

    test_all_episode_done = inference_traj_batch[0]
    test_all_total_rewards = inference_traj_batch[1]
    test_all_optimal_costs = inference_traj_batch[2]
    failure_counter = runner_state[-1]

    # Calculate competitive ratio without using pandas
    episode_numbers = jnp.cumsum(test_all_episode_done)
    shifted_episode_numbers = jnp.concatenate([jnp.array([0]), episode_numbers[:-1]])

    def aggregate_by_episode(episode_numbers, values, num_segments):
        return jax.ops.segment_sum(values, episode_numbers, num_segments=num_segments)

    # In order to jax jit, the number of episodes must be known. So we will take the first n episodes only
    min_num_episodes = num_testing_timesteps // (
        arguments["horizon_length_factor"] * arguments["n_node"] + 1
    )
    aggregated_rewards = aggregate_by_episode(
        shifted_episode_numbers, test_all_total_rewards, num_segments=min_num_episodes
    )
    aggregated_optimal_path_lengths = aggregate_by_episode(
        shifted_episode_numbers,
        test_all_optimal_costs,
        num_segments=min_num_episodes,
    )
    # Don't need to remove the last incomplete episode because we are using the first n complete episodes
    competitive_ratio = jnp.abs(aggregated_rewards) / aggregated_optimal_path_lengths

    # Get average competitive ratio
    average_competitive_ratio = jnp.mean(competitive_ratio)

    # ADDED LATER
    # Calculate failure rate -> number of times the reward is divisible by reward_exceed_horizon/(max_shifted_episode_numbers+1)
    failure_mask = jnp.where(
        (test_all_total_rewards % arguments["reward_exceed_horizon"]) == 0, 1, 0
    )
    failure_rate = failure_counter * 100 / (jnp.max(shifted_episode_numbers) + 1)
    # safety measure
    failure_rate = jax.lax.cond(
        failure_rate > 100, lambda _: 100.0, lambda _: failure_rate, operand=None
    )

    # To calculate mean competitive ratio excluding the failed episodes, make the rewards for failed episodes and all the later episodes all 0. When calculating mean, divide by number of successful_episodes
    failed_episodes = jax.ops.segment_max(
        failure_mask, shifted_episode_numbers, num_segments=min_num_episodes
    )
    num_successful_episodes = min_num_episodes - jnp.sum(failed_episodes)
    edited_rewards = jnp.where(
        failed_episodes[shifted_episode_numbers] == 1, 0, test_all_total_rewards
    )
    edited_optimal_costs = jnp.where(
        failed_episodes[shifted_episode_numbers] == 1, 0, test_all_optimal_costs
    )

    # aggregate - the failed ones will be 0
    aggregated_edited_rewards = jax.ops.segment_sum(
        edited_rewards, shifted_episode_numbers, min_num_episodes
    )
    aggregated_edited_optimal_costs = jax.ops.segment_sum(
        edited_optimal_costs, shifted_episode_numbers, min_num_episodes
    )
    # Don't need to remove the later episode because we are using the first n complete episodes
    aggregated_edited_optimal_costs = jnp.where(
        aggregated_edited_optimal_costs == 0, 1, aggregated_edited_optimal_costs
    )
    competitive_ratio_exclude_failures = (
        jnp.abs(aggregated_edited_rewards) / aggregated_edited_optimal_costs
    )
    mean_competitive_ratio_exclude_failures = (
        jnp.sum(competitive_ratio_exclude_failures) / num_successful_episodes
    )
    mean_competitive_ratio_exclude_failures = jax.lax.cond(
        num_successful_episodes == 0,
        lambda _: jnp.float16(10.0),
        lambda _: mean_competitive_ratio_exclude_failures,
        operand=None,
    )

    return (
        average_competitive_ratio,
        jnp.float16(mean_competitive_ratio_exclude_failures),
        jnp.float16(failure_rate),
    )
