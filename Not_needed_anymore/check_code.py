import jax.numpy as jnp
import jax

test_all_total_rewards = jnp.array([-0.2, -0.3, -1.5, -0.4, -3, -4.5, -0.9, -1.1])
test_all_optimal_costs = jnp.array([1.1, 0, 0, 1.1, 0, 1.1, 1.1, 0])
shifted_episode_numbers = jnp.array([0, 0, 0, 1, 1, 2, 3, 3])
min_num_episodes = 3

failure_mask = jnp.where((test_all_total_rewards % -1.5) == 0, 1, 0)
print(failure_mask)
failure_rate = jnp.sum(failure_mask) * 100 / (jnp.max(shifted_episode_numbers) + 1)
print(failure_rate)
# To calculate mean competitive ratio excluding the failed episodes, make the rewards for failed episodes and all the later episodes all 0. When calculating mean, divide by number of successful_episodes
failed_episodes = jax.ops.segment_max(
    failure_mask, shifted_episode_numbers, num_segments=min_num_episodes
)
num_successful_episodes = min_num_episodes - jnp.sum(failed_episodes)
print(num_successful_episodes)
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
print(aggregated_edited_rewards)
print(aggregated_edited_optimal_costs)
competitive_ratio_exclude_failures = (
    jnp.abs(aggregated_edited_rewards) / aggregated_edited_optimal_costs
)
mean_competitive_ratio_exclude_failures = (
    jnp.sum(competitive_ratio_exclude_failures) / num_successful_episodes
)
print(mean_competitive_ratio_exclude_failures)
