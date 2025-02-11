import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_environment
from Environment import CTP_generator
import pytest
import pytest_print as pp
import os
import warnings
import numpy as np
from Agents.ppo import PPO
from Networks.densenet import DenseNet_ActorCritic_Same
import flax.linen as nn
from Utils.optimal_combination import get_optimal_combination_and_cost
from Agents.optimistic_agent import Optimistic_Agent


@pytest.fixture
def environment():
    key = jax.random.PRNGKey(30)
    environment = CTP_environment.MA_CTP_General(
        2, 10, key, prop_stoch=0.4, grid_size=10, num_stored_graphs=1
    )
    return environment


# def test_env_step(environment: CTP_environment.MA_CTP_General):
# def test_loop(environment: CTP_environment.MA_CTP_General):
if __name__ == "__main__":
    n_node = 5
    n_agent = 2
    key = jax.random.PRNGKey(30)
    model = DenseNet_ActorCritic_Same(
        n_node,
        act_fn=nn.leaky_relu,
        densenet_kernel_init=nn.initializers.kaiming_normal(),
        bn_size=2,
        growth_rate=10,
        num_layers=tuple(map(int, ("2,2").split(","))),
    )
    agent = PPO(
        model,
        environment,
        1.0,
        0.95,
        0.1,
        0.1,
        0.1,
        batch_size=10,
        num_minibatches=1,
        horizon_length=2 * n_node,
        reward_exceed_horizon=-1.5,
        num_loops=5,
        anneal_ent_coeff=True,
        deterministic_inference_policy=False,
        ent_coeff_schedule="sigmoid",
        sigmoid_beginning_offset_num=0,
        sigmoid_total_nums_all=100,
        num_agents=2,
        reward_service_goal=-0.1,
    )
    state_shape = (6, 7, 5)
    init_params = model.init(jax.random.PRNGKey(0), jax.random.normal(key, state_shape))
    optimistic_agent = Optimistic_Agent(n_agent, n_node)

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
        actions, action_key = agent.act(action_key, init_params, current_belief_states)
        print(actions)
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
                timestep_in_episode >= (2 * n_node),
                lambda _: (
                    *environment.reset(reset_key),
                    jnp.full(n_agent, -1.5, dtype=jnp.float16),
                    0,
                    jnp.full(n_agent, True, dtype=bool),
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
            _, goals = jax.lax.top_k(jnp.diag(env_state[3, n_agent:, :]), n_agent)
            origins = jnp.argmax(env_state[0, :n_agent, :], axis=1)
            _, optimal_cost = get_optimal_combination_and_cost(
                env_state[1, n_agent:, :],
                env_state[0, n_agent:, :],
                origins,
                goals,
                n_agent,
            )
            optimal_cost_including_service_goal_costs = optimal_cost + (0.1) * 2
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

    environment = CTP_environment.MA_CTP_General(
        2, 5, key, prop_stoch=0.8, grid_size=10, num_stored_graphs=1
    )
    new_env_state, new_belief_states = environment.reset(key)
    runner_state = (
        new_env_state,
        new_belief_states,
        key,
        jnp.int32(0),
        jnp.bool_(True),
    )
    # for i in range(10):
    #    runner_state, transition = _one_step_inference(runner_state, None)
    # runner_state, inference_traj_batch = _one_step_inference(runner_state, None)
    runner_state, inference_traj_batch = jax.lax.scan(
        _one_step_inference, runner_state, jnp.arange(10)
    )
    test_all_episode_done = inference_traj_batch[0]
    test_all_total_rewards = inference_traj_batch[1]
    test_all_optimal_cost = inference_traj_batch[2]
    test_optimistic_baseline = inference_traj_batch[3]
    print(test_all_episode_done)
