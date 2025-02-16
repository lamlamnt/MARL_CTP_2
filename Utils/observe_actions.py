import jax
import jax.numpy as jnp
import os
import sys

sys.path.append("..")
import json
from Utils.plot_graph_env_state import plot_realised_graph_from_env_state
from Utils.optimal_combination import get_optimal_combination_and_cost


# Observe what actions the MA agent takes
# Output JSON file with the actions taken by the MA agent and individual rewards in one episode
def get_actions(
    initial_env_state,
    initial_belief_states,
    environment,
    agent,
    model_params,
    log_directory,
    key,
    args,
):
    episode_done = False
    new_belief_states = initial_belief_states
    new_env_state = initial_env_state
    info = ""

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
        return jnp.array(optimal_cost_including_service_goal_costs, dtype=jnp.float16)

    optimal_cost = _calculate_optimal_cost(new_env_state)
    info += f"Optimal cost: {optimal_cost}\n"
    timestep_in_episode = 0
    n_nodes = initial_env_state.shape[2]
    while (
        not episode_done and timestep_in_episode <= args.horizon_length_factor * n_nodes
    ):
        action_key, env_key = jax.random.split(key, 2)
        # Agent acts
        actions, action_key = agent.act(action_key, model_params, new_belief_states)
        new_env_state, new_belief_states, rewards, dones, env_key = environment.step(
            env_key, new_env_state, new_belief_states, actions
        )
        episode_done = jnp.all(dones)
        timestep_in_episode += 1
        _, key = jax.random.split(key)

        # Store actions and rewards to string
        info += f"Time step: {timestep_in_episode}\n"
        info += f"Actions: {actions}\n"
        info += f"Rewards: {rewards}\n"

    # Write actions and rewards to JSON file
    args_path = os.path.join(log_directory, "One_Episode_Example" + ".json")
    with open(args_path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, ensure_ascii=False, indent=4)
    plot_realised_graph_from_env_state(initial_env_state, log_directory)
    # for some reason - it works when used in isolation but during evaluation, it plots learning curve
