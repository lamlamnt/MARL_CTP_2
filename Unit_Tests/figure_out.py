import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment.CTP_environment import MA_CTP_General
from Environment import CTP_generator
from Utils.hand_crafted_graphs import get_go_past_goal_without_servicing_graph
from Agents.optimistic_agent import Optimistic_Agent

if __name__ == "__main__":
    gae = jnp.array([0.0, 0.0])
    reward = jnp.array([1.0, 1.0])
    done = jnp.array([False, False])
    critic_value = jnp.array([0.0, 0.0])
    next_value = jnp.array([0.0, 0.0])
    team_reward = jnp.sum(reward, axis=0)
    broadcasted_team_reward = jnp.broadcast_to(team_reward, reward.shape)
    individual_delta = reward + next_value * (1 - done) - critic_value[0]
    team_delta = broadcasted_team_reward + next_value * (1 - done) - critic_value[1]
    # separate delta and gae
    individual_gae = individual_delta + (1 - done) * gae[0]
    team_gae = team_delta + (1 - done) * gae[1]
    output = jnp.stack([individual_gae, team_gae])
    print(output.shape)
