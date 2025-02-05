import jax
import jax.numpy as jnp
import sys
import distrax
from distrax._src.utils import math

sys.path.append("..")
from Environment import CTP_generator


# Compared to single agent, has the added decision of whether service action is valid
# Service action is valid if the agent is at the goal and the goal has not been serviced
def decide_validity_of_action_space(current_belief_state: jnp.ndarray) -> jnp.array:
    # Return an array with size equal to num_nodes+1 in the graph where the element is
    # True if the action is valid and False if the action is invalid
    num_agents = current_belief_state.shape[1] - current_belief_state.shape[2]
    num_nodes = current_belief_state.shape[2]
    weights = current_belief_state[1, num_agents:, :]
    blocking_status = current_belief_state[0, num_agents:, :]
    valid = jnp.zeros(num_nodes + 1, dtype=jnp.float16)
    agent_index = jnp.argmin(jnp.sum(current_belief_state[1, num_agents:, :], axis=1))
    for i in range(num_nodes):
        is_invalid = jnp.logical_or(
            i == jnp.argmax(current_belief_state[0, :num_agents, :]),
            jnp.logical_or(
                weights[jnp.argmax(current_belief_state[0, :num_agents, :]), i]
                == CTP_generator.NOT_CONNECTED,
                blocking_status[jnp.argmax(current_belief_state[0, :num_agents, :]), i]
                == CTP_generator.BLOCKED,
            ),
        )
        valid = valid.at[i].set(jnp.where(is_invalid, -jnp.inf, 1.0))
    # get goals and goal service status

    return valid
