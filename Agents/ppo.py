from functools import partial
import jax.numpy as jnp
import jax
import sys
import optax
from flax.training.train_state import TrainState

sys.path.append("..")
import flax.linen as nn
from typing import Sequence, NamedTuple, Any
from Environment.CTP_environment import MA_CTP_General
from Environment import CTP_generator


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    critic_value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    belief_state: jnp.ndarray
    shortest_path: jnp.ndarray


class PPO:
    def __init__(
        self,
        model: nn.Module,
        environment: MA_CTP_General,
        discount_factor: float,
        gae_lambda: float,
        clip_eps: float,
        vf_coeff: float,
        ent_coeff: float,
        batch_size: int,
        num_minibatches: int,
        horizon_length: int,
        reward_exceed_horizon: float,
        num_loops: int,
        anneal_ent_coeff: bool,
        deterministic_inference_policy: bool,
        ent_coeff_schedule: str,
        division_plateau: int,
        sigmoid_beginning_offset_num: int,
        sigmoid_total_nums_all: int,
    ) -> None:
        self.model = model
        self.environment = environment
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coeff = vf_coeff
        self.ent_coeff = ent_coeff
        self.batch_size = batch_size
        self.num_minibatches = num_minibatches
        self.horizon_length = horizon_length
        self.reward_exceed_horizon = jnp.float16(reward_exceed_horizon)
        self.num_loops = num_loops
        self.anneal_ent_coeff = anneal_ent_coeff
        self.deterministic_inference_policy = deterministic_inference_policy
        self.ent_coeff_schedule = ent_coeff_schedule
        self.sigmoid_beginning_offset_num = sigmoid_beginning_offset_num
        self.sigmoid_total_nums_all = sigmoid_total_nums_all

    def _ent_coeff_schedule(self, loop_count):
        # linear or sigmoid or plateau schedule
        frac = jax.lax.cond(
            self.ent_coeff_schedule == "sigmoid",
            lambda _: 1 / (1 + jnp.exp(10 * (loop_count / self.num_loops - 0.5))),
            lambda _: 1
            / (
                1
                + jnp.exp(
                    10
                    * (
                        (
                            (loop_count + self.sigmoid_beginning_offset_num)
                            / self.sigmoid_total_nums_all
                        )
                        - 0.5
                    )
                )
            ),
            operand=None,
        )
        return self.ent_coeff * frac

    @partial(jax.jit, static_argnums=(0,))
    def act(
        self, key, params, belief_state, unused
    ) -> tuple[jnp.array, jax.random.PRNGKey]:
        pass

    @partial(jax.jit, static_argnums=(0,))
    def env_step(self, runner_state, unused):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def calculate_gae(self, traj_batch, last_critic_val):
        pass
