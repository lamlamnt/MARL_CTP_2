import sys

sys.path.append("..")
from Environment.CTP_environment import MA_CTP_General
import jax
from Evaluation.inference_during_training import get_average_testing_stats
from Utils.load_store_graphs import load_graphs
from Agents.ppo import PPO
from Networks.densenet import DenseNet_ActorCritic_Same
import flax.linen as nn
import argparse
from flax.core.frozen_dict import FrozenDict

# use the average_testing_stats function for 20 nodes 4 agents to figure out why it's being weird
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    parser = argparse.ArgumentParser(
        description="Parse command-line arguments for this unit test"
    )
    parser.add_argument("--n_agent", type=int, default=4)
    parser.add_argument("--n_node", type=int, default=20)
    parser.add_argument("--prop_stoch", type=float, default=0.8)
    parser.add_argument("--k_edges", type=int, default=None)
    parser.add_argument(
        "--graph_identifier",
        type=str,
        default="normalized_node_20_agent_4_prop_0.8",
    )
    args = parser.parse_args()
    # Temporarily modify load_graphs to go to Not_Pushed_to_Git
    training_graphs, inference_graphs, num_training_graphs, num_inference_graphs = (
        load_graphs(args)
    )
    environment = MA_CTP_General(
        num_agents=4,
        num_nodes=20,
        key=key,
        prop_stoch=0.8,
        loaded_graphs=inference_graphs,
    )
    model = DenseNet_ActorCritic_Same(
        20,
        act_fn=nn.leaky_relu,
        densenet_kernel_init=nn.initializers.kaiming_normal(),
        bn_size=2,
        growth_rate=5,
        num_layers=tuple(map(int, "1,1".split(","))),
    )
    model_params = model.init(
        jax.random.PRNGKey(0),
        jax.random.normal(key, (6, 24, 20)),
    )
    agent = PPO(
        model,
        environment,
        1,
        0.95,
        0.1,
        0.1,
        0.1,
        batch_size=40,
        num_minibatches=1,
        horizon_length=2 * 20,
        reward_exceed_horizon=-3,
        num_loops=5,
        anneal_ent_coeff=True,
        deterministic_inference_policy=True,
        ent_coeff_schedule="sigmoid",
        sigmoid_beginning_offset_num=0,
        sigmoid_total_nums_all=100000,
        num_agents=4,
        reward_service_goal=-0.1,
        individual_reward_weight=1,
        individual_reward_weight_schedule="constant",
    )
    arguments = FrozenDict(
        {
            "factor_testing_timesteps": 3,
            "n_node": 20,
            "reward_exceed_horizon": -3,
            "horizon_length_factor": 2,
            "random_seed_for_inference": 4,
            "n_agent": 4,
            "reward_service_goal": -0.1,
        }
    )
    average_competitive_ratio, average_competitive_ratio_exclude, failure_rate = (
        get_average_testing_stats(environment, agent, model_params, arguments)
    )
    print(average_competitive_ratio_exclude)
    print(failure_rate)
