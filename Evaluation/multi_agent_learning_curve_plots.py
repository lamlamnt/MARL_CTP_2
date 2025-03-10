import os
import sys

sys.path.append("..")
from Evaluation.pretty_plots import plot_learning_curve_general


def learning_curve_10_nodes_2_agents():
    names = [
        os.path.join(
            "10_nodes_2_agents", "experiment_10_nodes_2_agents_1_critic_individual"
        ),
        os.path.join("10_nodes_2_agents", "experiment_10_nodes_2_agents_1_critic_team"),
        os.path.join(
            "10_nodes_2_agents", "experiment_10_nodes_2_agents_2_critic_decay"
        ),
        os.path.join(
            "10_nodes_2_agents", "experiment_10_nodes_2_agents_1_critic_mixed_no_decay"
        ),
        os.path.join(
            "10_nodes_2_agents", "experiment_10_nodes_2_agents_1_critic_decay_best"
        ),
    ]
    plot_learning_curve_general(
        names,
        5,
        2300,
        20,
        "Learning Curves for 10 Nodes 2 Agents 80% Stochastic Edges",
        "10_nodes_2_agents",
    )


def learning_curve_20_nodes_2_agents():
    names = [
        os.path.join(
            "20_nodes_2_agents", "experiment_20_nodes_2_agents_1_critic_individual"
        ),
        os.path.join("20_nodes_2_agents", "experiment_20_nodes_2_agents_1_critic_team"),
        os.path.join(
            "20_nodes_2_agents", "experiment_20_nodes_2_agents_2_critic_decay"
        ),
        os.path.join(
            "20_nodes_2_agents", "experiment_20_nodes_2_agents_1_critic_mixed_no_decay"
        ),
        os.path.join(
            "20_nodes_2_agents", "experiment_20_nodes_2_agents_1_critic_decay_best"
        ),
    ]
    plot_learning_curve_general(
        names,
        5,
        5500,
        20,
        "Learning Curves for 20 Nodes 2 Agents 80% Stochastic Edges",
        "20_nodes_2_agents",
    )


def learning_curve_20_nodes_4_agents():
    pass


if __name__ == "__main__":
    learning_curve_10_nodes_2_agents()
    learning_curve_20_nodes_2_agents()
