import os
import sys

sys.path.append("..")
from Evaluation.new_plots_general import plot_learning_curve_general


def learning_curve_10_nodes_2_agents():
    names = [
        os.path.join(
            "10_nodes_2_agents", "experiment_10_nodes_2_agents_1_critic_individual"
        ),
        os.path.join("10_nodes_2_agents", "experiment_10_nodes_2_agents_1_critic_team"),
        os.path.join(
            "10_nodes_2_agents", "experiment_10_nodes_2_agents_1_critic_mixed_no_decay"
        ),
    ]
    plot_learning_curve_general(
        names,
        5,
        2300,
        20,
        "Learning Curves for 10 Nodes 2 Agents 80% Stochastic Edges",
        "New_10_nodes_2_agents_1",
        first_graph=True,
        ylim_top=7,
    )
    names = [
        os.path.join(
            "10_nodes_2_agents", "experiment_10_nodes_2_agents_1_critic_decay"
        ),
        os.path.join(
            "10_nodes_2_agents", "experiment_10_nodes_2_agents_2_critic_constant"
        ),
        os.path.join(
            "10_nodes_2_agents", "experiment_10_nodes_2_agents_2_critic_decay"
        ),
    ]
    plot_learning_curve_general(
        names,
        5,
        2300,
        20,
        "Learning Curves for 10 Nodes 2 Agents 80% Stochastic Edges",
        "New_10_nodes_2_agents_2",
        first_graph=False,
        ylim_top=7,
    )


def learning_curve_20_nodes_2_agents():
    names = [
        os.path.join(
            "20_nodes_2_agents", "experiment_20_nodes_2_agents_1_critic_individual"
        ),
        os.path.join("20_nodes_2_agents", "experiment_20_nodes_2_agents_1_critic_team"),
        os.path.join(
            "20_nodes_2_agents", "experiment_20_nodes_2_agents_1_critic_mixed_no_decay"
        ),
    ]
    plot_learning_curve_general(
        names,
        5,
        5500,
        20,
        "Learning Curves for 20 Nodes 2 Agents 80% Stochastic Edges",
        "New_20_nodes_2_agents_1",
        first_graph=True,
        ylim_top=11,
    )
    names = [
        os.path.join(
            "20_nodes_2_agents", "experiment_20_nodes_2_agents_1_critic_decay"
        ),
        os.path.join(
            "20_nodes_2_agents", "experiment_20_nodes_2_agents_2_critic_constant"
        ),
        os.path.join(
            "20_nodes_2_agents", "experiment_20_nodes_2_agents_2_critic_decay"
        ),
    ]
    plot_learning_curve_general(
        names,
        5,
        5500,
        20,
        "Learning Curves for 20 Nodes 2 Agents 80% Stochastic Edges",
        "New_20_nodes_2_agents_2",
        first_graph=False,
        ylim_top=11,
    )


def learning_curve_20_nodes_4_agents():
    names = [
        os.path.join(
            "20_nodes_4_agents", "combined_20_nodes_4_agents_1_critic_individual"
        ),
        os.path.join("20_nodes_4_agents", "combined_20_nodes_4_agents_1_critic_team"),
        os.path.join("20_nodes_4_agents", "combined_20_nodes_4_agents_1_critic_mixed"),
    ]
    plot_learning_curve_general(
        names,
        5,
        3000,
        20,
        "Learning Curves for 20 Nodes 4 Agents 80% Stochastic Edges",
        "New_20_nodes_4_agents_1",
        first_graph=True,
        ylim_top=6,
    )
    names = [
        os.path.join("20_nodes_4_agents", "combined_20_nodes_4_agents_1_critic_decay"),
        os.path.join(
            "20_nodes_4_agents", "combined_20_nodes_4_agents_2_critic_constant"
        ),
        os.path.join("20_nodes_4_agents", "combined_20_nodes_4_agents_2_critic_decay"),
    ]

    plot_learning_curve_general(
        names,
        5,
        3000,
        20,
        "Learning Curves for 20 Nodes 4 Agents 80% Stochastic Edges",
        "New_20_nodes_4_agents_2",
        first_graph=False,
        ylim_top=6,
    )


if __name__ == "__main__":
    learning_curve_10_nodes_2_agents()
    learning_curve_20_nodes_2_agents()
    learning_curve_20_nodes_4_agents()
