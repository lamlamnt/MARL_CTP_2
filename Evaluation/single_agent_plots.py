import sys

sys.path.append("..")
from Evaluation.pretty_plots import plot_bar_graph_plotly_general
from Evaluation.new_plots_general import (
    percentage_bar_plot_general,
    box_whisker_general,
)
import numpy as np


def single_agent_plot_percentage():
    groups = ["5 nodes", "10 nodes", "30 nodes"]
    x_axis_title = "Number of Nodes"
    y_axis_title = "Percentage (%)"

    title = "Single-Agent CTP with 80% Stochastic Edges"
    values = [
        [8.06, 87.95, 0.0],
        [26.37, 50.96, 0.80],
        [37.03, 18.11, 3.64],
    ]
    percentage_bar_plot_general(groups, values, title, x_axis_title, y_axis_title)

    title = "Single-Agent CTP with 40% Stochastic Edges"
    values = [[8.24, 86.37, 0], [21.68, 53.33, 0.3980], [30.09, 19.23, 0.1706]]
    percentage_bar_plot_general(groups, values, title, x_axis_title, y_axis_title)

    title = "Single-Agent CTP with Mixed-Percentage Stochastic Edges"
    values = [[9.62, 84.46, 0], [21.37, 53.62, 0.68], [30.08, 16.14, 0.35]]
    percentage_bar_plot_general(groups, values, title, x_axis_title, y_axis_title)


def single_agent_plot_box_whisker():
    groups = [
        "5 Nodes RL",
        "5 Nodes OB",
        "10 Nodes RL",
        "10 Nodes OB",
        "30 Nodes RL",
        "30 Nodes OB",
    ]
    title = "Statistics Excluding Failed Episodes for Single-Agent CTP with 80% Stochastic Edges"
    values_prop_80 = np.array(
        [
            [1.029, 1.0, 0.141, 1.0, 3.039],
            [1.058, 1.0, 0.200, 1.0, 2.429],
            [1.092, 1.0, 0.249, 1.0, 3.280],
            [1.147, 1.0, 0.332, 1.0, 3.784],
            [1.207, 1.103, 0.289, 1.0, 3.107],
            [1.223, 1.071, 0.351, 1.0, 4.253],
        ]
    )
    box_whisker_general(values_prop_80, title, groups)

    title = "Statistics Excluding Failed Episodes for Single-Agent CTP with 40% Stochastic Edges"
    values_prop_40 = np.array(
        [
            [1.0179, 1.0, 0.07934, 1.0, 1.6935],
            [1.0465, 1.0, 0.1572, 1.0, 2.1683],
            [1.04296, 1.0, 0.1231, 1.0, 2.56],
            [1.0678, 1.0, 0.1856, 1.0, 2.915],
            [1.0808, 1.0, 0.1104, 1.0, 2.23],
            [1.0704, 1.0, 0.1380, 1.0, 2.71],
        ]
    )
    box_whisker_general(values_prop_40, title, groups)

    title = "Statistics Excluding Failed Episodes for Single-Agent CTP with Mixed-Percentage Stochastic Edges"
    values_mixed = np.array(
        [
            [1.0236, 1.0, 0.1131, 1.0, 2.2609],
            [1.0542, 1.0, 0.1936, 1.0, 2.5644],
            [1.0596, 1.0, 0.1905, 1.0, 3.9065],
            [1.0802, 1.0, 0.2402, 1.0, 3.907],
            [1.1464, 1.0, 0.2147, 1.0, 3.016],
            [1.1344, 1.0, 0.2468, 1.0, 3.097],
        ]
    )
    box_whisker_general(values_mixed, title, groups)


def plot_learning_curve_30_nodes():
    pass


if __name__ == "__main__":
    # single_agent_plot_percentage()
    single_agent_plot_box_whisker()
