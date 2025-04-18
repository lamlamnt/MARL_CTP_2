import sys

sys.path.append("..")
from Evaluation.new_plots_general import (
    percentage_bar_plot_general,
    box_whisker_general,
)
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def single_agent_plot_percentage():
    groups = [
        "5 Nodes (40%)",
        "5 Nodes (80%)",
        "5 Nodes (Mixed)",
        "10 Nodes (40%)",
        "10 Nodes (80%)",
        "10 Nodes (Mixed)",
        "30 Nodes (40%)",
        "30 Nodes (80%)",
        "30 Nodes (Mixed)",
    ]
    x_axis_title = "Number of Nodes (Percentage of Stochastic Edges)"
    y_axis_title = "Percentage (%)"

    title = "Single-Agent CTP"
    values = [
        [8.2, 86.4, 0.0],
        [8.1, 88.0, 0.0],
        [9.6, 84.5, 0.0],
        [21.7, 53.3, 0.4],
        [26.4, 51.0, 0.8],
        [21.4, 53.6, 0.7],
        [30.1, 19.2, 0.2],
        [35.8, 16.5, 0.4],
        [30.1, 16.1, 0.3],
    ]
    colors = [
        [
            "#C8E6C9",
            "#66BB6A",
            "#2E7D32",
            "#C8E6C9",
            "#66BB6A",
            "#2E7D32",
            "#C8E6C9",
            "#66BB6A",
            "#2E7D32",
        ],
        [
            "#BBDEFB",
            "#42A5F5",
            "#1565C0",
            "#BBDEFB",
            "#42A5F5",
            "#1565C0",
            "#BBDEFB",
            "#42A5F5",
            "#1565C0",
        ],
        [
            "#FFCDD2",
            "#EF5350",
            "#C62828",
            "#FFCDD2",
            "#EF5350",
            "#C62828",
            "#FFCDD2",
            "#EF5350",
            "#C62828",
        ],
    ]
    percentage_bar_plot_general(
        groups, values, title, x_axis_title, y_axis_title, colors
    )


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
            [1.04296, 1.0, 0.1231, 1.0, 2.560],
            [1.0678, 1.0, 0.1856, 1.0, 2.915],
            [1.0808, 1.0, 0.1104, 1.0, 2.230],
            [1.0704, 1.0, 0.1380, 1.0, 2.710],
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
    folder = "C:\\Users\\shala\\Documents\\Oxford Undergrad\\4th Year\\4YP\\For Report Writing\\single_agent_node_30_combined"
    average_window = 5
    num_steps_before_update = 6000
    frequency_testing = 20
    csv_location = os.path.join(folder, "learning_curve_series.csv")
    learning_curve_series = pd.read_csv(csv_location, header=None).iloc[1:, 0]

    # Do rolling mean and std for each
    rolling_mean = learning_curve_series.rolling(
        window=average_window, min_periods=1
    ).mean()
    rolling_std = learning_curve_series.rolling(
        window=average_window, min_periods=1
    ).std()

    plt.plot(
        num_steps_before_update
        * frequency_testing
        * np.arange(len(learning_curve_series)),
        rolling_mean,
        linestyle="-",
        color="red",
    )
    plt.fill_between(
        num_steps_before_update
        * frequency_testing
        * np.arange(len(learning_curve_series)),
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        color="blue",
        alpha=0.2,
    )
    plt.axhline(y=1.0, color="green", linestyle="--")  # horizontal line
    # plt.title("Learning Curve for 30 Nodes and 80% Stochastic Edges")
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    plt.ylim(0.5, 3)
    plt.xlabel("Training Timesteps")
    plt.ylabel("Mean Competitive Ratio (Including Failed Episodes)")
    plt.savefig(
        os.path.join(folder, "learning_curve_full_new_30_nodes.pdf"),
        bbox_inches="tight",
    )
    plt.close()


def plot_learning_curve_10_nodes():
    folder = "C:\\Users\\shala\\Documents\\Oxford Undergrad\\4th Year\\4YP\\For Report Writing\\report_writing_10_nodes"
    average_window = 5
    num_steps_before_update = 3600
    frequency_testing = 10
    csv_location = os.path.join(folder, "learning_curve_series.csv")
    learning_curve_series = pd.read_csv(csv_location, header=None).iloc[1:, 0]

    # Do rolling mean and std for each
    rolling_mean = learning_curve_series.rolling(
        window=average_window, min_periods=1
    ).mean()
    rolling_std = learning_curve_series.rolling(
        window=average_window, min_periods=1
    ).std()

    plt.plot(
        num_steps_before_update
        * frequency_testing
        * np.arange(len(learning_curve_series)),
        rolling_mean,
        linestyle="-",
        color="red",
    )
    plt.fill_between(
        num_steps_before_update
        * frequency_testing
        * np.arange(len(learning_curve_series)),
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        color="blue",
        alpha=0.2,
    )
    plt.axhline(y=1.0, color="green", linestyle="--")  # horizontal line
    # plt.title("Learning Curve for 30 Nodes and 80% Stochastic Edges")
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    plt.ylim(0.5, 3)
    plt.xlabel("Training Timesteps")
    plt.ylabel("Mean Competitive Ratio (Including Failed Episodes)")
    plt.savefig(
        os.path.join(folder, "learning_curve_full_new_10_nodes.pdf"),
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    single_agent_plot_percentage()
    # single_agent_plot_box_whisker()
    # plot_learning_curve_30_nodes()
    # plot_learning_curve_10_nodes()
