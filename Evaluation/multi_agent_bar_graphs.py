import os
import sys

sys.path.append("..")
from Evaluation.pretty_plots import plot_bar_graph_plotly_general


def percentage_all_nodes_80():
    groups = ["10 nodes 2 agents", "20 nodes 2 agents", "20 nodes 4 agents"]
    categories = [
        "RL Beats OB",
        "RL Equals OB",
        "RL Beats or Equals OB",
        "Failure Rate for RL",
    ]
    colors = ["red", "blue", "green", "yellow"]
    title = "Multi-agent CTP with 80% Stochastic Edges"
    x_axis_title = "Number of Nodes"
    y_axis_title = "Percentage (%)"
    values = [
        [38.73, 34.93, 73.66, 0.98],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    plot_bar_graph_plotly_general(
        groups, categories, values, title, x_axis_title, y_axis_title, colors
    )


def percentage_all_nodes_40():
    groups = ["10 nodes 2 agents", "20 nodes 2 agents", "20 nodes 4 agents"]
    categories = [
        "RL Beats OB",
        "RL Equals OB",
        "RL Beats or Equals OB",
        "Failure Rate for RL",
    ]
    colors = ["red", "blue", "green", "yellow"]
    title = "Multi-agent CTP with 40% Stochastic Edges"
    x_axis_title = "Number of Nodes"
    y_axis_title = "Percentage (%)"
    values = [
        [21.79, 42.24, 64.03, 0.97],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    plot_bar_graph_plotly_general(
        groups, categories, values, title, x_axis_title, y_axis_title, colors
    )


def mean_median_std_all_nodes_80():
    groups = ["10 nodes 2 agents", "20 nodes 2 agents", "20 nodes 4 agents"]
    categories = [
        "Mean RL",
        "Median RL",
        "Std RL",
        "Mean OB",
        "Median OB",
        "Std OB",
    ]
    colors = [
        "rgb(139, 0, 0)",
        "rgb(0, 0, 139)",
        "rgb(0,139,0)",
        "rgb(255, 102, 102)",
        "rgb(102, 178, 255)",
        "rgb(119, 221, 119)",
    ]
    title = "Statistics Excluding Failed Episodes for MA-CTP with 80% Stochastic Edges"
    x_axis_title = "Number of Nodes"
    y_axis_title = "Value"
    values = [
        [1.0873, 1.0, 0.2009, 1.1358, 1.0241, 0.2839],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    plot_bar_graph_plotly_general(
        groups, categories, values, title, x_axis_title, y_axis_title, colors, 3, 0.1
    )


def mean_median_std_all_nodes_40():
    groups = ["10 nodes 2 agents", "20 nodes 2 agents", "20 nodes 4 agents"]
    categories = [
        "Mean RL",
        "Median RL",
        "Std RL",
        "Mean OB",
        "Median OB",
        "Std OB",
    ]
    colors = [
        "rgb(139, 0, 0)",
        "rgb(0, 0, 139)",
        "rgb(0,139,0)",
        "rgb(255, 102, 102)",
        "rgb(102, 178, 255)",
        "rgb(119, 221, 119)",
    ]
    title = "Statistics Excluding Failed Episodes for MA-CTP with 40% Stochastic Edges"
    x_axis_title = "Number of Nodes"
    y_axis_title = "Value"
    values = [
        [1.0573, 1.0, 0.1044, 1.1044, 1.0, 0.1169],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    plot_bar_graph_plotly_general(
        groups, categories, values, title, x_axis_title, y_axis_title, colors, 3, 0.1
    )


if __name__ == "__main__":
    percentage_all_nodes_80()
    percentage_all_nodes_40()
    mean_median_std_all_nodes_80()
    mean_median_std_all_nodes_40()
