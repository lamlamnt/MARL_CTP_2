import sys

sys.path.append("..")
from Evaluation.pretty_plots import plot_bar_graph_plotly_general


def single_agent_plot_percentage_all_num_nodes():
    groups = ["5 nodes", "10 nodes", "30 nodes"]
    categories = [
        "RL Beats OB",
        "RL Equals OB",
        "RL Beats or Equals OB",
        "Failure Rate for RL",
    ]
    colors = ["red", "blue", "green", "yellow"]
    title = "Single-agent CTP with 80% Stochastic Edges"
    x_axis_title = "Number of Nodes"
    y_axis_title = "Percentage (%)"
    values = [
        [8.06, 87.95, 96.01, 0],
        [26.37, 50.96, 77.33, 0.80],
        [37.03, 18.11, 55.14, 3.64],
    ]
    plot_bar_graph_plotly_general(
        groups, categories, values, title, x_axis_title, y_axis_title, colors
    )


def single_agent_plot_percentage_node_10():
    groups = ["20%", "40%", "80%", "Mixed"]
    categories = [
        "RL Beats OB",
        "RL Equals OB",
        "RL Beats or Equals OB",
        "Failure Rate for RL",
    ]
    colors = ["red", "blue", "green", "yellow"]
    title = (
        "Single-Agent CTP on 10-Node Graphs at Different Percent of Stochastic Edges"
    )
    x_axis_title = "Percentage of Stochastic Edges"
    y_axis_title = "Percentage (%)"
    values = [
        [13.38, 63.61, 76.99, 0.14],
        [21.68, 53.33, 75.01, 0.40],
        [26.37, 50.96, 77.33, 0.80],
        [21.37, 53.62, 74.99, 0.68],
    ]
    plot_bar_graph_plotly_general(
        groups, categories, values, title, x_axis_title, y_axis_title, colors
    )


def single_agent_plot_mean_median_all_num_nodes():
    groups = [
        "5 nodes",
        "10 nodes",
        "30 nodes",
    ]
    categories = [
        "Mean RL",
        "Median RL",
        "Std RL",
        "Max RL",
        "Mean OB",
        "Median OB",
        "Std OB",
        "Max OB",
    ]
    colors = [
        "rgb(139, 0, 0)",
        "rgb(0, 0, 139)",
        "rgb(0,139,0)",
        "rgb(204, 153, 0)",
        "rgb(255, 102, 102)",
        "rgb(102, 178, 255)",
        "rgb(119, 221, 119)",
        "rgb(255, 255, 153)",
    ]
    title = "Statistics Excluding Failed Episodes for Single-agent CTP with 80% Stochastic Edges"
    x_axis_title = "Number of Nodes"
    y_axis_title = "Competitive Ratio"
    values = [
        [1.029, 1.000, 0.141, 3.039, 1.058, 1.000, 0.200, 2.429],
        [1.092, 1.000, 0.249, 3.280, 1.147, 1.000, 0.332, 3.784],
        [1.207, 1.103, 0.289, 3.107, 1.223, 1.071, 0.351, 4.253],
    ]
    plot_bar_graph_plotly_general(
        groups,
        categories,
        values,
        title,
        x_axis_title,
        y_axis_title,
        colors,
        shift_num_columns=4,
        bar_width=0.1,
    )


def single_agent_plot_mean_median_node_10():
    groups = [
        "20%",
        "40%",
        "80%",
        "Mixed",
    ]
    categories = [
        "Mean RL",
        "Median RL",
        "Std RL",
        "Max RL",
        "Mean OB",
        "Median OB",
        "Std OB",
        "Max OB",
    ]
    colors = [
        "rgb(139, 0, 0)",
        "rgb(0, 0, 139)",
        "rgb(0,139,0)",
        "rgb(204, 153, 0)",
        "rgb(255, 102, 102)",
        "rgb(102, 178, 255)",
        "rgb(119, 221, 119)",
        "rgb(255, 255, 153)",
    ]
    title = "Statistics Excluding Failed Episodes for Single-Agent CTP on 10-Node Graphs at Different Levels of Stochasticity"
    x_axis_title = "Percentage of Stochastic Edges"
    y_axis_title = "Competitive Ratio"
    values = [
        [1.021, 1.000, 0.086, 2.862, 1.028, 1.000, 0.111, 2.225],
        [1.043, 1.000, 0.123, 2.560, 1.068, 1.000, 0.186, 2.915],
        [1.092, 1.000, 0.249, 3.280, 1.147, 1.000, 0.332, 3.784],
        [1.060, 1.000, 0.190, 3.907, 1.080, 1.000, 0.240, 3.907],
    ]
    plot_bar_graph_plotly_general(
        groups, categories, values, title, x_axis_title, y_axis_title, colors, 4, 0.11
    )


if __name__ == "__main__":
    # single_agent_plot_percentage_all_num_nodes()
    # single_agent_plot_percentage_node_10()
    # single_agent_plot_mean_median_all_num_nodes()
    single_agent_plot_mean_median_node_10()
