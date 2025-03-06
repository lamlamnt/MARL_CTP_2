import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.graph_objects as go
import pandas as pd


# Add optimistic baseline as well -> to get more box plots
def plot_box_and_whisker_plotly():
    box_stats = [
        {
            "whislo": 1,
            "q1": 1,
            "med": 1,
            "q3": 1.069,
            "whishi": 6.62,
            "mean": 1.0665,
            "mean_excluding": 1.0517,
        },  # RL for 10 nodes prop 0.4
        {
            "whislo": 1,
            "q1": 1,
            "med": 1,
            "q3": 1.048,
            "whishi": 2.12,
            "mean": 1.0528,
            "mean_excluding": 1.0528,
        },  # Optimistic baseline prop 0.4
        {
            "whislo": 1.0,
            "q1": 1.0,
            "med": 1.0,
            "q3": 1.104,
            "whishi": 7.774,
            "mean": 1.11,
            "mean_excluding": 1.08,
        },  # RL for 10 nodes prop 0.8
        {
            "whislo": 1.0,
            "q1": 1.0,
            "med": 1.025,
            "q3": 1.2037,
            "whishi": 7.48,
            "mean": 1.1368,
            "mean_excluding": 1.1368,
        },  # Optimistic baseline prop 0.8
        {
            "whislo": 1,
            "q1": 1.015,
            "med": 1.0875,
            "q3": 1.216,
            "whishi": 11.76,
            "mean": 1.243,
            "mean_excluding": 1.14,
        },  # RL for 20 nodes 2 agents prop 0.8
        {
            "whislo": 1,
            "q1": 1.002,
            "med": 1.082,
            "q3": 1.21875,
            "whishi": 2.72,
            "mean": 1.15,
            "mean_excluding": 1.15,
        },  # Optimistic baseline for 20 nodes 2 agents prop 0.8
    ]

    # Labels
    labels = [
        "10 nodes 2 agents prop 0.4 RL",
        "10 nodes 2 agents prop 0.4 Optimistic Baseline",
        "10 nodes 2 agents prop 0.8 RL",
        "10 nodes 2 agents prop 0.8 Optimistic Baseline",
        "20 nodes 2 agents prop 0.8 RL",
        "20 nodes 2 agents prop 0.8 Optimistic Baseline",
    ]
    label_legend = ["RL", "Optimistic Baseline"]
    colors = ["red", "blue"]

    # Create traces (one per box plot)
    fig = go.Figure()
    for i, stats in enumerate(box_stats):
        # Add the mean as a separate scatter point
        fig.add_trace(
            go.Scatter(
                x=[labels[i]],
                y=[stats["mean"]],
                mode="markers",
                marker=dict(
                    symbol="diamond",
                    size=10,
                    color="red",
                    line=dict(color="black", width=1),
                ),
                name=(f"Mean" if i == 0 else None),  # Only show in legend once
                legendgroup="mean",
                showlegend=(i == 0),  # Only show in legend once
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[labels[i]],
                y=[stats["mean_excluding"]],
                mode="markers",
                marker=dict(
                    symbol="diamond",
                    size=10,
                    color="yellow",
                    line=dict(color="black", width=1),
                ),
                name=(
                    f"Mean Excluding Failed Episodes" if i == 0 else None
                ),  # Only show in legend once
                legendgroup="mean",
                showlegend=(i == 0),  # Only show in legend once
            )
        )

        fig.add_trace(
            go.Box(
                y=[
                    stats["whislo"],
                    stats["q1"],
                    stats["med"],
                    stats["q3"],
                    stats["whishi"],
                ],
                boxpoints=False,  # Hide outliers
                name=labels[i],
                quartilemethod="inclusive",
                marker=dict(
                    symbol="diamond", size=8, color=colors[i % 2]
                ),  # Mean marker
            )
        )

    # Layout settings
    fig.update_layout(
        title="Box and Whisker Plot of the Competitive Ratio",
        xaxis_title="Different Experiments",
        yaxis_title="Competitive Ratio",
        width=700,
        height=500,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    # Do not save to a file. -> Will open a browser window
    fig.show()


def plot_learning_curve_general(
    names, average_window, num_steps_before_update, frequency_testing, title
):
    legend_labels = [
        "Individual",
        "Team",
        "2 Critic - Decay",
        "Mixed at 0.5",
        "Linear Decay From Mixed",
    ]
    colors = ["red", "black", "green", "blue", "yellow"]
    current_directory = os.getcwd()
    parent_dir = os.path.dirname(current_directory)
    log_directory_names = [os.path.join(parent_dir, "Logs", name) for name in names]
    plt.figure(figsize=(10, 6))

    for i, name in enumerate(log_directory_names):
        # Read values from csv files. Ignore the first value, which is 0 (csv adds 0 for some reason as the first value)
        csv_location = os.path.join(name, "learning_curve_series.csv")
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
            color=colors[i],
            label=legend_labels[i],
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
    plt.title(title)
    plt.xlabel("Training Timesteps")
    plt.ylabel("Average Competitive Ratio")
    plt.legend()
    location_of_plot = os.path.join(parent_dir, "Logs/Unit_Tests")
    plt.savefig(os.path.join(location_of_plot, "Many_Learning_Curves.png"))
    plt.close()


def plot_bar_graph_plotly_general(
    groups,
    categories,
    values,
    title,
    x_axis_title,
    y_axis_title,
    colors,
    shift_num_columns=None,
):
    # Bar positions
    x_positions = [i for i in range(len(groups))]  # One position for each group
    bar_width = 0.2  # Width of each bar

    # Create figure
    fig = go.Figure()

    # Add bars for each category
    for i, category in enumerate(categories):
        if shift_num_columns is not None:
            shift_num_columns = (i // 2) * 0.05
        else:
            shift = 0
        fig.add_trace(
            go.Bar(
                x=[
                    x - bar_width / 2 + i * bar_width + shift for x in x_positions
                ],  # Adjust bar positions
                y=[values[j][i] for j in range(len(groups))],  # Values for each group
                name=category,
                marker_color=colors[i],
                width=bar_width,
                text=[f"{values[j][i]:.2f}" for j in range(len(groups))],
                textposition="outside",
                texttemplate="%{text}",
            )
        )

    # Update layout for better visualization
    fig.update_layout(
        title=title,
        xaxis=dict(
            tickvals=x_positions,
            ticktext=groups,
            title=x_axis_title,
        ),
        yaxis=dict(title=y_axis_title),
        barmode="group",  # Grouped bars
        bargap=0.2,  # Space between groups
    )

    # Show figure
    fig.show()


# Failure rate and percentage for 10, 20, and 30 nodes
def plot_bar_graph_plotly():
    groups = [
        "10 nodes 2 agents prop 0.4",
        "10 nodes 2 agents prop 0.8",
        "20 nodes 2 agents prop 0.4",
        "20 nodes 2 agents prop 0.8",
        "20 nodes 4 agents prop 0.4",
        "20 nodes 4 agents prop 0.8",
        "30 nodes 2 agents prop 0.4",
        "30 nodes 2 agents prop 0.8",
        "30 nodes 4 agents prop 0.4",
        "30 nodes 4 agents prop 0.8",
    ]
    categories = [
        "Percentage of episodes where RL beats the optimistic baseline",
        "Percentage of episodes where RL equals to the optimistic baseline",
        "Failure rate of RL",
    ]  # Three bars per group
    values = [
        [65.68, 0.6214, 1],  # 10 nodes 2 agents prop 0.4
        [73.47, 1.36, 1],  # 10 nodes 2 agents prop 0.8
        [1, 1, 1],  # 20 nodes 2 agents prop 0.4
        [53.30, 4.96, 1],  # 20 nodes 2 agents prop 0.8
        [1, 1, 1],  # 20 nodes 4 agents prop 0.4
        [1, 1, 1],  # 20 nodes 4 agents prop 0.8
        [1, 1, 1],  # 30 nodes 2 agents prop 0.4
        [1, 1, 1],  # 30 nodes 2 agents prop 0.8
        [1, 1, 1],  # 30 nodes 4 agents prop 0.4
        [1, 1, 1],  # 30 nodes 4 agents prop 0.8
    ]

    # Bar positions
    x_positions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # One position for each group
    bar_width = 0.3  # Width of each bar

    # Colors for each category
    colors = ["red", "blue", "green"]

    # Create figure
    fig = go.Figure()

    # Add bars for each category
    for i, category in enumerate(categories):
        fig.add_trace(
            go.Bar(
                x=[
                    x - bar_width / 2 + i * bar_width for x in x_positions
                ],  # Adjust bar positions
                y=[values[j][i] for j in range(len(groups))],  # Values for each group
                name=category,
                marker_color=colors[i],
                width=bar_width,
                text=[f"{values[j][i]:.2f}" for j in range(len(groups))],
                textposition="outside",
                texttemplate="%{text}",
            )
        )

    # Update layout for better visualization
    fig.update_layout(
        title="Statistics for Different Number of Nodes and Agents",
        xaxis=dict(
            tickvals=x_positions,
            ticktext=groups,
            title="Number of Nodes and Agents",
        ),
        yaxis=dict(title="Percentage (%)"),
        barmode="group",  # Grouped bars
        bargap=0.2,  # Space between groups
    )

    # Show figure
    fig.show()


def plot_mean_median():
    groups = [
        "10 nodes 2 agents",
        "20 nodes 2 agents",
        "20 nodes 4 agents",
        "30 nodes 2 agents",
        "30 nodes 4 agents",
    ]
    categories = ["Mean RL", "Mean Optimistic", "Median RL", "Median Optimistic"]
    colors = [
        "rgb(139, 0, 0)",
        "rgb(255, 102, 102)",
        "rgb(0, 0, 139)",
        "rgb(102, 178, 255)",
    ]  # dark red, light, red, dark blue, light blue
    title = "Mean and Median of RL and Optimistic Baseline at Prop 0.8"
    x_axis_title = "Number of Nodes and Agents"
    y_axis_title = "Competitive Ratio"
    values = [
        [1.11, 1.08, 1.11, 1.08],
        [1.243, 1.14, 1.243, 1.14],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]
    plot_bar_graph_plotly_general(
        groups, categories, values, title, x_axis_title, y_axis_title, colors, 2
    )


def plot_percentage():
    groups = [
        "10 nodes 2 agents",
        "20 nodes 2 agents",
        "20 nodes 4 agents",
        "30 nodes 2 agents",
        "30 nodes 4 agents",
    ]
    categories = [
        "Percentage RL Beats Optimistic",
        "Percentage RL Equals Optimistic",
        "Failure Rate RL",
    ]
    colors = ["red", "blue", "green"]
    title = "Statistics for Different Number of Nodes and Agents for Prop 0.8"
    x_axis_title = "Number of Nodes and Agents"
    y_axis_title = "Percentage (%)"
    values = [
        [73.47, 1.36, 1],
        [53.30, 4.96, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    plot_bar_graph_plotly_general(
        groups, categories, values, title, x_axis_title, y_axis_title, colors
    )


def plot_learning_curve():
    names = [
        os.path.join("10_nodes_2_agents", "experiment_1_critic_individual"),
        os.path.join("10_nodes_2_agents", "experiment_1_critic_team"),
        os.path.join("10_nodes_2_agents", "experiment_2_critic_decay"),
        os.path.join("10_nodes_2_agents", "experiment_1_critic_mixed_no_decay"),
        os.path.join("10_nodes_2_agents", "experiment_1_critic_decay_best"),
    ]
    plot_learning_curve_general(
        names, 5, 5500, 20, "Learning Curves for 10 Nodes 2 Agents"
    )


if __name__ == "__main__":
    plot_mean_median()
    # plot_percentage()
    # plot_learning_curve()
