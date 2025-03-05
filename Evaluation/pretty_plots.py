import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.graph_objects as go
import pandas as pd


def plot_box_and_whisker():
    box_stats = [
        {"whislo": 5, "q1": 10, "med": 15, "q3": 20, "whishi": 25},  # First box
        {"whislo": 7, "q1": 12, "med": 18, "q3": 22, "whishi": 30},  # Second box
        {"whislo": 3, "q1": 8, "med": 14, "q3": 19, "whishi": 27},  # Third box
    ]

    # Create the box plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bxp(box_stats, showfliers=False)  # showfliers=False removes outliers

    # Set labels
    ax.set_title("Box and Whisker Plot of the Competitive Ratio")
    ax.set_xlabel("Different Methods")
    ax.set_ylabel("Competitive Ratio")
    ax.set_xticklabels(["Set 1", "Set 2", "Set 3"])  # Label each dataset

    # Show plot
    current_directory = os.getcwd()
    parent_dir = os.path.dirname(current_directory)
    log_directory = os.path.join(parent_dir, "Logs/Unit_Tests")
    plt.savefig(os.path.join(log_directory, "box_whisker.png"))
    plt.close()


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


# Not sure how it would look for proper learning curves yet
# Must have the same number of training steps and frequency of testing
# Deal with Missing/NaN values (first 2 or 3 values) - not sure why they are there.
# If can't figure out, then automatically interpolate
def plot_multiple_learning_curves():
    average_window = 5
    num_steps_before_update = 5500
    frequency_testing = 20
    """
    names = [
        os.path.join("10_nodes_2_agents", "experiment_1_critic_individual"),
        os.path.join("10_nodes_2_agents", "experiment_1_critic_team"),
        os.path.join("10_nodes_2_agents", "experiment_2_critic_decay"),
        os.path.join("10_nodes_2_agents", "experiment_1_critic_mixed_no_decay"),
        os.path.join("10_nodes_2_agents", "experiment_1_critic_decay_best"),
    ]
    colors = ["red", "black", "green", "blue", "yellow"]
    legend_labels = [
        "Individual",
        "Team",
        "2 Critic - Decay",
        "Mixed at 0.5",
        "Linear Decay From Mixed",
    ]
    """
    names = [
        os.path.join("20_nodes_2_agents", "peachy-sweep-4"),
        os.path.join("20_nodes_2_agents", "check_2_critics"),
    ]
    colors = ["red", "black"]
    legend_labels = ["1 critic linear decay", "2 critics linear decay"]
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
    plt.title(
        "Learning Curve with Rolling Average for 20 Nodes 2 Agents 80% Stochastic Edges"
    )
    plt.xlabel("Training Timesteps")
    plt.ylabel("Average Competitive Ratio")
    plt.legend()
    location_of_plot = os.path.join(parent_dir, "Logs/Unit_Tests")
    plt.savefig(os.path.join(location_of_plot, "Many_Learning_Curves.png"))
    plt.close()


# Failure rate and percentage beats for 10, 20, and 30 nodes
def plot_bar_graph_plotly():
    groups = [
        "10 nodes 2 agents prop 0.4",
        "10 nodes 2 agents prop 0.8",
        "20 nodes 2 agents prop 0.8",
    ]
    categories = [
        "Percentage of episodes Where RL beats or equals to the optimistic baseline",
        "Failure Rate of RL",
    ]  # Two bars per group
    values = [
        [65.68, 0.6214],  # 10 nodes 2 agents prop 0.4
        [73.47, 1.36],  # 10 nodes 2 agents prop 0.8
        [53.30, 4.96],  # 20 nodes 2 agents prop 0.8
    ]

    # Bar positions
    x_positions = [0, 1, 2]  # One position for each group
    bar_width = 0.3  # Width of each bar

    # Colors for each category
    colors = ["red", "blue"]

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
        title="Percentage of Episodes where RL Beats or Equals to the Optimistic Baseline and Failure Rate of RL",
        xaxis=dict(
            tickvals=x_positions,
            ticktext=groups,
            title="Experiments",
        ),
        yaxis=dict(title="Percentage (%)"),
        barmode="group",  # Grouped bars
        bargap=0.2,  # Space between groups
    )

    # Show figure
    fig.show()


def plot_bar_graph():
    # Problem: the legend being on the graph makes it hard to read the data
    # 10 nodes 2 agents 0.4 and 0.8
    # Data
    groups = ["0.4", "0.8"]  # 2 groups
    categories = [
        "Optimistic Baseline",
        "Individual Reward",
        "Combined",
        "Tean Reward",
    ]  # 4 bars per group
    values = np.array([[1.416, 1.071, 1.042, 1.046], [1.370, 1.275, 1.112, 1.173]])
    pastel_colors = ["#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9"]

    # Bar settings
    bar_width = 0.15
    x = np.arange(len(groups))  # Positions for groups

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    for i in range(len(categories)):
        bars = ax.bar(
            x + i * bar_width,
            values[:, i],
            width=bar_width,
            label=categories[i],
            color=pastel_colors[i],
        )
        # Add values on top of the bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # Labels and legend
    ax.set_xticks(x + bar_width * 1.5)  # Center x-axis labels
    ax.set_xticklabels(groups)
    ax.set_ylabel("Average Competitive Ratio")
    ax.set_title("10 Nodes 2 Agents")
    ax.legend(
        title="Prop Stoch",
        loc="upper right",
        bbox_to_anchor=(1.15, 1),
        fontsize=6,
        framealpha=0.8,
    )

    current_directory = os.getcwd()
    parent_dir = os.path.dirname(current_directory)
    log_directory = os.path.join(parent_dir, "Logs/Unit_Tests")
    plt.savefig(os.path.join(log_directory, "bar_graph.png"))
    plt.close()


if __name__ == "__main__":
    # plot_box_and_whisker_plotly()
    plot_multiple_learning_curves()
    # plot_bar_graph_plotly()
