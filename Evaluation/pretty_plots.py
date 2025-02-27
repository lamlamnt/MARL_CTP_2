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
            "whislo": 5,
            "q1": 10,
            "med": 15,
            "q3": 20,
            "whishi": 25,
            "mean": 13,
            "mean_excluding": 12,
        },  # First box
        {
            "whislo": 7,
            "q1": 12,
            "med": 18,
            "q3": 22,
            "whishi": 30,
            "mean": 15,
            "mean_excluding": 14,
        },  # Second box
        {
            "whislo": 3,
            "q1": 8,
            "med": 14,
            "q3": 19,
            "whishi": 27,
            "mean": 15,
            "mean_excluding": 14,
        },  # Third box
    ]

    # Labels
    labels = ["10 nodes 2 agents", "20 nodes 2 agents", "30 nodes 2 agents"]

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
                marker=dict(symbol="diamond", size=8, color="blue"),  # Mean marker
            )
        )

    # Layout settings
    fig.update_layout(
        title="Box and Whisker Plot of the Competitive Ratio",
        xaxis_title="Different Methods",
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
    average_window = 3
    num_steps_before_update = 2000
    frequency_testing = 5
    names = [
        os.path.join("Handcraft_single_instance", "sacrifice_choose_goals_team"),
        os.path.join("Handcraft_single_instance", "sacrifice_choose_goals_individual"),
    ]
    current_directory = os.getcwd()
    parent_dir = os.path.dirname(current_directory)
    log_directory_names = [os.path.join(parent_dir, "Logs", name) for name in names]
    plt.figure(figsize=(10, 6))

    for name in log_directory_names:
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
    plt.title("Learning Curve with Rolling Average")
    plt.xlabel("Training Timesteps")
    plt.ylabel("Average Competitive Ratio")
    plt.legend(["Team", "Individual"])
    location_of_plot = os.path.join(parent_dir, "Logs/Unit_Tests")
    plt.savefig(os.path.join(location_of_plot, "Many_Learning_Curves.png"))
    plt.close()


# might not be needed. very ugly because of the legend and spacing
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
