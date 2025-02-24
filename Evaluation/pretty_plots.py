import matplotlib.pyplot as plt
import numpy as np
import os


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


def plot_multiple_learning_curves():
    log_directory_names = ["log1", "log2", "log3"]

    for log_directory in log_directory_names:
        # Read values from csv files
        pass

        # Do rolling mean and std for each


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
    plot_box_and_whisker()
    # plot_bar_graph()
