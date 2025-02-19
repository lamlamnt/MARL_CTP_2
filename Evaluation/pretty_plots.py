import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == "__main__":
    # Plot box and whisker plot

    # Plot bar graph
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
