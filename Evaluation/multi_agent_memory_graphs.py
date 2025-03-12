import os
import sys

sys.path.append("..")
from Evaluation.pretty_plots import scatter_plot_general
import plotly.express as px
import pandas as pd


# 2 scatter plots - total memory - autoencoder and non-autoencoder AND multi-objective metric (side-by-side plots)
def plot_memory():
    autoencoder_memory = [10, 20, 30, 40, 50]
    non_autoencoder_memory = [6, 66, 79, 79, 83]
    names = [
        "10 nodes 2 agents",
        "20 nodes 2 agents",
        "20 nodes 4 agents",
        "30 nodes 2 agents",
        "30 nodes 4 agents",
    ]
    # Create DataFrame
    df = pd.DataFrame(
        {
            "Configuration": names * 2,  # Repeat names for both memory types
            "Memory Usage": autoencoder_memory
            + non_autoencoder_memory,  # Combine memory values
            "Type": ["Encoder"] * 5
            + ["Non-encoder"] * 5,  # Labels for color distinction
        }
    )

    # Create scatter plot
    fig = px.scatter(
        df,
        x="Configuration",
        y="Memory Usage",
        color="Type",
        title="Memory Usage Comparison",
        labels={
            "Configuration": "System Configuration",
            "Memory Usage": "Memory (GB)",
            "Type": "Model Type",
        },
        category_orders={
            "Configuration": list(dict.fromkeys(names))
        },  # Ensures correct order without duplicates
        size_max=10,
    )

    # Show plot
    fig.show()


def plot_performance():
    encoder_multi_objective_metric = [3, 4, 5, 9, 10]
    non_encoder_multi_objective_metric = [1, 2, 3, 4, 5]
    names = [
        "10 nodes 2 agents",
        "20 nodes 2 agents",
        "20 nodes 4 agents",
        "30 nodes 2 agents",
        "30 nodes 4 agents",
    ]
    # Create DataFrame
    df = pd.DataFrame(
        {
            "Configuration": names * 2,  # Repeat names for both memory types
            "Multi-Objective Metric": encoder_multi_objective_metric
            + non_encoder_multi_objective_metric,  # Combine memory values
            "Type": ["Encoder"] * 5
            + ["Non-Encoder"] * 5,  # Labels for color distinction
        }
    )

    # Create scatter plot
    fig = px.scatter(
        df,
        x="Configuration",
        y="Multi-Objective Metric",
        color="Type",
        title="Performance Comparison",
        labels={
            "Configuration": "System Configuration",
            "Multi-Objective Metric": "Multi-Objective Metric",
            "Type": "Model Type",
        },
        category_orders={
            "Configuration": list(dict.fromkeys(names))
        },  # Ensures correct order without duplicates
        size_max=10,
    )

    # Show plot
    fig.show()


if __name__ == "__main__":
    plot_memory()
    plot_performance()
