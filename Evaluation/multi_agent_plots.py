import os
import sys
import numpy as np
import plotly.graph_objects as go

sys.path.append("..")
from Evaluation.new_plots_general import (
    percentage_bar_plot_general,
    box_whisker_general,
)


def percentage_all_nodes_all_prop():
    groups = [
        "10, 2, 40%",
        "10, 2, 80%",
        "20, 2, 40%",
        "20, 2, 80%",
        "20, 4, 40%",
        "20, 4, 80%",
    ]
    x_axis_title = "Number of Nodes, Number of Agents, Percentage of Stochastic Edges"
    y_axis_title = "Percentage (%)"
    title = "Multi-Agent CTP"
    values = [
        [21.8, 42.2, 0.97],
        [38.7, 34.9, 0.98],
        [23.7, 13.9, 0.97],
        [42.7, 11.4, 3.4],
        [18.5, 2.7, 0.45],
        [37.7, 4.6, 2.3],
    ]
    colors = [
        [
            "#C8E6C9",
            "#66BB6A",
            "#C8E6C9",
            "#66BB6A",
            "#C8E6C9",
            "#66BB6A",
        ],
        [
            "#BBDEFB",
            "#42A5F5",
            "#BBDEFB",
            "#42A5F5",
            "#BBDEFB",
            "#42A5F5",
        ],
        [
            "#FFCDD2",
            "#EF5350",
            "#FFCDD2",
            "#EF5350",
            "#FFCDD2",
            "#EF5350",
        ],
    ]
    percentage_bar_plot_general(
        groups,
        values,
        title,
        x_axis_title,
        y_axis_title,
        colors,
        width=1500,
        fatness=2.0,
        x_axis_tickangle=0,
        x_axis_offset=0.05,
    )


def box_whisker_all_nodes_all_prop():
    groups = [
        "10 Nodes 2 Agents 40% RL",
        "10 Nodes 2 Agents 40% OB",
        "10 Nodes 2 Agents 80% RL",
        "10 Nodes 2 Agents 80% OB",
        "20 Nodes 2 Agents 40% RL",
        "20 Nodes 2 Agents 40% OB",
        "20 Nodes 2 Agents 80% RL",
        "20 Nodes 2 Agents 80% OB",
        "20 Nodes 4 Agents 40% RL",
        "20 Nodes 4 Agents 40% OB",
        "20 Nodes 4 Agents 80% RL",
        "20 Nodes 4 Agents 80% OB",
    ]
    title = "Statistics Excluding Failed Episodes for MA-CTP"
    values = np.array(
        [
            [1.057, 1.0, 0.104, 1.0, 2.328],
            [1.104, 1.0, 0.117, 1.0, 2.122],
            [1.087, 1.0, 0.201, 1.0, 5.602],
            [1.136, 1.024, 0.284, 1.0, 7.099],
            [1.088, 1.062, 0.093, 1.0, 2.671],
            [1.052, 1.008, 0.092, 1.0, 2.211],
            [1.138, 1.081, 0.178, 1.0, 3.765],
            [1.145, 1.077, 0.191, 1.0, 3.000],
            [1.165, 1.135, 0.141, 1.0, 2.726],
            [1.065, 1.044, 0.072, 1.0, 1.862],
            [1.163, 1.108, 0.202, 1.0, 2.776],
            [1.119, 1.087, 0.126, 1.0, 2.776],
        ]
    )
    box_whisker_general(values, title, groups, width=1800)


def bar_memory_performance_old():
    memory = [6, 66, 79, 79, 83]
    percentage = [73.7, 54.1, 42.3, 39.3, 14.8]
    groups = [
        "10 Nodes 2 Agents",
        "20 Nodes 2 Agents",
        "20 Nodes 4 Agents",
        "30 Nodes 2 Agents",
        "30 Nodes 4 Agents",
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=groups,
            y=memory,
            name="Memory",
            marker_color="lightcoral",
            text=memory,
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            x=groups,
            y=percentage,
            name="Percentage RL Beats or Equals OB",
            marker_color="lightgreen",
            text=percentage,
            textposition="auto",
        )
    )
    fig.update_layout(
        barmode="group",
        title="Memory and Peformance of RL",
        xaxis_title="Number of Nodes and Agents",
        yaxis_title="Percentage(%)",
        legend_title="",
        width=1000,
        height=500,
    )
    fig.show()


def bar_memory_performance():
    memory = [6, 66, 79, 79, 83]
    percentage = [73.7, 54.1, 42.3, 39.3, 14.8]
    groups = [
        "10 Nodes 2 Agents",
        "20 Nodes 2 Agents",
        "20 Nodes 4 Agents",
        "30 Nodes 2 Agents",
        "30 Nodes 4 Agents",
    ]

    fig = go.Figure()

    # First bar (percentage) on left y-axis (shown on the left in each group)
    fig.add_trace(
        go.Bar(
            x=groups,
            y=percentage,
            name="Percentage RL Beats or Equals OB",
            marker_color="lightgreen",
            text=percentage,
            textposition="outside",
            yaxis="y1",
            offsetgroup=0,
        )
    )

    # Second bar (memory) on right y-axis (shown on the right in each group)
    fig.add_trace(
        go.Bar(
            x=groups,
            y=memory,
            name="Memory (GB)",
            marker_color="lightcoral",
            text=memory,
            textposition="outside",
            yaxis="y2",
            offsetgroup=1,
        )
    )

    fig.update_layout(
        barmode="group",
        title="Memory and Performance of RL",
        xaxis_title="Number of Nodes and Agents",
        yaxis=dict(title="Percentage (%)", range=[0, 100], side="left"),
        yaxis2=dict(title="Memory (GB)", range=[0, 100], overlaying="y", side="right"),
        legend_title="",
        width=1000,
        height=500,
    )
    fig.update_layout(showlegend=False)

    fig.show()


def percentage_autoencoder():
    groups = [
        "Without Encoder",
        "Using Encoder - Latent Size 100",
    ]
    x_axis_title = "Method"
    y_axis_title = "Percentage (%)"
    title = "Multi-Agent CTP"
    values = [
        [38.73, 34.93, 0.98],
        [21.6, 24.7, 16.2],
    ]
    colors = [
        [
            "#66BB6A",
            "#66BB6A",
        ],
        [
            "#42A5F5",
            "#42A5F5",
        ],
        [
            "#EF5350",
            "#EF5350",
        ],
    ]
    percentage_bar_plot_general(
        groups,
        values,
        title,
        x_axis_title,
        y_axis_title,
        colors,
        width=800,
        fatness=0.5,
        x_axis_tickangle=0,
        height=500,
    )


if __name__ == "__main__":
    percentage_all_nodes_all_prop()
    # box_whisker_all_nodes_all_prop()
    # bar_memory_performance()
    # percentage_autoencoder()
