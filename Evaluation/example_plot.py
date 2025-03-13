import plotly.graph_objects as go
import numpy as np

# Generate example data for 3 groups of 2
np.random.seed(42)
group_data = {
    "Group 1A": np.random.normal(50, 10, 100),
    "Group 1B": np.random.normal(55, 12, 100),
    "Group 2A": np.random.normal(60, 8, 100),
    "Group 2B": np.random.normal(65, 10, 100),
    "Group 3A": np.random.normal(70, 9, 100),
    "Group 3B": np.random.normal(75, 11, 100),
}

fig = go.Figure()
colors = ["#FFA07A", "#87CEFA", "#FFA07A", "#87CEFA", "#FFA07A", "#87CEFA"]
legend_shown = {"std": False, "minmax": False, "median": False, "mean": False}

for i, (label, values) in enumerate(group_data.items()):
    mean = np.mean(values)
    std = np.std(values)
    median = np.median(values)
    min_val = np.min(values)
    max_val = np.max(values)

    # Add shaded region for standard deviation
    fig.add_trace(
        go.Scatter(
            x=[i - 0.2, i + 0.2, i - 0.2, i + 0.2],
            y=[mean - std, mean + std, mean + std, mean - std],
            fill="toself",
            mode="none",
            fillcolor=colors[i],
            opacity=0.4,
            name="Standard Deviation" if not legend_shown["std"] else None,
        )
    )
    legend_shown["std"] = True

    # Add min/max lines
    fig.add_trace(
        go.Scatter(
            x=[i, i],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="darkblue", width=2),
            name="Min/Max" if not legend_shown["minmax"] else None,
        )
    )
    legend_shown["minmax"] = True

    # Add median line
    fig.add_trace(
        go.Scatter(
            x=[i - 0.2, i + 0.2],
            y=[median, median],
            mode="lines",
            line=dict(color="black", width=3),
            name="Median" if not legend_shown["median"] else None,
        )
    )
    legend_shown["median"] = True

    # Add mean line
    fig.add_trace(
        go.Scatter(
            x=[i - 0.2, i + 0.2],
            y=[mean, mean],
            mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            name="Mean" if not legend_shown["mean"] else None,
        )
    )
    legend_shown["mean"] = True

fig.update_layout(
    title="Statistical Summary for Groups",
    xaxis=dict(
        tickvals=list(range(len(group_data))),
        ticktext=list(group_data.keys()),
        title="Groups",
    ),
    yaxis_title="Value",
    showlegend=True,
)

fig.show()
