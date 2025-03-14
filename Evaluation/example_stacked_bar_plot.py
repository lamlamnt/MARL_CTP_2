import plotly.graph_objects as go

# Sample data
groups = ["Group 1", "Group 2", "Group 3"]
values1 = [10, 20, 15]
values2 = [5, 10, 10]
values3 = [8, 12, 18]

# Create figure
fig = go.Figure()

# Add first stacked trace
fig.add_trace(
    go.Bar(
        x=[g + " (Stacked)" for g in groups],
        y=values1,
        name="Stacked Value 1",
        marker_color="blue",
    )
)

# Add second stacked trace
fig.add_trace(
    go.Bar(
        x=[g + " (Stacked)" for g in groups],
        y=values2,
        name="Stacked Value 2",
        marker_color="orange",
    )
)

# Add separate bar for third set of values
fig.add_trace(
    go.Bar(
        x=[g + " (Separate)" for g in groups],
        y=values3,
        name="Separate Value 3",
        marker_color="green",
    )
)

# Update layout for stacked bars and grouped bars
fig.update_layout(
    barmode="stack",
    title="Stacked and Grouped Bar Chart",
    xaxis_title="Groups",
    yaxis_title="Values",
    template="plotly_white",
)

# Show figure
fig.show()
