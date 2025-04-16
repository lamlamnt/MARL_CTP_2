import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import os
import pandas as pd


def percentage_bar_plot_general(
    group_names,
    values,
    title,
    x_axis_title,
    y_axis_title,
    colors,
    width=1400,
    fatness=2.7,
):
    fig = go.Figure()
    num_groups = len(values)  # num_rows
    x_positions = np.linspace(0, fatness, num_groups)

    for i, x in enumerate(x_positions):
        fig.add_trace(
            go.Bar(
                x=[x],
                y=[values[i][0]],
                name="Percentage of Episodes RL Beats OB",
                marker_color=colors[0][i],
                text=[values[i][0]],
                textposition="auto",
                showlegend=(i == 0),
                width=0.1,
            )
        )
        fig.add_trace(
            go.Bar(
                x=[x],
                y=[values[i][1]],
                name="Percentage of Episodes RL Equals OB",
                marker_color=colors[1][i],
                text=[values[i][1]],
                textposition="auto",
                showlegend=(i == 0),
                width=0.1,
            )
        )
        adjusted_value = max(values[i][2], 0.01)
        fig.add_trace(
            go.Bar(
                x=[x + 0.1],  # Offset for separation
                y=[adjusted_value],
                name="Failure Rate for RL",
                marker_color=colors[2][i],
                text=[values[i][2]],
                textposition="auto",
                showlegend=(i == 0),
                width=0.1,
            )
        )

    # Update layout
    fig.update_layout(
        barmode="stack",  # Stack within each group
        title=title,
        xaxis=dict(
            tickmode="array",
            tickvals=list(x_positions),  # Position ticks correctly
            ticktext=list(group_names),
            title=x_axis_title,  # Display correct labels
        ),
        yaxis=dict(
            title=y_axis_title,
            range=[0, 100],  # Force y-axis to show 0-100%
        ),
        template="plotly_white",
        xaxis_tickangle=-15,
        width=width,
        height=500,
    )

    # Show figure
    fig.show()


def box_whisker_general(all_values, title, group_names, width=900):
    # Create figure
    fig = go.Figure()
    colors = ["#FFA07A", "#87CEFA", "#FFA07A", "#87CEFA", "#FFA07A", "#87CEFA"]

    num_groups = len(all_values)
    x_positions = []  # New x positions with extra spacing

    # Generate x positions with extra space every two groups
    for i in range(num_groups):
        x_positions.append(i + (i // 2) * 0.7)  # Adds space after every 2 groups

    for i, x in enumerate(x_positions):
        mean = all_values[i, 0]
        median = all_values[i, 1]
        std = all_values[i, 2]
        min_val = all_values[i, 3]
        max_val = all_values[i, 4]

        # Add shaded region for standard deviation
        fig.add_trace(
            go.Scatter(
                x=[x - 0.1, x + 0.1, x + 0.1, x - 0.1, x - 0.1],
                y=[mean - std, mean - std, mean + std, mean + std, mean - std],
                fill="toself",
                mode="none",
                fillcolor=colors[i % len(colors)],
                opacity=0.4,
                name="Std" if i == 0 else None,
                showlegend=(i == 0),
            )
        )

        # Add min/max vertical line
        fig.add_trace(
            go.Scatter(
                x=[x, x],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="darkblue", width=2),
                name="Max" if i == 0 else None,
                showlegend=(i == 0),
            )
        )

        # Min/Max Horizontal Lines
        fig.add_trace(
            go.Scatter(
                x=[x - 0.1, x + 0.1],
                y=[max_val, max_val],
                mode="lines",
                line=dict(color="blue", width=2),
                showlegend=False,
            )
        )

        # Add median line
        fig.add_trace(
            go.Scatter(
                x=[x - 0.1, x + 0.1],
                y=[median, median],
                mode="lines",
                line=dict(color="black", width=2),
                name="Median" if i == 0 else None,
                showlegend=(i == 0),
            )
        )

        # Add mean line
        fig.add_trace(
            go.Scatter(
                x=[x - 0.1, x + 0.1],
                y=[mean, mean],
                mode="lines",
                line=dict(color="red", width=2),
                name="Mean" if i == 0 else None,
                showlegend=(i == 0),
            )
        )

        # Add text annotations
        fig.add_trace(
            go.Scatter(
                x=[x - 0.2],
                y=[mean],
                text=[f"{mean:.2f}"],
                mode="text",
                textposition="middle left",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[x + 0.2],
                y=[median],
                text=[f"{median:.2f}" if median > 1.0 else f"{median:.1f}"],
                mode="text",
                textposition="middle right",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[x + 0.2],
                y=[max_val],
                text=[f"{max_val:.2f}"],
                mode="text",
                textposition="middle right",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[min_val - 0.2],
                text=[f"{std:.2f}"],
                mode="text",
                textposition="middle center",
                showlegend=False,
            )
        )

    # Update layout with extra spacing between groups
    fig.update_layout(
        title=title,
        xaxis=dict(
            tickvals=x_positions,
            ticktext=group_names,
            title="Experiments",
        ),
        yaxis_title="Competitive Ratio",
        showlegend=True,
        width=width,  # Adjusted width
        height=600,
    )

    fig.show()


def plot_learning_curve_general(
    names,
    average_window,
    num_steps_before_update,
    frequency_testing,
    title,
    file_name,
    first_graph=True,
    ylim_top=10,
):
    if first_graph:
        legend_labels = [
            "1 Critic - Individual",
            "1 Critic - Team",
            "1 Critic - Mixed at 0.5",
        ]
        colors = ["red", "black", "blue"]
    else:
        legend_labels = [
            "1 Critic - Linear Decay",
            "2 Critics - Mixed at 0.5",
            "2 Critics - Linear Decay",
        ]
        colors = ["olive", "orange", "purple"]  # can try brown or pink as well
    current_directory = os.getcwd()
    parent_dir = os.path.dirname(current_directory)
    log_directory_names = [os.path.join(parent_dir, "Logs", name) for name in names]
    plt.figure(figsize=(10, 6))
    y_axis_names = [
        "Mean Competitive Ratio (Including Failed Episodes)",
        "Mean Competitive Ratio Excluding Failed Episodes",
        "Failure Rate",
    ]

    for j in range(3):
        for i, name in enumerate(log_directory_names):
            # Read values from csv files. Ignore the first value, which is 0 (csv adds 0 for some reason as the first value)
            csv_location = os.path.join(
                name, "learning_curve_series_" + str(j) + ".csv"
            )
            learning_curve_series = pd.read_csv(csv_location, header=None).iloc[1:, 0]

            # Replace the 10 with empty
            if j == 1:
                learning_curve_series = learning_curve_series.replace(10, np.nan)
                # If one of the values is NaN, then whole thing NaN
                rolling_mean = learning_curve_series.rolling(
                    window=average_window, min_periods=average_window
                ).mean()
                rolling_std = learning_curve_series.rolling(
                    window=average_window, min_periods=average_window
                ).std()
            else:
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
                color=colors[i],
                alpha=0.2,
            )
        if j == 2:
            plt.axhline(y=0.0, color="green", linestyle="--")
        else:
            plt.axhline(y=1.0, color="green", linestyle="--")  # horizontal line
        if j == 0:
            plt.ylim(0, ylim_top)
        plt.xlabel("Training Timesteps")
        plt.ylabel(y_axis_names[j])
        plt.legend()
        location_of_plot = os.path.join(parent_dir, "Logs/Unit_Tests")
        plt.savefig(
            os.path.join(location_of_plot, file_name + "_" + str(j) + ".pdf"),
            bbox_inches="tight",
        )
        plt.close()
