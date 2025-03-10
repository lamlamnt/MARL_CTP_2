import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px


def plot_learning_curve_general(
    names, average_window, num_steps_before_update, frequency_testing, title, file_name
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
    y_axis_names = [
        "Average Competitive Ratio",
        "Average Competitive Ratio Excluding Failed Episodes",
        "Failure Rate",
    ]

    for j in range(3):
        for i, name in enumerate(log_directory_names):
            # Read values from csv files. Ignore the first value, which is 0 (csv adds 0 for some reason as the first value)
            csv_location = os.path.join(
                name, "learning_curve_series_" + str(j) + ".csv"
            )
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
                color=colors[i],
                alpha=0.2,
            )
        plt.title(title)
        plt.xlabel("Training Timesteps")
        plt.ylabel(y_axis_names[j])
        plt.legend()
        location_of_plot = os.path.join(parent_dir, "Logs/Unit_Tests")
        plt.savefig(os.path.join(location_of_plot, file_name + "_" + str(j) + ".png"))
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
    bar_width=0.2,
):
    # Bar positions
    x_positions = [i for i in range(len(groups))]  # One position for each group
    bar_width = bar_width  # Width of each bar

    # Create figure
    fig = go.Figure()

    # Add bars for each category
    for i, category in enumerate(categories):
        if shift_num_columns is not None:
            shift = (i // shift_num_columns) * 0.05
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


# Scatter plot of GB, number of parameters, and total size of all output layers vs performance (maybe multi-objective metric)
def scatter_plot_general():
    data = {
        "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "y": [10, 15, 7, 12, 18, 5, 14, 19, 6, 11],
        "category": ["A"] * 5 + ["B"] * 5,  # Assign categories for coloring
    }

    df = pd.DataFrame(data)

    # Create scatter plot
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="category",
        title="Scatter Plot with Two Data Sets",
        labels={"x": "X Axis", "y": "Y Axis", "category": "Category"},
    )

    # Show plot
    fig.show()
