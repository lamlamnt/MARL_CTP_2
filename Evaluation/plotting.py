import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import wandb
import jax.numpy as jnp


def plot_learning_curve(testing_average_competitive_ratio, log_directory, args):
    learning_curve_values = testing_average_competitive_ratio[
        testing_average_competitive_ratio != 0
    ]
    # Plot rolling average (mean and std) of competitive ratio
    learning_curve_series = pd.Series(learning_curve_values)
    rolling_mean = learning_curve_series.rolling(
        window=args.learning_curve_average_window, min_periods=1
    ).mean()
    rolling_std = learning_curve_series.rolling(
        window=args.learning_curve_average_window, min_periods=1
    ).std()
    plt.figure(figsize=(10, 6))
    plt.plot(
        args.num_steps_before_update
        * args.frequency_testing
        * jnp.arange(len(learning_curve_values)),
        rolling_mean,
        linestyle="-",
        color="red",
    )
    plt.fill_between(
        args.num_steps_before_update
        * args.frequency_testing
        * jnp.arange(len(learning_curve_values)),
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        color="blue",
        alpha=0.2,
    )
    plt.title("Learning Curve with Rolling Average")
    plt.xlabel("Training Timesteps")
    plt.ylabel("Average Competitive Ratio")
    plt.savefig(os.path.join(log_directory, "Smoothened_Learning_Curve.png"))
    plt.close()


def save_data_and_plotting(
    all_episode_done,
    all_total_rewards,
    all_optimal_costs,
    directory,
    reward_exceed_horizon,
    num_agents,
    all_optimistic_baseline=None,
    training=True,
) -> dict[str, float]:
    if training == True:
        beginning_str = "training_"
    else:
        beginning_str = "testing_"
    if training == True:
        df = pd.DataFrame(
            data={
                "episode": all_episode_done.cumsum(),
                "reward": all_total_rewards,
                "optimal_cost": all_optimal_costs,
            },
        )
        df["episode"] = df["episode"].shift().fillna(0)
        episodes_df = (
            df.groupby("episode")
            .agg("sum")
            .astype(np.float32)
            .round({"reward": 3, "optimal_cost": 3})
        )
    else:
        # For inference, get the additional optimistic baseline
        df = pd.DataFrame(
            data={
                "episode": all_episode_done.cumsum(),
                "reward": all_total_rewards,
                "optimal_cost": all_optimal_costs,
                "optimistic_baseline": all_optimistic_baseline,
            },
        )
        df["episode"] = df["episode"].shift().fillna(0)
        episodes_df = (
            df.groupby("episode")
            .agg("sum")
            .astype(np.float32)
            .round({"reward": 3, "optimal_cost": 3, "optimistic_baseline": 3})
        )
        episodes_df["competitive_ratio_optimistic_baseline"] = (
            episodes_df["optimistic_baseline"] / episodes_df["optimal_cost"]
        )
    episodes_df = episodes_df.iloc[:-1]
    episodes_df["regret"] = episodes_df["reward"].abs() - episodes_df["optimal_cost"]
    episodes_df["competitive_ratio"] = (
        episodes_df["reward"].abs() / episodes_df["optimal_cost"]
    )
    if episodes_df.shape[0] < 1000000:
        episodes_df.to_excel(
            os.path.join(directory, beginning_str + "episode_output.xlsx"),
            sheet_name="Sheet1",
            index=False,
        )

    if df.shape[0] < 1000000:
        df.to_excel(
            os.path.join(directory, beginning_str + "timestep_output.xlsx"),
            sheet_name="Sheet1",
            index=False,
        )

    # Plot histogram of competitive ratio
    if training == False:
        plt.figure(figsize=(10, 6))
        plt.hist(episodes_df["competitive_ratio"], bins=10)
        plt.xlabel("Competitive Ratio")
        plt.ylabel("Frequency")
        plt.title("Histogram of Competitive Ratio")
        plt.savefig(
            os.path.join(directory, beginning_str + "histogram_competitive_ratio.png")
        )
        plt.close()

    # Get the mean competitive ratio excluding the failed episodes
    if training == False:
        filtered_df = df.groupby("episode").filter(
            lambda group: ((group["reward"] % reward_exceed_horizon) != 0).all()
        )
        filtered_episodes_df = (
            filtered_df.groupby("episode")
            .agg("sum")
            .astype(np.float32)
            .round({"reward": 3, "optimal_cost": 3})
        )
        filtered_episodes_df = filtered_episodes_df.iloc[:-1]
        filtered_episodes_df["competitive_ratio"] = (
            filtered_episodes_df["reward"].abs() / filtered_episodes_df["optimal_cost"]
        )
        # Maybe an alternative way is to get the difference in number of rows between the 2 dataframes
        num_reach_horizon = len(episodes_df) - len(filtered_episodes_df)

        result_dict = {
            "average_regret": float(episodes_df["regret"].mean()),
            "average_competitive_ratio": float(episodes_df["competitive_ratio"].mean()),
            "average_competitive_ratio_excluding_failed_episodes": float(
                filtered_episodes_df["competitive_ratio"].mean()
            ),
            "median_competitive_ratio": float(
                episodes_df["competitive_ratio"].median()
            ),
            "min_competitive_ratio": float(episodes_df["competitive_ratio"].min()),
            "first_quartile_competitive_ratio": float(
                episodes_df["competitive_ratio"].quantile(0.25)
            ),
            "third_quartile_competitive_ratio": float(
                episodes_df["competitive_ratio"].quantile(0.75)
            ),
            "max_competitive_ratio": float(episodes_df["competitive_ratio"].max()),
            "average_reward": float(episodes_df["reward"].mean()),
            "failure_rate (%)": float(num_reach_horizon * 100 / episodes_df.shape[0]),
            "standard deviation of competitive ratio": float(
                episodes_df["competitive_ratio"].std()
            ),
            "average_competitive_ratio_of_optimistic_baseline": float(
                episodes_df["competitive_ratio_optimistic_baseline"].mean()
            ),
            "max_competitive_ratio_of_optimistic_baseline": float(
                episodes_df["competitive_ratio_optimistic_baseline"].max()
            ),
            "median_competitive_ratio_of_optimistic_baseline": float(
                episodes_df["competitive_ratio_optimistic_baseline"].median()
            ),
            "min_competitive_ratio_of_optimistic_baseline": float(
                episodes_df["competitive_ratio_optimistic_baseline"].min()
            ),
            "first_quartile_competitive_ratio_of_optimistic_baseline": float(
                episodes_df["competitive_ratio_optimistic_baseline"].quantile(0.25)
            ),
            "third_quartile_competitive_ratio_of_optimistic_baseline": float(
                episodes_df["competitive_ratio_optimistic_baseline"].quantile(0.75)
            ),
            "standard_deviation_competitive_ratio_of_optimistic_baseline": float(
                episodes_df["competitive_ratio_optimistic_baseline"].std()
            ),
            "percentage_RL_beats_optimistic_baseline": float(
                (
                    episodes_df["competitive_ratio"]
                    < episodes_df["competitive_ratio_optimistic_baseline"]
                ).mean()
                * 100
            ),
            "percentage_RL_equals_to_optimistic_baseline": float(
                (
                    episodes_df["competitive_ratio"]
                    == episodes_df["competitive_ratio_optimistic_baseline"]
                ).mean()
                * 100
            ),
        }
        for key, value in result_dict.items():
            wandb.summary[key] = value
    else:
        result_dict = {}

    return result_dict
