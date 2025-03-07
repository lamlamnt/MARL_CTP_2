import pandas as pd
import numpy as np
import os
import json

# Read in different CSV files and merge them into one Pandas frame. Exclude the failed episodes and print statistics
# For single agent 10 nodes and 30 nodes

# directory_names
reward_exceed_horizon = -1.5
root_directory = "C:\\Users\\shala\\Documents\\Oxford Undergrad\\4th Year\\4YP\\Code\\Not_pushed_to_git"
overall_folder_name = "generalize_node10_results_3_random_seeds"
names = [
    "prop_mixed_30",
    "prop_mixed_32",
    "prop_mixed_34",
]
name_of_json_file = "10_nodes_prop_mixed.json"
df_list = []
for name in names:
    file_path = os.path.join(
        root_directory, overall_folder_name, name, "testing_timestep_output.xlsx"
    )
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    df_list.append(df)
combined_df = pd.concat(df_list, ignore_index=True)
print(combined_df.shape[0])
# I have already manually removed the last incomplete episode from each excel file
# For 30 nodes, I have only done this for the deterministic runs. If want results from non-deterministic runs, remember to delete the last incomplete episode
episodes_df = (
    df.groupby("episode")
    .agg("sum")
    .astype(np.float32)
    .round({"reward": 2, "optimal_path_length": 2, "optimistic_baseline": 2})
)
filtered_df = df.groupby("episode").filter(
    lambda group: ((group["reward"] % reward_exceed_horizon) != 0).all()
)
filtered_episodes_df = (
    filtered_df.groupby("episode")
    .agg("sum")
    .astype(np.float32)
    .round({"reward": 2, "optimal_path_length": 2, "optimistic_baseline": 2})
)
filtered_episodes_df["competitive_ratio"] = (
    filtered_episodes_df["reward"].abs() / filtered_episodes_df["optimal_path_length"]
)
filtered_episodes_df["competitive_ratio_optimistic_baseline"] = (
    filtered_episodes_df["optimistic_baseline"]
    / filtered_episodes_df["optimal_path_length"]
)
result_dict = {
    "average_competitive_ratio_excluding_failed_episodes": float(
        filtered_episodes_df["competitive_ratio"].mean()
    ),
    "median_competitive_ratio_exclude": float(
        filtered_episodes_df["competitive_ratio"].median()
    ),
    "min_competitive_ratio_exclude": float(
        filtered_episodes_df["competitive_ratio"].min()
    ),
    "first_quartile_competitive_ratio_exclude": float(
        filtered_episodes_df["competitive_ratio"].quantile(0.25)
    ),
    "third_quartile_competitive_ratio_exclude": float(
        filtered_episodes_df["competitive_ratio"].quantile(0.75)
    ),
    "max_competitive_ratio_exclude": float(
        filtered_episodes_df["competitive_ratio"].max()
    ),
    "standard_deviation_of_competitive_ratio_exclude": float(
        filtered_episodes_df["competitive_ratio"].std()
    ),
    "average_competitive_ratio_of_optimistic_baseline_exclude": float(
        filtered_episodes_df["competitive_ratio_optimistic_baseline"].mean()
    ),
    "max_competitive_ratio_of_optimistic_baseline_exclude": float(
        filtered_episodes_df["competitive_ratio_optimistic_baseline"].max()
    ),
    "median_competitive_ratio_of_optimistic_baseline_exclude": float(
        filtered_episodes_df["competitive_ratio_optimistic_baseline"].median()
    ),
    "min_competitive_ratio_of_optimistic_baseline_exclude": float(
        filtered_episodes_df["competitive_ratio_optimistic_baseline"].min()
    ),
    "first_quartile_competitive_ratio_of_optimistic_baseline_exclude": float(
        filtered_episodes_df["competitive_ratio_optimistic_baseline"].quantile(0.25)
    ),
    "third_quartile_competitive_ratio_of_optimistic_baseline_exclude": float(
        filtered_episodes_df["competitive_ratio_optimistic_baseline"].quantile(0.75)
    ),
    "standard_deviation_competitive_ratio_of_optimistic_baseline_exclude": float(
        filtered_episodes_df["competitive_ratio_optimistic_baseline"].std()
    ),
}
print(result_dict)
# store result dict into a json file
json_directory = "C:\\Users\\shala\\Documents\\Oxford Undergrad\\4th Year\\4YP\\Code\\MARL_CTP_2\\Logs\\Single_agent_report_writing_retrospect"
with open(os.path.join(json_directory, name_of_json_file), "w") as f:
    f.write("Results excluding failed episodes\n")
    json.dump(result_dict, f, indent=4)
