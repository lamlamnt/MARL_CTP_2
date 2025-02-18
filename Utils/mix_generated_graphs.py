import os
import numpy as np
import jax.numpy as jnp

if __name__ == "__main__":
    # Load the files, concatenate the list, and save the files
    names = [
        "node_20_agent_2_prop_0.8_random_1",
        "node_20_agent_2_prop_0.8_random_3",
        "node_20_agent_2_prop_0.8_random_5",
    ]
    mixed_name = "node_20_agent_2_prop_0.8"

    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    log_directory = os.path.join(parent_directory, "Generated_graphs")
    all_training_graphs = np.empty((0, 3, 20, 20))
    all_inference_graphs = np.empty((0, 3, 20, 20))

    for name in names:
        directory = os.path.join(log_directory, name)
        training_graph_npy_file = os.path.join(directory, "training_graphs.npy")
        training_graphs = np.load(training_graph_npy_file)
        inference_graph_npy_file = os.path.join(directory, "inference_graphs.npy")
        inference_graphs = np.load(inference_graph_npy_file)
        all_training_graphs = np.append(all_training_graphs, training_graphs, axis=0)
        all_inference_graphs = np.append(
            all_inference_graphs, inference_graphs[:667], axis=0
        )
    print(all_training_graphs.shape)
    print(all_inference_graphs.shape)
    # Save to a file
    new_directory = os.path.join(log_directory, mixed_name)
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
    new_training_graph_npy_file = os.path.join(new_directory, "training_graphs.npy")
    np.save(new_training_graph_npy_file, np.array(all_training_graphs))
    new_inference_graph_npy_file = os.path.join(new_directory, "inference_graphs.npy")
    np.save(new_inference_graph_npy_file, np.array(all_inference_graphs))
