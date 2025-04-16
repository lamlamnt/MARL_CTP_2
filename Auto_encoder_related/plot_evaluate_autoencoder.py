import time
import flax
import os
import sys
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
import wandb

sys.path.append("..")
from Evaluation.inference import extract_params


def plot_store_results_autoencoder(log_directory, start_time, model_params, out, args):
    with open(os.path.join(log_directory, "weights.flax"), "wb") as f:
        f.write(flax.serialization.to_bytes(model_params))
    # Put here to ensure timing is correct (plotting time is negligible)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Plot training and validation loss on the same plot
    plt.figure()
    plt.plot(out["training_loss"], label="Train loss")
    plt.plot(out["validation_loss"], label="Validation loss")
    plt.xlabel("Number of Updates")
    plt.ylabel("Loss")
    plt.legend()
    # plt.title("Training and Validation Loss")
    plt.savefig(
        os.path.join(log_directory, "training_validation_loss.pdf"), bbox_inches="tight"
    )
    plt.close()

    # Write to JSON file
    # Record hyperparameters and results in JSON file
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_time = {"current_datetime": current_datetime}
    dict_args = vars(args)
    args_path = os.path.join(log_directory, "Hyperparameters_Results" + ".json")
    with open(args_path, "w") as fh:
        json.dump(dict_args, fh)
        fh.write("\n")
        json.dump(date_time, fh, indent=4)
        fh.write("\n")
        json.dump({"Total training time in seconds": elapsed_time}, fh)
        fh.write("\n")
        json.dump({"Final training loss": float(out["training_loss"][-1])}, fh)
        fh.write("\n")
        json.dump({"Final validation loss": float(out["validation_loss"][-1])}, fh)
        fh.write("\n")
        fh.write("\nNetwork architecture: \n")
        for layer_name, weights in extract_params(model_params):
            fh.write(f"{layer_name}: {weights.shape}\n")
        total_num_params = sum(p.size for p in jax.tree_util.tree_leaves(model_params))
        fh.write("Total number of parameters in the network: " + str(total_num_params))

    # Log into wandb
    wandb.summary["final_training_loss"] = float(out["training_loss"][-1])
    wandb.summary["final_validation_loss"] = float(out["validation_loss"][-1])
    print("All done!")
