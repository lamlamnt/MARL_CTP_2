import jax.numpy as jnp
import numpy as np


def zero_pad_graphs(stored_graphs: jnp.ndarray) -> jnp.ndarray:
    pass


if __name__ == "__main__":
    # read from .npy file inference graphs
    stored_graphs = np.load("inference_graphs.npy")
