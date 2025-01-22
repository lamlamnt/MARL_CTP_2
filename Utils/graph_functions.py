import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_generator


@jax.jit
def sample_blocking_status(
    key: jax.random.PRNGKey, blocking_prob: jnp.ndarray
) -> jnp.ndarray:
    n_nodes = blocking_prob.shape[0]
    blocking_status = jnp.full(
        (n_nodes, n_nodes), CTP_generator.BLOCKED, dtype=jnp.float16
    )
    # Indices for the upper triangular part (excluding the diagonal)
    idx_upper = jnp.triu_indices(n_nodes, k=1)
    keys = jax.random.split(key, num=idx_upper[0].shape)
    for i in range(idx_upper[0].shape[0]):
        element_blocking_status = jax.random.bernoulli(
            keys[i], p=blocking_prob[idx_upper[0][i], idx_upper[1][i]]
        )
        blocking_status = blocking_status.at[idx_upper[0][i], idx_upper[1][i]].set(
            element_blocking_status
        )
        blocking_status = blocking_status.at[idx_upper[1][i], idx_upper[0][i]].set(
            element_blocking_status
        )
    return blocking_status
