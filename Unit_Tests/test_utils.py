import pytest
import pytest_print as pp
import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_generator
from Utils.normalize_add_expensive_edge import add_expensive_edge


@pytest.fixture
def graphRealisation():
    key = jax.random.PRNGKey(101)
    graphRealisation = CTP_generator.CTPGraph_Realisation(
        key, 10, prop_stoch=0.4, num_agents=2
    )
    return graphRealisation


def test_add_expensive_edge(
    printer, graphRealisation: CTP_generator.CTPGraph_Realisation
):
    key = jax.random.PRNGKey(50)
    blocking_status = graphRealisation.sample_blocking_status(key)
    new_weights, new_blocking_prob, new_blocking_status = add_expensive_edge(
        graphRealisation.graph.weights,
        graphRealisation.graph.blocking_prob,
        blocking_status,
        graphRealisation.graph.origin,
        graphRealisation.graph.goal,
    )
    assert jnp.array_equal(new_weights, graphRealisation.graph.weights)
    assert jnp.array_equal(new_blocking_prob, graphRealisation.graph.blocking_prob)
    assert jnp.array_equal(new_blocking_status, blocking_status)

    # An unsolvable graph. Change blocking prob to all blocked
    graphRealisation.graph.blocking_prob = jnp.ones(
        (graphRealisation.graph.n_nodes, graphRealisation.graph.n_nodes)
    )
    blocking_status = graphRealisation.sample_blocking_status(key)
    new_weights, new_blocking_prob, new_blocking_status = add_expensive_edge(
        graphRealisation.graph.weights,
        graphRealisation.graph.blocking_prob,
        blocking_status,
        graphRealisation.graph.origin,
        graphRealisation.graph.goal,
    )
