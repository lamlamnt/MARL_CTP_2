import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_generator
from Utils.normalize_add_expensive_edge import get_solvability_matrix
import argparse
import os
import pytest
import pytest_print as pp


@pytest.fixture
def graphRealisation():
    key = jax.random.PRNGKey(101)
    graphRealisation = CTP_generator.CTPGraph_Realisation(
        key, 10, prop_stoch=0.4, num_agents=2
    )
    return graphRealisation


def test_no_inf(graphRealisation: CTP_generator.CTPGraph_Realisation):
    assert not jnp.any(jnp.isinf(graphRealisation.graph.blocking_prob))
    assert not jnp.any(jnp.isinf(graphRealisation.graph.weights))
    key = jax.random.PRNGKey(50)
    blocking_status = graphRealisation.sample_blocking_status(key)
    assert not jnp.any(jnp.isinf(blocking_status))


def test_symmetric_adjacency_matrices(
    graphRealisation: CTP_generator.CTPGraph_Realisation,
):
    assert jnp.all(graphRealisation.graph.weights == graphRealisation.graph.weights.T)
    assert jnp.all(
        graphRealisation.graph.blocking_prob == graphRealisation.graph.blocking_prob.T
    )
    assert jnp.all(
        graphRealisation.graph.blocking_prob == graphRealisation.graph.blocking_prob.T
    )
    key = jax.random.PRNGKey(50)
    blocking_status = graphRealisation.sample_blocking_status(key)
    assert jnp.all(blocking_status == blocking_status.T)


def test_plotting(printer, graphRealisation: CTP_generator.CTPGraph_Realisation):
    current_directory = os.getcwd()
    parent_dir = os.path.dirname(current_directory)
    log_directory = os.path.join(parent_dir, "Logs/Unit_Tests")
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    graphRealisation.graph.plot_nx_graph(log_directory)
    key = jax.random.PRNGKey(50)
    blocking_status = graphRealisation.sample_blocking_status(key)
    graphRealisation.plot_realised_graph(blocking_status, log_directory)


def test_resample(graphRealisation: CTP_generator.CTPGraph_Realisation):
    key = jax.random.PRNGKey(51)
    key, subkey = jax.random.split(key)
    old_blocking_status = graphRealisation.sample_blocking_status(key)
    new_blocking_status = graphRealisation.sample_blocking_status(subkey)
    assert not jnp.array_equal(old_blocking_status, new_blocking_status)


def test_check_blocking_status(graphRealisation: CTP_generator.CTPGraph_Realisation):
    # Check that non-existent edges have blocking status of True
    # Check that deterministic edges have blocking status of False
    key = jax.random.PRNGKey(50)
    blocking_status = graphRealisation.sample_blocking_status(key)
    for i in range(graphRealisation.graph.n_nodes):
        for j in range(graphRealisation.graph.n_nodes):
            if graphRealisation.graph.weights[i, j] == CTP_generator.NOT_CONNECTED:
                assert int(blocking_status[i, j]) is CTP_generator.BLOCKED
            if graphRealisation.graph.blocking_prob[i, j] == 0:
                assert int(blocking_status[i, j]) is CTP_generator.UNBLOCKED


# test that the origins are the first n_agents nodes, the special node is the last node, and goals before special node
# test that the special node is not connected to any other node
def test_goal_origin_special_node(graphRealisation: CTP_generator.CTPGraph_Realisation):
    expected_origin_array = jnp.arange(0, graphRealisation.graph.num_agents)
    assert jnp.array_equal(expected_origin_array, graphRealisation.graph.origin)
    expected_goal_array = jnp.arange(
        graphRealisation.graph.n_nodes - graphRealisation.graph.num_agents - 1,
        graphRealisation.graph.n_nodes - 1,
    )
    assert jnp.array_equal(expected_goal_array, graphRealisation.graph.goal)
    assert graphRealisation.graph.n_nodes - 1 not in graphRealisation.graph.goal

    # Assert that the special node is not connected to any other node
    assert jnp.all(graphRealisation.graph.weights[-1, :] == CTP_generator.NOT_CONNECTED)
    assert jnp.all(graphRealisation.graph.weights[:, -1] == CTP_generator.NOT_CONNECTED)


# test the solvability function (in Utils, not in CTP_generator)
def test_solvability_pair(graphRealisation: CTP_generator.CTPGraph_Realisation):
    # graph plotted above
    key = jax.random.PRNGKey(50)
    blocking_status = graphRealisation.sample_blocking_status(key)
    assert jnp.array_equal(
        get_solvability_matrix(
            graphRealisation.graph.weights,
            blocking_status,
            graphRealisation.graph.origin,
            graphRealisation.graph.goal,
        ),
        jnp.array([[True, True], [True, True]]),
    )

    # An unsolvable graph. Change blocking prob to all blocked
    graphRealisation.graph.blocking_prob = jnp.ones(
        (graphRealisation.graph.n_nodes, graphRealisation.graph.n_nodes)
    )
    blocking_status = graphRealisation.sample_blocking_status(key)
    assert jnp.array_equal(
        get_solvability_matrix(
            graphRealisation.graph.weights,
            blocking_status,
            graphRealisation.graph.origin,
            graphRealisation.graph.goal,
        ),
        jnp.array([[False, False], [False, False]]),
    )
