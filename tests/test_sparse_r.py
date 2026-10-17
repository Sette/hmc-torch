"""Tests for the sparse R-matrix and the validation holdout of ARFF datasets."""

import networkx as nx
import numpy as np
import pytest
import torch

from hmc.datasets.gofun.dataset_arff import SampleData
from hmc.datasets.gofun.manager import _holdout_split
from hmc.models.global_classifier.constraint.utils import get_constr_out
from hmc.models.hierarchical.sparse_r import BlockDiagonalR

# DAG with a node that has two parents (d -> b, d -> c), child -> parent edges.
EDGES = [("a", "root"), ("b", "a"), ("c", "a"), ("d", "b"), ("d", "c"), ("e", "root")]


def _dense_r(nodes, column_index):
    """Dense R in the pipeline convention (R[i, j] = 1 iff i is an ancestor of j)."""
    g = nx.DiGraph(EDGES)
    n = len(nodes)
    r = np.eye(n)
    for node in nodes:
        for anc in nx.descendants(g, node):  # child->parent: descendants = ancestors
            r[column_index[anc], column_index[node]] = 1
    return torch.tensor(r).unsqueeze(0)


@pytest.fixture(name="graph")
def fixture_graph():
    return nx.DiGraph(EDGES)


@pytest.mark.parametrize("order", [["root", "a", "b", "c", "d", "e"], ["e", "d", "root", "c", "a", "b"]])
def test_reconcile_matches_dense_for_any_column_order(graph, order):
    column_index = {n: i for i, n in enumerate(order)}
    sparse = BlockDiagonalR(graph, column_index=column_index)
    scores = torch.rand(5, len(order), dtype=torch.float64)
    expected = get_constr_out(scores, _dense_r(order, column_index))
    assert torch.equal(sparse.reconcile(scores), expected)


def test_reconcile_is_differentiable_and_consistent(graph):
    order = ["root", "a", "b", "c", "d", "e"]
    sparse = BlockDiagonalR(graph, column_index={n: i for i, n in enumerate(order)})
    scores = torch.rand(3, len(order), requires_grad=True)
    out = sparse.reconcile(scores)
    out.sum().backward()
    assert scores.grad is not None
    # Every parent ends up at least as high as each of its children.
    for child, parent in EDGES:
        assert torch.all(out[:, order.index(parent)] >= out[:, order.index(child)])


class _FakeSplit:
    def __init__(self, n):
        self.samples = SampleData(
            x=np.arange(n * 2, dtype=float).reshape(n, 2),
            y=np.arange(n)[:, None].astype(float),
            y_nodes=[[str(i)] for i in range(n)],
            y_local=[[i] for i in range(n)],
        )


def test_holdout_split_is_disjoint_and_deterministic():
    split = _FakeSplit(50)
    train, valid = _holdout_split(split, 0.2)
    ids_train = set(train.samples.y[:, 0].astype(int))
    ids_valid = set(valid.samples.y[:, 0].astype(int))
    assert len(ids_valid) == 10 and len(ids_train) == 40
    assert not ids_train & ids_valid
    assert [n[0] for n in valid.samples.y_nodes] == [str(i) for i in sorted(ids_valid)]
    again_train, _ = _holdout_split(_FakeSplit(50), 0.2)
    assert np.array_equal(again_train.samples.x, train.samples.x)
