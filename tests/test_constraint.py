"""Max-constraint module, MCLoss outputs, R-matrix and the constrained MLP."""

import networkx as nx
import numpy as np
import pytest
import torch

from hmc.data.hierarchy import TreeHierarchy
from hmc.models.global_classifier.constraint.model import ConstrainedModel
from hmc.models.global_classifier.constraint.utils import (
    get_constr_out,
    mc_training_outputs,
)
from hmc.utils import build_digraph_from_labels
from hmc.utils.graph import ancestor_closure, build_r_matrix


def _random_dag(n, seed):
    """Child → parent adjacency of a random DAG (node 0 is the root)."""
    rng = np.random.default_rng(seed)
    adj = np.zeros((n, n))
    for child in range(1, n):
        k = int(rng.integers(1, min(child, 3) + 1))
        adj[child, rng.choice(child, size=k, replace=False)] = 1
    return adj


@pytest.fixture(name="dag")
def fixture_dag():
    """Adjacency, R-matrix and its networkx graph for a 25-node DAG."""
    adj = _random_dag(25, seed=0)
    return adj, build_r_matrix(adj), nx.DiGraph(adj)


def test_build_r_matrix_matches_networkx_ancestors(dag):
    """``R[0, a, c] == 1`` iff ``a`` is ``c`` or an ancestor of ``c``."""
    adj, r, graph = dag
    n = len(adj)
    assert r.shape == (1, n, n) and r.dtype == torch.float64
    for c in range(n):
        ancestors = nx.descendants(graph, c) | {c}  # edges point child → parent
        assert set(torch.nonzero(r[0, :, c]).flatten().tolist()) == ancestors


def test_ancestor_closure_without_edges_is_identity():
    """A graph with no edges only relates each node to itself."""
    assert np.array_equal(ancestor_closure(np.zeros((4, 4))), np.eye(4, dtype=bool))


def test_hierarchy_r_matrix_is_the_pipeline_transpose():
    """``Hierarchy.r_matrix`` is the row = descendant layout of the same relation."""
    h = TreeHierarchy.from_graph(build_digraph_from_labels(["a.b.c", "a.d", "e"]))
    r = h.r_matrix
    pipeline = build_r_matrix(h.adjacency)[0].numpy()
    assert r.dtype == np.float32
    assert np.array_equal(r, pipeline.T)
    idx = {name: i for i, name in enumerate(h.nodes)}
    assert r[idx["a.b.c"], idx["a"]] == 1 and r[idx["a"], idx["a.b.c"]] == 0


def test_get_constr_out_is_max_over_descendants(dag):
    """Each class takes the max of its own and its descendants' scores."""
    _, r, _ = dag
    torch.manual_seed(0)
    x = torch.rand(6, r.shape[-1])
    out = get_constr_out(x, r)
    expected = torch.stack(
        [
            torch.stack([x[b, r[0, i].bool()].max() for i in range(r.shape[-1])])
            for b in range(len(x))
        ]
    ).double()
    assert out.dtype == torch.float64
    assert torch.equal(out, expected)
    # (C, C) and (1, C, C) R-matrices are interchangeable
    assert torch.equal(get_constr_out(x, r[0].float()), out)


def test_get_constr_out_is_hierarchy_consistent(dag):
    """A parent never scores below any of its children after the constraint."""
    adj, r, _ = dag
    out = get_constr_out(torch.rand(32, len(adj)), r)
    child, parent = np.nonzero(adj)
    assert torch.all(out[:, parent] >= out[:, child])


def test_mc_training_outputs_branches(dag):
    """Negatives see MCM(h); positives only see their positive descendants."""
    _, r, _ = dag
    torch.manual_seed(1)
    h = torch.rand(8, r.shape[-1])
    y = (torch.rand(8, r.shape[-1]) > 0.6).double()
    out = mc_training_outputs(h, y, r)
    neg = get_constr_out(h, r)
    pos = get_constr_out(h * y, r)
    assert torch.equal(out[y == 0], neg[y == 0])
    assert torch.equal(out[y == 1], pos[y == 1])


def test_mc_training_outputs_gradient_flows_to_scores(dag):
    """The training outputs are differentiable w.r.t. the network outputs."""
    _, r, _ = dag
    h = torch.rand(4, r.shape[-1], requires_grad=True)
    y = (torch.rand(4, r.shape[-1]) > 0.5).double()
    mc_training_outputs(h, y, r).sum().backward()
    assert h.grad is not None and torch.any(h.grad != 0)


@pytest.mark.parametrize("num_layers", [2, 3, 4])
def test_constrained_model_train_raw_eval_constrained(dag, num_layers):
    """Raw sigmoid outputs in training, constrained outputs in eval."""
    _, r, _ = dag
    n = r.shape[-1]
    model = ConstrainedModel(
        input_dim=10,
        hidden_dim=16,
        output_dim=n,
        r_matrix=r,
        hyperparams={"num_layers": num_layers, "dropout": 0.0, "non_lin": "relu"},
    )
    assert len(model.fc) == num_layers
    assert model.fc[0].in_features == 10 and model.fc[-1].out_features == n
    x = torch.randn(5, 10)
    raw = model.train()(x)
    assert raw.shape == (5, n) and torch.all((raw >= 0) & (raw <= 1))
    constrained = model.eval()(x)
    assert torch.equal(constrained, get_constr_out(raw.detach(), r))
    model.baseline_model = True
    assert torch.equal(model(x), raw.detach())
