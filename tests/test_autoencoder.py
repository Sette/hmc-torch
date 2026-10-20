"""Tests for the autoencoder variants of experiments/run_autoencoder.py."""

import os
import sys

import networkx as nx
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from run_autoencoder import FeatDAE, GaeHybrid, LabelAE  # noqa: E402
from run_protocol import MLP, _fit  # noqa: E402

from hmc.models.hierarchical.sparse_r import BlockDiagonalR  # noqa: E402

D_IN, HID, V = 6, 16, 5


def _data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    g = nx.DiGraph()
    g.add_nodes_from(range(V))
    g.add_edges_from([(1, 0), (2, 0), (3, 1), (4, 1)])  # child -> parent
    y = (rng.random((n, V)) > 0.6).astype(np.float32)
    y[:, 0] = 1.0
    x = rng.normal(size=(n, D_IN)).astype(np.float32)
    return {
        "x": [x, x[:10], x[:10]],
        "y": [y, y[:10], y[:10]],
        "eval_mask": np.array([False] + [True] * (V - 1)),
        "sparse_r": BlockDiagonalR(g, column_index={i: i for i in range(V)}),
        "anc": np.array([0, 0, 1, 1, 0, 0]),
        "desc": np.array([1, 2, 3, 4, 3, 4]),
    }


def test_gae_hybrid_starts_as_the_mlp():
    torch.manual_seed(0)
    mlp = MLP(D_IN, HID, V, 0.5).eval()
    torch.manual_seed(0)
    hybrid = GaeHybrid(D_IN, HID, V, 0.5, np.random.default_rng(0).normal(size=(V, 4))).eval()
    x = torch.randn(3, D_IN)
    torch.testing.assert_close(hybrid(x), mlp(x))
    assert hybrid.z.shape == (V, 4)


def test_feat_dae_pretraining_reduces_reconstruction():
    torch.manual_seed(0)
    x = np.random.default_rng(1).normal(size=(64, D_IN)).astype(np.float32)
    model = FeatDAE(D_IN, HID, V, 0.0).pretrain(x, "cpu", epochs=30, batch=16)
    assert model.pretrain_losses[-1] < model.pretrain_losses[0]
    assert model(torch.from_numpy(x[:2])).shape == (2, V)


def test_feat_dae_aux_loss_zero_without_weight():
    model = FeatDAE(D_IN, HID, V, 0.0)
    assert float(model.aux_loss(torch.randn(4, D_IN), None)) == 0.0
    assert float(FeatDAE(D_IN, HID, V, 0.0, aux_weight=0.1).aux_loss(torch.randn(4, D_IN), None)) > 0


def test_label_ae_shapes_and_aux():
    model = LabelAE(D_IN, HID, V, 0.0, np.array([False] + [True] * (V - 1)), 1.0)
    x, y = torch.randn(4, D_IN), (torch.rand(4, V) > 0.5).float()
    assert model(x).shape == (4, V)
    assert float(model.aux_loss(x, y)) > 0


def test_aux_loss_is_used_by_fit():
    calls = []

    class Spy(MLP):
        def aux_loss(self, xb, yb):
            calls.append(len(xb))
            return xb.new_zeros(())

    data = _data()
    hp = {"hidden": HID, "batch": 8, "dropout": 0.0, "patience": 2}
    _fit(data, "mc", 0, "cpu", hp, data["x"][0], data["y"][0], 1,
         model_fn=lambda d_in, d_out: Spy(d_in, HID, d_out, 0.0))
    assert sum(calls) == 40
