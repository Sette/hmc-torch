"""Tests for hierarchy violations, graph autoencoders, and hierarchical contrastive learning."""

import os
import sys

import networkx as nx
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hmc.models.hierarchical.graph_autoencoder import (  # noqa: E402
    HierarchicalMultiLabelContrastiveLoss,
    contrastive_knn_predict,
    pretrain_label_gae,
    sibling_pairs,
)
from hmc.models.hierarchical.sparse_r import BlockDiagonalR  # noqa: E402
from hmc.utils.metrics.hierarchy_violation import global_hierarchy_violation  # noqa: E402

# A small tree: root -> A; A -> B, C; B -> D, E; C -> F.
ROOT, A, B, C, D, E, F = range(7)
EDGES = np.array([[A, ROOT], [B, A], [C, A], [D, B], [E, B], [F, C]])  # child -> parent
# The violations in this example are B < D, B < E, and A < E.
SCORES = np.array([[1.0, 0.7, 0.5, 0.6, 0.6, 0.8, 0.4]], dtype=np.float32)


def _pairs(edges, n):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(map(tuple, edges.tolist()))
    anc, desc = [], []
    for node in graph.nodes():
        for a in nx.descendants(graph, node):  # child -> parent: descendants = ancestors
            anc.append(a)
            desc.append(node)
    return graph, np.array(anc), np.array(desc)


GRAPH, ANC, DESC = _pairs(EDGES, 7)
EVAL = np.array([False] + [True] * 6)


class TestGlobalHierarchyViolation:
    def test_violation_example(self):
        out = global_hierarchy_violation(SCORES, ANC, DESC, EVAL)
        assert out["pairs"] == 8  # A-B, A-C, A-D, A-E, A-F, B-D, B-E, C-F
        assert out["violations"] == 3
        assert out["samples_with_violation"] == 1

    def test_low_confidence_is_irrelevant(self):
        low = SCORES * 0.1
        assert global_hierarchy_violation(low, ANC, DESC, EVAL)["violations"] == 3
        assert global_hierarchy_violation(low, ANC, DESC, EVAL, threshold=0.5)["violations"] == 0

    def test_relevant_violation_counted(self):
        # E predicted (0.8 >= 0.75), its parent B and grandparent A are not.
        out = global_hierarchy_violation(SCORES, ANC, DESC, EVAL, threshold=0.75)
        assert out["violations"] == 2

    def test_root_pairs_excluded(self):
        assert global_hierarchy_violation(SCORES, ANC, DESC)["pairs"] == 14

    def test_reconciled_scores_have_none(self):
        r = BlockDiagonalR(GRAPH, column_index={i: i for i in range(7)})
        rec = r.reconcile(torch.from_numpy(SCORES)).numpy()
        assert global_hierarchy_violation(rec, ANC, DESC, EVAL)["violations"] == 0

    def test_depth_offsets_preserve_zero_violations_and_boost_deep_levels(self):
        r = BlockDiagonalR(GRAPH, column_index={i: i for i in range(7)})
        # Boost depth >= 2 nodes (B, C, D, E, F) via positive logit offsets
        rec_cal = r.reconcile_with_depth_offsets(
            torch.from_numpy(SCORES), {0: 0.0, 1: 0.0, 2: 0.5, 3: 1.0}
        ).numpy()
        assert global_hierarchy_violation(rec_cal, ANC, DESC, EVAL)["violations"] == 0
        rec_base = r.reconcile(torch.from_numpy(SCORES)).numpy()
        # Deep leaves D, E, F (indices 4, 5, 6 at depth 3) strictly increase
        assert np.all(rec_cal[:, [D, E, F]] > rec_base[:, [D, E, F]])

    def test_chunks_agree(self):
        s = np.random.default_rng(0).random((50, 7)).astype(np.float32)
        full = global_hierarchy_violation(s, ANC, DESC, EVAL)
        assert global_hierarchy_violation(s, ANC, DESC, EVAL, chunk=7) == full


class TestGraphAutoencoder:
    @pytest.mark.parametrize("loss", ["sampled", "full", "sibling_repulsive"])
    def test_edges_score_above_non_edges(self, loss):
        z = torch.from_numpy(pretrain_label_gae(EDGES, 7, dim=8, hidden_dim=16, loss=loss))
        logits = z @ z.T
        edge = torch.zeros(7, 7, dtype=torch.bool)
        edge[EDGES[:, 0], EDGES[:, 1]] = edge[EDGES[:, 1], EDGES[:, 0]] = True
        off = ~edge & ~torch.eye(7, dtype=torch.bool)
        assert logits[edge].mean() > logits[off].mean()

    def test_sibling_pairs_extracted_correctly(self):
        # EDGES has A -> B, C (siblings B=2, C=3) and B -> D, E (siblings D=4, E=5).
        sib = sibling_pairs(EDGES)
        assert sib.tolist() == [[B, C], [D, E]]
        assert sibling_pairs(np.array([[1, 0], [2, 1]])).shape == (0, 2)

    def test_sibling_repulsive_reduces_sibling_cosine_collapse(self):
        # Multi-branch tree (root 0 -> 8 parents -> 4 leaf siblings each, N=41):
        # Standard GAE separates branches but collapses siblings within each branch.
        edges = [[b, 0] for b in range(1, 9)]
        for b in range(1, 9):
            edges.extend([[8 + (b - 1) * 4 + k, b] for k in range(1, 5)])
        tree_edges = np.asarray(edges, dtype=int)
        z_full = torch.from_numpy(
            pretrain_label_gae(tree_edges, 41, dim=16, hidden_dim=32, loss="full", seed=0)
        )
        z_sr = torch.from_numpy(
            pretrain_label_gae(
                tree_edges, 41, dim=16, hidden_dim=32, loss="sibling_repulsive", seed=0
            )
        )
        zn_full = torch.nn.functional.normalize(z_full, dim=1)
        zn_sr = torch.nn.functional.normalize(z_sr, dim=1)
        sib = sibling_pairs(tree_edges)
        cos_full = (zn_full[sib[:, 0]] * zn_full[sib[:, 1]]).sum(dim=1).mean()
        cos_sr = (zn_sr[sib[:, 0]] * zn_sr[sib[:, 1]]).sum(dim=1).mean()
        assert float(cos_full) > 0.55
        assert float(cos_sr) < 0.05
        assert float(cos_sr) < float(cos_full)

    def test_deterministic(self):
        a = pretrain_label_gae(EDGES, 7, dim=8, loss="full", seed=3)
        b = pretrain_label_gae(EDGES, 7, dim=8, loss="full", seed=3)
        np.testing.assert_allclose(a, b)
        c = pretrain_label_gae(EDGES, 7, dim=8, loss="sibling_repulsive", seed=3)
        d = pretrain_label_gae(EDGES, 7, dim=8, loss="sibling_repulsive", seed=3)
        np.testing.assert_allclose(c, d)

    def test_features(self):
        feats = np.random.default_rng(0).random((7, 5))
        assert pretrain_label_gae(EDGES, 7, dim=8, epochs=5, features=feats).shape == (7, 8)

    def test_unknown_loss(self):
        with pytest.raises(ValueError):
            pretrain_label_gae(EDGES, 7, loss="kl")

    def test_hierarchical_multilabel_contrastive_loss_and_knn(self):
        """H-MLCL computes finite gradients and k-NN preserves zero hierarchy violations."""
        z = pretrain_label_gae(
            EDGES, 7, dim=16, hidden_dim=32, epochs=20, loss="sibling_repulsive", seed=0
        )
        loss_fn = HierarchicalMultiLabelContrastiveLoss(z, EDGES, queue_size=8)
        u = torch.randn(4, 16, requires_grad=True)
        # Ancestor-closed labels on EDGES: (1->0, 2->1, 3->1, 4->2, 5->2, 6->3)
        y = torch.tensor(
            [
                [1, 1, 1, 0, 1, 0, 0],
                [1, 1, 1, 0, 0, 1, 0],
                [1, 1, 0, 1, 0, 0, 1],
                [1, 1, 0, 1, 0, 0, 0],
            ],
            dtype=torch.float32,
        )
        l1 = loss_fn(u, y)
        l1.backward()
        assert torch.isfinite(l1) and l1.item() > 0.0
        assert u.grad is not None and torch.isfinite(u.grad).all()
        assert int(loss_fn.queue_len[0].item()) == 4

        # Second step exercises the populated FIFO queue
        u2 = torch.randn(4, 16, requires_grad=True)
        l2 = loss_fn(u2, y)
        l2.backward()
        assert torch.isfinite(l2) and int(loss_fn.queue_len[0].item()) == 8

        # Contrastive k-NN preserves parent >= child on every edge
        u_eval = torch.randn(5, 16)
        knn_scores = contrastive_knn_predict(u.detach(), y, u_eval, k=3, temp=0.1)
        assert knn_scores.shape == (5, 7)
        for child, parent in EDGES:
            assert (knn_scores[:, parent] + 1e-6 >= knn_scores[:, child]).all()
