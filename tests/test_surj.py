"""Tests for Surj: classifier, graph-autoencoder losses, hierarchy violations, CLI."""

import json
import os
import sys

import networkx as nx
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hmc.models.hierarchical.graph_autoencoder import pretrain_label_gae  # noqa: E402
from hmc.models.hierarchical.sparse_r import BlockDiagonalR  # noqa: E402
from hmc.models.hierarchical.surj import SurjClassifier  # noqa: E402
from hmc.utils.metrics.hierarchy_violation import global_hierarchy_violation  # noqa: E402

# The subtree of the Surj paper's Figure 3: root -> A; A -> B, C; B -> D, E; C -> F.
ROOT, A, B, C, D, E, F = range(7)
EDGES = np.array([[A, ROOT], [B, A], [C, A], [D, B], [E, B], [F, C]])  # child -> parent
# Figure 3(a): B < D, B < E and A < E are the violations.
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
    def test_figure_3a(self):
        out = global_hierarchy_violation(SCORES, ANC, DESC, EVAL)
        assert out["pairs"] == 8  # A-B, A-C, A-D, A-E, A-F, B-D, B-E, C-F
        assert out["violations"] == 3
        assert out["samples_with_violation"] == 1

    def test_figure_3b_low_confidence_is_irrelevant(self):
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

    def test_chunks_agree(self):
        s = np.random.default_rng(0).random((50, 7)).astype(np.float32)
        full = global_hierarchy_violation(s, ANC, DESC, EVAL)
        assert global_hierarchy_violation(s, ANC, DESC, EVAL, chunk=7) == full


class TestSurjClassifier:
    def test_shapes_and_range(self):
        model = SurjClassifier(10, 7, torch.randn(7, 4), hidden_dim=16)
        out = model(torch.randn(5, 10))
        assert out.shape == (5, 7)
        assert torch.all((out > 0) & (out < 1))

    def test_label_embeddings_frozen_and_normalized(self):
        model = SurjClassifier(10, 7, torch.randn(7, 4), hidden_dim=16)
        assert "label_embeddings" not in dict(model.named_parameters())
        assert torch.allclose(model.label_embeddings.norm(dim=1), torch.ones(7))
        model(torch.randn(3, 10)).sum().backward()
        assert model.label_embeddings.grad is None

    def test_onehot_is_identity(self):
        model = SurjClassifier(10, 7, None, hidden_dim=16).eval()
        assert torch.equal(model.label_embeddings, torch.eye(7))
        x = torch.randn(3, 10)
        h = torch.nn.functional.normalize(model.encoder(x), dim=1)
        assert torch.allclose(model.similarities(x), h)

    def test_cosine_fc_off(self):
        model = SurjClassifier(10, 7, torch.randn(7, 4), hidden_dim=16, cosine_fc=False)
        assert not hasattr(model, "fc")
        assert model(torch.randn(2, 10)).shape == (2, 7)

    def test_wrong_rows_raise(self):
        with pytest.raises(ValueError):
            SurjClassifier(10, 7, torch.randn(6, 4))


class TestGraphAutoencoder:
    @pytest.mark.parametrize("loss", ["sampled", "full"])
    def test_edges_score_above_non_edges(self, loss):
        z = torch.from_numpy(pretrain_label_gae(EDGES, 7, dim=8, hidden_dim=16, loss=loss))
        logits = z @ z.T
        edge = torch.zeros(7, 7, dtype=torch.bool)
        edge[EDGES[:, 0], EDGES[:, 1]] = edge[EDGES[:, 1], EDGES[:, 0]] = True
        off = ~edge & ~torch.eye(7, dtype=torch.bool)
        assert logits[edge].mean() > logits[off].mean()

    def test_deterministic(self):
        a = pretrain_label_gae(EDGES, 7, dim=8, loss="full", seed=3)
        b = pretrain_label_gae(EDGES, 7, dim=8, loss="full", seed=3)
        np.testing.assert_allclose(a, b)

    def test_features(self):
        feats = np.random.default_rng(0).random((7, 5))
        assert pretrain_label_gae(EDGES, 7, dim=8, epochs=5, features=feats).shape == (7, 8)

    def test_unknown_loss(self):
        with pytest.raises(ValueError):
            pretrain_label_gae(EDGES, 7, loss="kl")


def test_cli_surj_smoke(tmp_path):
    """``--method surj`` end to end on a synthetic seq_FUN ARFF dataset."""
    from hmc.arguments import Args, DatasetConfig, TrainingConfig  # pylint: disable=import-outside-toplevel
    from hmc.pipeline.surj.main import train_surj  # pylint: disable=import-outside-toplevel
    from tests.fixtures.arff import SyntheticARFFFixture  # pylint: disable=import-outside-toplevel
    from tests.test_gofun_loading import (  # pylint: disable=import-outside-toplevel
        _patch_gofun_paths,
        _restore_gofun_paths,
    )

    with SyntheticARFFFixture(
        name="seq_FUN",
        hierarchy="root.CC.CC01,root.CC.CC02,root.MF.MF01,root.BP.BP01",
        num_train=30, num_valid=6, num_test=6,
    ) as fixture:
        _patch_gofun_paths(fixture._tmpdir.name)  # pylint: disable=protected-access
        try:
            args = Args(
                dataset=DatasetConfig(dataset_path=fixture._tmpdir.name, dataset_name="seq_FUN"),  # pylint: disable=protected-access
                output_path=str(tmp_path),
                method="surj",
                training=TrainingConfig(device="cpu", epochs=2, batch_size=8,
                                        best_threshold=True, surj_gae_epochs=5),
            )
            args.results_path = str(tmp_path / "run")
            metrics = train_surj("seq_FUN", args)
        finally:
            _restore_gofun_paths()
    assert metrics["rec"]["ghv"]["violations"] == 0
    assert 0.0 <= metrics["rec"]["auprc_micro"] <= 1.0
    with open(tmp_path / "run" / "metrics.json", encoding="utf-8") as f:
        assert json.load(f)["best_epoch"] >= 1
