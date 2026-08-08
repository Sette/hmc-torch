"""Unit tests for hierarchical heads, losses, and postprocessing (Marco 2)."""

import numpy as np
import pytest
import torch

from hmc.data.hierarchy import DagHierarchy, TreeHierarchy
from hmc.models.hierarchical.heads import (
    GlobalSigmoidHead,
    LocalLevelHead,
    TreePathHead,
)
from hmc.models.hierarchical.label_graph import LabelGAT, LabelGCN
from hmc.models.hierarchical.losses import (
    ContrastiveLabelLoss,
    FocalLoss,
    HierarchicalConsistencyLoss,
    WeightedBCELoss,
)
from hmc.models.hierarchical.postprocess import (
    IsotonicCalibrator,
    PlattCalibrator,
    reconcile,
    reconcile_dag,
    reconcile_tree,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fun_tree():
    return TreeHierarchy.from_fun_cat_terms(
        ["root.A.A1", "root.A.A2", "root.B.B1", "root.C"]
    )


@pytest.fixture(scope="module")
def level_sizes():
    return {0: 1, 1: 3, 2: 3}


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


class TestGlobalSigmoidHead:
    def test_output_shape(self):
        head = GlobalSigmoidHead(input_dim=64, n_nodes=10, hidden_dim=32, num_layers=2)
        x = torch.randn(4, 64)
        out = head(x)
        assert out.shape == (4, 10)
        assert (out >= 0).all() and (out <= 1).all()

    def test_linear_probe(self):
        """num_layers=0 should just be a linear + sigmoid."""
        head = GlobalSigmoidHead(input_dim=64, n_nodes=10, num_layers=0)
        x = torch.randn(4, 64)
        out = head(x)
        assert out.shape == (4, 10)


class TestLocalLevelHead:
    def test_output_structure(self, level_sizes):
        head = LocalLevelHead(
            input_dim=64, level_sizes=level_sizes, hidden_dim=32, num_layers=2
        )
        x = torch.randn(4, 64)
        out = head(x)
        assert set(out.keys()) == {"0", "1", "2"}
        assert out["0"].shape == (4, 1)
        assert out["1"].shape == (4, 3)
        assert out["2"].shape == (4, 3)

    def test_to_global(self, fun_tree, level_sizes):
        head = LocalLevelHead(
            input_dim=64, level_sizes=level_sizes, hidden_dim=32, num_layers=2
        )
        x = torch.randn(4, 64)
        level_preds = head(x)

        global_pred = head.to_global(
            level_preds,
            local_nodes_idx=(
                fun_tree.local_nodes_idx
                if hasattr(fun_tree, "local_nodes_idx")
                else {
                    0: {"root": 0},
                    1: {"root.A": 0, "root.B": 1, "root.C": 2},
                    2: {"root.A.A1": 0, "root.A.A2": 1, "root.B.B1": 2},
                }
            ),
            nodes_idx=fun_tree.node_index,
            n_nodes=fun_tree.n_nodes,
        )
        assert global_pred.shape == (4, fun_tree.n_nodes)


class TestTreePathHead:
    def test_output_structure(self, level_sizes):
        head = TreePathHead(input_dim=64, level_sizes=level_sizes, hidden_dim=32)
        x = torch.randn(4, 64)
        out = head(x)
        assert set(out.keys()) == {"0", "1", "2"}


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


class TestWeightedBCE:
    def test_scalar_output(self):
        loss_fn = WeightedBCELoss()
        inputs = torch.sigmoid(torch.randn(8, 10))
        targets = (torch.rand(8, 10) > 0.5).float()
        loss = loss_fn(inputs, targets)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_pos_weight(self):
        targets = torch.zeros(32, 5)
        targets[:, 0] = 1.0  # Only class 0 is positive
        pos_weight = WeightedBCELoss.compute_pos_weight(targets)
        # class 0 has 32 positives, others have 0 → very low weight for class 0
        assert pos_weight[0] < 1.0
        assert (pos_weight[1:] > 1.0).all()

    def test_with_pos_weight(self):
        pos_weight = torch.tensor([0.5, 2.0, 1.0])
        loss_fn = WeightedBCELoss(pos_weight=pos_weight)
        inputs = torch.sigmoid(torch.randn(8, 3))
        targets = (torch.rand(8, 3) > 0.5).float()
        loss = loss_fn(inputs, targets)
        assert loss.item() > 0


class TestFocalLoss:
    def test_scalar_output(self):
        loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
        inputs = torch.sigmoid(torch.randn(8, 10))
        targets = (torch.rand(8, 10) > 0.5).float()
        loss = loss_fn(inputs, targets)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_easy_examples_lower_loss(self):
        """Well-classified examples should have lower focal loss."""
        loss_fn = FocalLoss(gamma=2.0)
        # Confident and correct
        good = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        # Uncertain
        bad = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        loss_good = loss_fn(good, targets)
        loss_bad = loss_fn(bad, targets)
        assert loss_good < loss_bad


class TestHierarchicalConsistencyLoss:
    def test_no_violations_gives_zero(self, fun_tree):
        r_matrix = torch.tensor(fun_tree.r_matrix).unsqueeze(0)
        loss_fn = HierarchicalConsistencyLoss(r_matrix=r_matrix)

        # Create perfectly consistent probs: parent >= child
        probs = torch.zeros(2, fun_tree.n_nodes)
        root_idx = fun_tree.node_index["root"]
        a_idx = fun_tree.node_index["root.A"]
        a1_idx = fun_tree.node_index["root.A.A1"]
        probs[:, root_idx] = 0.9
        probs[:, a_idx] = 0.7
        probs[:, a1_idx] = 0.5

        loss = loss_fn(probs)
        assert loss.item() <= 1e-5

    def test_violation_gives_penalty(self, fun_tree):
        r_matrix = torch.tensor(fun_tree.r_matrix).unsqueeze(0)
        loss_fn = HierarchicalConsistencyLoss(r_matrix=r_matrix)

        # Violation: child > parent
        probs = torch.zeros(2, fun_tree.n_nodes)
        root_idx = fun_tree.node_index["root"]
        a1_idx = fun_tree.node_index["root.A.A1"]
        probs[:, root_idx] = 0.1
        probs[:, a1_idx] = 0.9

        loss = loss_fn(probs)
        assert loss.item() > 0


class TestContrastiveLoss:
    def test_scalar_output(self):
        loss_fn = ContrastiveLabelLoss(temperature=0.07)
        doc_emb = torch.randn(4, 128)
        label_emb = torch.randn(10, 128)
        labels = (torch.rand(4, 10) > 0.5).float()
        labels[0, :3] = 1.0  # Ensure at least one doc has positives

        loss = loss_fn(doc_emb, label_emb, labels)
        assert loss.ndim == 0


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------


class TestReconciliation:
    def test_tree_reconcile_no_violations(self, fun_tree):
        scores = np.random.rand(3, fun_tree.n_nodes).astype(np.float32)
        # Create violations: child > parent
        a_idx = fun_tree.node_index["root.A"]
        a1_idx = fun_tree.node_index["root.A.A1"]
        scores[:, a1_idx] = 0.9
        scores[:, a_idx] = 0.1

        reconciled = reconcile_tree(scores, fun_tree)
        # After reconciliation, parent >= child
        for b in range(3):
            assert reconciled[b, a_idx] + 1e-6 >= reconciled[b, a1_idx]

    def test_dag_reconcile_max_path(self):
        dag = DagHierarchy.from_go_terms(
            ["GO:0008150/GO:0009987/GO:0008152", "GO:0008150/GO:0005575/GO:0005623"]
        )
        scores = np.random.rand(2, dag.n_nodes).astype(np.float32)
        reconciled = reconcile_dag(scores, dag, strategy="max_path")
        assert reconciled.shape == scores.shape
        assert not np.isnan(reconciled).any()

    def test_reconcile_dispatcher(self, fun_tree):
        scores = np.random.rand(2, fun_tree.n_nodes).astype(np.float32)
        out = reconcile(scores, fun_tree)
        assert out.shape == scores.shape


class TestCalibrators:
    def test_platt_fit_calibrate(self):
        cal = PlattCalibrator()
        scores = np.random.rand(100, 3).astype(np.float32) * 0.8 + 0.1
        labels = (scores + np.random.randn(100, 3) * 0.1 > 0.5).astype(np.float32)

        cal.fit(scores, labels)
        calibrated = cal.calibrate(scores)
        assert calibrated.shape == scores.shape
        assert (calibrated >= 0).all() and (calibrated <= 1).all()

    def test_platt_skips_single_class(self):
        cal = PlattCalibrator()
        scores = np.random.rand(50, 2).astype(np.float32)
        labels = np.zeros((50, 2), dtype=np.float32)
        labels[:, 1] = 1.0  # Node 1 is always 1

        cal.fit(scores, labels)
        calibrated = cal.calibrate(scores)
        assert calibrated.shape == scores.shape

    def test_isotonic_fit_calibrate(self):
        cal = IsotonicCalibrator()
        scores = np.random.rand(100, 3).astype(np.float32) * 0.8 + 0.1
        labels = (scores > 0.5).astype(np.float32)

        cal.fit(scores, labels)
        calibrated = cal.calibrate(scores)
        assert calibrated.shape == scores.shape
        assert (calibrated >= 0).all() and (calibrated <= 1).all()


# ---------------------------------------------------------------------------
# Label graph
# ---------------------------------------------------------------------------


class TestLabelGCN:
    def test_output_shape(self):
        n_nodes, embed_dim = 8, 64
        gcn = LabelGCN(
            n_nodes=n_nodes, embed_dim=embed_dim, hidden_dim=32, num_layers=2
        )

        # Build undirected edge index
        edge_index = torch.tensor(
            [
                [0, 1, 1, 2, 0, 3],
                [1, 0, 2, 1, 3, 0],
            ],
            dtype=torch.long,
        )

        out = gcn(edge_index)
        assert out.shape == (n_nodes, embed_dim)

    def test_embeddings_change_after_forward(self):
        n_nodes, embed_dim = 5, 32
        gcn = LabelGCN(
            n_nodes=n_nodes, embed_dim=embed_dim, hidden_dim=16, num_layers=2
        )
        initial = gcn.label_embed.clone()

        edge_index = torch.tensor(
            [
                [0, 1, 2, 3, 4],
                [1, 2, 3, 4, 0],
            ],
            dtype=torch.long,
        )

        out = gcn(edge_index)
        # Output should differ from initial embeddings
        assert not torch.allclose(out, initial, atol=1e-4)


class TestLabelGAT:
    def test_output_shape(self):
        n_nodes, embed_dim = 8, 64
        gat = LabelGAT(
            n_nodes=n_nodes, embed_dim=embed_dim, hidden_dim=32, num_layers=2, heads=4
        )

        edge_index = torch.tensor(
            [
                [0, 1, 1, 2, 0, 3],
                [1, 0, 2, 1, 3, 0],
            ],
            dtype=torch.long,
        )

        out = gat(edge_index)
        assert out.shape == (n_nodes, embed_dim)
