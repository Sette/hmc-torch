"""Unit tests for hierarchy abstractions (Marco 1)."""

import numpy as np
import pytest

from hmc.data.hierarchy import DagHierarchy, TreeHierarchy

# ---------------------------------------------------------------------------
# TreeHierarchy
# ---------------------------------------------------------------------------

FUN_BRANCHES = ["root.A.A1", "root.A.A2", "root.B.B1", "root.C"]


@pytest.fixture(scope="module")
def fun_tree() -> TreeHierarchy:
    return TreeHierarchy.from_fun_cat_terms(FUN_BRANCHES)


class TestTreeHierarchy:
    def test_nodes(self, fun_tree):
        nodes = fun_tree.nodes
        assert "root" in nodes
        assert "root.A" in nodes
        assert "root.A.A1" in nodes
        # root(1) + A,B,C(3) = 4 at level 0+1; A1,A2,B1(3) = 3 at level 2 → 7
        assert len(nodes) == 7

    def test_parents(self, fun_tree):
        assert fun_tree.parents("root.A.A1") == {"root.A"}
        assert fun_tree.parents("root.A") == {"root"}
        assert fun_tree.parents("root") == set()
        # Tree: each non-root has exactly one parent
        for node in fun_tree.nodes:
            if node != "root":
                assert len(fun_tree.parents(node)) == 1, (
                    f"Tree node '{node}' should have exactly 1 parent"
                )

    def test_children(self, fun_tree):
        assert fun_tree.children("root") == {"root.A", "root.B", "root.C"}
        assert fun_tree.children("root.A") == {"root.A.A1", "root.A.A2"}

    def test_ancestors(self, fun_tree):
        assert fun_tree.ancestors("root.A.A1") == {"root", "root.A"}
        assert fun_tree.ancestors("root") == set()

    def test_node_index(self, fun_tree):
        idx = fun_tree.node_index
        assert idx["root"] == 0  # sorted first (depth 0)
        for node in fun_tree.nodes:
            assert idx[node] == fun_tree.nodes.index(node)

    def test_levels(self, fun_tree):
        levels = fun_tree.levels
        assert levels[0] == ["root"]
        assert set(levels[1]) == {"root.A", "root.B", "root.C"}
        assert set(levels[2]) == {"root.A.A1", "root.A.A2", "root.B.B1"}
        assert fun_tree.max_depth == 3

    def test_r_matrix(self, fun_tree):
        r = fun_tree.r_matrix
        assert r.shape == (7, 7)
        # Diagonal is 1 (every node is its own ancestor)
        assert (r.diagonal() == 1).all()
        # root (index 0) is an ancestor of every node:
        # column 0 = root, and every node has root as ancestor
        # But r is ancestors-by-node, so r[node, :] lists ancestors.
        # Check that every node (row) has root (col 0) set.
        root_idx = fun_tree.node_index["root"]
        assert (r[:, root_idx] == 1).all(), "every node should have root as ancestor"
        # A leaf (root.A.A1) has exactly 3 ancestors: root, root.A, root.A.A1
        leaf_idx = fun_tree.node_index["root.A.A1"]
        assert r[leaf_idx, :].sum() == 3.0

    def test_is_dag(self, fun_tree):
        assert not fun_tree.is_dag

    def test_validate_labels_passes(self, fun_tree):
        """Properly closed labels should have zero violations."""
        # Build a label where every leaf has ancestors set
        labels = np.zeros(fun_tree.n_nodes, dtype=np.float32)
        leaf = "root.A.A1"
        labels[fun_tree.node_index[leaf]] = 1.0
        for anc in fun_tree.ancestors(leaf):
            labels[fun_tree.node_index[anc]] = 1.0
        violations = fun_tree.validate_labels(labels)
        assert len(violations) == 0

    def test_validate_labels_detects_missing_ancestor(self, fun_tree):
        """Missing ancestor should be flagged."""
        labels = np.zeros(fun_tree.n_nodes, dtype=np.float32)
        labels[fun_tree.node_index["root.A.A1"]] = 1.0
        # root.A and root are NOT set
        violations = fun_tree.validate_labels(labels)
        assert len(violations) >= 1
        assert any("root.A" in v for v in violations)

    def test_reconcile_ancestor_max(self, fun_tree):
        """Reconciliation ensures parent >= max(children)."""
        scores = np.random.rand(2, fun_tree.n_nodes).astype(np.float32)
        reconciled = fun_tree.reconcile(scores, strategy="ancestor_max")

        # After reconciliation, every non-leaf node's score >= max child score
        for node in fun_tree.nodes:
            children = fun_tree.children(node)
            if not children:
                continue
            child_indices = [fun_tree.node_index[c] for c in children]
            node_idx = fun_tree.node_index[node]
            for b in range(scores.shape[0]):
                assert (
                    reconciled[b, node_idx] + 1e-6 >= reconciled[b, child_indices].max()
                ), f"Parent '{node}' score must be >= max child score"

    def test_reconcile_no_violations(self, fun_tree):
        """Reconciled scores should have no hierarchy violations."""
        scores = np.random.rand(3, fun_tree.n_nodes).astype(np.float32)
        # Make some children have higher scores than parents
        scores[:, fun_tree.node_index["root.A.A1"]] = 0.9
        scores[:, fun_tree.node_index["root.A"]] = 0.1
        reconciled = fun_tree.reconcile(scores, strategy="ancestor_max")
        # Now root.A should be at least 0.9
        assert (reconciled[:, fun_tree.node_index["root.A"]] >= 0.9).all()


# ---------------------------------------------------------------------------
# DagHierarchy
# ---------------------------------------------------------------------------

GO_BRANCHES = [
    "GO:0008150/GO:0009987/GO:0008152",
    "GO:0008150/GO:0009987/GO:0016043",
    "GO:0008150/GO:0005575/GO:0005623",
]


@pytest.fixture(scope="module")
def go_dag() -> DagHierarchy:
    return DagHierarchy.from_go_terms(GO_BRANCHES)


class TestDagHierarchy:
    def test_nodes(self, go_dag):
        nodes = go_dag.nodes
        assert "GO:0008150" in nodes
        assert len(nodes) >= 5

    def test_is_dag(self, go_dag):
        assert go_dag.is_dag

    def test_roots(self, go_dag):
        roots = go_dag.roots
        assert "GO:0008150" in roots

    def test_r_matrix(self, go_dag):
        r = go_dag.r_matrix
        assert r.shape[0] == go_dag.n_nodes
        assert r.shape[1] == go_dag.n_nodes
        assert (r.diagonal() == 1).all()

    def test_reconcile_max_path(self, go_dag):
        scores = np.random.rand(2, go_dag.n_nodes).astype(np.float32)
        reconciled = go_dag.reconcile(scores, strategy="max_path")
        assert reconciled.shape == scores.shape
        assert not np.isnan(reconciled).any()

    def test_reconcile_ancestor_max_dag(self, go_dag):
        scores = np.random.rand(2, go_dag.n_nodes).astype(np.float32)
        reconciled = go_dag.reconcile(scores, strategy="ancestor_max")
        assert reconciled.shape == scores.shape

    def test_reconcile_invalid_strategy(self, go_dag):
        scores = np.ones((1, go_dag.n_nodes))
        with pytest.raises(ValueError, match="supports"):
            go_dag.reconcile(scores, strategy="invalid")

    def test_validate_labels_dag(self, go_dag):
        labels = np.zeros(go_dag.n_nodes, dtype=np.float32)
        leaf = "GO:0008152"
        labels[go_dag.node_index[leaf]] = 1.0
        for anc in go_dag.ancestors(leaf):
            labels[go_dag.node_index[anc]] = 1.0
        violations = go_dag.validate_labels(labels)
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# Cross-hierarchy tests
# ---------------------------------------------------------------------------


class TestHierarchyRoundTrip:
    """Verify that TreeHierarchy and DagHierarchy produce consistent shapes."""

    def test_node_count_matches_terms(self, fun_tree):
        assert fun_tree.n_nodes == len(fun_tree.nodes)

    def test_index_mapping_is_bijective(self, fun_tree, go_dag):
        for h in (fun_tree, go_dag):
            idx = h.node_index
            assert len(idx) == h.n_nodes
            # Every index in 0..n_nodes-1 is used exactly once
            assert sorted(idx.values()) == list(range(h.n_nodes))

    def test_adjacency_is_square(self, fun_tree, go_dag):
        for h in (fun_tree, go_dag):
            adj = h.adjacency
            assert adj.shape == (h.n_nodes, h.n_nodes)
