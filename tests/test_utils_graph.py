"""Unit tests for build_digraph_from_labels."""

import networkx as nx
import pytest

from hmc.utils import build_digraph_from_labels


class TestBuildDigraphFromLabels:
    """Tests for the graph-building utility."""

    # ------------------------------------------------------------------
    # Basic cases
    # ------------------------------------------------------------------

    def test_empty_list_returns_graph_with_only_root(self):
        g = build_digraph_from_labels([])
        assert set(g.nodes()) == {"root"}
        assert len(g.edges()) == 0

    def test_single_top_level_label(self):
        g = build_digraph_from_labels(["cs"])
        assert "cs" in g.nodes()
        assert g.has_edge("cs", "root")

    def test_multiple_top_level_labels(self):
        g = build_digraph_from_labels(["cs", "stat", "math"])
        assert g.has_edge("cs", "root")
        assert g.has_edge("stat", "root")
        assert g.has_edge("math", "root")
        assert len(g.edges()) == 3

    # ------------------------------------------------------------------
    # Nested labels (inferred intermediate nodes)
    # ------------------------------------------------------------------

    def test_single_nested_label_infers_intermediate(self):
        g = build_digraph_from_labels(["cs.AI"])
        # Intermediate node "cs" must be inferred.
        assert "cs" in g.nodes()
        assert "cs.AI" in g.nodes()
        assert g.has_edge("cs.AI", "cs")
        assert g.has_edge("cs", "root")

    def test_deeply_nested_label_three_levels(self):
        g = build_digraph_from_labels(["a.b.c"])
        assert g.has_edge("a.b.c", "a.b")
        assert g.has_edge("a.b", "a")
        assert g.has_edge("a", "root")

    def test_mixed_flat_and_nested(self):
        g = build_digraph_from_labels(["cs", "cs.AI", "cs.LG", "stat", "stat.ML"])
        assert g.has_edge("cs", "root")
        assert g.has_edge("cs.AI", "cs")
        assert g.has_edge("cs.LG", "cs")
        assert g.has_edge("stat", "root")
        assert g.has_edge("stat.ML", "stat")

    # ------------------------------------------------------------------
    # Duplicate / overlap handling
    # ------------------------------------------------------------------

    def test_duplicate_labels_are_idempotent(self):
        g1 = build_digraph_from_labels(["cs", "cs.AI"])
        g2 = build_digraph_from_labels(["cs", "cs", "cs.AI", "cs.AI"])
        assert set(g1.nodes()) == set(g2.nodes())
        assert set(g1.edges()) == set(g2.edges())

    def test_child_listed_before_parent(self):
        """Order should not matter: child before parent still works."""
        g = build_digraph_from_labels(["cs.AI", "cs"])
        assert g.has_edge("cs.AI", "cs")
        assert g.has_edge("cs", "root")

    def test_intermediate_inferred_then_explicit(self):
        """When intermediate is inferred first, explicit later is fine."""
        g = build_digraph_from_labels(["cs.AI", "cs"])
        assert g.has_edge("cs.AI", "cs")
        assert g.has_edge("cs", "root")

    # ------------------------------------------------------------------
    # Edge direction (child → parent)
    # ------------------------------------------------------------------

    def test_edges_point_child_to_parent(self):
        g = build_digraph_from_labels(["cs", "cs.AI"])
        assert g.has_edge("cs", "root")  # cs → root
        assert g.has_edge("cs.AI", "cs")  # cs.AI → cs
        # Reverse should NOT exist.
        assert not g.has_edge("root", "cs")
        assert not g.has_edge("cs", "cs.AI")

    # ------------------------------------------------------------------
    # Graph properties
    # ------------------------------------------------------------------

    def test_graph_is_acyclic(self):
        g = build_digraph_from_labels(["cs", "cs.AI", "stat", "stat.ML"])
        assert nx.is_directed_acyclic_graph(g)

    def test_graph_is_directed(self):
        g = build_digraph_from_labels(["cs"])
        assert g.is_directed()

    # ------------------------------------------------------------------
    # Custom separator
    # ------------------------------------------------------------------

    def test_custom_separator_slash(self):
        g = build_digraph_from_labels(["cs/AI", "cs/LG"], sep="/")
        assert g.has_edge("cs/AI", "cs")
        assert g.has_edge("cs/LG", "cs")
        assert g.has_edge("cs", "root")

    # ------------------------------------------------------------------
    # Custom root
    # ------------------------------------------------------------------

    def test_custom_root_name(self):
        g = build_digraph_from_labels(["a", "a.b"], root="top")
        assert "top" in g.nodes()
        assert "root" not in g.nodes()
        assert g.has_edge("a", "top")
        assert g.has_edge("a.b", "a")

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_string_label_is_skipped(self):
        g = build_digraph_from_labels(["cs", "", "  ", "cs.AI"])
        assert g.has_edge("cs", "root")
        assert g.has_edge("cs.AI", "cs")

    def test_labels_with_trailing_dots(self):
        """Trailing dot produces an empty segment — treated as extra level."""
        g = build_digraph_from_labels(["cs."])
        # "cs." → parts = ["cs", ""], so child = "cs.", parent = "cs"
        assert g.has_edge("cs.", "cs")
        assert g.has_edge("cs", "root")

    def test_single_character_labels(self):
        g = build_digraph_from_labels(["a", "a.b", "x.y.z"])
        assert g.has_edge("a", "root")
        assert g.has_edge("a.b", "a")
        assert g.has_edge("x.y.z", "x.y")
        assert g.has_edge("x.y", "x")
        assert g.has_edge("x", "root")

    def test_all_nodes_have_at_least_one_outgoing_edge_except_root(self):
        g = build_digraph_from_labels(["cs", "cs.AI", "stat", "stat.ML"])
        for node in g.nodes():
            if node == "root":
                continue
            # Every non-root must have at least one parent (outgoing edge
            # in child→parent convention).
            assert g.out_degree(node) >= 1, f"Node '{node}' has no parent"
