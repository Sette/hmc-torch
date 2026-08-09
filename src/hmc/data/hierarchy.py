"""Hierarchy abstractions for HMC: trees, DAGs, and common operations.

Provides explicit :class:`TreeHierarchy` (FunCat) and :class:`DagHierarchy`
(Gene Ontology) so that multi-parent semantics cannot be silently conflated
with single-parent tree logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx
import numpy as np


@dataclass
class Hierarchy(ABC):
    """Abstract hierarchy over class nodes.

    Subclasses must populate the internal ``_graph`` (:class:`nx.DiGraph`)
    where every edge points **child → parent**.
    """

    _graph: nx.DiGraph = field(default_factory=nx.DiGraph, repr=False)

    # ------------------------------------------------------------------
    # Concrete properties (derived from _graph)
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> list[str]:
        """All node names, sorted by depth then lexicographically."""
        return sorted(
            self._graph.nodes(),
            key=lambda x: (self._node_depth(x), x),
        )

    @property
    def n_nodes(self) -> int:
        """Total number of class nodes."""
        return self._graph.number_of_nodes()

    @property
    def node_index(self) -> dict[str, int]:
        """Mapping from node name to its integer position in ``nodes``."""
        return {name: i for i, name in enumerate(self.nodes)}

    @property
    def node_levels(self) -> dict[str, int]:
        """Mapping from node name to its depth level (0 = root)."""
        return {node: self._node_depth(node) for node in self._graph.nodes()}

    @property
    def levels(self) -> dict[int, list[str]]:
        """Nodes grouped by depth level ``{level: [node_names]}``."""
        result: dict[int, list[str]] = defaultdict(list)
        for node, level in self.node_levels.items():
            result[level].append(node)
        return dict(result)

    @property
    def max_depth(self) -> int:
        """Number of depth levels."""
        return len(self.levels)

    @property
    def level_sizes(self) -> dict[int, int]:
        """Number of unique nodes per level ``{level: count}``."""
        return {k: len(v) for k, v in self.levels.items()}

    @property
    def adjacency(self) -> np.ndarray:
        """Dense adjacency matrix of shape ``(n_nodes, n_nodes)``.

        Entry ``[i, j] == 1`` iff node *j* is a **child** of node *i*
        (edge i → j exists in the parent→child transpose).
        """
        return nx.to_numpy_array(self._graph, nodelist=self.nodes)

    @property
    def r_matrix(self) -> np.ndarray:
        """Ancestor matrix **R** of shape ``(n_nodes, n_nodes)``.

        ``R[i, j] == 1`` iff node *i* is an ancestor of node *j*
        (or *i == j*).  Note: the matrix is **transposed** relative to
        the original adjacency so that rows represent ancestors and
        columns represent descendants.
        """
        g_parent_to_child = self._graph.reverse()
        n = self.n_nodes
        node_list = self.nodes
        node_idx = {name: i for i, name in enumerate(node_list)}

        r = np.zeros((n, n), dtype=np.float32)
        np.fill_diagonal(r, 1.0)

        for i, node in enumerate(node_list):
            descendants = list(nx.descendants(g_parent_to_child, node))
            if descendants:
                descendant_indices = [node_idx[d] for d in descendants]
                r[i, descendant_indices] = 1.0

        # Transpose so row i = ancestors of node i
        return r.transpose(1, 0)

    @property
    @abstractmethod
    def is_dag(self) -> bool:
        """``True`` for DAG hierarchies (multi-parent possible)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def parents(self, node: str) -> set[str]:
        """Direct parents of *node* (child→parent direction)."""
        return set(self._graph.successors(node))

    def children(self, node: str) -> set[str]:
        """Direct children of *node* (parent→child direction)."""
        return set(self._graph.predecessors(node))

    def ancestors(self, node: str) -> set[str]:
        """All ancestors of *node* (transitive closure of ``parents``)."""
        g_parent_to_child = self._graph.reverse()
        return nx.ancestors(g_parent_to_child, node)

    def descendants(self, node: str) -> set[str]:
        """All descendants of *node*."""
        g_parent_to_child = self._graph.reverse()
        return nx.descendants(g_parent_to_child, node)

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        """Check whether *ancestor* is an ancestor of *descendant*."""
        return ancestor in self.ancestors(descendant)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_labels(self, labels: np.ndarray, sample_id: str = "") -> list[str]:
        """Check that every positive label has all ancestors set to 1.

        Returns a list of violation messages (empty if valid).
        """
        violations: list[str] = []
        node_list = self.nodes
        g_parent_to_child = self._graph.reverse()

        positive_indices = np.where(labels > 0.5)[0]
        for idx in positive_indices:
            node = node_list[idx]
            for ancestor in nx.ancestors(g_parent_to_child, node):
                anc_idx = self.node_index.get(ancestor)
                if anc_idx is not None and labels[anc_idx] < 0.5:
                    violations.append(
                        f"{sample_id}: leaf '{node}' (idx={idx}) is positive "
                        f"but ancestor '{ancestor}' (idx={anc_idx}) is not"
                    )
        return violations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @abstractmethod
    def _node_depth(self, node: str) -> int:
        """Return the depth level of *node* (0 = root)."""
        raise NotImplementedError


# ======================================================================
# Tree hierarchy (FunCat)
# ======================================================================


class TreeHierarchy(Hierarchy):
    """Hierarchy where every child has **exactly one** parent (a tree).

    Used for FunCat-family datasets (``seq_FUN``, ``cellcycle_FUN``, etc.).
    The root node is ``"root"``.
    """

    # TreeHierarchy has no extra fields; _graph is inherited from Hierarchy.

    @property
    def is_dag(self) -> bool:
        return False

    def _node_depth(self, node: str) -> int:
        if node == "root":
            return 0
        # FunCat convention: depth = number of dot-separated segments - 1
        # e.g. "root.CC.CC01" has 3 segments → depth 2
        return len(node.split(".")) - 1

    @classmethod
    def from_graph(cls, graph: nx.DiGraph) -> TreeHierarchy:
        """Build directly from a NetworkX DiGraph (child→parent edges).

        No path extraction needed — uses the graph as-is.
        """
        return cls(_graph=graph.copy())

    @classmethod
    def from_fun_cat_terms(cls, branches: list[str]) -> TreeHierarchy:
        """Build a :class:`TreeHierarchy` from FunCat branch strings.

        Args:
            branches: List of dot-separated paths, e.g.
                ``["root.A.A1", "root.A.A2", "root.B"]``.

        Returns:
            A new :class:`TreeHierarchy` instance with the graph populated.
        """
        g = nx.DiGraph()  # child → parent
        for branch in branches:
            branch = branch.replace("/", ".")
            terms = branch.split(".")
            if len(terms) == 1:
                g.add_edge(terms[0], "root")
            else:
                for i in range(2, len(terms) + 1):
                    g.add_edge(
                        ".".join(terms[:i]),
                        ".".join(terms[: i - 1]),
                    )
        # Ensure root is in the graph (isolated if no edges reference it)
        if "root" not in g:
            g.add_node("root")
        return cls(_graph=g)

    def reconcile(
        self, scores: np.ndarray, strategy: str = "ancestor_max"
    ) -> np.ndarray:
        """Enforce hierarchical consistency on raw scores.

        Ensures that for every node, its score is at least as large as
        the maximum score of its children.

        Args:
            scores: Raw score matrix of shape ``(batch, n_nodes)``.
            strategy: Reconciliation strategy.  Currently only
                ``"ancestor_max"`` is supported for trees.

        Returns:
            Reconciled scores of the same shape.
        """
        if strategy != "ancestor_max":
            raise ValueError(
                f"TreeHierarchy only supports 'ancestor_max' strategy, got '{strategy}'"
            )
        result = scores.copy()
        # Bottom-up: for each non-leaf, ensure its score >= max(children)
        for level in sorted(self.levels.keys(), reverse=True):
            for node in self.levels[level]:
                children = self.children(node)
                if not children:
                    continue
                child_indices = [self.node_index[c] for c in children]
                node_idx = self.node_index[node]
                max_child = result[:, child_indices].max(axis=1)
                result[:, node_idx] = np.maximum(result[:, node_idx], max_child)
        return result


# ======================================================================
# DAG hierarchy (Gene Ontology)
# ======================================================================


@dataclass
class DagHierarchy(Hierarchy):
    """Hierarchy where nodes **may have multiple parents** (a DAG).

    Used for Gene Ontology datasets (``seq_GO``, ``cellcycle_GO``, etc.).
    The root nodes are the GO root terms.
    """

    _roots: list[str] = field(default_factory=list)

    @property
    def is_dag(self) -> bool:
        return True

    @property
    def roots(self) -> list[str]:
        """Top-level root terms of the DAG."""
        if self._roots:
            return self._roots
        # Nodes with no parents are roots
        return [n for n in self._graph.nodes() if self._graph.out_degree(n) == 0]

    def _node_depth(self, node: str) -> int:
        # Shortest path from any root to node (via parent→child edges)
        g_parent_to_child = self._graph.reverse()
        min_depth = float("inf")
        for root in self.roots:
            try:
                d = nx.shortest_path_length(g_parent_to_child, root, node)
                min_depth = min(min_depth, d)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        return int(min_depth) if min_depth != float("inf") else 0

    @classmethod
    def from_graph(
        cls, graph: nx.DiGraph, roots: list[str] | None = None
    ) -> DagHierarchy:
        """Build directly from a NetworkX DiGraph (child→parent edges)."""
        instance = cls(_graph=graph.copy())
        instance._roots = list(roots) if roots else []
        return instance

    @classmethod
    def from_go_terms(
        cls, branches: list[str], roots: list[str] | None = None
    ) -> DagHierarchy:
        """Build a :class:`DagHierarchy` from GO branch strings.

        Args:
            branches: List of ``/``-separated paths, e.g.
                ``["GO:0008150/GO:0009987/GO:0008152"]``.
            roots: Explicit list of GO root terms.  If ``None``, roots
                are inferred from the graph topology.

        Returns:
            A new :class:`DagHierarchy` instance.
        """
        g = nx.DiGraph()  # child → parent
        for branch in branches:
            branch = branch.replace(".", "/")
            terms = branch.split("/")
            if len(terms) >= 2:
                g.add_edge(terms[1], terms[0])
                for i in range(2, len(terms)):
                    g.add_edge(terms[i], terms[i - 1])
            elif len(terms) == 1:
                g.add_node(terms[0])

        instance = cls(_graph=g)
        instance._roots = list(roots) if roots else []
        return instance

    def reconcile(self, scores: np.ndarray, strategy: str = "max_path") -> np.ndarray:
        """Enforce hierarchical consistency on raw scores for a DAG.

        Args:
            scores: Raw score matrix of shape ``(batch, n_nodes)``.
            strategy:
                - ``"max_path"``: For each node, take the maximum score
                  along any path from a root to that node.
                - ``"ancestor_max"``: Same as tree reconciliation but
                  applied over all ancestors transitively.

        Returns:
            Reconciled scores of the same shape.
        """
        if strategy not in ("max_path", "ancestor_max"):
            raise ValueError(
                f"DagHierarchy supports 'max_path' and 'ancestor_max', got '{strategy}'"
            )
        result = scores.copy()
        node_list = self.nodes
        node_idx = self.node_index

        if strategy == "ancestor_max":
            # Bottom-up through levels, ensure ancestor >= max(descendants)
            for level in sorted(self.levels.keys(), reverse=True):
                if level == 0:
                    break
                for node in self.levels[level]:
                    parents = self.parents(node)
                    if not parents:
                        continue
                    parent_indices = [node_idx[p] for p in parents if p in node_idx]
                    if not parent_indices:
                        continue
                    n_idx = node_idx[node]
                    # parent score = max(parent_score, child_score)
                    for p_idx in parent_indices:
                        result[:, p_idx] = np.maximum(
                            result[:, p_idx], result[:, n_idx]
                        )
        else:  # max_path
            # Top-down: for each node, max over all ancestor paths
            g_parent_to_child = self._graph.reverse()
            for node in node_list:
                ancestors = nx.ancestors(g_parent_to_child, node)
                if not ancestors:
                    continue
                anc_indices = [node_idx[a] for a in ancestors if a in node_idx]
                n_idx = node_idx[node]
                max_ancestor = result[:, anc_indices].max(axis=1)
                result[:, n_idx] = np.maximum(result[:, n_idx], max_ancestor)

        return result
