"""
RCV1-V2 (Reuters Corpus Volume 1, v2) dataset structures.

Standard HTC benchmark with a 4-level topic hierarchy:
  - Root
    - 4 top-level categories: CCAT, ECAT, GCAT, MCAT
    - ~55 second-level sub-topics (e.g. CCAT/E21, GCAT/GPOL)
    - ~40 third-level sub-sub-topics
    - ~5 fourth-level leaves

Each document can have multiple topic labels (multi-label).
The hierarchy is a tree: each child has exactly one parent.
"""

import logging
from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class _SamplesHolder:
    """Mutable holder so the pipeline can attach tensor views."""

    x: Any | None = None
    y: Any | None = None


class RCV1Split:
    """ARFF-compatible data split for one RCV1 partition."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_local: list[list[np.ndarray]],
    ) -> None:
        self.x = x  # (N, feat_dim) float32
        self.y = y  # (N, total_labels) float32 — global binary labels
        self.y_local = y_local  # list[list[ndarray]] — per-sample, per-level
        self.samples = _SamplesHolder()


class RCV1HierarchyManager:
    """Manages the RCV1-V2 label hierarchy: root → 4 areas → ~103 topics.

    Topic paths use ``/`` as separator (e.g. ``"CCAT/E21/E211"``).
    The hierarchy is built automatically from the set of observed topic
    paths in the training data.
    """

    def __init__(self) -> None:
        self.g = nx.DiGraph()
        self.g_t = nx.DiGraph()
        self.levels: dict[int, list[str]] = defaultdict(list)
        self.levels_size: dict[int, int] = {}
        self.nodes_idx: dict[str, int] = {}
        self.local_nodes_idx: dict[int, dict[str, int]] = {}
        self.max_depth: int = 0
        self.terms: list[str] = []
        self.edge_index: dict[int, np.ndarray] = {}
        self.a: np.ndarray = np.array([])
        self._is_fitted: bool = False

    @classmethod
    def from_topic_paths(
        cls, all_topic_paths: list[list[str]]
    ) -> "RCV1HierarchyManager":
        """Build the hierarchy from a list of topic-path lists.

        Each element of *all_topic_paths* is a list of topic path strings
        for one document, e.g. ``["CCAT/E21/E211", "GCAT/GPOL"]``.
        """
        mgr = cls()
        mgr._build_graphs(all_topic_paths)
        mgr._build_edge_index()
        mgr.a = nx.to_numpy_array(mgr.g, nodelist=mgr.terms, dtype=np.float32)
        mgr._is_fitted = True
        logger.info(
            "RCV1 hierarchy: %d terms, levels %s",
            len(mgr.terms),
            dict(mgr.levels_size),
        )
        return mgr

    def _build_graphs(self, all_topic_paths: list[list[str]]) -> None:
        """Build DiGraph from observed topic paths."""
        self.g.add_node("root")
        self.levels[0].append("root")

        # Collect all unique nodes from topic paths
        all_nodes: set = {"root"}
        for paths in all_topic_paths:
            for path in paths:
                parts = path.split("/")
                # Add each prefix as a node
                for i in range(1, len(parts) + 1):
                    node = "/".join(parts[:i])
                    all_nodes.add(node)

        # Build edges: child → parent based on path prefix
        for node in all_nodes:
            if node == "root":
                continue
            parts = node.split("/")
            if len(parts) == 1:
                # Top-level category → root
                self.g.add_edge(node, "root")
            else:
                # Sub-topic → parent path
                parent = "/".join(parts[:-1])
                if parent in all_nodes:
                    self.g.add_edge(node, parent)
                else:
                    self.g.add_edge(node, "root")

        # Sort terms by depth then name
        self.g_t = self.g.reverse()
        self.terms = sorted(
            self.g.nodes(),
            key=lambda x: (
                nx.shortest_path_length(self.g, x, "root"),
                x,
            ),
        )
        self.nodes_idx = {node: idx for idx, node in enumerate(self.terms)}

        for label in self.terms:
            if label != "root":
                depth = nx.shortest_path_length(self.g_t, "root", label)
                if label not in self.levels[depth]:
                    self.levels[depth].append(label)

        self.levels_size = {k: len(v) for k, v in self.levels.items()}
        self.max_depth = max(self.levels_size.keys())
        self.local_nodes_idx = {
            depth: {node: i for i, node in enumerate(nodes)}
            for depth, nodes in self.levels.items()
        }

    def _build_edge_index(self) -> None:
        """Build parent→child adjacency matrices per level transition."""
        for depth in range(1, self.max_depth + 1):
            prev_nodes = self.levels[depth - 1]
            curr_nodes = self.levels[depth]

            matrix = np.zeros((len(prev_nodes), len(curr_nodes)), dtype=np.float32)
            parent_map = {node: i for i, node in enumerate(prev_nodes)}
            child_map = {node: i for i, node in enumerate(curr_nodes)}

            for c_node in curr_nodes:
                for p_node in self.g.successors(c_node):
                    if p_node in parent_map:
                        matrix[parent_map[p_node], child_map[c_node]] = 1.0

            self.edge_index[depth] = matrix

    def get_labels(self, topic_paths: list[str]) -> tuple[np.ndarray, list[np.ndarray]]:
        """Convert topic paths into global + local binary label vectors.

        Each topic path activates that node plus all its ancestors.

        Args:
            topic_paths: List of topic path strings
                         (e.g. ``["CCAT/E21", "GCAT/GPOL"]``).

        Returns:
            y_global: Binary vector of length ``len(self.terms)``.
            y_local: List of binary vectors, one per depth level.
        """
        y_global = np.zeros(len(self.terms), dtype=np.float32)
        y_local = [
            np.zeros(size, dtype=np.float32)
            for _, size in sorted(self.levels_size.items())
        ]

        # Root is always active
        y_global[self.nodes_idx["root"]] = 1.0
        y_local[0][self.local_nodes_idx[0]["root"]] = 1.0

        for path in topic_paths:
            if path not in self.nodes_idx:
                # Try to add partial paths
                parts = path.split("/")
                for i in range(1, len(parts) + 1):
                    partial = "/".join(parts[:i])
                    if partial in self.nodes_idx:
                        self._activate_node(partial, y_global, y_local)
                continue

            self._activate_node(path, y_global, y_local)

        return y_global, y_local

    def _activate_node(
        self,
        node: str,
        y_global: np.ndarray,
        y_local: list[np.ndarray],
    ) -> None:
        """Activate a node and all its ancestors."""
        y_global[self.nodes_idx[node]] = 1.0
        depth = nx.shortest_path_length(self.g_t, "root", node)
        y_local[depth][self.local_nodes_idx[depth][node]] = 1.0

        for ancestor in nx.ancestors(self.g_t, node):
            y_global[self.nodes_idx[ancestor]] = 1.0
            anc_depth = nx.shortest_path_length(self.g_t, "root", ancestor)
            y_local[anc_depth][self.local_nodes_idx[anc_depth][ancestor]] = 1.0
