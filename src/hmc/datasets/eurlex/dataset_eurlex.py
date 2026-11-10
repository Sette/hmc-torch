"""
EUR-Lex 57K dataset structures.

EU legal documents annotated with EUROVOC concepts (~4,271 labels).
The EUROVOC thesaurus is a deep tree (up to 8 levels) where each
concept has a unique parent. The hierarchy is loaded from a separate
concept-relations file or built from observed labels as fallback.

Reference: Chalkidis et al. (ACL 2019), "Large-Scale Multi-Label Text
Classification on EU Legislation"
"""

import json
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


class EURLexSplit:
    """ARFF-compatible data split for one EUR-Lex partition."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        y_local: list[list[np.ndarray]],
        g: nx.DiGraph | None = None,
    ) -> None:
        self.x = x  # (N, feat_dim) float32
        self.y = y  # (N, total_labels) float32 — global binary labels
        self.y_local = y_local  # list[list[ndarray]] — per-sample, per-level
        self.samples = _SamplesHolder()
        # DiGraph for sparse R reconciliation (needed by BlockDiagonalR)
        self.g = g if g is not None else nx.DiGraph()


class EURLexHierarchyManager:
    """Manages the EUROVOC label hierarchy for EUR-Lex 57K.

    The hierarchy can be loaded from a JSONL concept file (preferred)
    or built as a flat tree from observed labels (fallback).

    Concept file format (one JSON object per line):
        {"id": "192", "label": "international trade", "broader": ["18"], ...}

    The EUROVOC tree uses string IDs; parent→child edges are derived
    from the ``broader`` field.
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
    def from_concept_file(
        cls,
        concept_path: str,
        observed_labels: set[str],
    ) -> "EURLexHierarchyManager":
        """Build hierarchy from a EUROVOC concept JSONL file.

        The concept file should have one JSON object per line with at least
        ``"id"`` and optionally ``"broader"`` (list of parent concept IDs).
        Only concepts present in *observed_labels* (plus their ancestors)
        are included in the graph.
        """
        mgr = cls()
        mgr._load_concept_file(concept_path, observed_labels)
        mgr._build_edge_index()
        mgr.a = nx.to_numpy_array(mgr.g, nodelist=mgr.terms, dtype=np.float32)
        mgr._is_fitted = True
        logger.info(
            "EUR-Lex hierarchy (from concepts): %d terms, levels %s",
            len(mgr.terms),
            dict(mgr.levels_size),
        )
        return mgr

    @classmethod
    def from_labels(
        cls,
        all_labels_list: list[list[str]],
    ) -> "EURLexHierarchyManager":
        """Build a flat 2-level hierarchy from observed labels.

        Each concept is placed directly under root.  This is the fallback
        when no EUROVOC concept file is available.
        """
        mgr = cls()
        all_ids: set[str] = set()
        for labels in all_labels_list:
            all_ids.update(labels)

        mgr.g.add_node("root")
        mgr.levels[0].append("root")

        for cid in sorted(all_ids):
            mgr.g.add_edge(cid, "root")

        mgr._finalize_graph()
        mgr._build_edge_index()
        mgr.a = nx.to_numpy_array(mgr.g, nodelist=mgr.terms, dtype=np.float32)
        mgr._is_fitted = True
        logger.info(
            "EUR-Lex hierarchy (flat fallback): %d terms, 2 levels",
            len(mgr.terms),
        )
        return mgr

    @staticmethod
    def _parse_concept_jsonl(
        concept_path: str,
    ) -> dict[str, list[str]]:
        """Parse a EUROVOC concept JSONL file into a broader-mapping dict.

        Returns a dict mapping concept_id → list of parent (broader) concept IDs.
        """
        broader: dict[str, list[str]] = {}
        with open(concept_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cid = str(rec.get("id", rec.get("concept_id", "")))
                if not cid:
                    continue
                parents = rec.get("broader", rec.get("broader_concepts", []))
                if isinstance(parents, str):
                    parents = [parents]
                broader[cid] = [str(p) for p in parents]
        return broader

    @staticmethod
    def _collect_relevant_concepts(
        observed_labels: set[str],
        broader: dict[str, list[str]],
    ) -> set[str]:
        """Expand observed labels with their ancestors from the concept hierarchy."""
        relevant: set[str] = set(observed_labels)
        changed = True
        while changed:
            changed = False
            for cid in list(relevant):
                for parent_id in broader.get(cid, []):
                    if parent_id not in relevant:
                        relevant.add(parent_id)
                        changed = True
        return relevant

    def _load_concept_file(
        self,
        concept_path: str,
        observed_labels: set[str],
    ) -> None:
        """Load EUROVOC concept hierarchy from JSONL file."""
        broader = self._parse_concept_jsonl(concept_path)
        relevant = self._collect_relevant_concepts(observed_labels, broader)

        # Build graph
        self.g.add_node("root")
        self.levels[0].append("root")

        for cid in sorted(relevant):
            parents = broader.get(cid, [])
            if not parents:
                self.g.add_edge(cid, "root")
            else:
                for parent_id in parents:
                    if parent_id in relevant:
                        self.g.add_edge(cid, parent_id)
                    else:
                        self.g.add_edge(cid, "root")

        self._finalize_graph()

    def _finalize_graph(self) -> None:
        """Sort terms, assign indices, compute level assignments."""
        self.g_t = self.g.reverse()
        self.terms = sorted(
            self.g.nodes(),
            key=lambda x: (
                (
                    nx.shortest_path_length(self.g, x, "root")
                    if x in self.g and nx.has_path(self.g, x, "root")
                    else 999
                ),
                x,
            ),
        )
        self.nodes_idx = {node: idx for idx, node in enumerate(self.terms)}

        for label in self.terms:
            if label != "root":
                try:
                    depth = nx.shortest_path_length(self.g_t, "root", label)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    depth = 1
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

    def get_labels(self, concept_ids: list[str]) -> tuple[np.ndarray, list[np.ndarray]]:
        """Convert EUROVOC concept IDs into global + local binary labels.

        Each concept ID activates that node plus all its ancestors.

        Args:
            concept_ids: List of EUROVOC concept ID strings.

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

        for cid in concept_ids:
            if cid not in self.nodes_idx:
                continue
            self._activate_node(cid, y_global, y_local)

        return y_global, y_local

    def _activate_node(
        self,
        node: str,
        y_global: np.ndarray,
        y_local: list[np.ndarray],
    ) -> None:
        """Activate a node and all its ancestors."""
        y_global[self.nodes_idx[node]] = 1.0
        try:
            depth = nx.shortest_path_length(self.g_t, "root", node)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            depth = 1
        if depth < len(y_local):
            y_local[depth][self.local_nodes_idx[depth][node]] = 1.0

        try:
            for ancestor in nx.ancestors(self.g_t, node):
                y_global[self.nodes_idx[ancestor]] = 1.0
                anc_depth = nx.shortest_path_length(self.g_t, "root", ancestor)
                if anc_depth < len(y_local):
                    y_local[anc_depth][self.local_nodes_idx[anc_depth][ancestor]] = 1.0
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
