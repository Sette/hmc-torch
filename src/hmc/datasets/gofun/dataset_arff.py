"""
This module provides functionality to parse ARFF files and construct hierarchical datasets
for Hierarchical Multi-label Classification (HMC) models. It processes both Gene Ontology (GO)
and general hierarchical data formats, extracting features, labels, hierarchical levels,
and adjacency matrices representing the class hierarchy.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from hmc.utils.datasets.paths import to_skip

# Set a logger config
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


def get_depth_by_root(g_t, t, roots):
    """
    Calculates the depth of a node in a directed acyclic
    graph (DAG) by finding the shortest path to any of the specified roots.
    Args:
        g_t (nx.DiGraph): The transpose of the DAG.
        t (str): The node for which to calculate the depth.
        roots (list): A list of root nodes.
    Returns:
        int: The depth of the node.
    """
    for root in roots:
        depth = nx.shortest_path_length(g_t, t, root)
        if depth is not None:
            return depth
    return None


def _impute_column_means(x):
    """Replace every NaN of ``x`` in place with the mean of its column."""
    missing = np.isnan(x)
    if missing.any():
        column_means = np.nanmean(x, axis=0)
        x[missing] = np.take(column_means, np.nonzero(missing)[1])


def _branch_edges(path, is_go):
    """Child → parent edges declared by one entry of the hierarchy attribute.

    A GO entry is a single ``parent.child`` pair.  A FunCat entry is a full
    dotted path such as ``01.01.03``: it declares one edge per prefix, from the
    longer prefix to the shorter one, and a top-level path hangs from ``root``.
    """
    parts = path.split(".")
    if is_go:
        parent, child = parts[0], parts[1]
        return [(child, parent)]
    if len(parts) == 1:
        return [(parts[0], "root")]
    prefixes = [".".join(parts[:k]) for k in range(1, len(parts) + 1)]
    return list(zip(prefixes[1:], prefixes[:-1]))


@dataclass
class _Column:
    """One feature column of the ARFF header: numeric, or nominal (one-hot)."""

    categories: tuple = ()

    @property
    def width(self):
        """Number of values the column expands to."""
        return len(self.categories) or 1

    def encode(self, token):
        """Encode a raw token; ``?`` is NaN (numeric) or all zeros (nominal)."""
        if not self.categories:
            return [np.nan] if token == "?" else [float(token)]
        encoded = [0.0] * len(self.categories)
        if token in self.categories:
            encoded[self.categories.index(token)] = 1.0
        return encoded


@dataclass
class HierarchyData:  # pylint: disable=too-many-instance-attributes
    """Holds all graph and hierarchy structures derived from the @ATTRIBUTE class line."""

    g: Any = field(default_factory=nx.DiGraph)
    g_t: Any = field(default_factory=nx.DiGraph)
    levels: Any = field(default_factory=lambda: defaultdict(list))
    levels_size: dict = field(default_factory=dict)
    nodes_idx: dict = field(default_factory=dict)
    local_nodes_idx: dict = field(default_factory=dict)
    max_depth: int = 0
    terms: list = field(default_factory=list)
    edge_index: dict = field(default_factory=dict)
    a: Any = None
    root_depths: dict = field(default_factory=dict)
    node_ancestors: dict = field(default_factory=dict)


@dataclass
class SampleData:
    """Holds the feature matrix and label arrays parsed from the @DATA section."""

    x: Any = None
    y: Any = None
    y_nodes: list = field(default_factory=list)
    y_local: list = field(default_factory=list)


class HMCDatasetArff:
    """Dataset for Hierarchical Multi-label Classification (HMC) local classifiers.

    Parses an ARFF file that follows the HMC hierarchical format, builds the class
    hierarchy graph, and exposes features, labels, and adjacency structures as
    attributes ready for use in PyTorch or Keras models.

    Attributes:
        arff_file (str): Path to the source ARFF file.
        is_go (bool): Whether the hierarchy follows Gene Ontology (GO) conventions.
        hierarchy (HierarchyData): All graph and hierarchy structures.
        samples (SampleData): Feature matrix and label arrays.
        to_eval (list[bool]): Boolean mask indicating which terms should be evaluated
            (i.e. are not in the ``to_skip`` list).
    """

    def __init__(self, arff_file, is_go):
        """Initialise the dataset by parsing the given ARFF file.

        Args:
            arff_file (str): Path to the ARFF file to parse.
            is_go (bool): Set to ``True`` for Gene Ontology hierarchies, ``False``
                for FUN-style dot-separated hierarchies.
        """
        self.arff_file = arff_file
        self.is_go = is_go
        self.hierarchy = HierarchyData()
        self.samples = SampleData()

        self.parse_arff()

        self.to_eval = [t not in to_skip for t in self.hierarchy.terms]
        _impute_column_means(self.samples.x)

    # ------------------------------------------------------------------
    # Convenience properties that preserve the original flat attribute API
    # ------------------------------------------------------------------

    @property
    def x(self):
        """numpy.ndarray: Feature matrix of shape ``(n_samples, n_features)``."""
        return self.samples.x

    @property
    def y(self):
        """numpy.ndarray: Binary label matrix of shape ``(n_samples, n_nodes)``."""
        return self.samples.y

    @property
    def y_nodes(self):
        """list[list[str]]: Per-sample lists of active leaf label node names."""
        return self.samples.y_nodes

    @property
    def y_local(self):
        """list[list[numpy.ndarray]]: Per-sample, per-level local label vectors."""
        return self.samples.y_local

    @property
    def g(self):
        """nx.DiGraph: Directed hierarchy graph (child → parent edges)."""
        return self.hierarchy.g

    @property
    def g_t(self):
        """nx.DiGraph: Transpose of ``g`` (parent → child edges)."""
        return self.hierarchy.g_t

    @property
    def levels(self):
        """defaultdict[int, list[str]]: Mapping from depth level to node names."""
        return self.hierarchy.levels

    @property
    def levels_size(self):
        """dict[int, int]: Number of unique nodes at each depth level."""
        return self.hierarchy.levels_size

    @property
    def nodes_idx(self):
        """dict[str, int]: Mapping from node name to its global index."""
        return self.hierarchy.nodes_idx

    @property
    def local_nodes_idx(self):
        """dict[int, dict[str, int]]: Per-level mapping from node name to local index."""
        return self.hierarchy.local_nodes_idx

    @property
    def max_depth(self):
        """int: Total number of depth levels in the hierarchy."""
        return self.hierarchy.max_depth

    @property
    def terms(self):
        """list[str]: All node names sorted by depth then lexicographically."""
        return self.hierarchy.terms

    @property
    def edge_index(self):
        """dict[int, numpy.ndarray]: Per-level parent→child adjacency matrices."""
        return self.hierarchy.edge_index

    @property
    def a(self):
        """numpy.ndarray: Full adjacency matrix of the hierarchy graph."""
        return self.hierarchy.a

    # ------------------------------------------------------------------
    # Public build methods
    # ------------------------------------------------------------------

    def train_valid_split(self, valid_fraction=0.2, seed=42):
        """Hold out a deterministic validation slice from this dataset.

        Use this for releases that ship no validation file (e.g. the
        ``*_others`` datasets).  Aliasing the test split as validation would
        put test samples back into training as soon as a pipeline merges
        train and validation, so the slice is carved out of the training file
        instead.  The class hierarchy is derived from the label declarations
        and is therefore shared between the two halves.

        Args:
            valid_fraction (float): Fraction of rows held out for validation.
            seed (int): Seed for the row permutation, for reproducibility.

        Returns:
            tuple[HMCDatasetArff, HMCDatasetArff]: The (train, valid) datasets.
        """
        if not 0.0 < valid_fraction < 1.0:
            raise ValueError(f"valid_fraction must be in (0, 1), got {valid_fraction}")
        n_samples = len(self.samples.x)
        if n_samples < 2:
            raise ValueError(f"Need at least 2 samples to split, got {n_samples}")
        order = np.random.default_rng(seed).permutation(n_samples)
        n_valid = min(max(1, round(n_samples * valid_fraction)), n_samples - 1)
        train_rows = np.sort(order[n_valid:])
        valid_rows = np.sort(order[:n_valid])

        def _subset(rows):
            subset = object.__new__(type(self))
            subset.arff_file = self.arff_file
            subset.is_go = self.is_go
            subset.hierarchy = self.hierarchy
            subset.to_eval = self.to_eval
            subset.samples = SampleData(
                x=self.samples.x[rows],
                y=self.samples.y[rows],
                y_nodes=[self.samples.y_nodes[i] for i in rows],
                y_local=[self.samples.y_local[i] for i in rows],
            )
            return subset

        return _subset(train_rows), _subset(valid_rows)

    def build_hierarchy_graph(self, h):
        """Parse a hierarchical attribute declaration string into a directed graph.

        Iterates over the comma-separated branches in ``h``, builds a
        ``nx.DiGraph`` where each edge points from a child node to its parent,
        and groups node names by depth level.  Populates ``self.hierarchy.g``
        and ``self.hierarchy.levels`` as side effects.

        Args:
            h (str): The hierarchy string extracted from the ``@ATTRIBUTE class
                hierarchical`` line of the ARFF file (e.g.
                ``"root.A.A1,root.A.A2,root.B"``).

        Returns:
            tuple[nx.DiGraph, defaultdict[int, list[str]]]: The constructed
            directed graph and the levels mapping.
        """
        g = nx.DiGraph()
        levels = defaultdict(list)

        for entry in h.split(","):
            path = entry.replace("/", ".")
            if not self.is_go:
                levels[path.count(".")].append(path)
            g.add_edges_from(_branch_edges(path, self.is_go))

        self.hierarchy.g = g
        self.hierarchy.levels = levels
        return g, levels

    def build_node_structures(self):
        """Compute node indices, level sizes, and local node indices from ``self.hierarchy.g``.

        Sorts all graph nodes by depth (shortest path to root) and then
        lexicographically, assigns each a global integer index, reverses the
        graph to obtain ``g_t``, and builds per-level local index mappings.
        Populates ``self.hierarchy.terms``, ``self.hierarchy.nodes_idx``,
        ``self.hierarchy.g_t``, ``self.hierarchy.levels_size``,
        ``self.hierarchy.max_depth``, and ``self.hierarchy.local_nodes_idx``
        as side effects.

        Returns:
            list[str]: Sorted list of all node names (also stored in
            ``self.hierarchy.terms``).
        """
        g = self.hierarchy.g
        levels = self.hierarchy.levels
        g_t = g.reverse()
        root_depths = nx.shortest_path_length(g_t, "root") if self.is_go else {}

        # Columns are ordered by depth, then by name: the order the benchmark's
        # published results (and our saved R-matrices) are indexed with.
        if self.is_go:
            nodes = sorted(g.nodes(), key=lambda n: (root_depths[n], n))
        else:
            nodes = sorted(g.nodes(), key=lambda n: (n.count(".") + 1, n))
        nodes_idx = {node: i for i, node in enumerate(nodes)}

        if self.is_go:
            for label in nodes:
                if label != "root":
                    level = root_depths.get(label) - 1
                    levels[level].append(label)
        else:
            # Non-GO (FunCat): assign every node (including root and
            # intermediate nodes) to its correct depth level.
            for label in nodes:
                level = len(label.split(".")) - 1
                levels[level].append(label)

        levels_size = {key: len(set(value)) for key, value in levels.items()}
        max_depth = len(levels_size)
        # Deduplicate level nodes before indexing so that local_nodes_idx
        # maps each node to a valid 0..n-1 index within its level.
        levels_unique = {key: sorted(set(value)) for key, value in levels.items()}
        local_nodes_idx = {
            idx: {node: i for i, node in enumerate(level_nodes)}
            for idx, level_nodes in levels_unique.items()
        }

        self.hierarchy.terms = nodes
        self.hierarchy.nodes_idx = nodes_idx
        self.hierarchy.g_t = g_t
        self.hierarchy.levels_size = levels_size
        self.hierarchy.max_depth = max_depth
        self.hierarchy.local_nodes_idx = local_nodes_idx
        self.hierarchy.root_depths = root_depths
        self.hierarchy.node_ancestors = {n: list(nx.ancestors(g_t, n)) for n in nodes}
        return nodes

    def build_edge_index(self):
        """Build parent→child adjacency matrices between consecutive hierarchy levels.

        For each pair of adjacent depth levels, creates a binary
        ``numpy.ndarray`` of shape ``(n_parents, n_children)`` where entry
        ``[i, j]`` is ``1.0`` if node ``j`` at the current level is a child of
        node ``i`` at the previous level.  Populates and returns
        ``self.hierarchy.edge_index``.

        Returns:
            dict[int, numpy.ndarray]: Mapping from level index (starting at 1)
            to the corresponding adjacency matrix.
        """
        level_nodes_list = list(self.hierarchy.levels.values())
        edge_index = {}

        for idx, current_level_nodes in enumerate(level_nodes_list):
            if idx == 0:
                continue
            prev_level_nodes = level_nodes_list[idx - 1]
            shape = (len(prev_level_nodes), len(current_level_nodes))
            matrix = np.zeros(shape, dtype=np.float32)

            child_map = {node: i for i, node in enumerate(current_level_nodes)}
            parent_map = {node: i for i, node in enumerate(prev_level_nodes)}

            for c_node in current_level_nodes:
                if self.hierarchy.g.has_node(c_node):
                    for p in self.hierarchy.g.successors(c_node):
                        a = parent_map.get(p)
                        b = child_map.get(c_node)
                        if a is not None and b is not None:
                            matrix[a, b] = 1.0

            edge_index[idx] = matrix

        self.hierarchy.edge_index = edge_index
        return edge_index

    def parse_arff(self):
        """Parse ``self.arff_file`` and populate all dataset attributes.

        Opens the ARFF file, delegates attribute parsing to
        :meth:`_parse_attributes` and data-line parsing to
        :meth:`_parse_data_lines`, then calls :meth:`build_edge_index` and
        assembles the full adjacency matrix ``self.hierarchy.a``.  All results
        are stored directly on ``self.hierarchy`` and ``self.samples``; nothing
        is returned.
        """
        with open(self.arff_file, "r", encoding="utf-8") as f:
            feature_types = self._parse_attributes(f)
            self._parse_data_lines(f, feature_types)

        self.build_edge_index()
        self.hierarchy.a = np.array(
            nx.to_numpy_array(self.hierarchy.g, nodelist=self.hierarchy.terms)
        )

        logger.info(
            "Shape of edges matrix: %s",
            {k: v.shape for k, v in self.hierarchy.edge_index.items()},
        )
        logger.info("Parsed ARFF file: %s", self.arff_file)
        logger.info("Number of matrix: %d", len(self.hierarchy.edge_index))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _column_from_type(f_type):
        """Build the :class:`_Column` for the type of a non-class ``@ATTRIBUTE``.

        Args:
            f_type (str): ``numeric`` (any case) or a brace-enclosed category
                list such as ``{cat1,cat2,cat3}``.

        Returns:
            _Column: The column spec; nominal columns are one-hot encoded.
        """
        f_type = f_type.strip()
        if f_type.lower() == "numeric":
            return _Column()
        categories = [c.strip() for c in f_type.strip("{}").split(",")]
        return _Column(categories=tuple(categories))

    @staticmethod
    def _encode_features(tokens, columns):
        """Encode the feature tokens of one data line into a flat float list."""
        encoded = []
        for column, token in zip(columns, tokens):
            encoded.extend(column.encode(token))
        return encoded

    def _parse_sample_labels(self, lab):
        """Parse the label column for one sample into global and local label vectors.

        Splits the label string on ``"@"``, resolves each leaf label to its
        ancestors in the hierarchy, and sets the corresponding positions in the
        global binary vector ``y_`` and in the per-level local vectors
        ``y_local_``.

        Args:
            lab (str): Raw label string from the data line, with multiple
                labels separated by ``"@"`` and path separators as ``"/"`` or
                ``"."``.

        Returns:
            tuple[numpy.ndarray, list[str], list[numpy.ndarray]]:
                - ``y_`` — binary vector of length ``n_nodes`` with ``1`` for
                  every active node (leaf + ancestors).
                - ``y_nodes`` — list of active leaf node names for this sample.
                - ``y_local_`` — list of per-level binary vectors, one per
                  depth level.
        """
        h = self.hierarchy
        sorted_keys = sorted(h.levels_size.keys())
        y_ = np.zeros(len(h.terms))
        y_nodes = []
        y_local_ = [np.zeros(h.levels_size.get(key)) for key in sorted_keys]

        for t in lab.split("@"):
            y_node = t.replace("/", ".")
            y_nodes.append(y_node)
            ancs = h.node_ancestors.get(y_node)
            if ancs is None:
                ancs = list(nx.ancestors(h.g_t, y_node))
            # A sample belongs to its labels and to all of their ancestors.
            for node in (y_node, *ancs):
                y_[h.nodes_idx[node]] = 1

            if self.is_go:
                depth = h.root_depths.get(y_node) - 1
                y_local_[depth][h.local_nodes_idx[depth].get(y_node)] = 1
                for ancestor in ancs:
                    if ancestor != "root":
                        depth = h.root_depths.get(ancestor) - 1
                        y_local_[depth][h.local_nodes_idx[depth].get(ancestor)] = 1
            else:
                depth = y_node.count(".") + 1
                assert depth is not None
                for index in range(depth, 0, -1):
                    local_terms = y_node.split(".")[:index]
                    local_label = ".".join(local_terms)
                    local_depth = len(local_terms) - 1
                    y_local_[local_depth][
                        h.local_nodes_idx.get(local_depth).get(local_label)
                    ] = 1

        return y_, y_nodes, y_local_

    def _parse_attributes(self, f):
        """Read ``@ATTRIBUTE`` lines from an open file object until ``@DATA``.

        Dispatches the ``@ATTRIBUTE class hierarchical`` line to
        :meth:`build_hierarchy_graph` and :meth:`build_node_structures`, and
        collects a parsing callable for every other ``@ATTRIBUTE`` line via
        :meth:`_column_from_type`.

        Args:
            f (IO[str]): Open file object positioned at the start of the ARFF
                header.  The file cursor is advanced to the first data line
                (just after ``@DATA``) as a side effect.

        Returns:
            list[_Column]: One column spec per feature, in declaration order.
        """
        columns = []
        for line in f:
            if line.startswith("@DATA"):
                break
            if line.startswith("@ATTRIBUTE class"):
                declaration = line.partition("hierarchical")[2].strip()
                self.build_hierarchy_graph(declaration)
                self.build_node_structures()
            elif line.startswith("@ATTRIBUTE"):
                _, _, f_type = line.split(maxsplit=2)
                columns.append(self._column_from_type(f_type))
        return columns

    def _parse_data_lines(self, f, feature_types):
        """Read data lines from an open file object and populate ``self.samples``.

        Each line is split on ``","``; the first ``len(feature_types)`` tokens
        are parsed as features and the next token is parsed as the label
        string.  Results are stacked and stored in ``self.samples.x``,
        ``self.samples.y``, ``self.samples.y_nodes``, and
        ``self.samples.y_local``.

        Args:
            f (IO[str]): Open file object positioned just after the ``@DATA``
                line (i.e. at the first data record).
            feature_types (list[_Column]): Column specs as returned by
                :meth:`_parse_attributes`.
        """
        x = []
        y = []
        y_nodes = []
        y_local = []

        n_features = len(feature_types)
        for line in f:
            record = line.partition("%")[0].strip()
            if not record:
                continue
            tokens = record.split(",")
            x.append(self._encode_features(tokens[:n_features], feature_types))
            y_, sample_y_nodes, y_local_ = self._parse_sample_labels(
                tokens[n_features].strip()
            )
            y.append(y_)
            y_nodes.append(sample_y_nodes)
            y_local.append([np.stack(yy) for yy in y_local_])

        self.samples.x = np.array(x)
        self.samples.y = np.stack(y)
        self.samples.y_nodes = y_nodes
        self.samples.y_local = y_local
