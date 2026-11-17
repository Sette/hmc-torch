"""Graph-building utilities for hierarchical label structures."""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch


def build_digraph_from_labels(
    labels: list[str],
    root: str = "root",
    sep: str = ".",
) -> nx.DiGraph:
    """Build a child→parent DiGraph from a flat list of hierarchical labels.

    Each label is a dot-separated path (e.g. ``"cs.AI"``). Intermediate
    nodes are inferred automatically. Top-level labels (single segment)
    become direct children of *root*.

    Duplicate labels and edges are silently ignored — calling the function
    with ``["cs", "cs", "cs.AI"]`` produces the same graph as
    ``["cs", "cs.AI"]``.

    Args:
        labels: Flat list of label strings.  Order does not matter.
        root: Name for the root node (default ``"root"``).
        sep: Path separator (default ``"."``).

    Returns:
        ``nx.DiGraph`` where every edge points **child → parent**.

    Example:
        >>> g = build_digraph_from_labels(["cs", "cs.AI", "stat.ML"])
        >>> list(g.edges())
        [('cs', 'root'), ('cs.AI', 'cs'), ('stat', 'root'), ('stat.ML', 'stat')]
    """
    g = nx.DiGraph()
    g.add_node(root)

    for label in labels:
        if not label or not label.strip():
            continue
        parts = label.split(sep)
        # Single-segment: direct child of root.
        if len(parts) == 1:
            if not g.has_node(label):
                g.add_node(label)
            g.add_edge(label, root)
        else:
            # e.g. "cs.AI" → edges: cs.AI → cs, cs → root
            for i in range(len(parts), 0, -1):
                child = sep.join(parts[:i])
                parent = root if i == 1 else sep.join(parts[: i - 1])
                if not g.has_node(child):
                    g.add_node(child)
                if not g.has_node(parent):
                    g.add_node(parent)
                if not g.has_edge(child, parent):
                    g.add_edge(child, parent)

    return g


def ancestor_closure(adjacency) -> np.ndarray:
    """Reflexive-transitive closure of a child → parent adjacency matrix.

    Args:
        adjacency: ``(C, C)`` matrix with ``adjacency[c, p] != 0`` for every
            edge from child ``c`` to parent ``p`` (the layout of ``mgr.a``).

    Returns:
        np.ndarray: Boolean ``(C, C)``; entry ``[c, a]`` is true iff ``a`` is
        ``c`` itself or one of its ancestors.
    """
    edges = np.asarray(adjacency) != 0
    reach = edges | np.eye(edges.shape[0], dtype=bool)
    # Squaring the reachability relation doubles the path length it covers, so
    # a hierarchy of depth d converges in about log2(d) products.  float32
    # counts are exact here: they never exceed C, far below 2**24.
    while True:
        as_float = reach.astype(np.float32)
        extended = reach | ((as_float @ as_float) > 0)
        if np.array_equal(extended, reach):
            return reach
        reach = extended


def build_r_matrix(adjacency) -> torch.Tensor:
    """Constraint matrix of the training pipeline, shaped ``(1, C, C)``.

    ``R[0, a, c] = 1`` iff ``a`` is ``c`` or one of its ancestors (row =
    ancestor), in float64: the layout :func:`get_constr_out` expects.  The
    transpose (row = descendant) is :attr:`hmc.data.hierarchy.Hierarchy.r_matrix`.

    Args:
        adjacency: Child → parent adjacency, as in :func:`ancestor_closure`.
    """
    closure = ancestor_closure(adjacency)
    return torch.from_numpy(closure.T.astype(np.float64)).unsqueeze(0)
