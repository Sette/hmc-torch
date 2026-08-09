"""Graph-building utilities for hierarchical label structures."""

from __future__ import annotations

import networkx as nx


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
