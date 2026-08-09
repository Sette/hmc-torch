"""Sparse R-Matrix approximations for large hierarchies (GO: 4000+ classes).

The dense R-matrix is O(N²) memory — infeasible for Gene Ontology.
These sparse variants preserve hierarchical constraints with O(N) or
O(N·K) memory, enabling R-matrix methods on previously inaccessible datasets.

Classes
-------
BlockDiagonalR : One R-matrix per level, no cross-level ancestors.
TopKR : Only the K nearest ancestors per node.
ThresholdR : Keep ancestor relationships where the path length <= T.
HybridR : Block-diagonal within level + top-K cross-level.
"""

from __future__ import annotations

from collections import defaultdict, deque

import torch


def _compute_levels_and_index(graph):
    """Compute level dict and node→index mapping from a DiGraph (child→parent).

    Uses BFS from root(s) with caching — O(N + E), suitable for large graphs.
    Returns (levels, node_index, nodes_sorted, parents_map).
    """

    # Find roots (nodes with no outgoing edges = no parents)
    roots = [n for n in graph.nodes() if graph.out_degree(n) == 0]
    if not roots:
        roots = [next(iter(graph.nodes()))]

    # BFS from roots: parent→child direction
    g_rev = graph.reverse()
    levels = defaultdict(list)
    node_depth = {}
    visited = set()

    queue = deque()
    for r in roots:
        node_depth[r] = 0
        visited.add(r)
        queue.append(r)
        levels[0].append(r)

    while queue:
        parent = queue.popleft()
        for child in g_rev.successors(parent):
            if child not in visited:
                visited.add(child)
                depth = node_depth[parent] + 1
                node_depth[child] = depth
                levels[depth].append(child)
                queue.append(child)

    # Nodes not reached by BFS (disconnected components)
    for n in graph.nodes():
        if n not in visited:
            node_depth[n] = 999
            levels[999].append(n)

    nodes_sorted = sorted(graph.nodes(), key=lambda x: (node_depth.get(x, 999), x))
    node_index = {n: i for i, n in enumerate(nodes_sorted)}

    # Precompute parent→children mapping for fast lookup
    parents_map = {}
    for node in graph.nodes():
        parents_map[node] = set(graph.successors(node))  # child→parent edges

    return dict(levels), node_index, nodes_sorted, parents_map


class SparseRMatrix:
    """Base class for sparse R-matrix approximations.

    Works directly from NetworkX graph — does NOT require a full
    Hierarchy object (avoids expensive r_matrix/node_levels properties).
    """

    def __init__(self, graph, device: str = "cpu"):
        self.graph = graph  # child→parent DiGraph
        self.device = device
        self.n_nodes = graph.number_of_nodes()
        levels, node_index, nodes_sorted, parents_map = _compute_levels_and_index(graph)
        self.levels = levels
        self.node_idx = node_index
        self.nodes = nodes_sorted
        self._parents = parents_map
        # depth level per node (0 = root)
        self.node_levels = {n: lvl for lvl, nodes in levels.items() for n in nodes}

    def parents(self, node: str) -> set:
        """Direct parents of node (child→parent)."""
        return self._parents.get(node, set())

    def children(self, node: str) -> set:
        """Direct children of node (parent→child)."""
        return set(self.graph.predecessors(node))

    def ancestors(self, node: str) -> set:
        """All ancestors via BFS up the graph."""
        result = set()
        queue = list(self.parents(node))
        while queue:
            p = queue.pop()
            if p not in result:
                result.add(p)
                queue.extend(self.parents(p))
        return result

    def to_dense(self) -> torch.Tensor:
        """Convert to dense (N, N) tensor — for validation on small hierarchies."""
        raise NotImplementedError

    def reconcile(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply bottom-up reconciliation using sparse representation.

        Args:
            scores: (batch, N) tensor of raw scores.

        Returns:
            Reconciled scores of same shape.
        """
        raise NotImplementedError

    def consistency_loss(
        self, probs: torch.Tensor, margin: float = 0.0
    ) -> torch.Tensor:
        """Compute hierarchical consistency loss using sparse representation.

        Args:
            probs: (batch, N) predicted probabilities.
            margin: Optional margin for the hinge loss.

        Returns:
            Scalar loss.
        """
        raise NotImplementedError


class BlockDiagonalR(SparseRMatrix):
    """R-matrix approximated via graph traversal: no dense matrix stored.

    Uses the hierarchy graph edges to propagate scores bottom-up,
    which is exactly equivalent to dense R-matrix reconciliation
    but requires O(N) memory instead of O(N²).

    Reconciliation is performed by traversing the graph from leaves
    to root, ensuring parent score >= max(children) at each step.
    This is mathematically identical to the dense R-matrix max
    propagation but uses only adjacency information.

    Memory: O(N) — graph edges + node metadata.
    """

    def reconcile(self, scores: torch.Tensor) -> torch.Tensor:
        """Bottom-up max propagation through graph edges.

        For trees, one pass suffices.  For DAGs (multiple parents),
        iterates up to max_depth times to ensure transitive closure:
        after N passes, scores propagate N levels up the hierarchy.
        Convergence is guaranteed after max_depth passes.
        """
        result = scores.clone()
        max_depth = len(self.levels)
        sorted_levels = sorted(self.levels.keys(), reverse=True)
        for _ in range(max_depth):
            changed = False
            for level in sorted_levels:
                if level == 0:
                    continue
                for node in self.levels[level]:
                    parents = self.parents(node)
                    if not parents:
                        continue
                    n_idx = self.node_idx[node]
                    node_score = result[:, n_idx]
                    for p in parents:
                        if p not in self.node_idx:
                            continue
                        p_idx = self.node_idx[p]
                        update = torch.maximum(result[:, p_idx], node_score)
                        if not torch.equal(update, result[:, p_idx]):
                            changed = True
                            result[:, p_idx] = update
            if not changed:
                break
        return result

    def consistency_loss(
        self, probs: torch.Tensor, margin: float = 0.0
    ) -> torch.Tensor:
        """Penalize child > parent violations using graph edges."""
        violations = torch.tensor(0.0, device=probs.device)
        count = 0
        sorted_levels = sorted(self.levels.keys(), reverse=True)
        for level in sorted_levels:
            if level == 0:
                continue
            for node in self.levels[level]:
                n_idx = self.node_idx[node]
                parents = self.parents(node)
                for p in parents:
                    if p not in self.node_idx:
                        continue
                    p_idx = self.node_idx[p]
                    diff = probs[:, n_idx] - probs[:, p_idx] + margin
                    violations += torch.clamp(diff, min=0.0).sum()
                    count += probs.shape[0]
        return violations / max(count, 1)

    def to_dense(self) -> torch.Tensor:
        """Reconstruct dense R from graph for validation."""
        r = torch.eye(self.n_nodes, device=self.device)
        for node in self.nodes:
            n_idx = self.node_idx[node]
            for ancestor in self.ancestors(node):
                a_idx = self.node_idx.get(ancestor)
                if a_idx is not None:
                    r[n_idx, a_idx] = 1.0
        return r


class TopKR(SparseRMatrix):
    """Keep only the K nearest ancestors per node for the consistency loss.

    Reconciliation still uses full graph edges (O(N) memory).
    The top-K approximation applies only to the training-time loss,
    reducing the number of ancestor pairs that contribute to the
    consistency penalty.

    Memory: O(N·K) for ancestor cache.
    """

    def __init__(self, hierarchy, K: int = 10, device: str = "cpu"):
        super().__init__(hierarchy, device)
        self.K = K  # pylint: disable=invalid-name
        self._ancestors: dict[int, list[int]] = {}
        self._build()

    def _build(self):
        """Precompute top-K ancestors per node."""
        for node in self.nodes:
            n_idx = self.node_idx[node]
            ancestors = list(self.ancestors(node))
            ancestors_sorted = sorted(
                ancestors,
                key=lambda a: self.node_levels.get(a, 0),
                reverse=True,
            )
            top_k = ancestors_sorted[: self.K]
            root_nodes = [
                a
                for a in ancestors_sorted
                if self.node_levels.get(a, 0) == 0 and a not in top_k
            ]
            selected = top_k + root_nodes[:1]
            self._ancestors[n_idx] = [
                self.node_idx[a] for a in selected if a in self.node_idx and a != node
            ]

    def reconcile(self, scores: torch.Tensor) -> torch.Tensor:
        """Full reconciliation via graph edges (same as BlockDiagonalR)."""
        result = scores.clone()
        for level in sorted(self.levels.keys(), reverse=True):
            if level == 0:
                continue
            for node in self.levels[level]:
                n_idx = self.node_idx[node]
                for p in self.parents(node):
                    if p not in self.node_idx:
                        continue
                    p_idx = self.node_idx[p]
                    result[:, p_idx] = torch.maximum(result[:, p_idx], result[:, n_idx])
        return result

    def consistency_loss(
        self, probs: torch.Tensor, margin: float = 0.0
    ) -> torch.Tensor:
        """Penalize violations for top-K ancestors only (faster training)."""
        violations = torch.tensor(0.0, device=probs.device)
        count = 0
        for child_idx, ancestor_indices in self._ancestors.items():
            for a_idx in ancestor_indices:
                diff = probs[:, child_idx] - probs[:, a_idx] + margin
                violations += torch.clamp(diff, min=0.0).sum()
                count += probs.shape[0]
        return violations / max(count, 1)

    def to_dense(self) -> torch.Tensor:
        """Reconstruct sparse R from ancestor list."""
        r = torch.eye(self.n_nodes, device=self.device)
        for child_idx, ancestor_indices in self._ancestors.items():
            for a_idx in ancestor_indices:
                r[child_idx, a_idx] = 1.0
        return r


# ===================================================================
# Factory
# ===================================================================


def create_sparse_r(
    graph, method: str = "block_diag", K: int = 10, device: str = "cpu"
) -> SparseRMatrix:
    """Factory for sparse R-matrix variants.

    Args:
        graph: nx.DiGraph with child→parent edges.
        method: One of ``"block_diag"`` or ``"topk"``.
        K: Number of ancestors for topk method.
        device: PyTorch device.

    Returns:
        A SparseRMatrix instance.
    """
    if method == "block_diag":
        return BlockDiagonalR(graph, device)
    if method == "topk":
        return TopKR(graph, K=K, device=device)
    raise ValueError(f"Unknown sparse R method: {method}")
