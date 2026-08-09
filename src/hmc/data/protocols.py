"""Protocol definitions for dataset managers.

Every dataset manager — whether built-in or user-provided — must satisfy
:class:`DatasetManagerProtocol`.  This protocol is the contract between
dataset loading code and the training pipelines (global, local, tabular).

.. code-block:: python

    from hmc.data.protocols import DatasetManagerProtocol
    from hmc.data import DatasetRegistry

    class MyManager:
        # implement the protocol ...

    DatasetRegistry.register("my_data", lambda **kw: MyManager(**kw))
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DatasetManagerProtocol(Protocol):
    """Protocol that every HMC dataset manager must satisfy.

    Pipelines (global, local, tabular) access these attributes and methods.
    The protocol is **structural** — any object with these members works,
    no inheritance required.

    Required attributes
    -------------------
    input_dim : int
        Feature dimensionality (e.g. 768 for SPECTER2 embeddings).
    output_dim : int
        Total number of class nodes in the hierarchy (including root).
    levels_size : dict[int, int]
        Mapping from depth level to number of classes at that level.
        Level indices skip root (level 0 is the first meaningful level).
    max_depth : int
        Number of active training levels (excludes root).
    a : np.ndarray
        Full hierarchy adjacency matrix (includes root), shape
        ``(output_dim, output_dim)``.  Entry ``a[i, j] == 1`` if there is
        a directed edge from *i* to *j* (child → parent).
    edge_index : dict[int, np.ndarray]
        Per-level parent→child adjacency matrices.
        ``edge_index[level]`` has shape ``(2, n_edges_at_level)``.
    nodes_idx : dict[str, int]
        Mapping from node name to its global integer index (includes root).
    local_nodes_idx : dict[int, dict[str, int]]
        Per-level mapping from node name to its *local* index within that
        level.  Level keys skip root (0 = first meaningful level).
    to_eval : list[bool]
        Boolean mask over all nodes: ``True`` for evaluable classes,
        ``False`` for root (and possibly other auxiliary nodes).
    hierarchy_map : dict
        Mapping used by constrained models.  Empty dict for datasets that
        don't use it.

    Required methods
    ----------------
    get_datasets()
        Return ``(train, valid, test)`` splits.
    """

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    input_dim: int
    output_dim: int
    levels_size: dict[int, int]
    max_depth: int
    a: np.ndarray
    edge_index: dict[int, np.ndarray]
    nodes_idx: dict[str, int]
    local_nodes_idx: dict[int, dict[str, int]]
    to_eval: list[bool]
    hierarchy_map: dict

    # Optional but commonly accessed
    hierarchy_manager: Any
    hierarchy: Any

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def get_datasets(self) -> tuple[Any, Any | None, Any]:
        """Return ``(train, valid, test)`` splits.

        Each split object must provide:
          - ``.x`` : np.ndarray of shape ``(n_samples, input_dim)``
          - ``.y`` : np.ndarray of shape ``(n_samples, output_dim)``
          - ``.y_local`` : list of np.ndarray, one per level
          - ``.samples`` : object with ``.x`` and ``.y`` attributes
            (for direct tensor conversion in global pipeline)

        ``valid`` may be ``None`` for datasets without a validation split.
        """
        raise NotImplementedError
