"""Bridge from legacy ARFF parser to the new :class:`DatasetBundle` contract.

This adapter wraps :class:`hmc.datasets.gofun.dataset_arff.HMCDatasetArff` and
produces :class:`hmc.data.base.DatasetBundle` instances.  It exists so that
consumers can adopt the new contract immediately without waiting for the full
parser migration to ``hmc.data.gofun``.
"""

from __future__ import annotations

import logging

import numpy as np

from hmc.data.base import DatasetBundle, FeatureMetadata, Modality, Split
from hmc.data.hierarchy import DagHierarchy, Hierarchy, TreeHierarchy

logger = logging.getLogger(__name__)


def _build_hierarchy(is_go: bool, terms: list[str], graph) -> Hierarchy:
    """Build the appropriate :class:`Hierarchy` subclass from parsed data."""
    if is_go:
        # Extract branches from edges
        branches = []
        for parent, child in graph.edges():
            branches.append(f"{parent}/{child}")
        return DagHierarchy.from_go_terms(branches if branches else terms)
    # FunCat: branches are the dotted paths in terms
    fun_branches = [t for t in terms if t != "root"]
    if not fun_branches:
        fun_branches = terms
    return TreeHierarchy.from_fun_cat_terms(fun_branches)


def _build_local_labels(
    y_local: list[list[np.ndarray]], level_sizes: dict[int, int]
) -> list[np.ndarray]:
    """Convert legacy ``y_local`` to a list of per-level arrays.

    The legacy format is ``list[list[np.ndarray]]`` where each sample has
    a list of per-level 1-d arrays.  We stack them into 2-d arrays per level.
    """
    if not y_local:
        return []

    sorted_levels = sorted(level_sizes.keys())
    result = []
    for lvl_idx, lvl in enumerate(sorted_levels):
        # Each sample's y_local is a list; grab position lvl_idx
        arrs = []
        for sample_y in y_local:
            if lvl_idx < len(sample_y):
                arrs.append(np.asarray(sample_y[lvl_idx]).flatten())
            else:
                arrs.append(np.zeros(level_sizes[lvl], dtype=np.float32))
        result.append(np.stack(arrs))
    return result


def arff_to_bundle(
    train_arff_path: str,
    test_arff_path: str,
    valid_arff_path: str | None = None,
    is_go: bool = False,
) -> DatasetBundle:
    """Load ARFF files and return a :class:`DatasetBundle`.

    Args:
        train_arff_path: Path to the training ARFF file.
        test_arff_path: Path to the test ARFF file.
        valid_arff_path: Optional path to validation ARFF file.
        is_go: ``True`` for Gene Ontology datasets.

    Returns:
        A fully populated :class:`DatasetBundle`.
    """
    from hmc.datasets.gofun.dataset_arff import (  # pylint: disable=import-outside-toplevel
        HMCDatasetArff,
    )

    logger.info("Loading train ARFF: %s", train_arff_path)
    train_ds = HMCDatasetArff(train_arff_path, is_go=is_go)
    logger.info("Loading test ARFF: %s", test_arff_path)
    test_ds = HMCDatasetArff(test_arff_path, is_go=is_go)

    valid_ds = None
    if valid_arff_path and valid_arff_path != test_arff_path:
        logger.info("Loading valid ARFF: %s", valid_arff_path)
        valid_ds = HMCDatasetArff(valid_arff_path, is_go=is_go)

    # Build hierarchy
    hierarchy = _build_hierarchy(is_go, train_ds.terms, train_ds.g)

    # Build metadata
    n_features = int(train_ds.x.shape[1])
    has_missing = bool(np.isnan(train_ds.x).any())
    sparsity = float((train_ds.x == 0).mean())

    metadata = FeatureMetadata(
        modality=Modality.TABULAR,
        n_features=n_features,
        has_missing=has_missing,
        sparsity=sparsity,
        raw_available=False,
    )

    # Build splits
    def _make_split(ds: HMCDatasetArff) -> Split:
        local = _build_local_labels(ds.y_local, hierarchy.level_sizes)
        return Split(
            features=ds.x.astype(np.float32),
            labels=ds.y.astype(np.float32),
            local_labels=local,
            metadata=metadata,
        )

    train_split = _make_split(train_ds)
    test_split = _make_split(test_ds)
    valid_split = _make_split(valid_ds) if valid_ds else None

    return DatasetBundle(
        train=train_split,
        valid=valid_split,
        test=test_split,
        hierarchy=hierarchy,
        metadata=metadata,
    )
