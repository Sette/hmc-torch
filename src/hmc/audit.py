"""Dataset audit tool — generates ``dataset-card.json`` with statistics.

Usage::

    python -m hmc.audit --dataset_name seq_FUN --dataset_path ./data

Produces a JSON file with per-level node counts, term prevalence,
feature dimensions, missing value statistics, and hierarchy metadata.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from argparse import ArgumentParser

import numpy as np

from hmc.data.base import DatasetBundle, FeatureMetadata, Split
from hmc.data.hierarchy import Hierarchy

logger = logging.getLogger(__name__)


def _build_card(dataset_name: str, bundle: DatasetBundle, _output_dir: str) -> dict:
    """Produce a dataset-card dictionary from a :class:`DatasetBundle`."""
    h: Hierarchy = bundle.hierarchy
    train: Split = bundle.train

    # Per-level statistics
    levels = {}
    for lvl_key in sorted(h.level_sizes.keys()):
        level_nodes = h.levels[lvl_key]
        # Prevalence per term in this level
        prev = {}
        for node in level_nodes:
            gidx = h.node_index[node]
            prev[node] = float(train.labels[:, gidx].mean())
        levels[str(lvl_key)] = {
            "n_nodes": h.level_sizes[lvl_key],
            "nodes": level_nodes,
            "prevalence": prev,
        }

    # Overall statistics
    label_counts = train.labels.sum(axis=0)
    rare_terms = [h.nodes[i] for i, c in enumerate(label_counts) if c <= 5]
    n_samples = train.n_samples

    # Missing values
    missing_count = int(np.isnan(train.features).sum())
    missing_pct = float(missing_count / (train.features.size + 1e-9) * 100)

    card = {
        "dataset_name": dataset_name,
        "hierarchy_type": "dag" if h.is_dag else "tree",
        "n_nodes": h.n_nodes,
        "n_levels": h.max_depth,
        "n_samples": {
            "train": n_samples,
            "valid": bundle.valid.n_samples if bundle.valid else 0,
            "test": bundle.test.n_samples,
        },
        "n_features": bundle.n_features,
        "modality": bundle.metadata.modality.value,
        "missing_values": {
            "count": missing_count,
            "pct": f"{missing_pct:.2f}%",
        },
        "sparsity": float(bundle.metadata.sparsity),
        "levels": levels,
        "rare_terms": rare_terms,
        "roots": h.roots if h.is_dag else ["root"],
    }

    return card


def run_audit(dataset_name: str, dataset_path: str, output_dir: str = "./audit") -> str:
    """Load a dataset and write ``dataset-card.json``.

    Returns the path to the written JSON file.
    """
    from hmc.datasets.dataset_manager import initialize_dataset_experiments  # pylint: disable=import-outside-toplevel

    logger.info("Auditing dataset '%s' from %s", dataset_name, dataset_path)

    # Use legacy manager for now; will migrate to DatasetBundle adapter
    mgr = initialize_dataset_experiments(
        dataset_name,
        device="cpu",
        dataset_path=dataset_path,
        is_global=False,
    )
    train_legacy, valid_legacy, test_legacy = mgr.get_datasets()

    # Convert to Split objects
    from hmc.data.base import Modality  # pylint: disable=import-outside-toplevel
    from hmc.data.hierarchy import DagHierarchy, TreeHierarchy  # pylint: disable=import-outside-toplevel

    hierarchy: Hierarchy
    if mgr.dataset_values.get("is_go", False):
        branches = list(train_legacy.terms)
        hierarchy = DagHierarchy.from_go_terms(branches)
    else:
        branches = [t for t in train_legacy.terms if t != "root"]
        hierarchy = TreeHierarchy.from_fun_cat_terms(
            branches if branches else train_legacy.terms
        )

    has_missing = bool(np.isnan(train_legacy.x).any())
    sparsity = float((train_legacy.x == 0).mean())

    metadata = FeatureMetadata(
        modality=Modality.TABULAR,
        n_features=int(train_legacy.x.shape[1]),
        has_missing=has_missing,
        sparsity=sparsity,
        raw_available=False,
    )

    def _to_split(ds) -> Split:
        from hmc.data.gofun.adapter import _build_local_labels  # pylint: disable=import-outside-toplevel

        local = _build_local_labels(ds.y_local, hierarchy.level_sizes)
        return Split(
            features=ds.x.astype(np.float32),
            labels=ds.y.astype(np.float32),
            local_labels=local,
            metadata=metadata,
        )

    bundle = DatasetBundle(
        train=_to_split(train_legacy),
        valid=_to_split(valid_legacy) if valid_legacy else None,
        test=_to_split(test_legacy),
        hierarchy=hierarchy,
        metadata=metadata,
    )

    card = _build_card(dataset_name, bundle, output_dir)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}-dataset-card.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2, default=str)

    logger.info("Dataset card written to %s", output_path)
    return output_path


def main():
    """CLI entry point for dataset audit."""
    parser = ArgumentParser(description="Audit HMC dataset and write dataset-card.json")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./audit")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )

    try:
        output_path = run_audit(args.dataset_name, args.dataset_path, args.output_dir)
        print(f"Card written to: {output_path}")
    except (OSError, ValueError, RuntimeError):
        logger.exception("Audit failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
