"""TabFM local-conditional pipeline.

Implements the ``tabfm_local`` method.  For each node, trains a
conditional binary classifier where the context is built from rows
whose parent label is positive.

Produces the standard four artefacts:
  - ``scores_before_postprocess.npz``
  - ``scores_final.npz``
  - ``metrics.json``
  - ``run-config.json``
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np

from hmc.data.hierarchy import DagHierarchy, TreeHierarchy
from hmc.models.hierarchical.postprocess import reconcile
from hmc.models.tabular.preprocessing import TabularPreprocessor
from hmc.models.tabular.tabfm.adapter import TabFMAdapter, _check_tabfm
from hmc.models.tabular.tabfm.cache import TabFMCache
from hmc.models.tabular.tabfm.conditional import ConditionalNodeClassifier
from hmc.models.tabular.tabfm.context import StratifiedContextSampler

logger = logging.getLogger(__name__)


def _require_tabfm_installed():
    """Check TabFM availability at pipeline entry."""
    if not _check_tabfm():
        raise ImportError(
            "TabFM is required for the tabfm_local method. "
            "Install with: pip install tabfm[pytorch]\n"
            "Note: TabFM v1.0 weights have a non-commercial license."
        )


def train_tabfm_local(
    dataset_name: str,
    args,
    preprocessor: Optional[TabularPreprocessor] = None,
) -> dict:
    """Train a TabFM local-conditional model.

    This is the main entry point for ``--method tabfm_local``.

    Parameters
    ----------
    dataset_name : str
        Dataset identifier (e.g. ``"seq_FUN"``).
    args :
        Parsed :class:`Args` object.
    preprocessor :
        Optional pre-fitted preprocessor.

    Returns:
        Metrics dict.
    """
    _require_tabfm_installed()

    from hmc.datasets.dataset_manager import initialize_dataset_experiments

    logger.info("=== TabFM local-conditional for %s ===", dataset_name)

    # 1. Load data
    mgr = initialize_dataset_experiments(
        dataset_name,
        device="cpu",
        dataset_path=args.dataset.dataset_path,
        is_global=False,
    )
    train, valid, test = mgr.get_datasets()
    X_train = train.x.astype(np.float32)
    y_train = train.y.astype(np.float32)
    X_test = test.x.astype(np.float32)
    y_test = test.y.astype(np.float32)

    X_valid = None
    y_valid = None
    if valid is not None and id(valid) != id(test):
        X_valid = valid.x.astype(np.float32)
        y_valid = valid.y.astype(np.float32)

    eval_mask = np.array(mgr.to_eval, dtype=bool)
    is_go = mgr.dataset_values.get("is_go", False)

    hierarchy: TreeHierarchy | DagHierarchy
    if is_go:
        branches = [t for t in train.terms]
        hierarchy = DagHierarchy.from_go_terms(branches)
    else:
        branches = [t for t in train.terms if t != "root"]
        hierarchy = TreeHierarchy.from_fun_cat_terms(
            branches if branches else train.terms
        )

    # 2. Preprocess
    if preprocessor is None:
        sel = "variance" if X_train.shape[1] > 500 else None
        preprocessor = TabularPreprocessor(
            with_imputation=True, with_scaling=True,
            feature_selector=sel,
        )
    X_train_pp = preprocessor.fit_transform(X_train, y_train)
    X_test_pp = preprocessor.transform(X_test)
    X_valid_pp = preprocessor.transform(X_valid) if X_valid is not None else None

    # 3. TabFM adapter + context sampler
    adapter = TabFMAdapter(
        model_name=getattr(args.training, "tabfm_model", "tabfm-v1"),
        device=args.training.device,
    )

    sampler = StratifiedContextSampler(
        k_pos=50,
        k_neg=50,
        seed=args.training.seed,
        fallback="reduce",
    )

    n_contexts = getattr(args.training, "tabfm_n_contexts", 8)

    # 4. Train conditional classifiers
    cache_dir = getattr(args.training, "llm_cache_dir", "./.tabfm_cache") or "./.tabfm_cache"
    cache = TabFMCache(cache_dir=os.path.join(cache_dir, dataset_name))

    model = ConditionalNodeClassifier(
        adapter=adapter,
        context_sampler=sampler,
        n_contexts=n_contexts,
        calibrate=True,
    )
    model.fit(
        X_train_pp, y_train,
        hierarchy=hierarchy,
        eval_mask=eval_mask,
        X_valid=X_valid_pp,
        y_valid=y_valid,
    )

    # 5. Predict
    scores_raw = model.predict_proba(X_test_pp, y_train)

    # 6. Reconcile
    strategy = "max_path" if is_go else "ancestor_max"
    scores_final = reconcile(scores_raw, hierarchy, strategy=strategy)

    # 7. Metrics
    from hmc.pipeline.tabular.main import _compute_metrics

    metrics = _compute_metrics(y_test, scores_final, eval_mask)

    # 8. Save artefacts
    output_dir = os.path.join(
        getattr(args, "results_path", "./results"),
        dataset_name, "tabfm_local",
    )
    os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(output_dir, "scores_before_postprocess.npz"),
        scores=scores_raw,
    )
    np.savez_compressed(
        os.path.join(output_dir, "scores_final.npz"),
        scores=scores_final,
    )
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    run_config = {
        "method": "tabfm_local",
        "dataset": dataset_name,
        "hierarchy_type": "dag" if is_go else "tree",
        "n_contexts": n_contexts,
        "k_pos": sampler.k_pos,
        "k_neg": sampler.k_neg,
        "seed": sampler.seed,
        "calibrate": model.calibrate,
        "n_features_in": int(X_train.shape[1]),
        "n_features_out": int(X_train_pp.shape[1]),
        "preprocessing": preprocessor.get_params(),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "tabfm_model": adapter.model_name,
        "license_note": "TabFM v1.0 weights are non-commercial",
    }
    with open(os.path.join(output_dir, "run-config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, default=str)

    logger.info("TabFM Micro-F1: %.4f  AUPRC: %.4f",
                metrics["micro_f1"], metrics.get("auprc_micro", 0))
    return metrics
