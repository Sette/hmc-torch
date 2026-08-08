"""Tabular baseline training pipelines: GBDT One-vs-Rest and Residual MLP.

Each pipeline produces:
  - ``scores_before_postprocess.npz``
  - ``scores_final.npz``
  - ``metrics.json``
  - ``run-config.json``
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hmc.data.hierarchy import Hierarchy
from hmc.models.hierarchical.postprocess import reconcile
from hmc.models.tabular.preprocessing import TabularPreprocessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared evaluation
# ---------------------------------------------------------------------------


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eval_mask: np.ndarray,
    thresholds: tuple = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7),
) -> dict:
    """Compute Micro-F1, per-level F1, and best threshold.

    Returns a dict suitable for ``metrics.json``.
    """
    best_f1, best_thr = 0.0, 0.5
    best_bin = None

    for thr in thresholds:
        y_bin = (y_pred >= thr).astype(np.float32)
        tp = (y_bin[:, eval_mask] * y_true[:, eval_mask]).sum()
        fp = (y_bin[:, eval_mask] * (1 - y_true[:, eval_mask])).sum()
        fn = ((1 - y_bin[:, eval_mask]) * y_true[:, eval_mask]).sum()
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
            best_bin = y_bin

    y_bin = best_bin if best_bin is not None else (y_pred >= 0.5).astype(np.float32)

    tp_g = (y_bin[:, eval_mask] * y_true[:, eval_mask]).sum()
    fp_g = (y_bin[:, eval_mask] * (1 - y_true[:, eval_mask])).sum()
    fn_g = ((1 - y_bin[:, eval_mask]) * y_true[:, eval_mask]).sum()

    micro_p = tp_g / (tp_g + fp_g + 1e-9)
    micro_r = tp_g / (tp_g + fn_g + 1e-9)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-9)

    # AUPRC via sklearn if available
    auprc = 0.0
    try:
        from sklearn.metrics import average_precision_score  # pylint: disable=import-outside-toplevel

        auprc = float(
            average_precision_score(
                y_true[:, eval_mask], y_pred[:, eval_mask], average="micro"
            )
        )
    except ImportError:
        pass

    return {
        "micro_f1": float(micro_f1),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "best_threshold": float(best_thr),
        "auprc_micro": auprc,
    }


def _save_artefacts(
    output_dir: str,
    scores_raw: np.ndarray,
    scores_final: np.ndarray,
    metrics: dict,
    run_config: dict,
):
    """Persist scores, metrics, and config to *output_dir*."""
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
    with open(os.path.join(output_dir, "run-config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, default=str)

    logger.info("Artefacts saved to %s", output_dir)


# ---------------------------------------------------------------------------
# GBDT pipeline
# ---------------------------------------------------------------------------


def train_gbdt(
    dataset_name: str,
    args,
    preprocessor: TabularPreprocessor | None = None,
) -> dict:
    """Train a GBDT One-vs-Rest baseline.

    Returns the metrics dict.
    """
    from hmc.datasets.dataset_manager import initialize_dataset_experiments  # pylint: disable=import-outside-toplevel
    from hmc.models.tabular.gbdt_ovr import GBDTOvRClassifier  # pylint: disable=import-outside-toplevel

    logger.info("=== GBDT baseline for %s ===", dataset_name)

    # 1. Load data
    mgr = initialize_dataset_experiments(
        dataset_name,
        device="cpu",
        dataset_path=args.dataset.dataset_path,
        is_global=False,
    )
    train, valid, test = mgr.get_datasets()

    X_train, y_train = train.x.astype(np.float32), train.y.astype(np.float32)
    X_test, y_test = test.x.astype(np.float32), test.y.astype(np.float32)

    # Combine train + valid if valid is available and different from test
    if valid is not None and id(valid) != id(test):
        X_train = np.concatenate([X_train, valid.x.astype(np.float32)], axis=0)
        y_train = np.concatenate([y_train, valid.y.astype(np.float32)], axis=0)

    eval_mask = np.array(mgr.to_eval, dtype=bool)
    hierarchy = _build_hierarchy_from_manager(mgr)

    # 2. Preprocess
    if preprocessor is None:
        preprocessor = TabularPreprocessor(
            with_imputation=True,
            with_scaling=True,
            feature_selector="variance" if X_train.shape[1] > 500 else None,
        )
    X_train_pp = preprocessor.fit_transform(X_train, y_train)
    X_test_pp = preprocessor.transform(X_test)

    # 3. Train GBDT
    model = GBDTOvRClassifier(
        backend=getattr(args.training, "tabular_gbdt_backend", "histgb"),
        verbose=0,
    )
    model.fit(X_train_pp, y_train, eval_mask=eval_mask)

    # 4. Predict
    scores_raw = model.predict_proba(X_test_pp)

    # 5. Reconcile
    scores_final = reconcile(scores_raw, hierarchy, strategy="ancestor_max")

    # 6. Metrics
    metrics = _compute_metrics(y_test, scores_final, eval_mask)

    # 7. Save
    output_dir = _resolve_output_dir(args, dataset_name, "tabular_gbdt")
    run_config = {
        "method": "tabular_gbdt",
        "dataset": dataset_name,
        "backend": model.backend,
        "n_nodes_trained": model.n_estimators_trained,
        "preprocessing": preprocessor.get_params(),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features_in": int(X_train.shape[1]),
        "n_features_out": int(X_train_pp.shape[1]),
    }
    _save_artefacts(output_dir, scores_raw, scores_final, metrics, run_config)

    logger.info(
        "GBDT Micro-F1: %.4f  AUPRC: %.4f",
        metrics["micro_f1"],
        metrics.get("auprc_micro", 0),
    )
    return metrics


# ---------------------------------------------------------------------------
# Residual MLP pipeline
# ---------------------------------------------------------------------------


def train_tabular_mlp(
    dataset_name: str,
    args,
    preprocessor: TabularPreprocessor | None = None,
) -> dict:
    """Train a Residual MLP baseline.

    Returns the metrics dict.
    """
    logger.info("=== Tabular MLP baseline for %s ===", dataset_name)

    device = torch.device(args.training.device)

    # 1. Load and preprocess data
    data = _load_and_preprocess_tabular_data(args, dataset_name, preprocessor)
    defaults = args.registry.gofun_defaults

    # 2. Setup model, data, optimizer
    model, train_loader, test_loader, criterion, optimizer = _setup_mlp_training(
        data["n_features"],
        data["n_nodes"],
        data["X_train_pp"],
        data["y_train"],
        data["X_test_pp"],
        data["y_test"],
        defaults,
        device,
    )

    # 4. Train and predict
    _train_mlp_loop(
        model, train_loader, criterion, optimizer, device, args.training.epochs
    )
    scores_raw = _predict_mlp(model, test_loader, device)
    scores_final = reconcile(scores_raw, data["hierarchy"], strategy="ancestor_max")

    # 5. Metrics and save
    metrics = _compute_metrics(data["y_test"], scores_final, data["eval_mask"])
    _save_tabular_mlp_results(
        args,
        dataset_name,
        defaults,
        data,
        args.training.epochs,
        data["preprocessor"],
        scores_raw,
        scores_final,
        metrics,
    )
    return metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_hierarchy_from_manager(mgr) -> Hierarchy:
    """Instantiate the appropriate Hierarchy from a legacy manager."""
    from hmc.data.hierarchy import (  # pylint: disable=import-outside-toplevel
        DagHierarchy,
        TreeHierarchy,
    )

    train = mgr.get_datasets()[0]
    is_go = mgr.dataset_values.get("is_go", False)

    if is_go:
        branches = list(train.terms)
        return DagHierarchy.from_go_terms(branches)

    branches = [t for t in train.terms if t != "root"]
    return TreeHierarchy.from_fun_cat_terms(branches if branches else list(train.terms))


def _load_and_preprocess_tabular_data(
    args,
    dataset_name: str,
    preprocessor: TabularPreprocessor | None,
) -> dict:
    """Load, merge, and preprocess tabular data for MLP training."""
    from hmc.datasets.dataset_manager import (  # pylint: disable=import-outside-toplevel
        initialize_dataset_experiments,
    )

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

    if valid is not None and id(valid) != id(test):
        X_train = np.concatenate([X_train, valid.x.astype(np.float32)], axis=0)
        y_train = np.concatenate([y_train, valid.y.astype(np.float32)], axis=0)

    eval_mask = np.array(mgr.to_eval, dtype=bool)
    hierarchy = _build_hierarchy_from_manager(mgr)

    if preprocessor is None:
        preprocessor = TabularPreprocessor(
            with_imputation=True,
            with_scaling=True,
            feature_selector="variance" if X_train.shape[1] > 500 else None,
        )
    X_train_pp = preprocessor.fit_transform(X_train, y_train)
    X_test_pp = preprocessor.transform(X_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_pp": X_train_pp,
        "X_test_pp": X_test_pp,
        "eval_mask": eval_mask,
        "hierarchy": hierarchy,
        "preprocessor": preprocessor,
        "n_features": X_train_pp.shape[1],
        "n_nodes": y_train.shape[1],
    }


def _train_mlp_loop(
    model: nn.Module,
    train_loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    epochs: int,
) -> None:
    """Train an MLP model for the given number of epochs."""
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            logger.info("Epoch %d/%d — loss %.4f", epoch + 1, epochs, total_loss)


def _predict_mlp(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Run inference and return raw scores as numpy array."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            all_preds.append(model(batch_x).cpu().numpy())
    return np.concatenate(all_preds, axis=0)


def _resolve_output_dir(args, dataset_name: str, method: str) -> str:
    """Determine artefact output directory."""
    base = getattr(args, "results_path", "./results")
    return os.path.join(base, dataset_name, method)


def _create_mlp_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Create train and test DataLoaders from numpy arrays."""
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def _setup_mlp_training(
    n_features: int,
    n_nodes: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    defaults: dict,
    device: torch.device,
):
    """Create MLP model, dataloaders, loss, and optimizer."""
    from hmc.models.hierarchical.losses import (  # pylint: disable=import-outside-toplevel
        WeightedBCELoss,
    )
    from hmc.models.tabular.mlp import (  # pylint: disable=import-outside-toplevel
        TabularMLPModel,
    )

    model = TabularMLPModel(
        input_dim=n_features,
        n_nodes=n_nodes,
        hidden_dim=defaults["hidden_dim"],
        n_blocks=3,
        head_layers=defaults["num_layers"],
        dropout=defaults["dropout"],
    ).to(device)

    train_loader, test_loader = _create_mlp_dataloaders(
        X_train, y_train, X_test, y_test, defaults["batch_size"]
    )

    pos_weight = WeightedBCELoss.compute_pos_weight(
        torch.tensor(y_train, dtype=torch.float32)
    ).to(device)
    criterion = WeightedBCELoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=defaults["lr"],
        weight_decay=defaults["weight_decay"],
    )
    return model, train_loader, test_loader, criterion, optimizer


def _save_tabular_mlp_results(
    args,
    dataset_name: str,
    defaults: dict,
    data: dict,
    epochs: int,
    preprocessor,
    scores_raw: np.ndarray,
    scores_final: np.ndarray,
    metrics: dict,
) -> None:
    """Save tabular MLP artefacts and log results."""
    output_dir = _resolve_output_dir(args, dataset_name, "tabular_mlp")
    run_config = {
        "method": "tabular_mlp",
        "dataset": dataset_name,
        "hidden_dim": defaults["hidden_dim"],
        "n_blocks": 3,
        "head_layers": defaults["num_layers"],
        "dropout": defaults["dropout"],
        "lr": defaults["lr"],
        "epochs": epochs,
        "batch_size": defaults["batch_size"],
        "preprocessing": preprocessor.get_params(),
        "n_train": int(data["X_train"].shape[0]),
        "n_test": int(data["X_test"].shape[0]),
        "n_features_in": int(data["X_train"].shape[1]),
        "n_features_out": data["n_features"],
    }
    _save_artefacts(output_dir, scores_raw, scores_final, metrics, run_config)
    logger.info(
        "Tabular MLP Micro-F1: %.4f  AUPRC: %.4f",
        metrics["micro_f1"],
        metrics.get("auprc_micro", 0),
    )
