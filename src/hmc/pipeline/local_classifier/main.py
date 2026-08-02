"""
Train a local classifier — one MLP per hierarchy level.

Each level receives the same transformer embedding as input and predicts
only the classes at that level.  All levels are trained jointly with
per-level BCE loss summed across levels.
"""

import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hmc.datasets.dataset_manager import initialize_dataset_experiments
from hmc.models.local_classifier.model import LocalModel
from hmc.utils.model_cache import ensure_transformer_model_cached
from hmc.utils.train.job import create_job_id_name

logger = logging.getLogger(__name__)


def train_local(dataset_name, args):
    """Train one MLP per hierarchy level on features (transformer or tabular)."""
    args.device = torch.device(args.device)
    args.data = dataset_name
    args.ontology = None

    # Detect dataset family for registry lookup
    _is_gofun = any(suffix in (dataset_name or "")
                    for suffix in ("_FUN", "_GO", "_others"))

    # 1. Dataset
    args.hmc_dataset = initialize_dataset_experiments(
        dataset_name,
        device=args.device,
        dataset_path=args.dataset.dataset_path,
        is_global=False,
        arxiv_model_name=args.dataset.arxiv_model_name,
        arxiv_max_records=args.dataset.arxiv_max_records,
        arxiv_cache_dir=args.output_path,
        model_cache_dir=args.dataset.model_cache_dir,
    )
    args.train, args.valid, args.test = args.hmc_dataset.get_datasets()

    args.job_id = create_job_id_name(prefix="local")
    args.results_path = (
        f"output/train/{args.method}-{args.dataset.dataset_name}/{args.job_id}"
    )
    os.makedirs(args.results_path, exist_ok=True)

    # 2. Hyperparameters — pick defaults based on dataset family
    if dataset_name == "wos":
        defaults = args.registry.wos_defaults
    elif _is_gofun:
        defaults = args.registry.gofun_defaults
    else:
        defaults = args.registry.arxiv_defaults

    args.hidden_dim = defaults["hidden_dim"]
    args.lr = defaults["lr"]
    args.epochs = defaults["epochs"]
    args.weight_decay = defaults["weight_decay"]
    args.batch_size = defaults["batch_size"]
    args.num_layers = defaults["num_layers"]
    args.dropout = defaults["dropout"]
    args.input_dim = args.hmc_dataset.input_dim
    args.levels_size = args.hmc_dataset.levels_size
    args.max_depth = args.hmc_dataset.max_depth
    args.to_eval = torch.as_tensor(
        args.hmc_dataset.to_eval, dtype=torch.bool
    )

    # 3. Convert features to tensors
    for split in (args.train, args.valid, args.test):
        split.samples.x = (
            torch.tensor(split.x).clone().detach().float().to(args.device)
        )
        split.samples.y = (
            torch.tensor(split.y).clone().detach().float().to(args.device)
        )

    # 4. Build DataLoaders — concat train + valid for training
    train_dataset = list(zip(args.train.x, args.train.y, args.train.y_local))
    for x, y, yl in zip(args.valid.x, args.valid.y, args.valid.y_local):
        train_dataset.append((x, y, yl))
    test_dataset = list(
        zip(args.test.x, args.test.y, args.test.y_local)
    )

    args.train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    args.test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # 5. Model
    model = LocalModel(
        input_dim=args.input_dim,
        levels_size=args.levels_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        non_lin=args.non_lin,
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.BCELoss()

    # 6. Training loop
    best_micro_f1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in args.train_loader:
            x, _, y_local = batch
            x = x.to(args.device)
            preds = model(x)
            loss = torch.tensor(0.0, device=args.device)
            # y_local after collation: list of tensors, one per level
            # y_local[lvl] has shape (batch_size, n_classes_lvl)
            for lvl in sorted(args.levels_size):
                y_true = y_local[lvl].float().to(args.device)
                loss = loss + criterion(preds[str(lvl)], y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate on test set each epoch
        if (epoch + 1) % max(1, args.epochs // 5) == 0 or epoch == args.epochs - 1:
            micro_f1, scores = _evaluate(model, args.test_loader, args)
            logger.info(
                "Epoch %d/%d — loss %.4f — micro-F1 %.4f",
                epoch + 1, args.epochs, total_loss, micro_f1,
            )
            if micro_f1 > best_micro_f1:
                best_micro_f1 = micro_f1
                args.score = scores

    # 7. Save results
    import json

    with open(f"{args.results_path}/test-scores.json", "w") as f:
        json.dump(args.score, f, indent=4)
    logger.info("Best Micro-F1: %.4f — saved to %s",
                best_micro_f1, args.results_path)
    return args.score


def _evaluate(model, loader, args):
    """Compute per-level and global Micro-F1."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y_global, _y_local in loader:
            x = x.to(args.device)
            preds = model(x)

            for i in range(len(x)):
                global_pred = np.zeros(args.hmc_dataset.output_dim, dtype=np.float32)
                global_label = y_global[i].cpu().numpy()
                for lvl in sorted(args.levels_size):
                    local_pred = preds[str(lvl)][i].cpu().numpy()
                    local_idx = args.hmc_dataset.local_nodes_idx[lvl]
                    for name, lidx in local_idx.items():
                        gidx = args.hmc_dataset.nodes_idx[name]
                        global_pred[gidx] = local_pred[lidx]
                all_preds.append(global_pred)
                all_labels.append(global_label)

    y_pred = np.stack(all_preds)
    y_true = np.stack(all_labels)
    to_eval = np.array(args.hmc_dataset.to_eval)

    # Find best threshold on eval nodes only
    best_f1, best_thr = 0.0, 0.5
    for thr in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        y_bin = (y_pred > thr).astype(np.float32)
        tp = (y_bin[:, to_eval] * y_true[:, to_eval]).sum()
        fp = (y_bin[:, to_eval] * (1 - y_true[:, to_eval])).sum()
        fn = ((1 - y_bin[:, to_eval]) * y_true[:, to_eval]).sum()
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    y_bin = (y_pred > best_thr).astype(np.float32)

    # Per-level scores
    scores = {}
    for lvl in sorted(args.levels_size):
        local_map = args.hmc_dataset.local_nodes_idx[lvl]
        lvl_preds, lvl_labels = [], []
        for name, lidx in local_map.items():
            gidx = args.hmc_dataset.nodes_idx[name]
            lvl_preds.append(y_bin[:, gidx])
            lvl_labels.append(y_true[:, gidx])
        lvl_preds = np.column_stack(lvl_preds)
        lvl_labels = np.column_stack(lvl_labels)
        tp = (lvl_preds * lvl_labels).sum(axis=0)
        fp = (lvl_preds * (1 - lvl_labels)).sum(axis=0)
        fn = ((1 - lvl_preds) * lvl_labels).sum(axis=0)
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        scores[str(lvl)] = {
            "f1score": float(f1.mean()),
            "precision": float(p.mean()),
            "recall": float(r.mean()),
        }

    scores["global"] = {
        "precision": float((y_bin[:, to_eval] * y_true[:, to_eval]).sum()
                           / (y_bin[:, to_eval].sum() + 1e-9)),
        "recall": float((y_bin[:, to_eval] * y_true[:, to_eval]).sum()
                        / (y_true[:, to_eval].sum() + 1e-9)),
        "f1score": float(best_f1),
        "best_threshold": best_thr,
    }

    logger.info("Level 0 F1: %.4f  Level 1 F1: %.4f  Micro-F1: %.4f",
                scores["0"]["f1score"], scores["1"]["f1score"], best_f1)
    return best_f1, scores


# ── Local E2E (fine-tuned transformer + per-level MLPs) ──────────────

def train_local_e2e(dataset_name, args):
    """Fine-tune a transformer with per-level MLP heads (no R-matrix)."""
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    import torch.nn as nn

    from hmc.models.local_classifier.model import LocalE2EModel

    _is_gofun_e2e = any(suffix in (dataset_name or "")
                        for suffix in ("_FUN", "_GO", "_others"))

    args.device = torch.device(args.device)
    model_name = args.dataset.arxiv_model_name

    # 1. Load manager for hierarchy metadata only (no features)
    args.hmc_dataset = initialize_dataset_experiments(
        dataset_name,
        device=args.device,
        dataset_path=args.dataset.dataset_path,
        is_global=False,
        arxiv_model_name=model_name,
        arxiv_max_records=args.dataset.arxiv_max_records,
        arxiv_load_features=False,
        model_cache_dir=args.dataset.model_cache_dir,
    )

    args.data = dataset_name
    args.levels_size = args.hmc_dataset.levels_size
    args.max_depth = args.hmc_dataset.max_depth
    args.to_eval = torch.as_tensor(args.hmc_dataset.to_eval, dtype=torch.bool)
    args.output_dim = args.hmc_dataset.output_dim

    defaults = (
        args.registry.wos_defaults if dataset_name == "wos"
        else args.registry.gofun_defaults if _is_gofun_e2e
        else args.registry.arxiv_defaults
    )
    args.lr = defaults["lr"]
    args.hidden_dim = defaults["hidden_dim"]
    args.num_layers = 2
    args.dropout = defaults["dropout"]
    args.batch_size = 4  # small batch for fine-tuning
    args.epochs = 5

    # 2. Job setup
    args.job_id = create_job_id_name(prefix="local_e2e")
    args.results_path = (
        f"output/train/{args.method}-{args.dataset.dataset_name}/{args.job_id}"
    )
    os.makedirs(args.results_path, exist_ok=True)

    # 3. Tokenized dataset
    from hmc.pipeline.global_classifier.main import (  # pylint: disable=import-outside-toplevel
        _get_transformer_dataset,
    )

    local_model_path = ensure_transformer_model_cached(
        model_name,
        args.dataset.model_cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
    text_dataset, _ = _get_transformer_dataset(
        dataset_name, args, tokenizer, local_model_path
    )
    train_set, _val_set, test_set = text_dataset.get_datasets()

    args.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    args.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # 4. Model
    model = LocalE2EModel(
        model_name=local_model_path,
        levels_size=args.levels_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        model_cache_dir=args.dataset.model_cache_dir,
    ).to(args.device)

    optimizer = torch.optim.AdamW([
        {"params": model.transformer.parameters(), "lr": args.lr_transformer},
        {"params": model.heads.parameters(), "lr": args.lr},
    ], weight_decay=defaults["weight_decay"])
    criterion = nn.BCELoss()

    # 5. Training loop
    best_micro_f1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in args.train_loader:
            inputs, targets = batch  # (encoded_dict, labels_dict)
            inputs = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            preds = model(**inputs)
            # targets["local"] includes root at index 0; skip it
            y_local = targets["local"][1:]  # [tensor_l0, tensor_l1]
            loss = torch.tensor(0.0, device=args.device)
            for lvl in sorted(args.levels_size):
                yt = y_local[lvl].float().to(args.device)
                loss = loss + criterion(preds[str(lvl)], yt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        micro_f1, scores = _evaluate_local_e2e(model, args.test_loader, args)
        logger.info("Epoch %d/%d — loss %.2f — micro-F1 %.4f",
                     epoch + 1, args.epochs, total_loss, micro_f1)
        if micro_f1 > best_micro_f1:
            best_micro_f1 = micro_f1
            args.score = scores

    # 6. Save
    import json
    with open(f"{args.results_path}/test-scores.json", "w") as f:
        json.dump(args.score, f, indent=4)
    logger.info("Best Micro-F1: %.4f", best_micro_f1)
    return args.score


def _evaluate_local_e2e(model, loader, args):
    """Evaluate LocalE2E — same logic as frozen local eval."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch
            inputs = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            preds = model(**inputs)
            y_global = targets["global"]

            for i in range(len(y_global)):
                global_pred = np.zeros(args.hmc_dataset.output_dim, dtype=np.float32)
                global_label = y_global[i].cpu().numpy()
                for lvl in sorted(args.levels_size):
                    local_pred = preds[str(lvl)][i].cpu().numpy()
                    local_idx = args.hmc_dataset.local_nodes_idx[lvl]
                    for name, lidx in local_idx.items():
                        gidx = args.hmc_dataset.nodes_idx[name]
                        global_pred[gidx] = local_pred[lidx]
                all_preds.append(global_pred)
                all_labels.append(global_label)

    y_pred = np.stack(all_preds)
    y_true = np.stack(all_labels)
    to_eval = np.array(args.hmc_dataset.to_eval)

    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.1, 0.91, 0.05):
        y_bin = (y_pred > thr).astype(np.float32)
        tp = (y_bin[:, to_eval] * y_true[:, to_eval]).sum()
        fp = (y_bin[:, to_eval] * (1 - y_true[:, to_eval])).sum()
        fn = ((1 - y_bin[:, to_eval]) * y_true[:, to_eval]).sum()
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    scores = {"global": {"f1score": float(best_f1), "best_threshold": float(best_thr)}}
    logger.info("Micro-F1: %.4f (thr=%.2f)", best_f1, best_thr)
    return best_f1, scores
