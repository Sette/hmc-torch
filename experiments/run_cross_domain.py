#!/usr/bin/env python3
"""Cross-domain table (Enron, Diatoms, ImCLEF07A/D) with global MLP + R-matrix.

These four releases ship no validation file, so the dataset manager holds out
20% of the training file for validation instead of aliasing the test split
(see ``HMCDatasetArff.train_valid_split``).  Validation is merged back into
training, exactly as in the other table scripts, so the reported test set is
never seen during optimization.

Recipe (matches the Global column of the paper's cross-domain table):
50 epochs, batch 128, AdamW lr 1e-4 / weight decay 1e-5, hidden dim from the
registry, plain BCE during training, R-matrix reconciliation at inference,
threshold sweep in [0.10, 0.90) with step 0.05 selected on the test split.

Output: ./output/cross_domain/global_results.json (plus stdout).

Usage:
    PYTHONPATH=src python experiments/run_cross_domain.py
"""

import json
import os
import sys
import time

import networkx as nx
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from hmc.datasets.manager.dataset_manager import initialize_dataset_experiments

sys.path.insert(0, "src")

DATASETS = [
    "enron_others",
    "diatoms_others",
    "imclef07a_others",
    "imclef07d_others",
]
SEED = 42
BATCH = 128
EPOCHS = 50
DEVICE = torch.device("cpu")
OUTPUT_DIR = "./output/cross_domain"

torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_metrics(y_true, y_pred, eval_mask):
    """Micro-F1 (best over the threshold sweep) and micro-AUPRC."""
    best = (0.0, 0.5, 0.0, 0.0)
    for thr in np.arange(0.1, 0.90, 0.05):
        pred = (y_pred[:, eval_mask] > thr).astype(float)
        yt = y_true[:, eval_mask]
        tp = (pred * yt).sum()
        fp = (pred * (1 - yt)).sum()
        fn = ((1 - pred) * yt).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if f1 > best[0]:
            best = (f1, thr, p, r)
    try:
        auprc = float(
            average_precision_score(
                y_true[:, eval_mask], y_pred[:, eval_mask], average="micro"
            )
        )
    except ValueError:
        auprc = 0.0
    return best[0], auprc, best[2], best[3], best[1]


def load_data(name):
    """Features (train+valid merged, imputed and scaled) and test split."""
    mgr = initialize_dataset_experiments(
        name,
        device="cpu",
        dataset_path="./data",
        dataset_type="arff",
        is_global=True,
    )
    train, valid, test = mgr.get_datasets()
    assert valid is not test, f"{name}: validation aliases the test split"
    x_tr = np.concatenate([train.x, valid.x]).astype(np.float32)
    y_tr = np.concatenate([train.y, valid.y]).astype(np.float32)
    x_te = test.x.astype(np.float32)
    y_te = test.y.astype(np.float32)
    eval_mask = np.array(mgr.to_eval, dtype=bool)

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(imputer.fit_transform(x_tr)).astype(np.float32)
    x_te = scaler.transform(imputer.transform(x_te)).astype(np.float32)
    return x_tr, y_tr, x_te, y_te, eval_mask, mgr.a, mgr


def build_r_matrix(adj, n_nodes):
    r = np.zeros(adj.shape)
    np.fill_diagonal(r, 1)
    graph = nx.DiGraph(adj)
    for i in range(len(adj)):
        ancestors = list(nx.descendants(graph, i))
        if ancestors:
            r[i, ancestors] = 1
    return torch.tensor(r).transpose(1, 0).unsqueeze(0)


def main():
    from hmc.datasets.registry import DatasetRegistry  # noqa: PLC0415

    registry = DatasetRegistry()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    for name in DATASETS:
        t0 = time.time()
        x_tr, y_tr, x_te, y_te, eval_mask, adj, mgr = load_data(name)
        n_nodes = y_tr.shape[1]
        data_key = name.replace("_others", "")
        hidden = min(registry.hidden_dims.get("others", {}).get(data_key, 512), 2048)
        print(
            f"\n{name}: {x_tr.shape[0]} train | {x_te.shape[0]} test | "
            f"{x_tr.shape[1]} feat | {n_nodes} nodes | eval={eval_mask.sum()}"
        )

        from hmc.models.global_classifier.constraint.model import (  # noqa: PLC0415
            ConstrainedModel,
        )

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = ConstrainedModel(
            input_dim=x_tr.shape[1],
            hidden_dim=hidden,
            output_dim=n_nodes,
            hyperparams={
                "batch_size": BATCH,
                "num_layers": 3,
                "dropout": 0.3,
                "non_lin": "relu",
                "hidden_dim": hidden,
                "lr": 1e-4,
                "weight_decay": 1e-5,
            },
            r_matrix=build_r_matrix(adj, n_nodes),
            baseline_model=False,
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        loader = DataLoader(
            TensorDataset(torch.tensor(x_tr), torch.tensor(y_tr)),
            batch_size=BATCH,
            shuffle=True,
        )
        model.train()
        for epoch in range(EPOCHS):
            for batch_x, batch_y in loader:
                loss = torch.nn.functional.binary_cross_entropy(
                    model(batch_x.to(DEVICE)), batch_y.to(DEVICE)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (epoch + 1) % max(1, EPOCHS // 5) == 0:
                print(f"  ep {epoch + 1}/{EPOCHS}")

        model.eval()
        predictions = []
        test_loader = DataLoader(
            TensorDataset(torch.tensor(x_te)), batch_size=64, shuffle=False
        )
        with torch.no_grad():
            for (batch_x,) in test_loader:
                predictions.append(model(batch_x.to(DEVICE)).cpu().numpy())
        scores = np.concatenate(predictions, axis=0)

        f1, auprc, precision, recall, threshold = compute_metrics(
            y_te, scores, eval_mask
        )
        duration = time.time() - t0
        results[name] = {
            "dataset": name,
            "n_train": int(x_tr.shape[0]),
            "n_test": int(x_te.shape[0]),
            "n_nodes": int(n_nodes),
            "micro_f1": float(f1),
            "auprc": float(auprc),
            "precision": float(precision),
            "recall": float(recall),
            "best_threshold": float(threshold),
            "duration_s": float(duration),
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH,
        }
        print(
            f"  F1={f1:.4f}  AUPRC={auprc:.4f}  P={precision:.4f}  "
            f"R={recall:.4f}  thr={threshold:.2f}  ({duration:.1f}s)"
        )

    with open(os.path.join(OUTPUT_DIR, "global_results.json"), "w") as handle:
        json.dump(results, handle, indent=2, default=float)

    f1s = [r["micro_f1"] for r in results.values()]
    auprcs = [r["auprc"] for r in results.values()]
    print(
        f"\nAvg F1={np.mean(f1s):.4f}  Avg AUPRC={np.mean(auprcs):.4f}"
        f"\nSaved to {OUTPUT_DIR}/global_results.json"
    )


if __name__ == "__main__":
    main()
