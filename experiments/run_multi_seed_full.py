#!/usr/bin/env python3
"""Multi-seed stability: 5 seeds on 6 primary datasets.

Runs the ``global`` method (MLP + R-matrix) with 5 different random seeds
to establish robust mean ± std for SOTA claims.

Datasets: ArXiv, WOS, AAPD, cellcycle_FUN, seq_FUN, cellcycle_GO
Seeds: [42, 123, 456, 789, 1024]

Output: output/multi_seed_full/results.json
"""

from __future__ import annotations

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

from hmc.datasets.dataset_manager import initialize_dataset_experiments
from hmc.models.global_classifier.constraint.model import ConstrainedModel
from hmc.models.hierarchical.sparse_r import BlockDiagonalR

sys.path.insert(0, "src")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 128
SEEDS = [42, 123, 456, 789, 1024]

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, eval_mask: np.ndarray
) -> tuple[float, float, float, float, float]:
    """Compute micro-F1 (with threshold sweep) and micro-AUPRC."""
    best = (0.0, 0.5, 0.0, 0.0)
    for thr in np.arange(0.1, 0.90, 0.05):
        y_bin = (y_pred >= thr).astype(np.float32)
        tp = (y_bin[:, eval_mask] * y_true[:, eval_mask]).sum()
        fp = (y_bin[:, eval_mask] * (1 - y_true[:, eval_mask])).sum()
        fn = ((1 - y_bin[:, eval_mask]) * y_true[:, eval_mask]).sum()
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
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


def build_r_matrix(adj: np.ndarray) -> torch.Tensor:
    """Build dense R-matrix from adjacency."""
    r = np.zeros(adj.shape)
    np.fill_diagonal(r, 1)
    g = nx.DiGraph(adj)
    for i in range(len(adj)):
        ancestors = list(nx.descendants(g, i))
        if ancestors:
            r[i, ancestors] = 1
    return torch.tensor(r).transpose(1, 0).unsqueeze(0).to(DEVICE)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_seed(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    eval_mask: np.ndarray, adj: np.ndarray, n_nodes: int,
    go_graph: nx.DiGraph | None = None,
    hidden: int = 512, epochs: int = 50, seed: int = 42,
) -> dict:
    """Train and evaluate one seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # For large DAGs: use identity R in model (avoid OOM) + BlockDiagonalR post-hoc
    use_sparse = go_graph is not None or n_nodes > 2000
    if use_sparse:
        r_matrix = torch.eye(n_nodes).unsqueeze(0).to(DEVICE)
        baseline = True
    else:
        r_matrix = build_r_matrix(adj)
        baseline = False

    model = ConstrainedModel(
        input_dim=X_tr.shape[1],
        hidden_dim=min(hidden, 2048),
        output_dim=n_nodes,
        hyperparams={
            "batch_size": BATCH, "num_layers": 3, "dropout": 0.3,
            "non_lin": "relu", "hidden_dim": min(hidden, 2048),
            "lr": 1e-4, "weight_decay": 1e-5,
        },
        r_matrix=r_matrix, baseline_model=baseline,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    Xt_tr, yt_tr = torch.tensor(X_tr), torch.tensor(y_tr)
    tr_ldr = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=BATCH, shuffle=True)

    t0 = time.time()
    model.train()
    for _ in range(epochs):
        for bx, by in tr_ldr:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            loss = torch.nn.functional.binary_cross_entropy(
                model(bx)[:, eval_mask], by[:, eval_mask])
            opt.zero_grad()
            loss.backward()
            opt.step()
    train_time = time.time() - t0

    model.eval()
    preds: list[np.ndarray] = []
    te_ldr = DataLoader(TensorDataset(torch.tensor(X_te)),
                        batch_size=64, shuffle=False)
    with torch.no_grad():
        for (bx,) in te_ldr:
            preds.append(model(bx.to(DEVICE)).cpu().numpy())
    scores = np.concatenate(preds, axis=0)

    if go_graph is not None:
        sparse_r = BlockDiagonalR(go_graph, DEVICE)
        scores_t = torch.tensor(scores).to(DEVICE)
        scores = sparse_r.reconcile(scores_t).cpu().numpy()

    f1, au, p, r, thr = compute_metrics(y_te, scores, eval_mask)
    return {"f1": float(f1), "auprc": float(au), "precision": float(p),
            "recall": float(r), "threshold": float(thr),
            "time_s": float(train_time)}


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    results: dict[str, dict] = {}

    # ---- Text datasets ----
    for ds_name in ["arxiv", "wos", "aapd"]:
        print(f"\n{'='*50}\n  {ds_name}\n{'='*50}")
        mgr = initialize_dataset_experiments(
            ds_name, device="cpu", dataset_path="./data",
            is_global=True, model_cache_dir="./models",
        )
        tr, va, te = mgr.get_datasets()
        X_tr = np.concatenate([tr.x, va.x]).astype(np.float32)
        y_tr = np.concatenate([tr.y, va.y]).astype(np.float32)
        X_te = te.x.astype(np.float32)
        y_te = te.y.astype(np.float32)
        ev = np.array(mgr.to_eval, dtype=bool)
        adj = mgr.a
        n = mgr.output_dim
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr).astype(np.float32)
        X_te = scaler.transform(X_te).astype(np.float32)
        print(f"  Train: {X_tr.shape}  Test: {X_te.shape}  Nodes: {n}")

        seeds_data: dict[str, dict] = {}
        for s in SEEDS:
            print(f"    seed {s}...", end=" ", flush=True)
            r = train_one_seed(X_tr, y_tr, X_te, y_te, ev, adj, n,
                               hidden=512, epochs=50, seed=s)
            seeds_data[str(s)] = r
            print(f"F1={r['f1']:.4f} AUPRC={r['auprc']:.4f}")

        f1s = [v["f1"] for v in seeds_data.values()]
        aus = [v["auprc"] for v in seeds_data.values()]
        results[ds_name] = {
            "seeds": seeds_data,
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s, ddof=1)),
            "auprc_mean": float(np.mean(aus)),
            "auprc_std": float(np.std(aus, ddof=1)),
        }
        print(f"    F1={np.mean(f1s):.4f}±{np.std(f1s, ddof=1):.4f}  "
              f"AUPRC={np.mean(aus):.4f}±{np.std(aus, ddof=1):.4f}")

    # ---- Tabular datasets ----
    from hmc.datasets.manager.dataset_manager import \
        initialize_dataset_experiments as init_gofun

    for ds_name in ["cellcycle_FUN", "seq_FUN", "cellcycle_GO"]:
        print(f"\n{'='*50}\n  {ds_name}\n{'='*50}")
        mgr = init_gofun(ds_name, device="cpu", dataset_path="./data",
                          dataset_type="arff", is_global=False)
        tr, va, te = mgr.get_datasets()
        X_tr = np.concatenate([tr.x, va.x]).astype(np.float32)
        y_tr = np.concatenate([tr.y, va.y]).astype(np.float32)
        X_te = te.x.astype(np.float32)
        y_te = te.y.astype(np.float32)
        ev = np.array(mgr.to_eval, dtype=bool)
        adj = mgr.a
        n = len(mgr.nodes_idx)
        imp = SimpleImputer(strategy="mean")
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(imp.fit_transform(X_tr)).astype(np.float32)
        X_te = scaler.transform(imp.transform(X_te)).astype(np.float32)
        go_graph = nx.DiGraph(adj) if "GO" in ds_name else None
        print(f"  Train: {X_tr.shape}  Test: {X_te.shape}  Nodes: {n}")

        seeds_data = {}
        for s in SEEDS:
            print(f"    seed {s}...", end=" ", flush=True)
            r = train_one_seed(X_tr, y_tr, X_te, y_te, ev, adj, n,
                               go_graph=go_graph, hidden=256, epochs=30, seed=s)
            seeds_data[str(s)] = r
            print(f"F1={r['f1']:.4f} AUPRC={r['auprc']:.4f}")

        f1s = [v["f1"] for v in seeds_data.values()]
        aus = [v["auprc"] for v in seeds_data.values()]
        results[ds_name] = {
            "seeds": seeds_data,
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s, ddof=1)),
            "auprc_mean": float(np.mean(aus)),
            "auprc_std": float(np.std(aus, ddof=1)),
        }
        print(f"    F1={np.mean(f1s):.4f}±{np.std(f1s, ddof=1):.4f}  "
              f"AUPRC={np.mean(aus):.4f}±{np.std(aus, ddof=1):.4f}")

    # ---- Save ----
    os.makedirs("./output/multi_seed_full", exist_ok=True)
    out_path = "./output/multi_seed_full/results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\n{'='*70}")
    print("MULTI-SEED SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<20} {'F1':>16} {'AUPRC':>16}")
    print("-" * 54)
    for ds, r in results.items():
        print(f"{ds:<20} {r['f1_mean']:.4f}±{r['f1_std']:<.4f}   "
              f"{r['auprc_mean']:.4f}±{r['auprc_std']:<.4f}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
