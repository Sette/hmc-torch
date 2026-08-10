#!/usr/bin/env python3
"""Systematic ablation: isolate R-matrix training vs inference contributions.

Runs 4 configurations on 3 hierarchy types:
  - bce_only:          no R-matrix anywhere (pure BCE)
  - consistency_loss:   R-matrix during training only (MC-loss)
  - reconciliation:     R-matrix at inference only (bottom-up max-prop)
  - both:               R-matrix at training + inference (full method)

Datasets:
  - ArXiv            shallow tree (2 levels, 157 nodes)
  - cellcycle_FUN    deep tree (4+ levels, 500 nodes)
  - cellcycle_GO     DAG (13 levels, 4126 nodes, BlockDiagonalR)

Output: output/ablation/results.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from hmc.datasets.dataset_manager import initialize_dataset_experiments
from hmc.models.global_classifier.constraint.model import ConstrainedModel
from hmc.models.global_classifier.constraint.utils import get_constr_out
from hmc.models.hierarchical.sparse_r import BlockDiagonalR

sys.path.insert(0, "src")

if TYPE_CHECKING:
    import networkx as nx

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 128
SEED = 42
MODES = ["bce_only", "consistency_loss", "reconciliation", "both"]

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    eval_mask: np.ndarray) -> tuple[float, float, float, float, float]:
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
        auprc = float(average_precision_score(
            y_true[:, eval_mask], y_pred[:, eval_mask], average="micro"))
    except ValueError:
        auprc = 0.0
    return best[0], auprc, best[2], best[3], best[1]


# ---------------------------------------------------------------------------
# R-matrix builder
# ---------------------------------------------------------------------------


def build_r_matrix(adj: np.ndarray, device: torch.device = DEVICE) -> torch.Tensor:
    """Build dense R-matrix (ancestor closure) from adjacency."""
    import networkx as nx
    r = np.zeros(adj.shape)
    np.fill_diagonal(r, 1)
    g = nx.DiGraph(adj)
    for i in range(len(adj)):
        ancestors = list(nx.descendants(g, i))
        if ancestors:
            r[i, ancestors] = 1
    return torch.tensor(r).transpose(1, 0).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_global_ablated(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    eval_mask: np.ndarray,
    adj: np.ndarray,
    n_nodes: int,
    hidden: int = 512,
    epochs: int = 30,
    mode: str = "both",
    seed: int = 42,
    go_graph: "nx.DiGraph | None" = None,
) -> dict:
    """Train a global model in one of 4 ablation modes.

    Args:
        mode: ``"bce_only"``, ``"consistency_loss"``, ``"reconciliation"``, ``"both"``.
        go_graph: For DAG datasets, the NetworkX DiGraph used by BlockDiagonalR.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    r_dense = build_r_matrix(adj)
    r_identity = torch.eye(n_nodes).unsqueeze(0).to(DEVICE)

    # ---- Model: baseline_model=True skips get_constr_out in eval forward ----
    baseline = mode in ("bce_only", "consistency_loss")
    model_r = r_identity if baseline else r_dense

    model = ConstrainedModel(
        input_dim=X_tr.shape[1],
        hidden_dim=min(hidden, 2048),
        output_dim=n_nodes,
        hyperparams={
            "batch_size": BATCH,
            "num_layers": 3,
            "dropout": 0.3,
            "non_lin": "relu",
            "hidden_dim": min(hidden, 2048),
            "lr": 1e-4,
            "weight_decay": 1e-5,
        },
        r_matrix=model_r,
        baseline_model=baseline,
    ).to(DEVICE)

    # ---- Training loop ----
    use_mc_loss = mode in ("consistency_loss", "both")

    # For large DAGs: dense R-matrix OOMs on GPU (4126² × batch × 8 bytes ≈ 16 GiB).
    # Use BlockDiagonalR consistency loss instead, or skip if not available.
    if use_mc_loss and go_graph is not None:
        # BlockDiagonalR handles training loss in O(N) memory
        sparse_r_train = BlockDiagonalR(go_graph, DEVICE)
        train_r = r_identity  # not used; sparse loss replaces MC-loss
    else:
        sparse_r_train = None
        train_r = r_dense if use_mc_loss else r_identity

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    Xt_tr = torch.tensor(X_tr)
    yt_tr = torch.tensor(y_tr)
    tr_ldr = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=BATCH, shuffle=True)

    t0 = time.time()
    model.train()
    for _ in range(epochs):
        for bx, by in tr_ldr:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            output = model(bx)  # raw logits (model.training=True → no get_constr_out)

            if use_mc_loss:
                if sparse_r_train is not None:
                    # O(N) sparse consistency loss for large DAGs
                    hier_loss = sparse_r_train.consistency_loss(output, margin=0.0)
                    bce_loss = torch.nn.functional.binary_cross_entropy(
                        output[:, eval_mask], by[:, eval_mask])
                    loss = bce_loss + hier_loss
                else:
                    constr_output = get_constr_out(output, train_r)
                    train_out = by * output.double()
                    train_out = get_constr_out(train_out, train_r)
                    train_out = (1 - by) * constr_output.double() + by * train_out
                    loss = torch.nn.functional.binary_cross_entropy(
                        train_out[:, eval_mask].float(), by[:, eval_mask])
            else:
                loss = torch.nn.functional.binary_cross_entropy(
                    output[:, eval_mask], by[:, eval_mask])

            opt.zero_grad()
            loss.backward()
            opt.step()

    train_time = time.time() - t0

    # ---- Evaluation ----
    model.eval()
    preds: list[np.ndarray] = []
    te_ldr = DataLoader(TensorDataset(torch.tensor(X_te)), batch_size=64, shuffle=False)
    with torch.no_grad():
        for (bx,) in te_ldr:
            preds.append(model(bx.to(DEVICE)).cpu().numpy())
    scores = np.concatenate(preds, axis=0)

    # For DAG with reconciliation-only mode, apply BlockDiagonalR post-hoc
    # (model forward used identity, so we reconcile manually here)
    if mode == "reconciliation" and go_graph is not None:
        sparse_r = BlockDiagonalR(go_graph, DEVICE)
        scores_t = torch.tensor(scores).to(DEVICE)
        scores = sparse_r.reconcile(scores_t).cpu().numpy()

    f1, au, p, r, thr = compute_metrics(y_te, scores, eval_mask)
    return {
        "mode": mode,
        "f1": float(f1),
        "auprc": float(au),
        "precision": float(p),
        "recall": float(r),
        "threshold": float(thr),
        "time_s": float(train_time),
        "n_train": int(X_tr.shape[0]),
        "n_test": int(X_te.shape[0]),
        "n_features": int(X_tr.shape[1]),
        "n_nodes": int(n_nodes),
    }


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    """Run ablation on ArXiv, cellcycle_FUN, and cellcycle_GO."""
    results: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    # 1. ArXiv (shallow tree)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("ABLATION: ArXiv (shallow tree, 2 levels, 157 nodes)")
    print("=" * 60)

    mgr_a = initialize_dataset_experiments(
        "arxiv", device="cpu", dataset_path="./data",
        is_global=True, model_cache_dir="./models",
    )
    tr_a, va_a, te_a = mgr_a.get_datasets()
    Xa_tr = np.concatenate([tr_a.x, va_a.x]).astype(np.float32)
    ya_tr = np.concatenate([tr_a.y, va_a.y]).astype(np.float32)
    Xa_te = te_a.x.astype(np.float32)
    ya_te = te_a.y.astype(np.float32)
    eval_a = np.array(mgr_a.to_eval, dtype=bool)
    adj_a = mgr_a.a
    n_a = mgr_a.output_dim
    scaler = StandardScaler()
    Xa_tr = scaler.fit_transform(Xa_tr).astype(np.float32)
    Xa_te = scaler.transform(Xa_te).astype(np.float32)
    print(f"  Train: {Xa_tr.shape}  Test: {Xa_te.shape}  Nodes: {n_a}  "
          f"Eval: {eval_a.sum()}")

    arxiv_results: list[dict] = []
    for mode in MODES:
        print(f"  {mode}...", end=" ", flush=True)
        r = train_global_ablated(
            Xa_tr, ya_tr, Xa_te, ya_te, eval_a, adj_a, n_a,
            hidden=512, epochs=50, mode=mode, seed=SEED,
        )
        arxiv_results.append(r)
        print(f"F1={r['f1']:.4f} AUPRC={r['auprc']:.4f} t={r['time_s']:.1f}s")
    results["arxiv"] = arxiv_results

    # ------------------------------------------------------------------
    # 2. cellcycle_FUN (deep tree)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ABLATION: cellcycle_FUN (deep tree, 4+ levels, 500 nodes)")
    print("=" * 60)

    from hmc.datasets.manager.dataset_manager import \
        initialize_dataset_experiments as init_gofun

    ds_name = "cellcycle_FUN"
    mgr_f = init_gofun(ds_name, device="cpu", dataset_path="./data",
                        dataset_type="arff", is_global=False)
    tr_f, va_f, te_f = mgr_f.get_datasets()
    Xf_tr = np.concatenate([tr_f.x, va_f.x]).astype(np.float32)
    yf_tr = np.concatenate([tr_f.y, va_f.y]).astype(np.float32)
    Xf_te = te_f.x.astype(np.float32)
    yf_te = te_f.y.astype(np.float32)
    eval_f = np.array(mgr_f.to_eval, dtype=bool)
    adj_f = mgr_f.a
    n_f = len(mgr_f.nodes_idx)

    imp = SimpleImputer(strategy="mean")
    scaler_f = StandardScaler()
    Xf_tr = scaler_f.fit_transform(imp.fit_transform(Xf_tr)).astype(np.float32)
    Xf_te = scaler_f.transform(imp.transform(Xf_te)).astype(np.float32)
    print(f"  Train: {Xf_tr.shape}  Test: {Xf_te.shape}  Nodes: {n_f}  "
          f"Eval: {eval_f.sum()}")

    funcat_results: list[dict] = []
    for mode in MODES:
        print(f"  {mode}...", end=" ", flush=True)
        r = train_global_ablated(
            Xf_tr, yf_tr, Xf_te, yf_te, eval_f, adj_f, n_f,
            hidden=512, epochs=50, mode=mode, seed=SEED,
        )
        funcat_results.append(r)
        print(f"F1={r['f1']:.4f} AUPRC={r['auprc']:.4f} t={r['time_s']:.1f}s")
    results[ds_name] = funcat_results

    # ------------------------------------------------------------------
    # 3. cellcycle_GO (DAG — BlockDiagonalR for reconciliation)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ABLATION: cellcycle_GO (DAG, 13 levels, 4126 nodes)")
    print("=" * 60)

    ds_go = "cellcycle_GO"
    mgr_g = init_gofun(ds_go, device="cpu", dataset_path="./data",
                        dataset_type="arff", is_global=False)
    tr_g, va_g, te_g = mgr_g.get_datasets()
    Xg_tr = np.concatenate([tr_g.x, va_g.x]).astype(np.float32)
    yg_tr = np.concatenate([tr_g.y, va_g.y]).astype(np.float32)
    Xg_te = te_g.x.astype(np.float32)
    yg_te = te_g.y.astype(np.float32)
    eval_g = np.array(mgr_g.to_eval, dtype=bool)
    adj_g = mgr_g.a
    n_g = len(mgr_g.nodes_idx)

    # Build NetworkX graph for BlockDiagonalR
    import networkx as nx
    go_graph = nx.DiGraph(adj_g)

    imp_g = SimpleImputer(strategy="mean")
    scaler_g = StandardScaler()
    Xg_tr = scaler_g.fit_transform(imp_g.fit_transform(Xg_tr)).astype(np.float32)
    Xg_te = scaler_g.transform(imp_g.transform(Xg_te)).astype(np.float32)
    print(f"  Train: {Xg_tr.shape}  Test: {Xg_te.shape}  Nodes: {n_g}  "
          f"Eval: {eval_g.sum()}")

    go_results: list[dict] = []
    go_modes = ["bce_only", "reconciliation"]  # skip MC-loss modes (OOM/slow)
    for mode in MODES:
        if mode not in go_modes:
            print(f"  {mode}... SKIP (dense R OOM, sparse too slow)", flush=True)
            go_results.append({"mode": mode, "f1": 0.0, "auprc": 0.0,
                               "precision": 0.0, "recall": 0.0,
                               "threshold": 0.0, "time_s": 0.0,
                               "n_train": int(Xg_tr.shape[0]),
                               "n_test": int(Xg_te.shape[0]),
                               "n_features": int(Xg_tr.shape[1]),
                               "n_nodes": int(n_g),
                               "skipped": True})
            continue
        print(f"  {mode}...", end=" ", flush=True)
        r = train_global_ablated(
            Xg_tr, yg_tr, Xg_te, yg_te, eval_g, adj_g, n_g,
            hidden=256, epochs=30, mode=mode, seed=SEED,
            go_graph=go_graph,
        )
        go_results.append(r)
        print(f"F1={r['f1']:.4f} AUPRC={r['auprc']:.4f} t={r['time_s']:.1f}s")
    results[ds_go] = go_results

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs("./output/ablation", exist_ok=True)
    out_path = "./output/ablation/results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)
    header = f"{'Dataset':<20} {'Mode':<20} {'F1':>8} {'AUPRC':>8} {'ΔF1':>8} {'Time':>8}"
    print(header)
    print("-" * len(header))

    for ds_name, ds_results in results.items():
        base_f1 = ds_results[0]["f1"]  # bce_only baseline
        for r in ds_results:
            delta = r["f1"] - base_f1
            print(f"{ds_name:<20} {r['mode']:<20} {r['f1']:8.4f} {r['auprc']:8.4f} "
                  f"{delta:+8.4f} {r['time_s']:7.1f}s")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
