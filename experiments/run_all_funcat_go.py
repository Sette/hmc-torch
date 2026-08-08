#!/usr/bin/env python3
"""Run global (MLP+R) on ALL remaining FUN + GO datasets."""

import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "src")


DEVICE = torch.device("cuda")
torch.manual_seed(42)
np.random.seed(42)


def compute_metrics(y_true, y_pred, eval_mask):
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
    except:
        auprc = 0.0
    return best[0], auprc, best[2], best[3], best[1]


def load_data(name):
    from hmc.datasets.manager.dataset_manager import initialize_dataset_experiments

    is_go = "_GO" in name
    mgr = initialize_dataset_experiments(
        name, device="cpu", dataset_path="./data", dataset_type="arff", is_global=False
    )
    train, valid, test = mgr.get_datasets()
    X_tr = np.concatenate([train.x, valid.x]).astype(np.float32)
    y_tr = np.concatenate([train.y, valid.y]).astype(np.float32)
    X_te = test.x.astype(np.float32)
    y_te = test.y.astype(np.float32)
    eval_mask = np.array(mgr.to_eval, dtype=bool)
    g = train.g
    adj = mgr.a
    n_nodes = len(mgr.nodes_idx)
    imp = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(imp.fit_transform(X_tr)).astype(np.float32)
    X_te = scaler.transform(imp.transform(X_te)).astype(np.float32)
    return X_tr, y_tr, X_te, y_te, eval_mask, g, adj, n_nodes


def run_fun(name, X_tr, y_tr, X_te, y_te, eval_mask, adj, n_nodes):
    """Global MLP + dense R-matrix on FUN (500 nodes, fits in GPU)."""
    import networkx as nx

    from hmc.datasets.registry import DatasetRegistry
    from hmc.models.global_classifier.constraint.model import ConstrainedModel

    n_feat = X_tr.shape[1]
    dk = name.replace("_FUN", "")
    reg = DatasetRegistry()
    hidden = reg.hidden_dims.get("FUN", {}).get(dk, 512)
    rec_ep = reg.all_epochs.get("FUN", {}).get(dk, 50)
    epochs = min(rec_ep, 50)

    r = np.zeros(adj.shape)
    np.fill_diagonal(r, 1)
    g_nx = nx.DiGraph(adj)
    for i in range(len(adj)):
        ancestors = list(nx.descendants(g_nx, i))
        if ancestors:
            r[i, ancestors] = 1
    r_matrix = torch.tensor(r).transpose(1, 0).unsqueeze(0).to(DEVICE)

    t0 = time.time()
    model = ConstrainedModel(
        input_dim=n_feat,
        hidden_dim=min(hidden, 2048),
        output_dim=n_nodes,
        hyperparams={
            "batch_size": 128,
            "num_layers": 3,
            "dropout": 0.3,
            "non_lin": "relu",
            "hidden_dim": min(hidden, 2048),
            "lr": 1e-4,
            "weight_decay": 1e-5,
        },
        r_matrix=r_matrix,
        baseline_model=False,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    Xt_tr = torch.tensor(X_tr)
    yt_tr = torch.tensor(y_tr)
    tr_ldr = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=128, shuffle=True)
    model.train()
    for ep in range(epochs):
        for bx, by in tr_ldr:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            loss = torch.nn.functional.binary_cross_entropy(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    preds = []
    te_ldr = DataLoader(TensorDataset(torch.tensor(X_te)), batch_size=64, shuffle=False)
    with torch.no_grad():
        for (bx,) in te_ldr:
            preds.append(model(bx.to(DEVICE)).cpu().numpy())
    scores = np.concatenate(preds, axis=0)
    f1, au, p, r, thr = compute_metrics(y_te, scores, eval_mask)
    return f1, au, p, r, thr, time.time() - t0, epochs


def run_go(name, X_tr, y_tr, X_te, y_te, eval_mask, g, n_nodes):
    """MLP + sparse R reconciliation on GO (4000+ nodes, dense R OOMs)."""
    from hmc.models.global_classifier.constraint.model import ConstrainedModel
    from hmc.models.hierarchical.sparse_r import BlockDiagonalR

    n_feat = X_tr.shape[1]
    sparse_r = BlockDiagonalR(g)
    r_matrix = (
        torch.eye(n_nodes).unsqueeze(0).to(DEVICE)
    )  # identity, sparse R at inference

    t0 = time.time()
    model = ConstrainedModel(
        input_dim=n_feat,
        hidden_dim=min(512, n_feat),
        output_dim=n_nodes,
        hyperparams={
            "batch_size": 64,
            "num_layers": 2,
            "dropout": 0.3,
            "non_lin": "relu",
            "hidden_dim": min(512, n_feat),
            "lr": 1e-4,
            "weight_decay": 1e-5,
        },
        r_matrix=r_matrix,
        baseline_model=False,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    Xt_tr = torch.tensor(X_tr)
    yt_tr = torch.tensor(y_tr)
    tr_ldr = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=64, shuffle=True)
    model.train()
    epochs = 20
    for ep in range(epochs):
        for bx, by in tr_ldr:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            loss = torch.nn.functional.binary_cross_entropy(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    preds = []
    te_ldr = DataLoader(TensorDataset(torch.tensor(X_te)), batch_size=32, shuffle=False)
    with torch.no_grad():
        for (bx,) in te_ldr:
            preds.append(model(bx.to(DEVICE)).cpu().numpy())
    scores_raw = np.concatenate(preds, axis=0)
    scores_rec = sparse_r.reconcile(torch.tensor(scores_raw).to(DEVICE)).cpu().numpy()
    f1, au, p, r, thr = compute_metrics(y_te, scores_rec, eval_mask)
    return f1, au, p, r, thr, time.time() - t0, epochs


# ===== RUN =====
all_results = []

# FUN: church + pheno
for ds in ["church_FUN", "pheno_FUN"]:
    print(f"\n{'='*50}")
    print(f"FUN: {ds}")
    try:
        X_tr, y_tr, X_te, y_te, em, g, adj, n = load_data(ds)
        print(
            f"  {X_tr.shape[0]} train | {X_te.shape[0]} test | {X_tr.shape[1]} feat | {n} nodes"
        )
        f1, au, p, r, thr, dur, ep = run_fun(ds, X_tr, y_tr, X_te, y_te, em, adj, n)
        all_results.append(
            {
                "dataset": ds,
                "method": "global",
                "F1": float(f1),
                "AUPRC": float(au),
                "P": float(p),
                "R": float(r),
                "thr": float(thr),
                "time": dur,
                "epochs": ep,
                "nodes": n,
            }
        )
        print(
            f"  => F1={f1:.4f} AUPRC={au:.4f} P={p:.4f} R={r:.4f} t={dur:.1f}s ep={ep}"
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        all_results.append({"dataset": ds, "error": str(e)})

# GO: remaining 5
for ds in ["gasch1_GO", "gasch2_GO", "seq_GO", "spo_GO", "pheno_GO"]:
    print(f"\n{'='*50}")
    print(f"GO: {ds}")
    try:
        X_tr, y_tr, X_te, y_te, em, g, adj, n = load_data(ds)
        print(
            f"  {X_tr.shape[0]} train | {X_te.shape[0]} test | {X_tr.shape[1]} feat | {n} nodes"
        )
        f1, au, p, r, thr, dur, ep = run_go(ds, X_tr, y_tr, X_te, y_te, em, g, n)
        all_results.append(
            {
                "dataset": ds,
                "method": "sparse_R",
                "F1": float(f1),
                "AUPRC": float(au),
                "P": float(p),
                "R": float(r),
                "thr": float(thr),
                "time": dur,
                "epochs": ep,
                "nodes": n,
            }
        )
        print(
            f"  => F1={f1:.4f} AUPRC={au:.4f} P={p:.4f} R={r:.4f} t={dur:.1f}s ep={ep}"
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        all_results.append({"dataset": ds, "error": str(e)})

# ===== REPORT =====
print("\n\n" + "=" * 70)
print("ALL FUN + GO RESULTS")
print("=" * 70)
for family in ["FUN", "GO"]:
    ds_list = [r for r in all_results if family in r["dataset"]]
    if not ds_list:
        continue
    f1s = [r["F1"] for r in ds_list if "F1" in r]
    aus = [r["AUPRC"] for r in ds_list if "AUPRC" in r]
    print(f"\n{family} ({len(ds_list)} datasets):")
    for r in sorted(ds_list, key=lambda x: -(x.get("F1") or 0)):
        if "error" in r:
            print(f"  {r['dataset']:<20} ERROR: {r['error'][:50]}")
        else:
            print(
                f"  {r['dataset']:<20} F1={r['F1']:.4f} AUPRC={r['AUPRC']:.4f} "
                f"t={r['time']:.1f}s ep={r['epochs']}"
            )
    if f1s:
        print(f"  {'AVG':<20} F1={np.mean(f1s):.4f} AUPRC={np.mean(aus):.4f}")

# Merge with existing results and save
os.makedirs("./output/all_funcat_go", exist_ok=True)
with open("./output/all_funcat_go/results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved to ./output/all_funcat_go/results.json")
