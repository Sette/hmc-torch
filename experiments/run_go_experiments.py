#!/usr/bin/env python3
"""GO experiments: global model with BlockDiagonalR vs no constraint."""

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
BATCH = 64  # smaller batch for 4000+ output nodes


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


def load_go_data(name):
    from hmc.datasets.manager.dataset_manager import initialize_dataset_experiments

    mgr = initialize_dataset_experiments(
        name, device="cpu", dataset_path="./data", dataset_type="arff", is_global=False
    )
    train, valid, test = mgr.get_datasets()
    X_tr = np.concatenate([train.x, valid.x]).astype(np.float32)
    y_tr = np.concatenate([train.y, valid.y]).astype(np.float32)
    X_te = test.x.astype(np.float32)
    y_te = test.y.astype(np.float32)
    eval_mask = np.array(mgr.to_eval, dtype=bool)
    g = train.g  # child→parent DiGraph
    imp = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(imp.fit_transform(X_tr)).astype(np.float32)
    X_te = scaler.transform(imp.transform(X_te)).astype(np.float32)
    return X_tr, y_tr, X_te, y_te, eval_mask, g


def train_and_eval_go(X_tr, y_tr, X_te, y_te, eval_mask, graph, epochs=20, seed=42):
    """Train MLP on GO (no R-matrix during training — identity only).
    At inference, compare raw scores vs BlockDiagonalR reconciliation.
    This tests the inference-time benefit of sparse R, which is where
    the R-matrix provides most of its gain (see ablation, Sec 5.5)."""
    from hmc.models.global_classifier.constraint.model import ConstrainedModel
    from hmc.models.hierarchical.sparse_r import BlockDiagonalR

    torch.manual_seed(seed)
    np.random.seed(seed)
    n_feat = X_tr.shape[1]
    n_nodes = y_tr.shape[1]
    t0 = time.time()

    # Build sparse R for inference-time reconciliation
    sparse_r = BlockDiagonalR(graph)
    print(
        f"    SparseR: {sparse_r.n_nodes} nodes, {len(sparse_r.levels)} levels, "
        f"build={time.time()-t0:.3f}s"
    )

    # Train with identity (no constraint) — dense R would OOM for GO
    r_matrix = torch.eye(n_nodes).unsqueeze(0).to(DEVICE)

    model = ConstrainedModel(
        input_dim=n_feat,
        hidden_dim=min(512, n_feat),
        output_dim=n_nodes,
        hyperparams={
            "batch_size": BATCH,
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
    tr_ldr = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=BATCH, shuffle=True)

    model.train()
    for ep in range(epochs):
        for bx, by in tr_ldr:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            loss = torch.nn.functional.binary_cross_entropy(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"    ep {ep+1}/{epochs}")

    # Inference
    model.eval()
    preds = []
    te_ldr = DataLoader(TensorDataset(torch.tensor(X_te)), batch_size=32, shuffle=False)
    with torch.no_grad():
        for (bx,) in te_ldr:
            preds.append(model(bx.to(DEVICE)).cpu().numpy())
    scores_raw = np.concatenate(preds, axis=0)

    # Reconcile via sparse R
    scores_rec = sparse_r.reconcile(torch.tensor(scores_raw).to(DEVICE)).cpu().numpy()

    f1_raw, au_raw, p_raw, r_raw, th_raw = compute_metrics(y_te, scores_raw, eval_mask)
    f1_rec, au_rec, p_rec, r_rec, th_rec = compute_metrics(y_te, scores_rec, eval_mask)
    dur = time.time() - t0

    return (
        f1_raw,
        au_raw,
        p_raw,
        r_raw,
        th_raw,
        f1_rec,
        au_rec,
        p_rec,
        r_rec,
        th_rec,
        dur,
    )


# ===== Run =====
GO_DATASETS = ["cellcycle_GO", "derisi_GO", "eisen_GO", "expr_GO"]

all_results = []

for ds_name in GO_DATASETS:
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}")
    print(f"{'='*60}")
    try:
        X_tr, y_tr, X_te, y_te, eval_mask, g = load_go_data(ds_name)
        n_nodes = y_tr.shape[1]
        print(
            f"  {X_tr.shape[0]} train | {X_te.shape[0]} test | "
            f"{X_tr.shape[1]} feat | {n_nodes} nodes | eval={eval_mask.sum()}"
        )

        (
            f1_raw,
            au_raw,
            p_raw,
            r_raw,
            th_raw,
            f1_rec,
            au_rec,
            p_rec,
            r_rec,
            th_rec,
            dur,
        ) = train_and_eval_go(X_tr, y_tr, X_te, y_te, eval_mask, g, epochs=20, seed=42)

        delta_f1 = f1_rec - f1_raw
        delta_au = au_rec - au_raw
        print(
            f"  Raw:      F1={f1_raw:.4f} AUPRC={au_raw:.4f} P={p_raw:.4f} R={r_raw:.4f}"
        )
        print(
            f"  Sparse R: F1={f1_rec:.4f} AUPRC={au_rec:.4f} P={p_rec:.4f} R={r_rec:.4f}"
        )
        print(f"  Δ: F1={delta_f1:+.4f} AUPRC={delta_au:+.4f} t={dur:.1f}s")

        all_results.append(
            {
                "dataset": ds_name,
                "nodes": n_nodes,
                "raw_F1": f1_raw,
                "raw_AUPRC": au_raw,
                "raw_P": p_raw,
                "raw_R": r_raw,
                "sparse_F1": f1_rec,
                "sparse_AUPRC": au_rec,
                "sparse_P": p_rec,
                "sparse_R": r_rec,
                "delta_F1": delta_f1,
                "delta_AUPRC": delta_au,
                "duration_s": dur,
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        all_results.append({"dataset": ds_name, "error": str(e)})

# Report
print("\n\n" + "=" * 70)
print("GO EXPERIMENTS — Sparse R vs No Constraint")
print("=" * 70)
print(
    f"{'Dataset':<18} {'Nodes':>6} {'Raw F1':>10} {'Sparse F1':>10} {'ΔF1':>8} {'ΔAUPRC':>8}"
)
print("-" * 70)
for r in all_results:
    if "error" in r:
        print(f"{r['dataset']:<18} ERROR: {r['error'][:40]}")
    else:
        print(
            f"{r['dataset']:<18} {r['nodes']:>6} {r['raw_F1']:>10.4f} "
            f"{r['sparse_F1']:>10.4f} {r['delta_F1']:>+8.4f} {r['delta_AUPRC']:>+8.4f}"
        )

os.makedirs("./output/go_experiments", exist_ok=True)
with open("./output/go_experiments/results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved to ./output/go_experiments/results.json")
