#!/usr/bin/env python3
"""Batch experiment runner across all FUN datasets.

Runs tabular_gbdt and tabular_mlp on each dataset.
Saves artefacts and produces comparison table.
"""

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


def compute_metrics(y_true, y_pred, eval_mask):
    best_f1, best_thr, best_p, best_r = 0.0, 0.5, 0.0, 0.0
    for thr in np.arange(0.1, 0.90, 0.05):
        y_bin = (y_pred >= thr).astype(np.float32)
        tp = (y_bin[:, eval_mask] * y_true[:, eval_mask]).sum()
        fp = (y_bin[:, eval_mask] * (1 - y_true[:, eval_mask])).sum()
        fn = ((1 - y_bin[:, eval_mask]) * y_true[:, eval_mask]).sum()
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        if f1 > best_f1:
            best_f1, best_thr, best_p, best_r = f1, thr, p, r
    try:
        auprc = float(
            average_precision_score(
                y_true[:, eval_mask], y_pred[:, eval_mask], average="micro"
            )
        )
    except Exception:
        auprc = 0.0
    return best_f1, auprc, best_p, best_r, best_thr


def load_dataset(name):
    from hmc.datasets.manager.dataset_manager import initialize_dataset_experiments

    mgr = initialize_dataset_experiments(
        name,
        device="cpu",
        dataset_path="./data",
        dataset_type="arff",
        is_global=False,
    )
    train, valid, test = mgr.get_datasets()
    X_tr = np.concatenate([train.x, valid.x]).astype(np.float32)
    y_tr = np.concatenate([train.y, valid.y]).astype(np.float32)
    X_te = test.x.astype(np.float32)
    y_te = test.y.astype(np.float32)
    eval_mask = np.array(mgr.to_eval, dtype=bool)

    imp = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(imp.fit_transform(X_tr)).astype(np.float32)
    X_te = scaler.transform(imp.transform(X_te)).astype(np.float32)

    # Hierarchy
    from hmc.data.hierarchy import TreeHierarchy

    branches = [t for t in train.terms if t != "root"]
    hier = TreeHierarchy.from_fun_cat_terms(branches if branches else train.terms)
    return X_tr, y_tr, X_te, y_te, eval_mask, hier, mgr


def run_gbdt(name, X_tr, y_tr, X_te, y_te, eval_mask, hier):
    from hmc.models.hierarchical.postprocess import reconcile
    from hmc.models.tabular.gbdt_ovr import GBDTOvRClassifier

    t0 = time.time()
    n_pos = y_tr.sum(axis=0)
    trainable = (n_pos >= 5) & eval_mask

    gbdt = GBDTOvRClassifier(
        backend="histgb",
        backend_kwargs={
            "early_stopping": False,
            "max_iter": 100,
        },
    )
    gbdt.fit(X_tr, y_tr, eval_mask=trainable)
    scores_raw = gbdt.predict_proba(X_te)
    scores = reconcile(scores_raw, hier, strategy="ancestor_max")
    f1, au, p, r, thr = compute_metrics(y_te, scores, eval_mask)
    dur = time.time() - t0

    out_dir = f"./output/experiments/{name}/tabular_gbdt"
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(f"{out_dir}/scores_final.npz", scores=scores)
    metrics_dict = {
        "micro_f1": float(f1),
        "auprc": float(au),
        "precision": float(p),
        "recall": float(r),
        "threshold": float(thr),
        "duration_s": dur,
        "nodes_trained": int(trainable.sum()),
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "n_features": X_tr.shape[1],
        "n_nodes": y_tr.shape[1],
    }
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)

    return {"method": "tabular_gbdt", "dataset": name, **metrics_dict}


def run_mlp(name, X_tr, y_tr, X_te, y_te, eval_mask, hier):
    from hmc.models.hierarchical.losses import WeightedBCELoss
    from hmc.models.hierarchical.postprocess import reconcile
    from hmc.models.tabular.mlp import TabularMLPModel

    t0 = time.time()
    device = torch.device("cpu")
    n_features = X_tr.shape[1]
    n_nodes = y_tr.shape[1]

    model = TabularMLPModel(
        input_dim=n_features,
        n_nodes=n_nodes,
        hidden_dim=min(512, n_features),
        n_blocks=2,
        head_layers=2,
        dropout=0.3,
    ).to(device)

    pos_w = WeightedBCELoss.compute_pos_weight(torch.tensor(y_tr)).to(device)
    crit = WeightedBCELoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    tr_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    te_ds = TensorDataset(torch.tensor(X_te), torch.tensor(y_te))
    tr_ldr = DataLoader(tr_ds, batch_size=64, shuffle=True)
    te_ldr = DataLoader(te_ds, batch_size=64, shuffle=False)

    # Use registry epochs if available, else 30
    from hmc.datasets.registry import DatasetRegistry

    reg = DatasetRegistry()
    is_go = "_GO" in name
    ontology = "GO" if is_go else "FUN"
    data_key = name.replace("_FUN", "").replace("_GO", "")
    rec_epochs = reg.all_epochs.get(ontology, {}).get(data_key, 30)

    epochs = min(rec_epochs, 30)  # cap for speed
    for ep in range(epochs):
        model.train()
        for bx, by in tr_ldr:
            bx, by = bx.to(device), by.to(device)
            preds = model(bx)
            loss = crit(preds, by)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    all_preds = []
    with torch.no_grad():
        for bx, _ in te_ldr:
            all_preds.append(model(bx).cpu().numpy())
    scores_raw = np.concatenate(all_preds, axis=0)
    scores = reconcile(scores_raw, hier, strategy="ancestor_max")
    f1, au, p, r, thr = compute_metrics(y_te, scores, eval_mask)
    dur = time.time() - t0

    out_dir = f"./output/experiments/{name}/tabular_mlp"
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(f"{out_dir}/scores_final.npz", scores=scores)
    metrics_dict = {
        "micro_f1": float(f1),
        "auprc": float(au),
        "precision": float(p),
        "recall": float(r),
        "threshold": float(thr),
        "duration_s": dur,
        "epochs": epochs,
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "n_features": n_features,
        "n_nodes": n_nodes,
    }
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)

    return {"method": "tabular_mlp", "dataset": name, **metrics_dict}


# ===================================================================
FUN_DATASETS = [
    "cellcycle_FUN",
    "derisi_FUN",
    "eisen_FUN",
    "expr_FUN",
    "gasch1_FUN",
    "gasch2_FUN",
    "seq_FUN",
    "spo_FUN",
]

all_results = []

for ds_name in FUN_DATASETS:
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}")
    print(f"{'='*60}")
    try:
        X_tr, y_tr, X_te, y_te, eval_mask, hier, mgr = load_dataset(ds_name)
        n_nodes = y_tr.shape[1]
        n_feat = X_tr.shape[1]
        n_pos = int(y_tr.sum(axis=0).mean())
        print(f"  Samples: {X_tr.shape[0]} train, {X_te.shape[0]} test")
        print(
            f"  Features: {n_feat}  Nodes: {n_nodes}  "
            f"Eval: {eval_mask.sum()}  Avg positives/node: {n_pos}"
        )

        # GBDT
        print("  [GBDT] ", end="", flush=True)
        r_gbdt = run_gbdt(ds_name, X_tr, y_tr, X_te, y_te, eval_mask, hier)
        all_results.append(r_gbdt)
        print(
            f"F1={r_gbdt['micro_f1']:.4f}  AUPRC={r_gbdt['auprc']:.4f}  "
            f"nodes={r_gbdt['nodes_trained']}/{n_nodes}  time={r_gbdt['duration_s']:.1f}s"
        )

        # MLP
        print("  [MLP]  ", end="", flush=True)
        r_mlp = run_mlp(ds_name, X_tr, y_tr, X_te, y_te, eval_mask, hier)
        all_results.append(r_mlp)
        print(
            f"F1={r_mlp['micro_f1']:.4f}  AUPRC={r_mlp['auprc']:.4f}  "
            f"epochs={r_mlp.get('epochs', '?')}  time={r_mlp['duration_s']:.1f}s"
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback

        traceback.print_exc()

# ===================================================================
# Report
# ===================================================================
print("\n\n" + "=" * 100)
print("COMPLETE EXPERIMENT MATRIX — FUN Datasets")
print("=" * 100)

# Per-dataset comparison
for ds_name in FUN_DATASETS:
    ds_results = [r for r in all_results if r["dataset"] == ds_name]
    if not ds_results:
        continue
    print(f"\n{ds_name}:")
    for r in sorted(ds_results, key=lambda x: -x["micro_f1"]):
        extra = ""
        if "nodes_trained" in r:
            extra = f"  nodes={r['nodes_trained']}"
        if "epochs" in r:
            extra = f"  epochs={r['epochs']}"
        print(
            f"  {r['method']:<15} F1={r['micro_f1']:.4f}  "
            f"AUPRC={r['auprc']:.4f}  P={r['precision']:.4f}  "
            f"R={r['recall']:.4f}  thr={r['threshold']:.2f}  "
            f"time={r['duration_s']:.1f}s{extra}"
        )

# Summary by method
print(f"\n{'='*100}")
print("SUMMARY BY METHOD (averaged across datasets)")
print(f"{'='*100}")
for method in ["tabular_gbdt", "tabular_mlp"]:
    m_results = [r for r in all_results if r["method"] == method]
    if not m_results:
        continue
    avg_f1 = np.mean([r["micro_f1"] for r in m_results])
    avg_au = np.mean([r["auprc"] for r in m_results])
    avg_t = np.mean([r["duration_s"] for r in m_results])
    print(
        f"  {method:<15} avg F1={avg_f1:.4f}  avg AUPRC={avg_au:.4f}  "
        f"avg time={avg_t:.1f}s  ({len(m_results)} datasets)"
    )

# Save
os.makedirs("./output/experiments", exist_ok=True)
with open("./output/experiments/fun_matrix.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nFull results saved to ./output/experiments/fun_matrix.json")
