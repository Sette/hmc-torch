#!/usr/bin/env python3
"""Compare tabular baselines vs global (MLP + R-matrix) — the existing SOTA method."""

import json, os, sys, time
import numpy as np
import torch
import networkx as nx

sys.path.insert(0, "src")

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset


def compute_metrics(y_true, y_pred, eval_mask):
    best_f1, best_thr, best_p, best_r = 0.0, 0.5, 0.0, 0.0
    for thr in np.arange(0.1, 0.90, 0.05):
        y_bin = (y_pred >= thr).astype(np.float32)
        tp = (y_bin[:, eval_mask] * y_true[:, eval_mask]).sum()
        fp = (y_bin[:, eval_mask] * (1 - y_true[:, eval_mask])).sum()
        fn = ((1 - y_bin[:, eval_mask]) * y_true[:, eval_mask]).sum()
        p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        if f1 > best_f1: best_f1, best_thr, best_p, best_r = f1, thr, p, r
    try:
        auprc = float(average_precision_score(y_true[:, eval_mask], y_pred[:, eval_mask], average="micro"))
    except: auprc = 0.0
    return best_f1, auprc, best_p, best_r, best_thr


def load_data(name):
    from hmc.datasets.manager.dataset_manager import initialize_dataset_experiments
    mgr = initialize_dataset_experiments(name, device="cpu", dataset_path="./data",
                                          dataset_type="arff", is_global=True)
    train, valid, test = mgr.get_datasets()
    X_tr = np.concatenate([train.x, valid.x]).astype(np.float32)
    y_tr = np.concatenate([train.y, valid.y]).astype(np.float32)
    X_te = test.x.astype(np.float32); y_te = test.y.astype(np.float32)
    eval_mask = np.array(mgr.to_eval, dtype=bool)
    a = mgr.a  # adjacency matrix
    imp = SimpleImputer(strategy="mean"); scaler = StandardScaler()
    X_tr = scaler.fit_transform(imp.fit_transform(X_tr)).astype(np.float32)
    X_te = scaler.transform(imp.transform(X_te)).astype(np.float32)
    return X_tr, y_tr, X_te, y_te, eval_mask, a, mgr


def build_r_matrix(adj, device="cpu"):
    """Build ancestor R-matrix from adjacency."""
    r = np.zeros(adj.shape)
    np.fill_diagonal(r, 1)
    g = nx.DiGraph(adj)
    for i in range(len(adj)):
        ancestors = list(nx.descendants(g, i))
        if ancestors: r[i, ancestors] = 1
    r = torch.tensor(r).transpose(1, 0).unsqueeze(0).to(device)
    return r


DATASETS = ["cellcycle_FUN", "derisi_FUN", "eisen_FUN", "expr_FUN",
            "gasch1_FUN", "gasch2_FUN", "seq_FUN", "spo_FUN"]

all_results = []

for ds_name in DATASETS:
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}")
    X_tr, y_tr, X_te, y_te, eval_mask, adj, mgr = load_data(ds_name)
    n_nodes = y_tr.shape[1]; n_feat = X_tr.shape[1]
    r_matrix = build_r_matrix(adj)
    print(f"  Train: {X_tr.shape}  Test: {X_te.shape}  "
          f"Features: {n_feat}  Nodes: {n_nodes}  Eval: {eval_mask.sum()}")

    # ---- global (MLP + R-matrix constraint) ----
    from hmc.models.global_classifier.constraint.model import ConstrainedModel
    from hmc.pipeline.global_classifier.core.train import train_step

    t0 = time.time()

    # Config from registry
    from hmc.datasets.registry import DatasetRegistry
    reg = DatasetRegistry()
    ontology = "GO" if "_GO" in ds_name else "FUN"
    data_key = ds_name.replace("_FUN", "").replace("_GO", "")
    hidden = reg.hidden_dims.get(ontology, {}).get(data_key, 512)
    rec_epochs = reg.all_epochs.get(ontology, {}).get(data_key, 50)
    epochs = min(rec_epochs, 50)

    device = torch.device("cpu")
    model = ConstrainedModel(
        input_dim=n_feat, hidden_dim=hidden, output_dim=n_nodes,
        hyperparams={"batch_size": 32, "num_layers": 3, "dropout": 0.3,
                     "non_lin": "relu", "hidden_dim": hidden,
                     "lr": 1e-4, "weight_decay": 1e-5},
        r_matrix=r_matrix, baseline_model=False,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    crit = torch.nn.BCELoss()

    tr_ds = list(zip(torch.tensor(X_tr), torch.tensor(y_tr)))
    tr_ldr = DataLoader(tr_ds, batch_size=32, shuffle=True)

    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for bx, by in tr_ldr:
            preds = model(bx)
            loss = crit(preds, by)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"  global epoch {ep+1}/{epochs} loss={total_loss:.2f}")

    model.eval()
    with torch.no_grad():
        scores_global = model(torch.tensor(X_te)).numpy()

    f1, au, p, r, thr = compute_metrics(y_te, scores_global, eval_mask)
    dur = time.time() - t0
    all_results.append({"method": "global", "dataset": ds_name,
                        "micro_f1": float(f1), "auprc": float(au),
                        "precision": float(p), "recall": float(r),
                        "threshold": float(thr), "duration_s": dur,
                        "epochs": epochs, "hidden_dim": hidden})
    print(f"  => global     F1={f1:.4f}  AUPRC={au:.4f}  "
          f"P={p:.4f}  R={r:.4f}  time={dur:.1f}s  epochs={epochs}")

    # Save scores
    out_dir = f"./output/experiments/{ds_name}/global"
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(f"{out_dir}/scores_final.npz", scores=scores_global)
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(all_results[-1], f, indent=2)

# Load previous GBDT/MLP results
try:
    with open("./output/experiments/fun_matrix.json") as f:
        prev = json.load(f)
    all_results.extend(prev)
except Exception:
    pass

# ===================================================================
# FINAL COMPARISON
# ===================================================================
print("\n\n" + "=" * 100)
print("SOTA COMPARISON — 8 FUN Datasets")
print("=" * 100)

for ds_name in DATASETS:
    ds_res = [r for r in all_results if r["dataset"] == ds_name]
    if not ds_res: continue
    print(f"\n{ds_name}:")
    for r in sorted(ds_res, key=lambda x: -x["micro_f1"]):
        extra = ""
        if "epochs" in r: extra += f"  ep={r['epochs']}"
        if "nodes_trained" in r: extra += f"  nodes={r['nodes_trained']}"
        print(f"  {r['method']:<15} F1={r['micro_f1']:.4f}  "
              f"AUPRC={r['auprc']:.4f}  P={r['precision']:.4f}  "
              f"R={r['recall']:.4f}  thr={r['threshold']:.2f}  "
              f"t={r['duration_s']:.1f}s{extra}")

# Summary
print(f"\n{'='*100}")
print("SUMMARY BY METHOD (averaged across datasets)")
for method in ["global", "tabular_gbdt", "tabular_mlp"]:
    m_res = [r for r in all_results if r["method"] == method]
    if not m_res: continue
    datasets_covered = set(r["dataset"] for r in m_res)
    avg_f1 = np.mean([r["micro_f1"] for r in m_res])
    avg_au = np.mean([r["auprc"] for r in m_res])
    avg_t = np.mean([r["duration_s"] for r in m_res])
    print(f"  {method:<15} avg F1={avg_f1:.4f}  avg AUPRC={avg_au:.4f}  "
          f"avg time={avg_t:.1f}s  ({len(datasets_covered)} datasets)")

with open("./output/experiments/sota_comparison.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to ./output/experiments/sota_comparison.json")
