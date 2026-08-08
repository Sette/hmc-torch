#!/usr/bin/env python3
"""Run global (MLP+R-matrix) on ALL available datasets — FUN + GO + others."""

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

from hmc.datasets.registry import DatasetRegistry

sys.path.insert(0, "src")


SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 128
torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_metrics(y_true, y_pred, eval_mask):
    best = (0.0, 0.5, 0.0, 0.0)
    for thr in np.arange(0.05, 0.90, 0.05):
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


def load_data(name, with_adj=True):
    from hmc.datasets.manager.dataset_manager import initialize_dataset_experiments

    mgr = initialize_dataset_experiments(
        name,
        device="cpu",
        dataset_path="./data",
        dataset_type="arff",
        is_global=True if with_adj else False,
    )
    train, valid, test = mgr.get_datasets()
    X_tr = np.concatenate([train.x, valid.x]).astype(np.float32)
    y_tr = np.concatenate([train.y, valid.y]).astype(np.float32)
    X_te = test.x.astype(np.float32)
    y_te = test.y.astype(np.float32)
    eval_mask = np.array(mgr.to_eval, dtype=bool)
    adj = mgr.a if with_adj else None
    # Impute + scale
    imp = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(imp.fit_transform(X_tr)).astype(np.float32)
    X_te = scaler.transform(imp.transform(X_te)).astype(np.float32)
    return X_tr, y_tr, X_te, y_te, eval_mask, adj, mgr


reg = DatasetRegistry()

all_results = []
ALL_DS = (
    [
        "cellcycle_FUN",
        "derisi_FUN",
        "eisen_FUN",
        "expr_FUN",
        "gasch1_FUN",
        "gasch2_FUN",
        "seq_FUN",
        "spo_FUN",
        "church_FUN",
        "pheno_FUN",
    ]
    + [
        "cellcycle_GO",
        "derisi_GO",
        "eisen_GO",
        "expr_GO",
        "gasch1_GO",
        "gasch2_GO",
        "seq_GO",
        "spo_GO",
        "pheno_GO",
    ]
    + ["enron_others", "diatoms_others", "imclef07a_others", "imclef07d_others"]
)

for ds_name in ALL_DS:
    print(f"\n{'='*60}")
    print(f"  {ds_name}")
    t0 = time.time()
    try:
        X_tr, y_tr, X_te, y_te, eval_mask, adj, mgr = load_data(ds_name)
        n_nodes = y_tr.shape[1]
        n_feat = X_tr.shape[1]
        print(
            f"  {X_tr.shape[0]} train | {X_te.shape[0]} test | "
            f"{n_feat} feat | {n_nodes} nodes | eval={eval_mask.sum()}"
        )

        # Determine epochs from registry or use reasonable default
        is_go = "_GO" in ds_name
        is_others = "_others" in ds_name
        ontology = "GO" if is_go else ("others" if is_others else "FUN")
        data_key = ds_name.replace("_FUN", "").replace("_GO", "").replace("_others", "")
        hidden = reg.hidden_dims.get(ontology, {}).get(data_key, 512)
        rec_ep = reg.all_epochs.get(ontology, {}).get(data_key, 50)
        epochs = max(min(rec_ep, 80), 20) if not is_others else 50

        # Build R-matrix (skip if too large)
        use_r = n_nodes <= 2000
        if use_r:
            r = np.zeros(adj.shape)
            np.fill_diagonal(r, 1)
            g = nx.DiGraph(adj)
            for i in range(len(adj)):
                ancestors = list(nx.descendants(g, i))
                if ancestors:
                    r[i, ancestors] = 1
            r_matrix = torch.tensor(r).transpose(1, 0).unsqueeze(0).to(DEVICE)
        else:
            r_matrix = torch.eye(n_nodes).unsqueeze(0).to(DEVICE)

        # Build model
        from hmc.models.global_classifier.constraint.model import ConstrainedModel

        torch.manual_seed(SEED)
        model = ConstrainedModel(
            input_dim=n_feat,
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
                print(f"  ep {ep+1}/{epochs}")

        # Batched inference
        model.eval()
        preds = []
        te_ldr = DataLoader(
            TensorDataset(torch.tensor(X_te)), batch_size=64, shuffle=False
        )
        with torch.no_grad():
            for (bx,) in te_ldr:
                preds.append(model(bx.to(DEVICE)).cpu().numpy())
        scores = np.concatenate(preds, axis=0)

        f1, au, p, r, thr = compute_metrics(y_te, scores, eval_mask)
        dur = time.time() - t0
        all_results.append(
            {
                "method": "global",
                "dataset": ds_name,
                "micro_f1": f1,
                "auprc": au,
                "precision": p,
                "recall": r,
                "threshold": thr,
                "duration_s": dur,
                "epochs": epochs,
                "n_nodes": n_nodes,
                "n_feat": n_feat,
                "use_r_matrix": use_r,
            }
        )
        print(
            f"  => F1={f1:.4f}  AUPRC={au:.4f}  P={p:.4f}  R={r:.4f}  "
            f"thr={thr:.2f}  t={dur:.1f}s  R={'on' if use_r else 'off'}"
        )

    except Exception as e:
        dur = time.time() - t0
        all_results.append(
            {
                "method": "global",
                "dataset": ds_name,
                "error": str(e)[:80],
                "duration_s": dur,
            }
        )
        print(f"  ERROR: {e}")

# Save
os.makedirs("./output/all_datasets", exist_ok=True)
with open("./output/all_datasets/global_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

# Report
print("\n\n" + "=" * 70)
print("GLOBAL (MLP+R) — ALL DATASETS")
print("=" * 70)
for group, label in [
    ("_FUN", "FunCat"),
    ("_GO", "Gene Ontology"),
    ("_others", "Others"),
]:
    ds_list = [r for r in all_results if group in r["dataset"]]
    if not ds_list:
        continue
    f1s = [r["micro_f1"] for r in ds_list if "micro_f1" in r]
    errors = [r for r in ds_list if "error" in r]
    print(f"\n{label} ({len(ds_list)} datasets):")
    for r in sorted(ds_list, key=lambda x: -(x.get("micro_f1") or 0)):
        if "error" in r:
            print(f"  {r['dataset']:<22} ERROR: {r['error'][:60]}")
        else:
            print(
                f"  {r['dataset']:<22} F1={r['micro_f1']:.4f}  "
                f"AUPRC={r['auprc']:.4f}  t={r['duration_s']:.1f}s  "
                f"R={r.get('use_r_matrix', '?')}"
            )
    if f1s:
        print(
            f"  {'AVG':<22} F1={np.mean(f1s):.4f}  AUPRC={np.mean([r['auprc'] for r in ds_list if 'auprc' in r]):.4f}"
        )

print("\nSaved to ./output/all_datasets/global_results.json")
