#!/usr/bin/env python3
"""Fast HMC experiment comparison on real seq_FUN data.

Strategy:
  - GBDT: train on nodes with >=10 positive examples only
  - MLP: 30 epochs
  - local: 20 epochs
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

from hmc.data.hierarchy import TreeHierarchy
from hmc.datasets.manager.dataset_manager import initialize_dataset_experiments
from hmc.models.hierarchical.losses import WeightedBCELoss
from hmc.models.hierarchical.postprocess import reconcile
from hmc.models.local_classifier.baseline.model import HMCLocalModel
from hmc.models.tabular.gbdt_ovr import GBDTOvRClassifier
from hmc.models.tabular.mlp import TabularMLPModel

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


def save_artefacts(out_dir, scores_raw, scores_final, metrics, config):
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(f"{out_dir}/scores_before_postprocess.npz", scores=scores_raw)
    np.savez_compressed(f"{out_dir}/scores_final.npz", scores=scores_final)
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{out_dir}/run-config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)


# ===================================================================
print("=" * 60)
print("Loading seq_FUN...")

mgr = initialize_dataset_experiments(
    "seq_FUN",
    device="cpu",
    dataset_path="./data",
    dataset_type="arff",
    is_global=False,
)
train, valid, test = mgr.get_datasets()

X_train_raw = np.concatenate([train.x, valid.x]).astype(np.float32)
y_train_raw = np.concatenate([train.y, valid.y]).astype(np.float32)
X_test_raw = test.x.astype(np.float32)
y_test_raw = test.y.astype(np.float32)

eval_mask = np.array(mgr.to_eval, dtype=bool)
n_nodes = len(mgr.nodes_idx)
n_features = X_train_raw.shape[1]

imp = SimpleImputer(strategy="mean")
scaler = StandardScaler()
X_train = scaler.fit_transform(imp.fit_transform(X_train_raw)).astype(np.float32)
X_test = scaler.transform(imp.transform(X_test_raw)).astype(np.float32)

print(
    f"  Train: {X_train.shape}  Test: {X_test.shape}  "
    f"Features: {n_features}  Nodes: {n_nodes}  Eval: {eval_mask.sum()}"
)

results = []
base_dir = "./output/experiments/seq_FUN"

# ===================================================================
# 1. tabular_gbdt (only trainable nodes)
# ===================================================================
print("\n" + "=" * 60)
print("[1/3] tabular_gbdt — HistGradientBoosting One-vs-Rest")
t0 = time.time()


# Only train nodes with enough positive examples
n_pos = y_train_raw.sum(axis=0)
trainable = (n_pos >= 5) & eval_mask
gbdt_mask = trainable
print(f"  Training on {gbdt_mask.sum()}/{eval_mask.sum()} eval nodes (>=5 positives)")

gbdt = GBDTOvRClassifier(
    backend="histgb",
    backend_kwargs={
        "early_stopping": False,
        "max_iter": 100,
    },
)
gbdt.fit(X_train, y_train_raw, eval_mask=gbdt_mask)
scores_gbdt_raw = gbdt.predict_proba(X_test)

branches = [t for t in train.terms if t != "root"]
hier = TreeHierarchy.from_fun_cat_terms(branches if branches else train.terms)
scores_gbdt = reconcile(scores_gbdt_raw, hier, strategy="ancestor_max")

f1, au, p, r, thr = compute_metrics(y_test_raw, scores_gbdt, eval_mask)
dur = time.time() - t0
results.append(
    {
        "method": "tabular_gbdt",
        "dataset": "seq_FUN",
        "micro_f1": float(f1),
        "auprc": float(au),
        "precision": float(p),
        "recall": float(r),
        "threshold": float(thr),
        "duration_s": dur,
        "nodes_trained": int(gbdt_mask.sum()),
    }
)
print(
    f"  F1={f1:.4f}  AUPRC={au:.4f}  P={p:.4f}  R={r:.4f}  "
    f"thr={thr:.2f}  time={dur:.1f}s  nodes={gbdt_mask.sum()}"
)

save_artefacts(
    f"{base_dir}/tabular_gbdt",
    scores_gbdt_raw,
    scores_gbdt,
    results[-1],
    {
        "method": "tabular_gbdt",
        "dataset": "seq_FUN",
        "backend": "histgb",
        "seed": 42,
        "nodes_trained": int(gbdt_mask.sum()),
        "n_nodes_total": n_nodes,
    },
)

# ===================================================================
# 2. tabular_mlp
# ===================================================================
print("\n" + "=" * 60)
print("[2/3] tabular_mlp — Residual MLP + GlobalSigmoidHead")
t0 = time.time()


device = torch.device("cpu")
model_mlp = TabularMLPModel(
    input_dim=n_features,
    n_nodes=n_nodes,
    hidden_dim=512,
    n_blocks=3,
    head_layers=2,
    dropout=0.3,
).to(device)

pos_w = WeightedBCELoss.compute_pos_weight(torch.tensor(y_train_raw)).to(device)
crit = WeightedBCELoss(pos_weight=pos_w)
opt = torch.optim.AdamW(model_mlp.parameters(), lr=1e-4, weight_decay=1e-5)

tr_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train_raw))
te_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test_raw))
tr_ldr = DataLoader(tr_ds, batch_size=64, shuffle=True)
te_ldr = DataLoader(te_ds, batch_size=64, shuffle=False)

epochs = 30
for ep in range(epochs):
    model_mlp.train()
    total_loss = 0.0
    for bx, by in tr_ldr:
        bx, by = bx.to(device), by.to(device)
        preds = model_mlp(bx)
        loss = crit(preds, by)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    if (ep + 1) % 10 == 0:
        print(f"  epoch {ep+1}/{epochs}  loss={total_loss:.2f}")

model_mlp.eval()
all_preds = []
with torch.no_grad():
    for bx, _ in te_ldr:
        all_preds.append(model_mlp(bx).cpu().numpy())
scores_mlp_raw = np.concatenate(all_preds, axis=0)
scores_mlp = reconcile(scores_mlp_raw, hier, strategy="ancestor_max")

f1, au, p, r, thr = compute_metrics(y_test_raw, scores_mlp, eval_mask)
dur = time.time() - t0
results.append(
    {
        "method": "tabular_mlp",
        "dataset": "seq_FUN",
        "micro_f1": float(f1),
        "auprc": float(au),
        "precision": float(p),
        "recall": float(r),
        "threshold": float(thr),
        "duration_s": dur,
    }
)
print(
    f"  F1={f1:.4f}  AUPRC={au:.4f}  P={p:.4f}  R={r:.4f}  "
    f"thr={thr:.2f}  time={dur:.1f}s"
)

save_artefacts(
    f"{base_dir}/tabular_mlp",
    scores_mlp_raw,
    scores_mlp,
    results[-1],
    {
        "method": "tabular_mlp",
        "dataset": "seq_FUN",
        "epochs": epochs,
        "seed": 0,
        "hidden_dim": 512,
        "n_blocks": 3,
        "dropout": 0.3,
    },
)

# ===================================================================
# 3. local (HMCLocalModel)
# ===================================================================
print("\n" + "=" * 60)
print("[3/3] local — HMCLocalModel (per-level MLPs)")
t0 = time.time()


levels_list = [mgr.levels_size[k] for k in sorted(mgr.levels_size)]

model_loc = HMCLocalModel(
    levels_size=levels_list,
    input_size=n_features,
    hidden_dims=[[256, 128], [256, 128], [128, 64], [128, 64], [64, 32], [64, 32]],
    results_path=f"{base_dir}/local",
    num_layers=[2, 2, 2, 2, 2, 2],
    dropouts=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    active_levels=[0, 1, 2, 3, 4, 5],
)

opt_loc = torch.optim.AdamW(model_loc.parameters(), lr=1e-4, weight_decay=1e-5)
crit_loc = torch.nn.BCELoss()

tr_y_local = [[np.stack(yy) for yy in yl] for yl in train.y_local] + [
    [np.stack(yy) for yy in yl] for yl in valid.y_local
]
te_y_local = [[np.stack(yy) for yy in yl] for yl in test.y_local]

tr_ds_loc = list(zip(torch.tensor(X_train), torch.tensor(y_train_raw), tr_y_local))
te_ds_loc = list(zip(torch.tensor(X_test), torch.tensor(y_test_raw), te_y_local))
tr_ldr_loc = DataLoader(tr_ds_loc, batch_size=64, shuffle=True)
te_ldr_loc = DataLoader(te_ds_loc, batch_size=64, shuffle=False)

epochs_loc = 20
for ep in range(epochs_loc):
    model_loc.train()
    total_loss = 0.0
    for bx, _, byl in tr_ldr_loc:
        outputs = model_loc(bx)
        loss = None
        for lvl_idx in range(len(levels_list)):
            if lvl_idx in outputs:
                l = crit_loc(outputs[lvl_idx], byl[lvl_idx].float())
                loss = l if loss is None else loss + l
        if loss is None:
            continue
        opt_loc.zero_grad()
        loss.backward()
        opt_loc.step()
        total_loss += loss.item()
    if (ep + 1) % 5 == 0:
        print(f"  epoch {ep+1}/{epochs_loc}  loss={total_loss:.2f}")

model_loc.eval()
all_preds_loc = []
with torch.no_grad():
    for bx, yg, _ in te_ldr_loc:
        outputs = model_loc(bx)
        for i in range(len(bx)):
            gp = np.zeros(n_nodes, dtype=np.float32)
            for lvl_idx in range(len(levels_list)):
                if lvl_idx in outputs:
                    lp = outputs[lvl_idx][i].detach().numpy()
                    lvl_key = sorted(mgr.levels_size.keys())[lvl_idx]
                    for name, li in mgr.local_nodes_idx[lvl_key].items():
                        if li < len(lp):
                            gp[mgr.nodes_idx[name]] = lp[li]
            all_preds_loc.append(gp)
scores_loc = np.stack(all_preds_loc)

f1, au, p, r, thr = compute_metrics(y_test_raw, scores_loc, eval_mask)
dur = time.time() - t0
results.append(
    {
        "method": "local",
        "dataset": "seq_FUN",
        "micro_f1": float(f1),
        "auprc": float(au),
        "precision": float(p),
        "recall": float(r),
        "threshold": float(thr),
        "duration_s": dur,
    }
)
print(
    f"  F1={f1:.4f}  AUPRC={au:.4f}  P={p:.4f}  R={r:.4f}  "
    f"thr={thr:.2f}  time={dur:.1f}s"
)

save_artefacts(
    f"{base_dir}/local",
    scores_loc,
    scores_loc,
    results[-1],
    {"method": "local", "dataset": "seq_FUN", "epochs": epochs_loc, "seed": 0},
)

# ===================================================================
print("\n" + "=" * 75)
print("FINAL RESULTS — seq_FUN (529 features, 500 nodes, 3919 samples)")
print("=" * 75)
print(
    f"{'Method':<18} {'Micro-F1':>10} {'AUPRC':>10} {'Precision':>10} "
    f"{'Recall':>10} {'Thresh':>8} {'Time':>8}"
)
print("-" * 75)
for r in sorted(results, key=lambda x: -x["micro_f1"]):
    extra = f"  nodes={r.get('nodes_trained', 'all')}" if "nodes_trained" in r else ""
    print(
        f"{r['method']:<18} {r['micro_f1']:>10.4f} {r['auprc']:>10.4f} "
        f"{r['precision']:>10.4f} {r['recall']:>10.4f} "
        f"{r['threshold']:>8.2f} {r['duration_s']:>7.1f}s{extra}"
    )
print("-" * 75)

with open(f"{base_dir}/comparison.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nArtefacts saved to {base_dir}/")
