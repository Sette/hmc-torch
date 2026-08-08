#!/usr/bin/env python3
"""GPU-accelerated HMC benchmark: global vs tabular_mlp vs gbdt_platt.

All GPU methods use seed=42. GBDT runs on CPU (scikit-learn).
Metrics computed with Platt calibration + R-matrix reconciliation.
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

from hmc.datasets.registry import DatasetRegistry

sys.path.insert(0, "src")


SEED = 42
DEVICE = torch.device("cuda")
BATCH = 256
torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_metrics(y_true, y_pred, eval_mask):
    y_t = torch.tensor(y_true).cuda()
    y_p = torch.tensor(y_pred).cuda()
    e_m = torch.tensor(eval_mask).cuda()
    best = (0.0, 0.5, 0.0, 0.0)
    for thr in np.arange(0.02, 0.90, 0.02):
        y_bin = (y_p >= thr).float()
        tp = (y_bin[:, e_m] * y_t[:, e_m]).sum()
        fp = (y_bin[:, e_m] * (1 - y_t[:, e_m])).sum()
        fn = ((1 - y_bin[:, e_m]) * y_t[:, e_m]).sum()
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        if f1 > best[0]:
            best = (f1.item(), thr, p.item(), r.item())
    try:
        auprc = float(
            average_precision_score(
                y_true[:, eval_mask], y_pred[:, eval_mask], average="micro"
            )
        )
    except:
        auprc = 0.0
    return best[0], auprc, best[2], best[3], best[1]


def build_r_matrix(adj):
    r = np.zeros(adj.shape)
    np.fill_diagonal(r, 1)
    g = nx.DiGraph(adj)
    for i in range(len(adj)):
        ancestors = list(nx.descendants(g, i))
        if ancestors:
            r[i, ancestors] = 1
    return torch.tensor(r).transpose(1, 0).unsqueeze(0).cuda()


def load_data(name):
    from hmc.datasets.manager.dataset_manager import initialize_dataset_experiments

    mgr = initialize_dataset_experiments(
        name, device="cpu", dataset_path="./data", dataset_type="arff", is_global=False
    )
    train, valid, test = mgr.get_datasets()
    X_tr = np.concatenate([train.x, valid.x]).astype(np.float32)
    y_tr = np.concatenate([train.y, valid.y]).astype(np.float32)
    X_va = valid.x.astype(np.float32)
    y_va = valid.y.astype(np.float32)
    X_te = test.x.astype(np.float32)
    y_te = test.y.astype(np.float32)
    eval_mask = np.array(mgr.to_eval, dtype=bool)
    terms = train.terms
    adj = mgr.a

    imp = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    all_X = np.concatenate([X_tr, X_te])
    scaler.fit(imp.fit_transform(all_X))
    X_tr = scaler.transform(imp.transform(X_tr)).astype(np.float32)
    X_va = scaler.transform(imp.transform(X_va)).astype(np.float32)
    X_te = scaler.transform(imp.transform(X_te)).astype(np.float32)

    from hmc.data.hierarchy import TreeHierarchy

    branches = [t for t in terms if t != "root"]
    hier = TreeHierarchy.from_fun_cat_terms(branches if branches else terms)
    return X_tr, y_tr, X_va, y_va, X_te, y_te, eval_mask, adj, hier


DATASETS = [
    "cellcycle_FUN",
    "derisi_FUN",
    "eisen_FUN",
    "expr_FUN",
    "gasch1_FUN",
    "gasch2_FUN",
    "seq_FUN",
    "spo_FUN",
]

# Get registry epochs

reg = DatasetRegistry()

all_results = []

for ds_name in DATASETS:
    print(f"\n{'='*65}")
    print(f"  {ds_name}")
    print(f"{'='*65}")
    X_tr, y_tr, X_va, y_va, X_te, y_te, eval_mask, adj, hier = load_data(ds_name)
    n_nodes = y_tr.shape[1]
    n_feat = X_tr.shape[1]
    r_matrix = build_r_matrix(adj)
    data_key = ds_name.replace("_FUN", "")
    rec_ep = reg.all_epochs.get("FUN", {}).get(data_key, 50)
    hidden = reg.hidden_dims.get("FUN", {}).get(data_key, 512)
    epochs_g = min(rec_ep, 50)

    Xt_tr = torch.tensor(X_tr)
    yt_tr = torch.tensor(y_tr)
    Xt_te = torch.tensor(X_te)
    print(
        f"  samples: {X_tr.shape[0]} train | {X_te.shape[0]} test | "
        f"{n_feat} feat | {n_nodes} nodes | eval={eval_mask.sum()} | "
        f"ep={epochs_g} hidden={hidden}"
    )

    # ===== 1. GLOBAL (MLP + R-matrix) GPU =====
    from hmc.models.global_classifier.constraint.model import ConstrainedModel
    from hmc.models.hierarchical.postprocess import PlattCalibrator, reconcile

    torch.manual_seed(SEED)
    t0 = time.time()
    model_g = ConstrainedModel(
        input_dim=n_feat,
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
        r_matrix=r_matrix,
        baseline_model=False,
    ).to(DEVICE)

    opt_g = torch.optim.AdamW(model_g.parameters(), lr=1e-4, weight_decay=1e-5)
    tr_ldr_g = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=BATCH, shuffle=True)

    model_g.train()
    for ep in range(epochs_g):
        for bx, by in tr_ldr_g:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            loss = torch.nn.functional.binary_cross_entropy(model_g(bx), by)
            opt_g.zero_grad()
            loss.backward()
            opt_g.step()
        if (ep + 1) % max(1, epochs_g // 5) == 0:
            print(f"  [global] ep {ep+1}/{epochs_g}")

    model_g.eval()
    # Batched inference to avoid OOM from R-matrix expansion
    s_g_list = []
    te_ldr_g = DataLoader(
        TensorDataset(torch.tensor(X_te)), batch_size=64, shuffle=False
    )
    with torch.no_grad():
        for (bx,) in te_ldr_g:
            s_g_list.append(model_g(bx.to(DEVICE)).cpu().numpy())
    s_g = np.concatenate(s_g_list, axis=0)
    f1_g, au_g, p_g, r_g, th_g = compute_metrics(y_te, s_g, eval_mask)
    dur_g = time.time() - t0
    all_results.append(
        {
            "method": "global",
            "dataset": ds_name,
            "micro_f1": f1_g,
            "auprc": au_g,
            "precision": p_g,
            "recall": r_g,
            "threshold": th_g,
            "duration_s": dur_g,
            "epochs": epochs_g,
        }
    )
    print(
        f"  => global        F1={f1_g:.4f}  AUPRC={au_g:.4f}  "
        f"P={p_g:.4f}  R={r_g:.4f}  t={dur_g:.1f}s"
    )

    # Save
    od = f"./output/gpu/{ds_name}/global"
    os.makedirs(od, exist_ok=True)
    np.savez_compressed(f"{od}/scores.npz", scores=s_g)

    # ===== 2. TABULAR_MLP (Residual + SigmoidHead) GPU =====
    from hmc.models.hierarchical.losses import WeightedBCELoss
    from hmc.models.tabular.mlp import TabularMLPModel

    torch.manual_seed(SEED)
    t0 = time.time()
    mlp = TabularMLPModel(
        input_dim=n_feat,
        n_nodes=n_nodes,
        hidden_dim=min(512, max(128, n_feat)),
        n_blocks=3,
        head_layers=2,
        dropout=0.3,
    ).to(DEVICE)

    pos_w = WeightedBCELoss.compute_pos_weight(yt_tr).to(DEVICE)
    crit_m = WeightedBCELoss(pos_weight=pos_w)
    opt_m = torch.optim.AdamW(mlp.parameters(), lr=1e-4, weight_decay=1e-5)
    tr_ldr_m = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=BATCH, shuffle=True)

    epochs_m = max(epochs_g, 30)
    mlp.train()
    for ep in range(epochs_m):
        for bx, by in tr_ldr_m:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            loss = crit_m(mlp(bx), by)
            opt_m.zero_grad()
            loss.backward()
            opt_m.step()
        if (ep + 1) % max(1, epochs_m // 5) == 0:
            print(f"  [mlp] ep {ep+1}/{epochs_m}")

    mlp.eval()
    with torch.no_grad():
        s_m = mlp(Xt_te.to(DEVICE)).cpu().numpy()
    s_m = reconcile(s_m, hier)
    f1_m, au_m, p_m, r_m, th_m = compute_metrics(y_te, s_m, eval_mask)
    dur_m = time.time() - t0
    all_results.append(
        {
            "method": "tabular_mlp",
            "dataset": ds_name,
            "micro_f1": f1_m,
            "auprc": au_m,
            "precision": p_m,
            "recall": r_m,
            "threshold": th_m,
            "duration_s": dur_m,
            "epochs": epochs_m,
        }
    )
    print(
        f"  => tabular_mlp   F1={f1_m:.4f}  AUPRC={au_m:.4f}  "
        f"P={p_m:.4f}  R={r_m:.4f}  t={dur_m:.1f}s"
    )

    od = f"./output/gpu/{ds_name}/tabular_mlp"
    os.makedirs(od, exist_ok=True)
    np.savez_compressed(f"{od}/scores.npz", scores=s_m)

    # ===== 3. GBDT + Platt + Reconcile (CPU) =====
    from hmc.models.tabular.gbdt_ovr import GBDTOvRClassifier

    np.random.seed(SEED)
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
    s_va = gbdt.predict_proba(X_va)
    s_te = gbdt.predict_proba(X_te)

    cal = PlattCalibrator()
    cal.fit(s_va, y_va, eval_mask=eval_mask)
    s_te_cal = cal.calibrate(s_te)
    s_b = reconcile(s_te_cal, hier)

    f1_b, au_b, p_b, r_b, th_b = compute_metrics(y_te, s_b, eval_mask)
    dur_b = time.time() - t0
    all_results.append(
        {
            "method": "gbdt_platt_r",
            "dataset": ds_name,
            "micro_f1": f1_b,
            "auprc": au_b,
            "precision": p_b,
            "recall": r_b,
            "threshold": th_b,
            "duration_s": dur_b,
            "nodes": int(trainable.sum()),
        }
    )
    print(
        f"  => gbdt+platt+r  F1={f1_b:.4f}  AUPRC={au_b:.4f}  "
        f"P={p_b:.4f}  R={r_b:.4f}  t={dur_b:.1f}s  n={trainable.sum()}"
    )

    od = f"./output/gpu/{ds_name}/gbdt_platt"
    os.makedirs(od, exist_ok=True)
    np.savez_compressed(f"{od}/scores.npz", scores=s_b)

# ===================================================================
print("\n\n" + "=" * 75)
print("GPU BENCHMARK — 8 FUN Datasets")
print("=" * 75)

for ds_name in DATASETS:
    ds_res = [r for r in all_results if r["dataset"] == ds_name]
    if not ds_res:
        continue
    best_f1 = max(r["micro_f1"] for r in ds_res)
    print(f"\n{ds_name}:")
    for r in sorted(ds_res, key=lambda x: -x["micro_f1"]):
        mark = " <<<" if abs(r["micro_f1"] - best_f1) < 0.001 else ""
        extra = (
            f" ep={r.get('epochs', '?')}"
            if "epochs" in r
            else f" n={r.get('nodes', '?')}"
        )
        print(
            f"  {r['method']:<16} F1={r['micro_f1']:.4f}  "
            f"AUPRC={r['auprc']:.4f}  P={r['precision']:.4f}  "
            f"R={r['recall']:.4f}  thr={r['threshold']:.2f}  "
            f"t={r['duration_s']:.1f}s{extra}{mark}"
        )

print(f"\n{'='*75}")
print("SUMMARY (avg across 8 datasets)")
for method in ["global", "gbdt_platt_r", "tabular_mlp"]:
    m_res = [r for r in all_results if r["method"] == method]
    if not m_res:
        continue
    avg_f1 = np.mean([r["micro_f1"] for r in m_res])
    avg_au = np.mean([r["auprc"] for r in m_res])
    avg_t = np.mean([r["duration_s"] for r in m_res])
    wins = sum(
        1
        for ds in DATASETS
        for r in all_results
        if r["dataset"] == ds
        and r["method"] == method
        and abs(
            r["micro_f1"]
            - max(rr["micro_f1"] for rr in all_results if rr["dataset"] == ds)
        )
        < 0.001
    )
    print(
        f"  {method:<16} F1={avg_f1:.4f}  AUPRC={avg_au:.4f}  "
        f"t={avg_t:.1f}s  wins={wins}/8"
    )

os.makedirs("./output/gpu", exist_ok=True)
with open("./output/gpu/comparison.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved to ./output/gpu/comparison.json")
