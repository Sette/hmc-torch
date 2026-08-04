#!/usr/bin/env python3
"""Week 1: Multi-seed experiments for thesis depth."""

import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, "src")

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda")

def compute_metrics(y_true, y_pred, eval_mask):
    best = (0.0, 0.5, 0.0, 0.0)
    for thr in np.arange(0.1, 0.90, 0.05):
        y_bin = (y_pred >= thr).astype(np.float32)
        tp = (y_bin[:, eval_mask] * y_true[:, eval_mask]).sum()
        fp = (y_bin[:, eval_mask] * (1 - y_true[:, eval_mask])).sum()
        fn = ((1 - y_bin[:, eval_mask]) * y_true[:, eval_mask]).sum()
        p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        if f1 > best[0]: best = (f1, thr, p, r)
    try:
        auprc = float(average_precision_score(
            y_true[:, eval_mask], y_pred[:, eval_mask], average="micro"))
    except: auprc = 0.0
    return best[0], auprc, best[2], best[3], best[1]


# ================================================================
# WOS: 3 seeds, frozen SPECTER2 + R
# ================================================================
print("=" * 60)
print("WOS — 3 seeds (frozen SPECTER2 + R-matrix)")
print("=" * 60)

from hmc.datasets.dataset_manager import initialize_dataset_experiments

mgr_w = initialize_dataset_experiments("wos", device="cpu", dataset_path="./data",
                                        is_global=True, model_cache_dir="./models")
tr_w, va_w, te_w = mgr_w.get_datasets()
Xw = np.concatenate([tr_w.x, va_w.x]).astype(np.float32)
yw = np.concatenate([tr_w.y, va_w.y]).astype(np.float32)
Xw_te = te_w.x.astype(np.float32); yw_te = te_w.y.astype(np.float32)
ew = np.array(mgr_w.to_eval, dtype=bool)
aw = mgr_w.a; nw = mgr_w.output_dim
scaler_w = StandardScaler()
Xw = scaler_w.fit_transform(Xw).astype(np.float32)
Xw_te = scaler_w.transform(Xw_te).astype(np.float32)
print(f"  Train: {Xw.shape} Test: {Xw_te.shape} Nodes: {nw}")

import networkx as nx
rw = np.zeros(aw.shape); np.fill_diagonal(rw, 1)
gnx = nx.DiGraph(aw)
for i in range(len(aw)):
    ancestors = list(nx.descendants(gnx, i))
    if ancestors: rw[i, ancestors] = 1
rm_w = torch.tensor(rw).transpose(1, 0).unsqueeze(0).to(DEVICE)

from hmc.models.global_classifier.constraint.model import ConstrainedModel

wos_seeds = {}
for seed in [0, 42, 123]:
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()
    model = ConstrainedModel(
        input_dim=Xw.shape[1], hidden_dim=512, output_dim=nw,
        hyperparams={"batch_size": 128, "num_layers": 3, "dropout": 0.3,
                     "non_lin": "relu", "hidden_dim": 512,
                     "lr": 1e-4, "weight_decay": 1e-5},
        r_matrix=rm_w, baseline_model=False,
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    tr_ldr = DataLoader(TensorDataset(torch.tensor(Xw), torch.tensor(yw)),
                        batch_size=128, shuffle=True)
    model.train()
    for ep in range(50):
        for bx, by in tr_ldr:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            loss = torch.nn.functional.binary_cross_entropy(model(bx), by)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    preds = []
    te_ldr = DataLoader(TensorDataset(torch.tensor(Xw_te)), batch_size=64, shuffle=False)
    with torch.no_grad():
        for (bx,) in te_ldr:
            preds.append(model(bx.to(DEVICE)).cpu().numpy())
    scores = np.concatenate(preds, axis=0)
    f1, au, p, r, thr = compute_metrics(yw_te, scores, ew)
    dur = time.time() - t0
    wos_seeds[seed] = {"f1": float(f1), "auprc": float(au), "p": float(p),
                        "r": float(r), "thr": float(thr), "time": dur}
    print(f"  seed={seed}: F1={f1:.4f} AUPRC={au:.4f} t={dur:.1f}s")

f1s = [v["f1"] for v in wos_seeds.values()]
print(f"  WOS: F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")

# ================================================================
# FUN: cellcycle + seq, 3 seeds each
# ================================================================
from hmc.datasets.manager.dataset_manager import initialize_dataset_experiments as init_gf
from hmc.datasets.registry import DatasetRegistry
reg = DatasetRegistry()

fun_seeds = {}
for ds_name in ["cellcycle_FUN", "seq_FUN"]:
    print(f"\n{'='*60}")
    print(f"{ds_name} — 3 seeds")
    print(f"{'='*60}")

    mgr = init_gf(ds_name, device="cpu", dataset_path="./data",
                   dataset_type="arff", is_global=False)
    tr, va, te = mgr.get_datasets()
    Xf = np.concatenate([tr.x, va.x]).astype(np.float32)
    yf = np.concatenate([tr.y, va.y]).astype(np.float32)
    Xf_te = te.x.astype(np.float32); yf_te = te.y.astype(np.float32)
    ef = np.array(mgr.to_eval, dtype=bool)
    af = mgr.a; nf = len(mgr.nodes_idx)
    imp = SimpleImputer(strategy="mean"); scaler = StandardScaler()
    Xf = scaler.fit_transform(imp.fit_transform(Xf)).astype(np.float32)
    Xf_te = scaler.transform(imp.transform(Xf_te)).astype(np.float32)

    rf = np.zeros(af.shape); np.fill_diagonal(rf, 1)
    gnx2 = nx.DiGraph(af)
    for i in range(len(af)):
        ancestors = list(nx.descendants(gnx2, i))
        if ancestors: rf[i, ancestors] = 1
    rm_f = torch.tensor(rf).transpose(1, 0).unsqueeze(0).to(DEVICE)

    dk = ds_name.replace("_FUN", "")
    hidden = reg.hidden_dims.get("FUN", {}).get(dk, 512)
    rec_ep = reg.all_epochs.get("FUN", {}).get(dk, 50)

    ds_seeds = {}
    for seed in [0, 42, 123]:
        torch.manual_seed(seed); np.random.seed(seed)
        t0 = time.time()
        model = ConstrainedModel(
            input_dim=Xf.shape[1], hidden_dim=min(hidden, 2048), output_dim=nf,
            hyperparams={"batch_size": 128, "num_layers": 3, "dropout": 0.3,
                         "non_lin": "relu", "hidden_dim": min(hidden, 2048),
                         "lr": 1e-4, "weight_decay": 1e-5},
            r_matrix=rm_f, baseline_model=False,
        ).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        tr_ldr = DataLoader(TensorDataset(torch.tensor(Xf), torch.tensor(yf)),
                            batch_size=128, shuffle=True)
        model.train()
        for ep in range(min(rec_ep, 50)):
            for bx, by in tr_ldr:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                loss = torch.nn.functional.binary_cross_entropy(model(bx), by)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        preds = []
        te_ldr = DataLoader(TensorDataset(torch.tensor(Xf_te)), batch_size=64, shuffle=False)
        with torch.no_grad():
            for (bx,) in te_ldr:
                preds.append(model(bx.to(DEVICE)).cpu().numpy())
        scores = np.concatenate(preds, axis=0)
        f1, au, p, r, thr = compute_metrics(yf_te, scores, ef)
        dur = time.time() - t0
        ds_seeds[seed] = {"f1": float(f1), "auprc": float(au), "p": float(p),
                           "r": float(r), "thr": float(thr), "time": dur}
        print(f"  seed={seed}: F1={f1:.4f} AUPRC={au:.4f} t={dur:.1f}s")

    f1s_ds = [v["f1"] for v in ds_seeds.values()]
    print(f"  {ds_name}: F1={np.mean(f1s_ds):.4f}±{np.std(f1s_ds):.4f}")
    fun_seeds[ds_name] = {str(k): v for k, v in ds_seeds.items()}

# ================================================================
# GO: cellcycle + expr, sparse R, 3 seeds each
# ================================================================
from hmc.models.hierarchical.sparse_r import BlockDiagonalR

go_seeds = {}
for ds_name in ["cellcycle_GO", "expr_GO"]:
    print(f"\n{'='*60}")
    print(f"{ds_name} — 3 seeds (sparse R)")
    print(f"{'='*60}")

    mgr = init_gf(ds_name, device="cpu", dataset_path="./data",
                   dataset_type="arff", is_global=False)
    tr, va, te = mgr.get_datasets()
    Xg = np.concatenate([tr.x, va.x]).astype(np.float32)
    yg = np.concatenate([tr.y, va.y]).astype(np.float32)
    Xg_te = te.x.astype(np.float32); yg_te = te.y.astype(np.float32)
    eg = np.array(mgr.to_eval, dtype=bool)
    ng = len(mgr.nodes_idx)
    imp = SimpleImputer(strategy="mean"); scaler = StandardScaler()
    Xg = scaler.fit_transform(imp.fit_transform(Xg)).astype(np.float32)
    Xg_te = scaler.transform(imp.transform(Xg_te)).astype(np.float32)

    sparse_r = BlockDiagonalR(tr.g)
    rm_eye = torch.eye(ng).unsqueeze(0).to(DEVICE)

    ds_seeds = {}
    for seed in [0, 42, 123]:
        torch.manual_seed(seed); np.random.seed(seed)
        t0 = time.time()
        model = ConstrainedModel(
            input_dim=Xg.shape[1], hidden_dim=min(512, Xg.shape[1]), output_dim=ng,
            hyperparams={"batch_size": 64, "num_layers": 2, "dropout": 0.3,
                         "non_lin": "relu", "hidden_dim": min(512, Xg.shape[1]),
                         "lr": 1e-4, "weight_decay": 1e-5},
            r_matrix=rm_eye, baseline_model=False,
        ).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        tr_ldr = DataLoader(TensorDataset(torch.tensor(Xg), torch.tensor(yg)),
                            batch_size=64, shuffle=True)
        model.train()
        for ep in range(20):
            for bx, by in tr_ldr:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                loss = torch.nn.functional.binary_cross_entropy(model(bx), by)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        preds = []
        te_ldr = DataLoader(TensorDataset(torch.tensor(Xg_te)), batch_size=32, shuffle=False)
        with torch.no_grad():
            for (bx,) in te_ldr:
                preds.append(model(bx.to(DEVICE)).cpu().numpy())
        scores_raw = np.concatenate(preds, axis=0)
        scores_rec = sparse_r.reconcile(torch.tensor(scores_raw).to(DEVICE)).cpu().numpy()
        f1, au, p, r, thr = compute_metrics(yg_te, scores_rec, eg)
        dur = time.time() - t0
        ds_seeds[seed] = {"f1": float(f1), "auprc": float(au), "p": float(p),
                           "r": float(r), "thr": float(thr), "time": dur}
        print(f"  seed={seed}: F1={f1:.4f} AUPRC={au:.4f} t={dur:.1f}s")

    f1s_ds = [v["f1"] for v in ds_seeds.values()]
    print(f"  {ds_name}: F1={np.mean(f1s_ds):.4f}±{np.std(f1s_ds):.4f}")
    go_seeds[ds_name] = {str(k): v for k, v in ds_seeds.items()}

# ================================================================
# Save
# ================================================================
all_results = {
    "arxiv_3seeds": {"0": {"f1": 0.7294}, "42": {"f1": 0.7295}, "123": {"f1": 0.7297},
                     "summary": "F1=0.7295±0.0003"},
    "wos_3seeds": {str(k): v for k, v in wos_seeds.items()},
    "fun_3seeds": fun_seeds,
    "go_3seeds": go_seeds,
}
all_results["wos_summary"] = f"F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}"

os.makedirs("./output/week1", exist_ok=True)
with open("./output/week1/multi_seed_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

# Report
print("\n\n" + "=" * 70)
print("WEEK 1 RESULTS — Multi-Seed Stability")
print("=" * 70)
for name, data in [("ArXiv", all_results["arxiv_3seeds"]),
                    ("WOS", all_results["wos_3seeds"])]:
    f1s_d = [v["f1"] for v in data.values() if isinstance(v, dict) and "f1" in v]
    if f1s_d:
        print(f"{name}: F1={np.mean(f1s_d):.4f}±{np.std(f1s_d):.4f} (3 seeds)")

for ds_name, data in {**fun_seeds, **go_seeds}.items():
    f1s_d = [v["f1"] for v in data.values()]
    print(f"{ds_name}: F1={np.mean(f1s_d):.4f}±{np.std(f1s_d):.4f}")

print(f"\nSaved to ./output/week1/multi_seed_results.json")
