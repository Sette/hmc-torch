#!/usr/bin/env python3
"""Phase 1: Multi-seed + ablation experiments for the paper.

Runs:
  1. ArXiv global (frozen SPECTER2) — 3 seeds
  2. WOS global (frozen SPECTER2) — 3 seeds
  3. R-matrix ablation (on/off) on ArXiv, WOS, cellcycle_FUN, seq_FUN
  4. Saves all with mean ± std
"""

import json, os, sys, time
import numpy as np
import torch
import networkx as nx

sys.path.insert(0, "src")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 128

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


def build_r_matrix(adj, device=DEVICE):
    r = np.zeros(adj.shape); np.fill_diagonal(r, 1)
    g = nx.DiGraph(adj)
    for i in range(len(adj)):
        ancestors = list(nx.descendants(g, i))
        if ancestors: r[i, ancestors] = 1
    return torch.tensor(r).transpose(1, 0).unsqueeze(0).to(device)


def train_global(X_tr, y_tr, X_te, y_te, eval_mask, adj, n_nodes,
                 hidden=512, epochs=30, use_r=True, seed=42):
    """Train one global model. Returns (f1, auprc, p, r, thr, duration, scores)."""
    torch.manual_seed(seed); np.random.seed(seed)
    n_feat = X_tr.shape[1]
    if use_r:
        r_matrix = build_r_matrix(adj)
    else:
        r_matrix = torch.eye(n_nodes).unsqueeze(0).to(DEVICE)

    from hmc.models.global_classifier.constraint.model import ConstrainedModel
    t0 = time.time()
    model = ConstrainedModel(
        input_dim=n_feat, hidden_dim=min(hidden, 2048), output_dim=n_nodes,
        hyperparams={"batch_size": BATCH, "num_layers": 3, "dropout": 0.3,
                     "non_lin": "relu", "hidden_dim": min(hidden, 2048),
                     "lr": 1e-4, "weight_decay": 1e-5},
        r_matrix=r_matrix, baseline_model=False,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    Xt_tr = torch.tensor(X_tr); yt_tr = torch.tensor(y_tr)
    tr_ldr = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=BATCH, shuffle=True)

    model.train()
    for ep in range(epochs):
        for bx, by in tr_ldr:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            loss = torch.nn.functional.binary_cross_entropy(model(bx), by)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    preds = []
    te_ldr = DataLoader(TensorDataset(torch.tensor(X_te)), batch_size=64, shuffle=False)
    with torch.no_grad():
        for (bx,) in te_ldr:
            preds.append(model(bx.to(DEVICE)).cpu().numpy())
    scores = np.concatenate(preds, axis=0)
    f1, au, p, r, thr = compute_metrics(y_te, scores, eval_mask)
    return f1, au, p, r, thr, time.time() - t0, scores


# ===================================================================
# 1. ArXiv — 3 seeds (frozen SPECTER2)
# ===================================================================
print("=" * 60)
print("PHASE 1A: ArXiv — 3 seeds (frozen SPECTER2 + R-matrix)")
print("=" * 60)

from hmc.datasets.dataset_manager import initialize_dataset_experiments
print("Loading ArXiv...")
mgr_a = initialize_dataset_experiments("arxiv", device="cpu", dataset_path="./data",
                                        is_global=True, model_cache_dir="./models")
tr_a, va_a, te_a = mgr_a.get_datasets()
Xa_tr = np.concatenate([tr_a.x, va_a.x]).astype(np.float32)
ya_tr = np.concatenate([tr_a.y, va_a.y]).astype(np.float32)
Xa_te = te_a.x.astype(np.float32); ya_te = te_a.y.astype(np.float32)
eval_a = np.array(mgr_a.to_eval, dtype=bool)
adj_a = mgr_a.a; n_a = mgr_a.output_dim
scaler_a = StandardScaler()
Xa_tr = scaler_a.fit_transform(Xa_tr).astype(np.float32)
Xa_te = scaler_a.transform(Xa_te).astype(np.float32)
print(f"  Train: {Xa_tr.shape}  Test: {Xa_te.shape}  Nodes: {n_a}  Eval: {eval_a.sum()}")

arxiv_seeds = {}
for s in [0, 42, 123]:
    print(f"  Seed {s}...", end=" ", flush=True)
    f1, au, p, r, thr, dur, scores = train_global(
        Xa_tr, ya_tr, Xa_te, ya_te, eval_a, adj_a, n_a, hidden=512, epochs=50, seed=s)
    arxiv_seeds[s] = {"f1": f1, "auprc": au, "p": p, "r": r, "thr": thr, "time": dur}
    print(f"F1={f1:.4f} AUPRC={au:.4f} t={dur:.1f}s")

f1s = [v["f1"] for v in arxiv_seeds.values()]
aus = [v["auprc"] for v in arxiv_seeds.values()]
print(f"  ArXiv: F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}  AUPRC={np.mean(aus):.4f}±{np.std(aus):.4f}")

# ===================================================================
# 2. WOS — 3 seeds (frozen SPECTER2)
# ===================================================================
print("\n" + "=" * 60)
print("PHASE 1B: WOS — 3 seeds (frozen SPECTER2 + R-matrix)")
print("=" * 60)

mgr_w = initialize_dataset_experiments("wos", device="cpu", dataset_path="./data",
                                        is_global=True, model_cache_dir="./models")
tr_w, va_w, te_w = mgr_w.get_datasets()
Xw_tr = np.concatenate([tr_w.x, va_w.x]).astype(np.float32)
yw_tr = np.concatenate([tr_w.y, va_w.y]).astype(np.float32)
Xw_te = te_w.x.astype(np.float32); yw_te = te_w.y.astype(np.float32)
eval_w = np.array(mgr_w.to_eval, dtype=bool)
adj_w = mgr_w.a; n_w = mgr_w.output_dim
scaler_w = StandardScaler()
Xw_tr = scaler_w.fit_transform(Xw_tr).astype(np.float32)
Xw_te = scaler_w.transform(Xw_te).astype(np.float32)
print(f"  Train: {Xw_tr.shape}  Test: {Xw_te.shape}  Nodes: {n_w}  Eval: {eval_w.sum()}")

wos_seeds = {}
for s in [0, 42, 123]:
    print(f"  Seed {s}...", end=" ", flush=True)
    f1, au, p, r, thr, dur, scores = train_global(
        Xw_tr, yw_tr, Xw_te, yw_te, eval_w, adj_w, n_w, hidden=512, epochs=50, seed=s)
    wos_seeds[s] = {"f1": f1, "auprc": au, "p": p, "r": r, "thr": thr, "time": dur}
    print(f"F1={f1:.4f} AUPRC={au:.4f} t={dur:.1f}s")

f1s_w = [v["f1"] for v in wos_seeds.values()]
aus_w = [v["auprc"] for v in wos_seeds.values()]
print(f"  WOS: F1={np.mean(f1s_w):.4f}±{np.std(f1s_w):.4f}  AUPRC={np.mean(aus_w):.4f}±{np.std(aus_w):.4f}")

# ===================================================================
# 3. R-matrix ablation — ArXiv, WOS, cellcycle_FUN, seq_FUN
# ===================================================================
print("\n" + "=" * 60)
print("PHASE 1C: R-matrix ablation (on vs off)")
print("=" * 60)

ablation_results = {}

# ArXiv ablation
print("\nArXiv ablation:")
for use_r in [True, False]:
    label = "R=on" if use_r else "R=off"
    f1, au, p, r, thr, dur, _ = train_global(
        Xa_tr, ya_tr, Xa_te, ya_te, eval_a, adj_a, n_a, hidden=512, epochs=50, use_r=use_r, seed=42)
    ablation_results[f"arxiv_{label}"] = {"f1": f1, "auprc": au, "p": p, "r": r, "thr": thr, "time": dur}
    print(f"  {label}: F1={f1:.4f} AUPRC={au:.4f} P={p:.4f} R={r:.4f} t={dur:.1f}s")

# WOS ablation
print("\nWOS ablation:")
for use_r in [True, False]:
    label = "R=on" if use_r else "R=off"
    f1, au, p, r, thr, dur, _ = train_global(
        Xw_tr, yw_tr, Xw_te, yw_te, eval_w, adj_w, n_w, hidden=512, epochs=50, use_r=use_r, seed=42)
    ablation_results[f"wos_{label}"] = {"f1": f1, "auprc": au, "p": p, "r": r, "thr": thr, "time": dur}
    print(f"  {label}: F1={f1:.4f} AUPRC={au:.4f} P={p:.4f} R={r:.4f} t={dur:.1f}s")

# FunCat ablation (cellcycle_FUN + seq_FUN)
for ds_name in ["cellcycle_FUN", "seq_FUN"]:
    print(f"\n{ds_name} ablation:")
    mgr_f = initialize_dataset_experiments(ds_name, device="cpu", dataset_path="./data",
                                            dataset_type="arff", is_global=False)
    tr_f, va_f, te_f = mgr_f.get_datasets()
    Xf_tr = np.concatenate([tr_f.x, va_f.x]).astype(np.float32)
    yf_tr = np.concatenate([tr_f.y, va_f.y]).astype(np.float32)
    Xf_te = te_f.x.astype(np.float32); yf_te = te_f.y.astype(np.float32)
    eval_f = np.array(mgr_f.to_eval, dtype=bool)
    adj_f = mgr_f.a; n_f = len(mgr_f.nodes_idx)

    imp_f = SimpleImputer(strategy="mean"); scaler_f = StandardScaler()
    Xf_tr = scaler_f.fit_transform(imp_f.fit_transform(Xf_tr)).astype(np.float32)
    Xf_te = scaler_f.transform(imp_f.transform(Xf_te)).astype(np.float32)

    # Use registry hidden dim
    from hmc.datasets.registry import DatasetRegistry
    reg = DatasetRegistry()
    dk = ds_name.replace("_FUN", "")
    hidden_f = reg.hidden_dims.get("FUN", {}).get(dk, 512)
    rec_ep = reg.all_epochs.get("FUN", {}).get(dk, 50)
    epochs_f = min(rec_ep, 50)

    for use_r in [True, False]:
        label = "R=on" if use_r else "R=off"
        f1, au, p, r, thr, dur, _ = train_global(
            Xf_tr, yf_tr, Xf_te, yf_te, eval_f, adj_f, n_f,
            hidden=hidden_f, epochs=epochs_f, use_r=use_r, seed=42)
        ablation_results[f"{ds_name}_{label}"] = {"f1": f1, "auprc": au, "p": p, "r": r, "thr": thr, "time": dur}
        print(f"  {label}: F1={f1:.4f} AUPRC={au:.4f} P={p:.4f} R={r:.4f} t={dur:.1f}s")

# ===================================================================
# Save everything
# ===================================================================
from sklearn.impute import SimpleImputer

os.makedirs("./output/phase1", exist_ok=True)
results = {
    "arxiv_3seeds": {str(k): v for k, v in arxiv_seeds.items()},
    "wos_3seeds": {str(k): v for k, v in wos_seeds.items()},
    "ablation": ablation_results,
    "arxiv_summary": f"F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}",
    "wos_summary": f"F1={np.mean(f1s_w):.4f}±{np.std(f1s_w):.4f}",
}
with open("./output/phase1/results.json", "w") as f:
    json.dump(results, f, indent=2)

# Final report
print("\n\n" + "=" * 70)
print("PHASE 1 RESULTS")
print("=" * 70)
print(f"\nArXiv (frozen SPECTER2 + R):")
print(f"  F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"  AUPRC = {np.mean(aus):.4f} ± {np.std(aus):.4f}")
for s in [0, 42, 123]:
    v = arxiv_seeds[s]
    print(f"    seed {s}: F1={v['f1']:.4f} AUPRC={v['auprc']:.4f} t={v['time']:.1f}s")

print(f"\nWOS (frozen SPECTER2 + R):")
print(f"  F1 = {np.mean(f1s_w):.4f} ± {np.std(f1s_w):.4f}")
print(f"  AUPRC = {np.mean(aus_w):.4f} ± {np.std(aus_w):.4f}")
for s in [0, 42, 123]:
    v = wos_seeds[s]
    print(f"    seed {s}: F1={v['f1']:.4f} AUPRC={v['auprc']:.4f} t={v['time']:.1f}s")

print(f"\nR-Matrix Ablation:")
for ds in ["arxiv", "wos", "cellcycle_FUN", "seq_FUN"]:
    on = ablation_results[f"{ds}_R=on"]
    off = ablation_results[f"{ds}_R=off"]
    delta = on["f1"] - off["f1"]
    delta_au = on["auprc"] - off["auprc"]
    print(f"  {ds:<20} R=on F1={on['f1']:.4f}  R=off F1={off['f1']:.4f}  "
          f"ΔF1={delta:+.4f}  ΔAUPRC={delta_au:+.4f}")

print(f"\nSaved to ./output/phase1/results.json")
