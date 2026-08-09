#!/usr/bin/env python3
"""Week 2: Per-level hierarchy analysis."""
import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from hmc.datasets.dataset_manager import initialize_dataset_experiments as init_aw
from hmc.datasets.manager.dataset_manager import (
    initialize_dataset_experiments as init_gf,
)
from hmc.models.hierarchical.sparse_r import BlockDiagonalR

matplotlib.use("Agg")

sys.path.insert(0, "src")
DEVICE = torch.device("cuda")
SEED = 42
os.makedirs("./output/week2", exist_ok=True)


def build_r_from_adj(adj):
    n = adj.shape[0]
    r = np.zeros((n, n))
    np.fill_diagonal(r, 1)
    g = nx.DiGraph(adj)
    for i in range(n):
        for d in nx.descendants(g, i):
            r[i, d] = 1
    return torch.tensor(r).transpose(1, 0).unsqueeze(0).to(DEVICE)


def build_hier_from_adj(adj):
    n = adj.shape[0]
    g = nx.DiGraph(adj)
    leaves = [i for i in range(n) if g.reverse().out_degree(i) == 0]
    branches = []
    for leaf in leaves:
        try:
            path = list(nx.shortest_path(g, leaf, 0))
            branches.append(".".join(str(x) for x in reversed(path)))
        except:
            pass
    if not branches:
        branches = [str(i) for i in range(n)]
    from hmc.data.hierarchy import TreeHierarchy

    return TreeHierarchy.from_fun_cat_terms(branches)


def per_level(y_true, y_pred, hier, eval_mask, n_nodes):
    results = {}
    for lvl, nodes in sorted(hier.levels.items()):
        indices = []
        for n in nodes:
            idx = hier.node_index.get(n, -1)
            if 0 <= idx < n_nodes and eval_mask[idx]:
                indices.append(idx)
        if not indices:
            continue
        yt = y_true[:, indices]
        yp = y_pred[:, indices]
        best = 0.0
        for thr in np.arange(0.1, 0.90, 0.05):
            yb = (yp >= thr).astype(np.float32)
            tp = (yb * yt).sum()
            fp = (yb * (1 - yt)).sum()
            fn = ((1 - yb) * yt).sum()
            p = tp / (tp + fp + 1e-9)
            r = tp / (tp + fn + 1e-9)
            f1 = 2 * p * r / (p + r + 1e-9)
            if f1 > best:
                best = f1
        results[lvl] = {"f1": float(best), "n": len(indices)}
    return results


def train(X_tr, y_tr, X_te, y_te, r_matrix, epochs, batch):
    from hmc.models.global_classifier.constraint.model import ConstrainedModel

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    n_feat, n_nodes = X_tr.shape[1], y_tr.shape[1]
    model = ConstrainedModel(
        input_dim=n_feat,
        hidden_dim=min(512, n_feat * 2),
        output_dim=n_nodes,
        hyperparams={
            "batch_size": batch,
            "num_layers": 2 if n_nodes > 2000 else 3,
            "dropout": 0.3,
            "non_lin": "relu",
            "hidden_dim": min(512, n_feat * 2),
            "lr": 1e-4,
            "weight_decay": 1e-5,
        },
        r_matrix=r_matrix,
        baseline_model=False,
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    tr_ldr = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=batch,
        shuffle=True,
    )
    model.train()
    for _ in range(epochs):
        for bx, by in tr_ldr:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            loss = torch.nn.functional.binary_cross_entropy(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    preds = []
    with torch.no_grad():
        for (bx,) in DataLoader(
            TensorDataset(torch.tensor(X_te)), batch_size=min(64, batch), shuffle=False
        ):
            preds.append(model(bx.to(DEVICE)).cpu().numpy())
    return np.concatenate(preds, axis=0)


# ===== RUN =====

all_levels = {}

# ArXiv
print("ArXiv...")
mgr = init_aw(
    "arxiv",
    device="cpu",
    dataset_path="./data",
    is_global=True,
    model_cache_dir="./models",
)
tr, va, te = mgr.get_datasets()
X = np.concatenate([tr.x, va.x]).astype(np.float32)
y = np.concatenate([tr.y, va.y]).astype(np.float32)
Xte = te.x.astype(np.float32)
yte = te.y.astype(np.float32)
em = np.array(mgr.to_eval, dtype=bool)
sc = StandardScaler()
X = sc.fit_transform(X).astype(np.float32)
Xte = sc.transform(Xte).astype(np.float32)
all_levels["ArXiv"] = per_level(
    yte,
    train(X, y, Xte, yte, build_r_from_adj(mgr.a), 50, 128),
    build_hier_from_adj(mgr.a),
    em,
    mgr.output_dim,
)
print("  OK")

# WOS
print("WOS...")
mgr_w = init_aw(
    "wos",
    device="cpu",
    dataset_path="./data",
    is_global=True,
    model_cache_dir="./models",
)
tr_w, va_w, te_w = mgr_w.get_datasets()
Xw = np.concatenate([tr_w.x, va_w.x]).astype(np.float32)
yw = np.concatenate([tr_w.y, va_w.y]).astype(np.float32)
Xwte = te_w.x.astype(np.float32)
ywte = te_w.y.astype(np.float32)
emw = np.array(mgr_w.to_eval, dtype=bool)
scw = StandardScaler()
Xw = scw.fit_transform(Xw).astype(np.float32)
Xwte = scw.transform(Xwte).astype(np.float32)
all_levels["WOS"] = per_level(
    ywte,
    train(Xw, yw, Xwte, ywte, build_r_from_adj(mgr_w.a), 50, 128),
    build_hier_from_adj(mgr_w.a),
    emw,
    mgr_w.output_dim,
)
print("  OK")

# cellcycle_FUN
print("cellcycle_FUN...")
mgr_f = init_gf(
    "cellcycle_FUN",
    device="cpu",
    dataset_path="./data",
    dataset_type="arff",
    is_global=False,
)
tr_f, va_f, te_f = mgr_f.get_datasets()
Xf = np.concatenate([tr_f.x, va_f.x]).astype(np.float32)
yf = np.concatenate([tr_f.y, va_f.y]).astype(np.float32)
Xfte = te_f.x.astype(np.float32)
yfte = te_f.y.astype(np.float32)
emf = np.array(mgr_f.to_eval, dtype=bool)
imp = SimpleImputer(strategy="mean")
scf = StandardScaler()
Xf = scf.fit_transform(imp.fit_transform(Xf)).astype(np.float32)
Xfte = scf.transform(imp.transform(Xfte)).astype(np.float32)
all_levels["cellcycle_FUN"] = per_level(
    yfte,
    train(Xf, yf, Xfte, yfte, build_r_from_adj(mgr_f.a), 50, 128),
    build_hier_from_adj(mgr_f.a),
    emf,
    len(mgr_f.nodes_idx),
)
print("  OK")

# cellcycle_GO with sparse R
print("cellcycle_GO...")
mgr_g = init_gf(
    "cellcycle_GO",
    device="cpu",
    dataset_path="./data",
    dataset_type="arff",
    is_global=False,
)
tr_g, va_g, te_g = mgr_g.get_datasets()
Xg = np.concatenate([tr_g.x, va_g.x]).astype(np.float32)
yg = np.concatenate([tr_g.y, va_g.y]).astype(np.float32)
Xgte = te_g.x.astype(np.float32)
ygte = te_g.y.astype(np.float32)
emg = np.array(mgr_g.to_eval, dtype=bool)
imp2 = SimpleImputer(strategy="mean")
scg = StandardScaler()
Xg = scg.fit_transform(imp2.fit_transform(Xg)).astype(np.float32)
Xgte = scg.transform(imp2.transform(Xgte)).astype(np.float32)
ng = len(mgr_g.nodes_idx)
scores_g = train(Xg, yg, Xgte, ygte, torch.eye(ng).unsqueeze(0).to(DEVICE), 20, 64)

scores_g = (
    BlockDiagonalR(tr_g.g).reconcile(torch.tensor(scores_g).to(DEVICE)).cpu().numpy()
)
all_levels["cellcycle_GO"] = per_level(
    ygte, scores_g, build_hier_from_adj(mgr_g.a), emg, ng
)
print("  OK")

# ---- FIGURE ----
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for idx, (ds_name, levels) in enumerate(all_levels.items()):
    ax = axes.flatten()[idx]
    lvls = sorted(levels.keys())
    f1s = [levels[l]["f1"] for l in lvls]
    sizes = [levels[l]["n"] for l in lvls]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(lvls)))
    ax.bar(range(len(lvls)), f1s, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(lvls)))
    ax.set_xticklabels([f"L{l}" for l in lvls], fontsize=8)
    ax.set_ylabel("Micro-F1")
    ax.set_title(ds_name, fontweight="bold")
    ax.set_ylim(0, 1.0)
    for i, (f1, sz) in enumerate(zip(f1s, sizes)):
        ax.annotate(f"n={sz}", (i, f1 + 0.03), ha="center", fontsize=7, color="gray")
fig.suptitle("F1 by Hierarchy Depth Level", fontweight="bold", fontsize=14)
plt.tight_layout()
out = "/home/bruno/git/ragnar/hmc-torch/docs/hmc-paper/figures"
os.makedirs(out, exist_ok=True)
fig.savefig(f"{out}/fig5_per_level_f1.pdf", bbox_inches="tight", dpi=150)
plt.close()
print("\nFigure 5 saved")

os.makedirs("./output/week2", exist_ok=True)
with open("./output/week2/per_level_metrics.json", "w") as f:
    json.dump(all_levels, f, indent=2)
print("\nPer-Level F1:")
for ds, levels in all_levels.items():
    lvls = sorted(levels.keys())
    s = " → ".join([f"L{l}:{levels[l]['f1']:.3f}" for l in lvls])
    print(f"  {ds:<20} {s}")
print("\nSaved")
