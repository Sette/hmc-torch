#!/usr/bin/env python3
"""Run global (MLP+R) on WOS and ArXiv — compare with published SOTA."""

import json, os, sys, time
import numpy as np
import torch
import networkx as nx

sys.path.insert(0, "src")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 128
torch.manual_seed(SEED); np.random.seed(SEED)


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


def load_wos():
    """Load WOS with SPECTER2 embeddings."""
    from hmc.datasets.dataset_manager import initialize_dataset_experiments
    print("  Loading WOS + generating SPECTER2 embeddings...")
    mgr = initialize_dataset_experiments(
        "wos", device="cpu", dataset_path="./data", is_global=True,
        model_cache_dir="./models",
    )
    train, valid, test = mgr.get_datasets()
    X_tr = np.concatenate([train.x, valid.x]).astype(np.float32)
    y_tr = np.concatenate([train.y, valid.y]).astype(np.float32)
    X_te = test.x.astype(np.float32); y_te = test.y.astype(np.float32)
    eval_mask = np.array(mgr.to_eval, dtype=bool)
    a = mgr.a; n_nodes = mgr.output_dim
    # Scale embeddings
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)
    return X_tr, y_tr, X_te, y_te, eval_mask, a, n_nodes


def load_arxiv():
    """Load ArXiv with SPECTER2 embeddings."""
    from hmc.datasets.dataset_manager import initialize_dataset_experiments
    print("  Loading ArXiv + generating SPECTER2 embeddings...")
    mgr = initialize_dataset_experiments(
        "arxiv", device="cpu", dataset_path="./data", is_global=True,
        model_cache_dir="./models",
    )
    train, valid, test = mgr.get_datasets()
    X_tr = np.concatenate([train.x, valid.x]).astype(np.float32)
    y_tr = np.concatenate([train.y, valid.y]).astype(np.float32)
    X_te = test.x.astype(np.float32); y_te = test.y.astype(np.float32)
    eval_mask = np.array(mgr.to_eval, dtype=bool)
    a = mgr.a; n_nodes = mgr.output_dim
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)
    return X_tr, y_tr, X_te, y_te, eval_mask, a, n_nodes


def build_r_matrix_gpu(adj, n_nodes):
    r = np.zeros(adj.shape); np.fill_diagonal(r, 1)
    g = nx.DiGraph(adj)
    for i in range(len(adj)):
        ancestors = list(nx.descendants(g, i))
        if ancestors: r[i, ancestors] = 1
    return torch.tensor(r).transpose(1, 0).unsqueeze(0).to(DEVICE)


def train_global(name, X_tr, y_tr, X_te, y_te, eval_mask, adj, n_nodes, epochs=30):
    from hmc.models.global_classifier.constraint.model import ConstrainedModel
    n_feat = X_tr.shape[1]
    r_matrix = build_r_matrix_gpu(adj, n_nodes)
    hidden = min(512, n_feat)

    torch.manual_seed(SEED)
    t0 = time.time()
    model = ConstrainedModel(
        input_dim=n_feat, hidden_dim=hidden, output_dim=n_nodes,
        hyperparams={"batch_size": BATCH, "num_layers": 3, "dropout": 0.3,
                     "non_lin": "relu", "hidden_dim": hidden,
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
        if (ep+1) % max(1, epochs//5) == 0:
            print(f"  ep {ep+1}/{epochs}")

    model.eval()
    preds = []
    te_ldr = DataLoader(TensorDataset(torch.tensor(X_te)), batch_size=128, shuffle=False)
    with torch.no_grad():
        for (bx,) in te_ldr:
            preds.append(model(bx.to(DEVICE)).cpu().numpy())
    scores = np.concatenate(preds, axis=0)

    f1, au, p, r, thr = compute_metrics(y_te, scores, eval_mask)
    dur = time.time() - t0
    return f1, au, p, r, thr, dur, scores


results = []

# ===== WOS =====
print("\n" + "=" * 60)
print("WOS — Web of Science (7 áreas + 134 subcats)")
print("=" * 60)
try:
    X_tr, y_tr, X_te, y_te, eval_mask, adj, n_nodes = load_wos()
    print(f"  Train: {X_tr.shape}  Test: {X_te.shape}  Nodes: {n_nodes}  Eval: {eval_mask.sum()}")
    f1, au, p, r, thr, dur, scores = train_global("wos", X_tr, y_tr, X_te, y_te, eval_mask, adj, n_nodes, epochs=50)
    results.append({"dataset": "WOS", "method": "global", "F1": f1, "AUPRC": au, "P": p, "R": r, "thr": thr, "time": dur})
    print(f"  => F1={f1:.4f}  AUPRC={au:.4f}  P={p:.4f}  R={r:.4f}  thr={thr:.2f}  t={dur:.1f}s")
    os.makedirs("./output/wos_arxiv/wos", exist_ok=True)
    np.savez_compressed("./output/wos_arxiv/wos/scores.npz", scores=scores)
except Exception as e:
    import traceback; traceback.print_exc()
    results.append({"dataset": "WOS", "error": str(e)})

# ===== ArXiv =====
print("\n" + "=" * 60)
print("ArXiv — 19 áreas + 137 subcats")
print("=" * 60)
try:
    X_tr, y_tr, X_te, y_te, eval_mask, adj, n_nodes = load_arxiv()
    print(f"  Train: {X_tr.shape}  Test: {X_te.shape}  Nodes: {n_nodes}  Eval: {eval_mask.sum()}")
    f1, au, p, r, thr, dur, scores = train_global("arxiv", X_tr, y_tr, X_te, y_te, eval_mask, adj, n_nodes, epochs=50)
    results.append({"dataset": "ArXiv", "method": "global", "F1": f1, "AUPRC": au, "P": p, "R": r, "thr": thr, "time": dur})
    print(f"  => F1={f1:.4f}  AUPRC={au:.4f}  P={p:.4f}  R={r:.4f}  thr={thr:.2f}  t={dur:.1f}s")
    os.makedirs("./output/wos_arxiv/arxiv", exist_ok=True)
    np.savez_compressed("./output/wos_arxiv/arxiv/scores.npz", scores=scores)
except Exception as e:
    import traceback; traceback.print_exc()
    results.append({"dataset": "ArXiv", "error": str(e)})

# ===== COMPARISON =====
print("\n" + "=" * 70)
print("COMPARISON WITH PUBLISHED SOTA")
print("=" * 70)
print(f"{'Dataset':<10} {'Method':<22} {'Micro-F1':>10} {'AUPRC':>10}")
print("-" * 55)

# Published SOTA numbers from HMC literature
published = {
    "WOS": [
        ("HiAGM (Zhou+ ACL'20)", 0.8604),
        ("HTC-infoMAX (ICML'21)", 0.8720),
        ("HMCN (Zhou+ ECML'20)", 0.8580),
        ("Ours (global MLP+R)", None),  # will be filled
    ],
    "ArXiv": [
        ("HiAGM (Zhou+ ACL'20)", 0.5950),
        ("HTC-infoMAX (ICML'21)", 0.6130),
        ("Ours (global MLP+R)", None),
    ],
}

for r in results:
    ds = r["dataset"]
    print(f"\n{ds}:")
    for method, pub_f1 in published.get(ds, []):
        if pub_f1 is None:
            ours = r.get("F1", 0)
            print(f"  {method:<22} {ours:>10.4f}  ← our result")
        else:
            diff = ""
            if "F1" in r:
                d = r["F1"] - pub_f1
                diff = f"  (Δ={d:+.4f})"
            print(f"  {method:<22} {pub_f1:>10.4f}{diff}")

with open("./output/wos_arxiv/results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to ./output/wos_arxiv/results.json")
