#!/usr/bin/env python3
"""Scaling laws for HMC: collect performance metrics across all datasets.

Runs the ``global`` method (MLP + R-matrix) on every built-in dataset and
records hierarchy properties alongside performance, enabling analysis of
how F1/AUPRC scale with: number of classes, hierarchy depth, branching factor,
and training size.

Output: output/scaling_laws/results.json
"""

from __future__ import annotations

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

from hmc.datasets.dataset_manager import initialize_dataset_experiments
from hmc.models.global_classifier.constraint.model import ConstrainedModel
from hmc.models.hierarchical.sparse_r import BlockDiagonalR

sys.path.insert(0, "src")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 128
SEED = 42

# ---------------------------------------------------------------------------
# Dataset list — tuples of (name, dataset_type, is_global)
# ---------------------------------------------------------------------------
TEXT_DATASETS = [
    ("arxiv", "arxiv", True),
    ("wos", "wos", True),
    ("aapd", "aapd", True),
]

FUN_DATASETS = [
    ("cellcycle_FUN", "arff", False),
    ("church_FUN", "arff", False),
    ("derisi_FUN", "arff", False),
    ("eisen_FUN", "arff", False),
    ("expr_FUN", "arff", False),
    ("gasch1_FUN", "arff", False),
    ("gasch2_FUN", "arff", False),
    ("pheno_FUN", "arff", False),
    ("seq_FUN", "arff", False),
    ("spo_FUN", "arff", False),
]

GO_DATASETS = [
    ("cellcycle_GO", "arff", False),
    ("derisi_GO", "arff", False),
    ("eisen_GO", "arff", False),
    ("expr_GO", "arff", False),
    ("gasch1_GO", "arff", False),
    ("gasch2_GO", "arff", False),
    ("pheno_GO", "arff", False),
    ("seq_GO", "arff", False),
    ("spo_GO", "arff", False),
]

OTHER_DATASETS = [
    ("enron_others", "arff", False),
    ("diatoms_others", "arff", False),
    ("imclef07a_others", "arff", False),
    ("imclef07d_others", "arff", False),
]

ALL_DATASETS = TEXT_DATASETS + FUN_DATASETS + GO_DATASETS + OTHER_DATASETS


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, eval_mask: np.ndarray
) -> tuple[float, float]:
    """Compute micro-F1 and micro-AUPRC."""
    best_f1 = 0.0
    for thr in np.arange(0.1, 0.90, 0.05):
        y_bin = (y_pred >= thr).astype(np.float32)
        tp = (y_bin[:, eval_mask] * y_true[:, eval_mask]).sum()
        fp = (y_bin[:, eval_mask] * (1 - y_true[:, eval_mask])).sum()
        fn = ((1 - y_bin[:, eval_mask]) * y_true[:, eval_mask]).sum()
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        if f1 > best_f1:
            best_f1 = f1
    try:
        auprc = float(
            average_precision_score(
                y_true[:, eval_mask], y_pred[:, eval_mask], average="micro"
            )
        )
    except ValueError:
        auprc = 0.0
    return best_f1, auprc


# ---------------------------------------------------------------------------
# R-matrix builder
# ---------------------------------------------------------------------------


def build_r_matrix(adj: np.ndarray) -> torch.Tensor:
    """Build dense R-matrix from adjacency."""
    r = np.zeros(adj.shape)
    np.fill_diagonal(r, 1)
    g = nx.DiGraph(adj)
    for i in range(len(adj)):
        ancestors = list(nx.descendants(g, i))
        if ancestors:
            r[i, ancestors] = 1
    return torch.tensor(r).transpose(1, 0).unsqueeze(0).to(DEVICE)


# ---------------------------------------------------------------------------
# Hierarchy properties
# ---------------------------------------------------------------------------


def compute_hierarchy_props(
    adj: np.ndarray, nodes_idx: dict
) -> dict:
    """Compute depth, branching factor, and other hierarchy stats."""
    g = nx.DiGraph(adj)
    # Depth = longest path from any root to any node
    roots = [n for n in g.nodes() if g.out_degree(n) == 0]
    max_depth = 0
    g_rev = g.reverse()
    for root in roots:
        try:
            lengths = nx.single_source_shortest_path_length(g_rev, root)
            max_depth = max(max_depth, max(lengths.values()))
        except (nx.NetworkXNoPath, nx.NodeNotFound, ValueError):
            pass

    # Branching factor = avg children per non-leaf node
    non_leaves = [n for n in g.nodes() if g.in_degree(n) > 0]
    if non_leaves:
        avg_branching = float(np.mean([g.in_degree(n) for n in non_leaves]))
    else:
        avg_branching = 0.0

    # Leaf count
    n_leaves = sum(1 for n in g.nodes() if g.in_degree(n) == 0)

    return {
        "n_nodes": int(len(g.nodes())),
        "n_edges": int(len(g.edges())),
        "max_depth": int(max_depth),
        "n_roots": int(len(roots)),
        "n_leaves": int(n_leaves),
        "avg_branching": float(avg_branching),
        "n_eval": len(nodes_idx) - len(roots),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_dataset(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    eval_mask: np.ndarray,
    adj: np.ndarray,
    n_nodes: int,
    is_go: bool = False,
    go_graph: nx.DiGraph | None = None,
    hidden: int = 256,
    epochs: int = 30,
) -> dict:
    """Train global model with R-matrix on one dataset."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # For large DAGs (>2000 nodes): use identity R in model + BlockDiagonalR
    # post-hoc to avoid OOM from dense (C×C×batch) in get_constr_out.
    use_sparse = is_go or n_nodes > 2000
    if use_sparse:
        r_matrix = torch.eye(n_nodes).unsqueeze(0).to(DEVICE)
        baseline = True
    else:
        r_matrix = build_r_matrix(adj)
        baseline = False

    model = ConstrainedModel(
        input_dim=X_tr.shape[1],
        hidden_dim=min(hidden, 1024),
        output_dim=n_nodes,
        hyperparams={
            "batch_size": BATCH,
            "num_layers": 3,
            "dropout": 0.3,
            "non_lin": "relu",
            "hidden_dim": min(hidden, 1024),
            "lr": 1e-4,
            "weight_decay": 1e-5,
        },
        r_matrix=r_matrix,
        baseline_model=baseline,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    Xt_tr = torch.tensor(X_tr)
    yt_tr = torch.tensor(y_tr)
    tr_ldr = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=BATCH, shuffle=True)

    t0 = time.time()
    model.train()
    for _ in range(epochs):
        for bx, by in tr_ldr:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            output = model(bx)
            loss = torch.nn.functional.binary_cross_entropy(
                output[:, eval_mask], by[:, eval_mask]
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    train_time = time.time() - t0

    # Eval
    model.eval()
    preds: list[np.ndarray] = []
    te_ldr = DataLoader(
        TensorDataset(torch.tensor(X_te)), batch_size=64, shuffle=False
    )
    with torch.no_grad():
        for (bx,) in te_ldr:
            preds.append(model(bx.to(DEVICE)).cpu().numpy())
    scores = np.concatenate(preds, axis=0)

    # For GO: apply BlockDiagonalR
    if is_go and go_graph is not None:
        sparse_r = BlockDiagonalR(go_graph, DEVICE)
        scores_t = torch.tensor(scores).to(DEVICE)
        scores = sparse_r.reconcile(scores_t).cpu().numpy()

    f1, auprc = compute_metrics(y_te, scores, eval_mask)

    return {
        "f1": float(f1),
        "auprc": float(auprc),
        "time_s": float(train_time),
    }


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_text_dataset(ds_name: str) -> tuple:
    """Load a text dataset (arxiv, wos, aapd). Returns X, y, eval_mask, adj, n_nodes."""
    mgr = initialize_dataset_experiments(
        ds_name, device="cpu", dataset_path="./data",
        is_global=True, model_cache_dir="./models",
    )
    tr, va, te = mgr.get_datasets()
    X_tr = np.concatenate([tr.x, va.x]).astype(np.float32)
    y_tr = np.concatenate([tr.y, va.y]).astype(np.float32)
    X_te = te.x.astype(np.float32)
    y_te = te.y.astype(np.float32)
    eval_mask = np.array(mgr.to_eval, dtype=bool)
    adj = mgr.a
    n_nodes = mgr.output_dim
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)
    props = compute_hierarchy_props(adj, mgr.nodes_idx)
    props["modality"] = "text"
    return X_tr, y_tr, X_te, y_te, eval_mask, adj, n_nodes, props


def load_tabular_dataset(ds_name: str) -> tuple:
    """Load a tabular dataset (FUN, GO, others)."""
    from hmc.datasets.manager.dataset_manager import \
        initialize_dataset_experiments as init_gofun

    mgr = init_gofun(
        ds_name, device="cpu", dataset_path="./data",
        dataset_type="arff", is_global=False,
    )
    tr, va, te = mgr.get_datasets()
    X_tr = np.concatenate([tr.x, va.x]).astype(np.float32)
    y_tr = np.concatenate([tr.y, va.y]).astype(np.float32)
    X_te = te.x.astype(np.float32)
    y_te = te.y.astype(np.float32)
    eval_mask = np.array(mgr.to_eval, dtype=bool)
    adj = mgr.a
    n_nodes = len(mgr.nodes_idx)

    imp = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(imp.fit_transform(X_tr)).astype(np.float32)
    X_te = scaler.transform(imp.transform(X_te)).astype(np.float32)

    props = compute_hierarchy_props(adj, mgr.nodes_idx)
    if "GO" in ds_name:
        props["modality"] = "tabular_dag"
    elif "others" in ds_name:
        props["modality"] = "tabular_other"
    else:
        props["modality"] = "tabular_tree"
    return X_tr, y_tr, X_te, y_te, eval_mask, adj, n_nodes, props


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    results: list[dict] = []
    errors: list[dict] = []

    for ds_name, ds_type, is_global in ALL_DATASETS:
        print(f"\n{'='*50}")
        print(f"  {ds_name} ({ds_type})")
        print(f"{'='*50}")

        try:
            if ds_type in ("arxiv", "wos", "aapd"):
                X_tr, y_tr, X_te, y_te, ev, adj, n_nodes, props = \
                    load_text_dataset(ds_name)
                is_go = False
                go_graph = None
                hidden = 512
                epochs = 50
            else:
                X_tr, y_tr, X_te, y_te, ev, adj, n_nodes, props = \
                    load_tabular_dataset(ds_name)
                is_go = "GO" in ds_name
                go_graph = nx.DiGraph(adj) if is_go else None
                hidden = 256
                epochs = 30

            print(f"  n_train={X_tr.shape[0]} n_test={X_te.shape[0]} "
                  f"n_feat={X_tr.shape[1]} nodes={n_nodes} "
                  f"depth={props['max_depth']} eval={ev.sum()}")
        except Exception as exc:
            print(f"  SKIP: {exc}")
            errors.append({"dataset": ds_name, "error": str(exc)})
            continue

        try:
            metrics = train_one_dataset(
                X_tr, y_tr, X_te, y_te, ev, adj, n_nodes,
                is_go=is_go, go_graph=go_graph, hidden=hidden, epochs=epochs,
            )
        except Exception as exc:
            print(f"  TRAIN ERROR: {exc}")
            errors.append({"dataset": ds_name, "error": str(exc)})
            continue

        entry = {**props, **metrics,
                 "dataset": ds_name,
                 "n_train": int(X_tr.shape[0]),
                 "n_test": int(X_te.shape[0]),
                 "n_features": int(X_tr.shape[1]),
                 "ratio": float(X_tr.shape[0] / max(n_nodes, 1))}
        results.append(entry)
        print(f"  F1={metrics['f1']:.4f} AUPRC={metrics['auprc']:.4f} "
              f"t={metrics['time_s']:.1f}s")

    # Save
    os.makedirs("./output/scaling_laws", exist_ok=True)
    out = {
        "results": results,
        "n_total": len(results),
        "n_errors": len(errors),
        "errors": errors,
    }
    out_path = "./output/scaling_laws/results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {len(results)} results + {len(errors)} errors to {out_path}")


if __name__ == "__main__":
    main()
