#!/usr/bin/env python3
"""Run global (MLP+R) experiments on new text datasets: AAPD, RCV1-V2, EUR-Lex.

Usage:
    python run_new_datasets.py --dataset aapd --seeds 3
    python run_new_datasets.py --dataset rcv1 --seeds 1
    python run_new_datasets.py --dataset eurlex --seeds 3
    python run_new_datasets.py --dataset all --seeds 3
"""

import argparse
import json
import os
import sys
import time

import networkx as nx
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "src")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, eval_mask):
    """Micro-F1 (threshold sweep), AUPRC, Precision, Recall."""
    best = (0.0, 0.5, 0.0, 0.0)
    for thr in np.arange(0.1, 0.90, 0.05):
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
        auprc = float(average_precision_score(
            y_true[:, eval_mask], y_pred[:, eval_mask], average="micro"))
    except Exception:
        auprc = 0.0
    return best[0], auprc, best[2], best[3], best[1]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(dataset_name):
    """Load a text dataset via unified dispatch. Returns all needed arrays."""
    from hmc.datasets.dataset_manager import initialize_dataset_experiments

    mgr = initialize_dataset_experiments(
        dataset_name, device="cpu", dataset_path="./data",
        is_global=True, model_cache_dir="./models",
    )
    train, valid, test = mgr.get_datasets()

    X_tr = np.concatenate([train.x, valid.x]).astype(np.float32)
    y_tr = np.concatenate([train.y, valid.y]).astype(np.float32)
    X_te = test.x.astype(np.float32)
    y_te = test.y.astype(np.float32)

    eval_mask = np.array(mgr.to_eval, dtype=bool)
    adj = mgr.a
    n_nodes = mgr.output_dim
    input_dim = mgr.input_dim

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    # For sparse R: pass the label DiGraph if available
    label_g = getattr(test, 'g', None)

    return X_tr, y_tr, X_te, y_te, eval_mask, adj, n_nodes, input_dim, label_g


def build_dense_r(adj):
    """Build dense R-matrix (ancestor matrix) from adjacency."""
    n = adj.shape[0]
    r = np.zeros((n, n))
    np.fill_diagonal(r, 1)
    g = nx.DiGraph(adj)
    for i in range(n):
        for d in nx.descendants(g, i):
            r[i, d] = 1
    return torch.tensor(r).transpose(1, 0).unsqueeze(0).to(DEVICE)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(X_tr, y_tr, r_matrix, n_nodes, hidden_dim, epochs,
                batch_size, seed):
    """Train ConstrainedModel and return the model."""
    from hmc.models.global_classifier.constraint.model import ConstrainedModel

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_feat = X_tr.shape[1]

    model = ConstrainedModel(
        input_dim=n_feat,
        hidden_dim=min(hidden_dim, n_feat * 2),
        output_dim=n_nodes,
        hyperparams={
            "batch_size": batch_size, "num_layers": 3, "dropout": 0.3,
            "non_lin": "relu", "hidden_dim": min(hidden_dim, n_feat * 2),
            "lr": 1e-4, "weight_decay": 1e-5,
        },
        r_matrix=r_matrix,
        baseline_model=False,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    tr_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=batch_size, shuffle=True,
    )

    model.train()
    for _ in range(epochs):
        for bx, by in tr_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            loss = torch.nn.functional.binary_cross_entropy(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return model


def predict(model, X_te, batch_size):
    """Run inference and return numpy scores."""
    model.eval()
    preds = []
    te_loader = DataLoader(
        TensorDataset(torch.tensor(X_te)),
        batch_size=min(64, batch_size), shuffle=False,
    )
    with torch.no_grad():
        for (bx,) in te_loader:
            preds.append(model(bx.to(DEVICE)).cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_and_eval(X_tr, y_tr, X_te, y_te, eval_mask, r_matrix, n_nodes,
                   hidden_dim=512, epochs=50, batch_size=32, seed=42,
                   sparse_r=None):
    """Train + inference + optional sparse R reconciliation."""
    t0 = time.time()

    model = train_model(X_tr, y_tr, r_matrix, n_nodes,
                        hidden_dim, epochs, batch_size, seed)
    scores = predict(model, X_te, batch_size)

    # Sparse R reconciliation (for large label spaces)
    if sparse_r is not None:
        scores = sparse_r.reconcile(
            torch.tensor(scores).to(DEVICE)
        ).cpu().numpy()

    f1, au, p, r, thr = compute_metrics(y_te, scores, eval_mask)
    dur = time.time() - t0

    return {
        "f1": float(f1), "auprc": float(au),
        "precision": float(p), "recall": float(r),
        "threshold": float(thr), "time_s": dur, "epochs": epochs,
    }


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------

def run_dataset(dataset_name, seeds, epochs=50, hidden_dim=512, batch_size=32):
    """Run full experiment (with-R + without-R ablation) on one dataset."""
    print(f"\n{'=' * 70}")
    print(f"DATASET: {dataset_name}")
    print(f"{'=' * 70}")

    X_tr, y_tr, X_te, y_te, eval_mask, adj, n_nodes, input_dim, label_g = \
        load_data(dataset_name)
    print(
        f"  Train: {X_tr.shape}  Test: {X_te.shape}  "
        f"Features: {input_dim}  Nodes: {n_nodes}  "
        f"Eval labels: {eval_mask.sum()}"
    )

    # Decide: dense R (small/medium) or sparse R (large label space)
    use_sparse_r = n_nodes > 1000
    if use_sparse_r:
        print(f"  >>> {n_nodes} nodes — sparse R (BlockDiagonalR) <<<")
        from hmc.models.hierarchical.sparse_r import BlockDiagonalR
        r_matrix = torch.eye(n_nodes).unsqueeze(0).to(DEVICE)
        if label_g is not None:
            sparse_r = BlockDiagonalR(label_g)
        else:
            # Build DiGraph from adjacency
            g = nx.DiGraph(adj)
            sparse_r = BlockDiagonalR(g)
    else:
        r_matrix = build_dense_r(adj)
        sparse_r = None

    results = {
        "dataset": dataset_name, "n_nodes": n_nodes,
        "n_features": input_dim, "use_sparse_r": use_sparse_r,
    }

    # --- With R ---
    print("\n  --- With R-matrix ---")
    with_r = {}
    for seed in seeds:
        print(f"    seed={seed}...", end=" ", flush=True)
        r = train_and_eval(
            X_tr, y_tr, X_te, y_te, eval_mask, r_matrix, n_nodes,
            hidden_dim, epochs, batch_size, seed, sparse_r,
        )
        with_r[str(seed)] = r
        print(f"F1={r['f1']:.4f} AUPRC={r['auprc']:.4f} t={r['time_s']:.1f}s")

    f1s = [v["f1"] for v in with_r.values()]
    if len(f1s) > 1:
        print(f"    Mean±Std: F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")
    results["with_R"] = with_r
    results["with_R_summary"] = {
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)) if len(f1s) > 1 else 0.0,
    }

    # --- Without R (ablation) ---
    print("\n  --- Without R-matrix (ablation) ---")
    identity_r = torch.eye(n_nodes).unsqueeze(0).to(DEVICE)
    without_r = {}
    for seed in seeds:
        print(f"    seed={seed}...", end=" ", flush=True)
        r = train_and_eval(
            X_tr, y_tr, X_te, y_te, eval_mask, identity_r, n_nodes,
            hidden_dim, epochs, batch_size, seed,
            # No sparse R reconciliation for ablation
        )
        without_r[str(seed)] = r
        print(f"F1={r['f1']:.4f} AUPRC={r['auprc']:.4f} t={r['time_s']:.1f}s")

    f1s_no_r = [v["f1"] for v in without_r.values()]
    if len(f1s_no_r) > 1:
        print(f"    Mean±Std: F1={np.mean(f1s_no_r):.4f}±{np.std(f1s_no_r):.4f}")
    results["without_R"] = without_r
    results["without_R_summary"] = {
        "f1_mean": float(np.mean(f1s_no_r)),
        "f1_std": float(np.std(f1s_no_r)) if len(f1s_no_r) > 1 else 0.0,
    }

    delta = np.mean(f1s) - np.mean(f1s_no_r)
    results["delta_F1"] = float(delta)
    print(f"\n  >>> ΔF1 (R-matrix gain): {delta:+.4f}")

    # Save
    output_dir = f"./output/new_datasets/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {output_dir}/results.json")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run HMC experiments on new text datasets"
    )
    parser.add_argument(
        "--dataset", type=str, default="all",
        choices=["aapd", "rcv1", "eurlex", "all"],
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=512)
    args = parser.parse_args()

    seeds = [0, 42, 123][: args.seeds]
    datasets = ["aapd", "rcv1", "eurlex"] if args.dataset == "all" else [args.dataset]

    all_results = {}
    for ds in datasets:
        try:
            all_results[ds] = run_dataset(
                ds, seeds=seeds, epochs=args.epochs,
                hidden_dim=args.hidden_dim, batch_size=args.batch_size,
            )
        except FileNotFoundError as e:
            print(f"\n  SKIP {ds}: {e}")
            print(f"  Run: make download-{ds}")
            all_results[ds] = {"error": str(e)}
        except Exception as e:
            print(f"\n  FAIL {ds}: {e}")
            import traceback
            traceback.print_exc()
            all_results[ds] = {"error": str(e)}

    # Final summary
    print(f"\n{'=' * 70}")
    print("SUMMARY — New Datasets")
    print(f"{'=' * 70}")
    print(f"{'Dataset':<12} {'With R':>10} {'Without R':>10} {'ΔF1':>8}  Notes")
    print("-" * 60)
    for ds, data in all_results.items():
        if "error" in data:
            print(f"{ds:<12} ERROR: {data['error'][:40]}")
        else:
            f1_r = data.get("with_R_summary", {}).get("f1_mean", 0)
            f1_no = data.get("without_R_summary", {}).get("f1_mean", 0)
            delta = data.get("delta_F1", 0)
            sr = " (sparse R)" if data.get("use_sparse_r") else ""
            print(f"{ds:<12} {f1_r:>10.4f} {f1_no:>10.4f} {delta:>+8.4f}{sr}")

    os.makedirs("./output/new_datasets", exist_ok=True)
    with open("./output/new_datasets/summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to ./output/new_datasets/summary.json")


if __name__ == "__main__":
    main()
