#!/usr/bin/env python3
"""GBDT + Platt calibration + reconciliation on all FUN datasets.

Trains GBDT One-vs-Rest, calibrates each node with Platt scaling
fitted on the validation split, reconciles, and compares against
uncalibrated GBDT and global SOTA.
"""

import json
import os
import sys
import time

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

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
    except:
        auprc = 0.0
    return best_f1, auprc, best_p, best_r, best_thr


def load_data(name):
    from hmc.datasets.manager.dataset_manager import initialize_dataset_experiments

    mgr = initialize_dataset_experiments(
        name, device="cpu", dataset_path="./data", dataset_type="arff", is_global=False
    )
    train, valid, test = mgr.get_datasets()

    X_tr = train.x.astype(np.float32)
    y_tr = train.y.astype(np.float32)
    X_va = valid.x.astype(np.float32)
    y_va = valid.y.astype(np.float32)
    X_te = test.x.astype(np.float32)
    y_te = test.y.astype(np.float32)

    eval_mask = np.array(mgr.to_eval, dtype=bool)
    terms = train.terms

    imp = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(imp.fit_transform(X_tr)).astype(np.float32)
    X_va = scaler.transform(imp.transform(X_va)).astype(np.float32)
    X_te = scaler.transform(imp.transform(X_te)).astype(np.float32)

    from hmc.data.hierarchy import TreeHierarchy

    branches = [t for t in terms if t != "root"]
    hier = TreeHierarchy.from_fun_cat_terms(branches if branches else terms)
    return X_tr, y_tr, X_va, y_va, X_te, y_te, eval_mask, hier, mgr


def run_gbdt_platt(name, X_tr, y_tr, X_va, y_va, X_te, y_te, eval_mask, hier):
    """GBDT + Platt + Reconcile pipeline."""
    from hmc.models.hierarchical.postprocess import PlattCalibrator, reconcile
    from hmc.models.tabular.gbdt_ovr import GBDTOvRClassifier

    t0 = time.time()
    n_pos_tr = y_tr.sum(axis=0)
    trainable = (n_pos_tr >= 5) & eval_mask

    # 1. Train GBDT
    gbdt = GBDTOvRClassifier(
        backend="histgb",
        backend_kwargs={
            "early_stopping": False,
            "max_iter": 100,
        },
    )
    gbdt.fit(X_tr, y_tr, eval_mask=trainable)

    # 2. Raw scores on validation
    scores_va_raw = gbdt.predict_proba(X_va)
    scores_te_raw = gbdt.predict_proba(X_te)

    # 3. Platt calibration (fit on validation)
    cal = PlattCalibrator()
    cal.fit(scores_va_raw, y_va, eval_mask=eval_mask)
    scores_te_cal = cal.calibrate(scores_te_raw)

    # 4. Reconcile
    scores_te_final = reconcile(scores_te_cal, hier, strategy="ancestor_max")

    dur = time.time() - t0
    return scores_te_raw, scores_te_cal, scores_te_final, dur, int(trainable.sum())


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

all_results = []

for ds_name in DATASETS:
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}")
    X_tr, y_tr, X_va, y_va, X_te, y_te, eval_mask, hier, mgr = load_data(ds_name)
    n_nodes = y_tr.shape[1]
    print(
        f"  Train: {X_tr.shape}  Valid: {X_va.shape}  Test: {X_te.shape}  "
        f"Features: {X_tr.shape[1]}  Nodes: {n_nodes}  Eval: {eval_mask.sum()}"
    )

    # ---- GBDT + Platt ----
    scores_raw, scores_cal, scores_final, dur, n_trained = run_gbdt_platt(
        ds_name, X_tr, y_tr, X_va, y_va, X_te, y_te, eval_mask, hier
    )

    # Metrics at each stage
    f1_raw, au_raw, p_raw, r_raw, th_raw = compute_metrics(y_te, scores_raw, eval_mask)
    f1_cal, au_cal, p_cal, r_cal, th_cal = compute_metrics(y_te, scores_cal, eval_mask)
    f1_fin, au_fin, p_fin, r_fin, th_fin = compute_metrics(
        y_te, scores_final, eval_mask
    )

    print(
        f"  GBDT raw:       F1={f1_raw:.4f}  AUPRC={au_raw:.4f}  "
        f"P={p_raw:.4f}  R={r_raw:.4f}  thr={th_raw:.2f}"
    )
    print(
        f"  GBDT + Platt:   F1={f1_cal:.4f}  AUPRC={au_cal:.4f}  "
        f"P={p_cal:.4f}  R={r_cal:.4f}  thr={th_cal:.2f}"
    )
    print(
        f"  GBDT + Platt+R: F1={f1_fin:.4f}  AUPRC={au_fin:.4f}  "
        f"P={p_fin:.4f}  R={r_fin:.4f}  thr={th_fin:.2f}"
    )
    print(f"  Time: {dur:.1f}s  Nodes trained: {n_trained}/{n_nodes}")

    all_results.append(
        {
            "method": "gbdt_raw",
            "dataset": ds_name,
            "micro_f1": float(f1_raw),
            "auprc": float(au_raw),
            "precision": float(p_raw),
            "recall": float(r_raw),
            "threshold": float(th_raw),
            "duration_s": dur,
        }
    )
    all_results.append(
        {
            "method": "gbdt_platt",
            "dataset": ds_name,
            "micro_f1": float(f1_cal),
            "auprc": float(au_cal),
            "precision": float(p_cal),
            "recall": float(r_cal),
            "threshold": float(th_cal),
            "duration_s": dur,
        }
    )
    all_results.append(
        {
            "method": "gbdt_platt_r",
            "dataset": ds_name,
            "micro_f1": float(f1_fin),
            "auprc": float(au_fin),
            "precision": float(p_fin),
            "recall": float(r_fin),
            "threshold": float(th_fin),
            "duration_s": dur,
        }
    )

    # Save
    out_dir = f"./output/experiments/{ds_name}/gbdt_platt"
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(f"{out_dir}/scores_raw.npz", scores=scores_raw)
    np.savez_compressed(f"{out_dir}/scores_calibrated.npz", scores=scores_cal)
    np.savez_compressed(f"{out_dir}/scores_final.npz", scores=scores_final)

# Load previous results
try:
    with open("./output/experiments/sota_comparison.json") as f:
        prev = json.load(f)
    # Keep only global and gbdt methods
    all_results.extend(
        [r for r in prev if r["method"] in ("global", "tabular_gbdt", "tabular_mlp")]
    )
except Exception:
    pass

# ===================================================================
print("\n\n" + "=" * 100)
print("FINAL COMPARISON — GBDT + Platt Calibration vs SOTA")
print("=" * 100)

for ds_name in DATASETS:
    ds_res = [r for r in all_results if r["dataset"] == ds_name]
    if not ds_res:
        continue
    print(f"\n{ds_name}:")
    for r in sorted(ds_res, key=lambda x: -x["micro_f1"]):
        tag = ""
        if "gbdt" in r["method"]:
            tag = ""
        elif "global" in r["method"]:
            tag = " ← SOTA"
        print(
            f"  {r['method']:<18} F1={r['micro_f1']:.4f}  "
            f"AUPRC={r['auprc']:.4f}  P={r['precision']:.4f}  "
            f"R={r['recall']:.4f}  thr={r['threshold']:.2f}{tag}"
        )

# Summary
print(f"\n{'='*100}")
print("SUMMARY BY METHOD (averaged across 8 datasets)")
for method in [
    "global",
    "gbdt_platt_r",
    "gbdt_platt",
    "tabular_gbdt",
    "gbdt_raw",
    "tabular_mlp",
]:
    m_res = [r for r in all_results if r["method"] == method]
    if not m_res:
        continue
    avg_f1 = np.mean([r["micro_f1"] for r in m_res])
    avg_au = np.mean([r["auprc"] for r in m_res])
    wins = sum(
        1
        for ds in DATASETS
        for r in all_results
        if r["dataset"] == ds
        and r["method"] == method
        and r["micro_f1"]
        == max(rr["micro_f1"] for rr in all_results if rr["dataset"] == ds)
    )
    print(f"  {method:<18} avg F1={avg_f1:.4f}  avg AUPRC={avg_au:.4f}  wins={wins}/8")

with open("./output/experiments/gbdt_platt_comparison.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved to ./output/experiments/gbdt_platt_comparison.json")
