#!/usr/bin/env python3
"""Benchmark SR-GAE (Ponto 1) + Depth-Level Logit Calibration (Ponto 2) across HMC datasets.

Evaluates:
  1. Ontology embedding geometry: parent-child and sibling cosine similarities
     for Standard GAE (``loss="full"``) vs. Sibling-Repulsive GAE (``loss="sibling_repulsive"``).
  2. Pure cosine-head classification (``SrGaeCosine``) over frozen Std-GAE vs. SR-GAE embeddings.
  3. Global model (``MLP`` vs. ``SrGaeHybrid``) with and without consistency-preserving
     depth-level logit calibration (``BlockDiagonalR.reconcile_with_depth_offsets``),
     reporting Test Micro-AUPRC, Test Micro-F1, Deep-Level F1, and hierarchy violations.

Usage:
    PYTHONPATH=src python experiments/run_sr_gae_depth.py --datasets enron_others,diatoms_others,cellcycle_FUN
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(__file__))

from run_autoencoder import SrGaeCosine, SrGaeHybrid  # noqa: E402
from run_protocol import (  # noqa: E402
    FUNCAT,
    GO,
    OTHERS,
    _fit,
    _reconcile,
    auprc,
    evaluate,
    hyperparams,
    load_dataset,
    pick_threshold,
    predict,
    prf,
)

from hmc.models.hierarchical.graph_autoencoder import (  # noqa: E402
    pretrain_label_gae,
    sibling_pairs,
)
from hmc.utils.metrics.hierarchy_violation import (  # noqa: E402
    global_hierarchy_violation,
)

OUT_DIR = "output/sr_gae_depth"
GROUPS = {"funcat": FUNCAT, "others": OTHERS, "go": GO}


def geometry_stats(z: np.ndarray, edges: np.ndarray) -> dict[str, float]:
    """Compute parent-child and sibling cosine similarity statistics."""
    zn = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-8, None)
    edge_cos = (zn[edges[:, 0]] * zn[edges[:, 1]]).sum(axis=1)
    sib = sibling_pairs(edges)
    if len(sib) == 0:
        return {
            "edge_cos_mean": float(edge_cos.mean()),
            "sib_pairs": 0,
            "sib_cos_mean": 0.0,
            "sib_cos_median": 0.0,
            "sib_cos_p90": 0.0,
            "sib_cos_max": 0.0,
        }
    sib_cos = (zn[sib[:, 0]] * zn[sib[:, 1]]).sum(axis=1)
    return {
        "edge_cos_mean": float(edge_cos.mean()),
        "sib_pairs": int(len(sib)),
        "sib_cos_mean": float(sib_cos.mean()),
        "sib_cos_median": float(np.median(sib_cos)),
        "sib_cos_p90": float(np.percentile(sib_cos, 90)),
        "sib_cos_max": float(sib_cos.max()),
    }


def fit_depth_offsets(
    data: dict, s_val_raw: np.ndarray, device: str
) -> dict[int, float]:
    """Learn L depth-level logit offsets on the validation split only."""
    m = data["eval_mask"]
    y_val = data["y"][1]
    depth = data["depth"]
    sparse_r = data["sparse_r"]
    levels = sorted(set(depth[m].tolist()))
    raw_t = torch.from_numpy(s_val_raw).to(device)
    base_rec = sparse_r.reconcile(raw_t).cpu().numpy()
    base_thr = pick_threshold(y_val[:, m], base_rec[:, m])
    base_auprc = auprc(y_val[:, m], base_rec[:, m])
    base_f1 = prf(y_val[:, m], base_rec[:, m], base_thr)[2]
    best_obj = 0.5 * base_auprc + 0.5 * base_f1
    best_offsets = {int(lvl): 0.0 for lvl in levels}

    for _ in range(2):
        for lvl in levels:
            for cand in np.linspace(-1.2, 1.2, 25):
                trial = dict(best_offsets)
                trial[int(lvl)] = float(cand)
                rec = (
                    sparse_r.reconcile_with_depth_offsets(raw_t, trial)
                    .cpu()
                    .numpy()
                )
                sc_auprc = auprc(y_val[:, m], rec[:, m])
                sc_f1 = prf(y_val[:, m], rec[:, m], base_thr)[2]
                obj = 0.5 * sc_auprc + 0.5 * sc_f1
                if obj > best_obj + 1e-6 and sc_auprc >= base_auprc - 0.0005:
                    best_obj = obj
                    best_offsets = trial
    return best_offsets


def eval_with_depth(
    data: dict,
    s_val_raw: np.ndarray,
    s_test_raw: np.ndarray,
    offsets: dict[int, float],
    device: str,
) -> dict:
    """Evaluate reconciled predictions with and without depth-level logit offsets."""
    m = data["eval_mask"]
    depth = data["depth"]
    max_d = int(depth[m].max())
    deep_cutoff = max(2, int(np.ceil(0.65 * max_d)))
    deep_mask = m & (depth >= deep_cutoff)
    sparse_r = data["sparse_r"]

    sv_rec = _reconcile(data, s_val_raw, device)
    st_rec = _reconcile(data, s_test_raw, device)
    base_ev = evaluate(data, sv_rec, st_rec)
    base_ev["deep_cutoff"] = deep_cutoff
    base_ev["deep_f1"] = (
        prf(data["y"][2][:, deep_mask], st_rec[:, deep_mask], base_ev["threshold"])[2]
        if data["y"][2][:, deep_mask].sum() > 0
        else 0.0
    )
    base_ev["ghv_violations"] = int(
        global_hierarchy_violation(st_rec, data["anc"], data["desc"], m)["violations"]
    )

    sv_drec = (
        sparse_r.reconcile_with_depth_offsets(
            torch.from_numpy(s_val_raw).to(device), offsets
        )
        .cpu()
        .numpy()
    )
    st_drec = (
        sparse_r.reconcile_with_depth_offsets(
            torch.from_numpy(s_test_raw).to(device), offsets
        )
        .cpu()
        .numpy()
    )
    depth_ev = evaluate(data, sv_drec, st_drec)
    depth_ev["deep_cutoff"] = deep_cutoff
    depth_ev["deep_f1"] = (
        prf(data["y"][2][:, deep_mask], st_drec[:, deep_mask], depth_ev["threshold"])[2]
        if data["y"][2][:, deep_mask].sum() > 0
        else 0.0
    )
    depth_ev["ghv_violations"] = int(
        global_hierarchy_violation(st_drec, data["anc"], data["desc"], m)["violations"]
    )
    return {"rec": base_ev, "depth_rec": depth_ev, "offsets": offsets}


def run_one(name: str, seed: int, device: str, out_dir: str, dim: int = 64) -> dict:
    """Run geometry + CosineHead + Global/SrGaeHybrid + DepthCal on one dataset."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        print(f"[cached] {name:<16} loaded from {path}", flush=True)
        return rec

    t0 = time.time()
    data = load_dataset(name)
    hp = hyperparams(name) | {"batch": 32}
    edges = data["edges"]
    n = data["y"][0].shape[1]

    z_std = pretrain_label_gae(
        edges, n, dim=dim, hidden_dim=32, epochs=200, lr=1e-2, loss="full", seed=seed, device=device
    )
    z_sr = pretrain_label_gae(
        edges,
        n,
        dim=dim,
        hidden_dim=max(2 * dim, 64),
        epochs=200,
        lr=1e-2,
        loss="sibling_repulsive",
        seed=seed,
        device=device,
    )
    geom_std = geometry_stats(z_std, edges)
    geom_sr = geometry_stats(z_sr, edges)

    # 1. Pure Cosine Head (lr=1e-3, matching Surj's cosine head optimizer)
    hp_cos = hp | {"lr": 1e-3}
    mod_cos_std, _, val_cos_std = _fit(
        data,
        "mc",
        seed,
        device,
        hp_cos,
        data["x"][0],
        data["y"][0],
        hp_cos["max_epochs"],
        x_val=data["x"][1],
        model_fn=lambda d_in, d_out: SrGaeCosine(
            d_in, hp_cos["hidden"], d_out, hp_cos["dropout"], z_std
        ),
    )
    mod_cos_sr, _, val_cos_sr = _fit(
        data,
        "mc",
        seed,
        device,
        hp_cos,
        data["x"][0],
        data["y"][0],
        hp_cos["max_epochs"],
        x_val=data["x"][1],
        model_fn=lambda d_in, d_out: SrGaeCosine(
            d_in, hp_cos["hidden"], d_out, hp_cos["dropout"], z_sr
        ),
    )
    ev_cos_std = evaluate(
        data,
        _reconcile(data, predict(mod_cos_std, data["x"][1], device), device),
        _reconcile(data, predict(mod_cos_std, data["x"][2], device), device),
    )
    ev_cos_sr = evaluate(
        data,
        _reconcile(data, predict(mod_cos_sr, data["x"][1], device), device),
        _reconcile(data, predict(mod_cos_sr, data["x"][2], device), device),
    )

    # 2. Global MLP vs. SrGaeHybrid + Depth-Level Calibration
    mod_mlp, ep_mlp, val_mlp = _fit(
        data, "mc", seed, device, hp, data["x"][0], data["y"][0], hp["max_epochs"], x_val=data["x"][1]
    )
    sv_mlp, st_mlp = predict(mod_mlp, data["x"][1], device), predict(mod_mlp, data["x"][2], device)
    off_mlp = fit_depth_offsets(data, sv_mlp, device)
    res_mlp = eval_with_depth(data, sv_mlp, st_mlp, off_mlp, device)

    mod_hyb, ep_hyb, val_hyb = _fit(
        data,
        "mc",
        seed,
        device,
        hp,
        data["x"][0],
        data["y"][0],
        hp["max_epochs"],
        x_val=data["x"][1],
        model_fn=lambda d_in, d_out: SrGaeHybrid(
            d_in, hp["hidden"], d_out, hp["dropout"], z_sr
        ),
    )
    sv_hyb, st_hyb = predict(mod_hyb, data["x"][1], device), predict(mod_hyb, data["x"][2], device)
    off_hyb = fit_depth_offsets(data, sv_hyb, device)
    res_hyb = eval_with_depth(data, sv_hyb, st_hyb, off_hyb, device)

    record = {
        "dataset": name,
        "seed": seed,
        "dim": dim,
        "geometry": {"std_gae": geom_std, "sr_gae": geom_sr},
        "cosine_head": {
            "std_gae": {"val_auprc": val_cos_std, "test": ev_cos_std},
            "sr_gae": {"val_auprc": val_cos_sr, "test": ev_cos_sr},
        },
        "mlp": {"best_epoch": ep_mlp, "val_auprc": val_mlp, **res_mlp},
        "sr_gae_hybrid": {"best_epoch": ep_hyb, "val_auprc": val_hyb, **res_hyb},
        "wall_time_s": time.time() - t0,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, default=float)

    print(
        f"{name:<16} | SibCos med: {geom_std['sib_cos_median']:.4f}->{geom_sr['sib_cos_median']:.4f} | "
        f"Cos Test AUPRC: {ev_cos_std['auprc']:.4f}->{ev_cos_sr['auprc']:.4f} ({ev_cos_sr['auprc']-ev_cos_std['auprc']:+.4f}) | "
        f"MLP F1: {res_mlp['rec']['f1']:.4f}->{res_mlp['depth_rec']['f1']:.4f} (Deep: {res_mlp['rec']['deep_f1']:.4f}->{res_mlp['depth_rec']['deep_f1']:.4f}) | "
        f"Hyb+Depth F1: {res_hyb['depth_rec']['f1']:.4f} (Deep: {res_hyb['depth_rec']['deep_f1']:.4f}) [{record['wall_time_s']:.0f}s]",
        flush=True,
    )
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--datasets", help="Comma-separated dataset names")
    ap.add_argument("--group", choices=[*GROUPS, "all"], default="others")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    if args.datasets:
        names = args.datasets.split(",")
    elif args.group == "all":
        names = OTHERS + FUNCAT
    else:
        names = GROUPS[args.group]

    for name in names:
        run_one(name, args.seed, device, args.out_dir, args.dim)


if __name__ == "__main__":
    main()
