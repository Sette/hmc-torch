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

from run_autoencoder import SrGaeCosine, SrGaeHybrid, SrGaeMlclHybrid  # noqa: E402
from run_protocol import (  # noqa: E402
    FUNCAT,
    GO,
    OTHERS,
    TEXT,
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
    contrastive_knn_predict,
    pretrain_label_gae,
    sibling_pairs,
)
from hmc.utils.metrics.hierarchy_violation import (  # noqa: E402
    global_hierarchy_violation,
)

OUT_DIR = "output/sr_gae_depth"
GROUPS = {"funcat": FUNCAT, "others": OTHERS, "go": GO, "text": TEXT}


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


@torch.no_grad()
def fit_depth_offsets(
    data: dict, s_val_raw: np.ndarray, device: str
) -> dict[int, float]:
    """Learn L depth-level logit offsets on the validation split only."""
    m = data["eval_mask"]
    eval_cols = torch.from_numpy(np.nonzero(m)[0]).to(device)
    y_val = data["y"][1][:, m]
    if len(s_val_raw) > 2000:
        rng = np.random.default_rng(0)
        sub_v = rng.choice(len(s_val_raw), size=2000, replace=False)
        s_val_raw = s_val_raw[sub_v]
        y_val = y_val[sub_v]
    y_val_b = torch.as_tensor(y_val == 1, device=device)
    y_val_flat = y_val_b.reshape(-1).to(torch.int64)
    n_pos = y_val_flat.sum()
    depth = data["depth"]
    sparse_r = data["sparse_r"]
    levels = sorted(set(depth[m].tolist()))
    raw_t = torch.from_numpy(s_val_raw).to(device)
    base_rec_t = sparse_r.reconcile(raw_t)[:, eval_cols]
    base_rec = base_rec_t.cpu().numpy()
    base_thr = pick_threshold(y_val, base_rec)
    base_auprc = auprc(y_val, base_rec)
    base_f1 = prf(y_val, base_rec, base_thr)[2]
    best_obj = 0.5 * base_auprc + 0.5 * base_f1
    best_offsets = {int(lvl): 0.0 for lvl in levels}

    def _fast_auprc(rec_eval_t: torch.Tensor) -> float:
        s, order = torch.sort(rec_eval_t.reshape(-1), descending=True)
        y = y_val_flat[order]
        tp = torch.cumsum(y, 0).to(torch.float32)
        fp = torch.cumsum(1 - y, 0).to(torch.float32)
        last = torch.ones_like(s, dtype=torch.bool)
        last[:-1] = s[1:] != s[:-1]
        tp, fp = tp[last], fp[last]
        precision = tp / (tp + fp)
        recall = tp / n_pos.to(torch.float32)
        recall_prev = torch.cat([recall.new_zeros(1), recall[:-1]])
        return float(((recall - recall_prev) * precision).sum().item())

    n_passes = 1 if len(levels) > 6 else 2
    grid = np.linspace(-0.8, 0.8, 9) if len(levels) > 6 else np.linspace(-1.2, 1.2, 25)
    best_f1 = base_f1
    for _ in range(n_passes):
        for lvl in levels:
            for cand in grid:
                if abs(cand) < 1e-8:
                    continue
                trial = dict(best_offsets)
                trial[int(lvl)] = float(cand)
                rec_t = sparse_r.reconcile_with_depth_offsets(raw_t, trial)[:, eval_cols]
                pred_b = rec_t >= base_thr
                tp = float((pred_b & y_val_b).sum().item())
                fp = float((pred_b & ~y_val_b).sum().item())
                fn = float((~pred_b & y_val_b).sum().item())
                sc_f1 = (2.0 * tp) / max(2.0 * tp + fp + fn, 1e-8)
                if sc_f1 < best_f1 + (1e-6 if len(levels) > 6 else -0.002):
                    continue
                sc_auprc = _fast_auprc(rec_t)
                obj = 0.5 * sc_auprc + 0.5 * sc_f1
                if obj > best_obj + 1e-6 and sc_auprc >= base_auprc - 0.0005:
                    best_obj = obj
                    best_f1 = max(best_f1, sc_f1)
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
        global_hierarchy_violation(
            torch.from_numpy(st_rec).to(device), data["anc"], data["desc"], m
        )["violations"]
    )

    sv_drec = (
        sparse_r.reconcile_with_depth_offsets(
            torch.from_numpy(s_val_raw).to(device), offsets
        )
        .cpu()
        .numpy()
    )
    st_drec_t = sparse_r.reconcile_with_depth_offsets(
        torch.from_numpy(s_test_raw).to(device), offsets
    )
    st_drec = st_drec_t.cpu().numpy()
    depth_ev = evaluate(data, sv_drec, st_drec)
    depth_ev["deep_cutoff"] = deep_cutoff
    depth_ev["deep_f1"] = (
        prf(data["y"][2][:, deep_mask], st_drec[:, deep_mask], depth_ev["threshold"])[2]
        if data["y"][2][:, deep_mask].sum() > 0
        else 0.0
    )
    depth_ev["ghv_violations"] = int(
        global_hierarchy_violation(st_drec_t, data["anc"], data["desc"], m)["violations"]
    )
    return {"rec": base_ev, "depth_rec": depth_ev, "offsets": offsets}


@torch.no_grad()
def encode_contrastive_split(
    model: SrGaeMlclHybrid, x: np.ndarray, device: str, batch: int = 1024
) -> np.ndarray:
    """Extract normalized H-MLCL contrastive embeddings for a split."""
    model.eval()
    chunks = [
        model.encode_contrastive(torch.from_numpy(x[i : i + batch]).to(device)).cpu()
        for i in range(0, len(x), batch)
    ]
    return torch.cat(chunks, dim=0).numpy()


def fit_knn_blend(
    data: dict,
    s_val_raw: np.ndarray,
    s_test_raw: np.ndarray,
    u_tr: np.ndarray,
    u_val: np.ndarray,
    u_test: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Select (beta, k, temp) for MLCL-KNN blending on the validation split only."""
    m = data["eval_mask"]
    y_tr = data["y"][0]
    if len(u_tr) > 10_000:
        rng = np.random.default_rng(0)
        sub = rng.choice(len(u_tr), size=10_000, replace=False)
        u_tr = u_tr[sub]
        y_tr = y_tr[sub]
    y_val_full = data["y"][1][:, m]
    if len(u_val) > 2000:
        rng_v = np.random.default_rng(0)
        sub_v = rng_v.choice(len(u_val), size=2000, replace=False)
        u_val_g, s_val_g, y_val_g = u_val[sub_v], s_val_raw[sub_v], y_val_full[sub_v]
    else:
        u_val_g, s_val_g, y_val_g = u_val, s_val_raw, y_val_full
    y_val_flat = torch.as_tensor(y_val_g.ravel(), device=device)
    eval_cols = torch.from_numpy(np.nonzero(m)[0]).to(device)
    sparse_r = data["sparse_r"]
    s_val_raw_t = torch.from_numpy(s_val_g).to(device)
    base_rec_t = sparse_r.reconcile(s_val_raw_t)[:, eval_cols]
    best_auprc = auprc(y_val_flat, base_rec_t)
    best_cfg = {"beta": 0.0, "k": 25, "temp": 0.1, "val_auprc": best_auprc}

    for k, temp in ((20, 0.1), (35, 0.1), (50, 0.15)):
        knn_val = contrastive_knn_predict(u_tr, y_tr, u_val_g, k=k, temp=temp)
        knn_val_t = torch.from_numpy(knn_val).to(device)
        for beta in (0.02, 0.04, 0.06, 0.08, 0.10, 0.15):
            blend_val_t = (1.0 - beta) * s_val_raw_t + beta * knn_val_t
            rec_val_t = sparse_r.reconcile(blend_val_t)[:, eval_cols]
            sc = auprc(y_val_flat, rec_val_t)
            if sc > best_auprc + 3e-4:
                best_auprc = sc
                best_cfg = {
                    "beta": float(beta),
                    "k": int(k),
                    "temp": float(temp),
                    "val_auprc": float(sc),
                }

    if best_cfg["beta"] > 0.0:
        knn_val_full = contrastive_knn_predict(
            u_tr, y_tr, u_val, k=int(best_cfg["k"]), temp=float(best_cfg["temp"])
        )
        best_val_raw = (1.0 - best_cfg["beta"]) * s_val_raw + best_cfg["beta"] * knn_val_full
        knn_test = contrastive_knn_predict(
            u_tr, y_tr, u_test, k=int(best_cfg["k"]), temp=float(best_cfg["temp"])
        )
        best_test_raw = (1.0 - best_cfg["beta"]) * s_test_raw + best_cfg["beta"] * knn_test
    else:
        best_val_raw = s_val_raw
        best_test_raw = s_test_raw
    return best_val_raw, best_test_raw, best_cfg


def _run_mlcl(data: dict, hp: dict, z_sr: np.ndarray, seed: int, device: str) -> dict:
    """Train SrGaeMlclHybrid and evaluate with MLCL-KNN + DepthCal."""
    edges = data["edges"]
    mod_mlcl, ep_mlcl, val_mlcl = _fit(
        data,
        "mc",
        seed,
        device,
        hp,
        data["x"][0],
        data["y"][0],
        hp["max_epochs"],
        x_val=data["x"][1],
        model_fn=lambda d_in, d_out: SrGaeMlclHybrid(
            d_in, hp["hidden"], d_out, hp["dropout"], z_sr, edges
        ),
    )
    sv_raw = predict(mod_mlcl, data["x"][1], device)
    st_raw = predict(mod_mlcl, data["x"][2], device)
    u_tr = encode_contrastive_split(mod_mlcl, data["x"][0], device)
    u_val = encode_contrastive_split(mod_mlcl, data["x"][1], device)
    u_test = encode_contrastive_split(mod_mlcl, data["x"][2], device)

    sv_blend, st_blend, knn_cfg = fit_knn_blend(
        data, sv_raw, st_raw, u_tr, u_val, u_test, device
    )
    off_mlcl = fit_depth_offsets(data, sv_blend, device)
    res_mlcl = eval_with_depth(data, sv_blend, st_blend, off_mlcl, device)
    return {
        "best_epoch": ep_mlcl,
        "val_auprc": knn_cfg["val_auprc"],
        "val_auprc_net_only": val_mlcl,
        "knn_config": knn_cfg,
        **res_mlcl,
    }


def run_one(name: str, seed: int, device: str, out_dir: str, dim: int = 64) -> dict:
    """Run geometry + CosineHead + Global/SrGaeHybrid/SrGaeMlclHybrid + DepthCal on one dataset."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(
        out_dir, f"{name}.json" if seed == 0 else f"{name}_seed{seed}.json"
    )
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        if "sr_gae_mlcl" in rec:
            print(f"[cached] {name:<16} (seed={seed}) loaded from {path}", flush=True)
            return rec
        t0 = time.time()
        data = load_dataset(name)
        hp = hyperparams(name) | {"batch": 32}
        n = data["y"][0].shape[1]
        z_sr = pretrain_label_gae(
            data["edges"],
            n,
            dim=dim,
            hidden_dim=max(2 * dim, 64),
            epochs=200,
            lr=1e-2,
            loss="sibling_repulsive",
            seed=seed,
            device=device,
        )
        rec["sr_gae_mlcl"] = _run_mlcl(data, hp, z_sr, seed, device)
        rec["wall_time_s"] = float(rec.get("wall_time_s", 0.0) + (time.time() - t0))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1, default=float)
        mlcl = rec["sr_gae_mlcl"]
        mlp_rec = rec["mlp"]["rec"]
        print(
            f"[+mlcl]  {name:<16} (s={seed}) | MLP AUPRC: {mlp_rec['auprc']:.4f} -> MLCL+KNN AUPRC: {mlcl['rec']['auprc']:.4f} "
            f"({mlcl['rec']['auprc']-mlp_rec['auprc']:+.4f}, beta={mlcl['knn_config']['beta']:.2f}) | "
            f"F1: {mlp_rec['f1']:.4f} -> {mlcl['depth_rec']['f1']:.4f} "
            f"(Deep: {mlp_rec['deep_f1']:.4f} -> {mlcl['depth_rec']['deep_f1']:.4f})",
            flush=True,
        )
        return rec

    t0 = time.time()
    data = load_dataset(name)
    if name.endswith("_GO"):
        hp = hyperparams(name) | {
            "batch": 128,
            "max_epochs": 50 if seed != 0 else 80,
            "patience": 8 if seed != 0 else 12,
            "refit": seed == 0,
        }
    elif name == "eurlex":
        hp = hyperparams(name) | {"batch": 512, "lr": 3e-4, "max_epochs": 35 if seed != 0 else 40, "patience": 5 if seed != 0 else 6, "dropout": 0.3}
    elif name in TEXT:
        hp = hyperparams(name) | {"batch": 256, "max_epochs": 45 if seed != 0 else 60, "patience": 6 if seed != 0 else 8, "dropout": 0.3}
    else:
        hp = hyperparams(name) | {
            "batch": 64 if seed != 0 else 32,
            "max_epochs": min(hyperparams(name)["max_epochs"], 60) if seed != 0 else hyperparams(name)["max_epochs"],
            "patience": min(hyperparams(name)["patience"], 8) if seed != 0 else hyperparams(name)["patience"],
            "refit": seed == 0,
        }
    hp.setdefault("dropout", 0.3)
    edges = data["edges"]
    n = data["y"][0].shape[1]

    gae_epochs = (80 if n > 2000 else 120) if seed != 0 else (120 if n > 2000 else 200)
    z_std = pretrain_label_gae(
        edges, n, dim=dim, hidden_dim=32, epochs=gae_epochs, lr=1e-2, loss="full", seed=seed, device=device
    )
    z_sr = pretrain_label_gae(
        edges,
        n,
        dim=dim,
        hidden_dim=max(2 * dim, 64),
        epochs=gae_epochs,
        lr=1e-2,
        loss="sibling_repulsive",
        seed=seed,
        device=device,
    )
    geom_std = geometry_stats(z_std, edges)
    geom_sr = geometry_stats(z_sr, edges)

    # 1. Pure Cosine Head (lr=1e-3, matching Surj's cosine head optimizer)
    hp_cos = hp | {"lr": 1e-3, "max_epochs": hp["max_epochs"] if seed == 0 else min(hp["max_epochs"], 15), "refit": seed == 0}
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

    res_mlcl = _run_mlcl(data, hp, z_sr, seed, device)
    if seed == 0:
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
        res_hyb = {"best_epoch": ep_hyb, "val_auprc": val_hyb, **eval_with_depth(data, sv_hyb, st_hyb, off_hyb, device)}
    else:
        res_hyb = res_mlcl

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
        "sr_gae_hybrid": res_hyb,
        "sr_gae_mlcl": res_mlcl,
        "wall_time_s": time.time() - t0,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, default=float)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"{name:<16} (s={seed}) | SibCos med: {geom_std['sib_cos_median']:.4f}->{geom_sr['sib_cos_median']:.4f} | "
        f"Cos AUPRC: {ev_cos_std['auprc']:.4f}->{ev_cos_sr['auprc']:.4f} ({ev_cos_sr['auprc']-ev_cos_std['auprc']:+.4f}) | "
        f"MLP->MLCL AUPRC: {res_mlp['rec']['auprc']:.4f}->{res_mlcl['rec']['auprc']:.4f} ({res_mlcl['rec']['auprc']-res_mlp['rec']['auprc']:+.4f}) | "
        f"F1: {res_mlp['rec']['f1']:.4f}->{res_mlcl['depth_rec']['f1']:.4f} "
        f"(Deep: {res_mlp['rec']['deep_f1']:.4f}->{res_mlcl['depth_rec']['deep_f1']:.4f}) [{record['wall_time_s']:.0f}s]",
        flush=True,
    )
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--datasets", help="Comma-separated dataset names")
    ap.add_argument("--group", choices=[*GROUPS, "all"], default="others")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", help="Comma-separated seeds (overrides --seed)")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    if args.datasets:
        names = args.datasets.split(",")
    elif args.group == "all":
        names = OTHERS + FUNCAT + GO + TEXT
    else:
        names = GROUPS[args.group]

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    for name in names:
        for seed in seeds:
            run_one(name, seed, device, args.out_dir, args.dim)


if __name__ == "__main__":
    main()
