#!/usr/bin/env python3
"""Paired evaluation of SR-GAE + H-MLCL+KNN + DepthCal against the protocol baseline.

For each seed, retrains the protocol trunk (``run_protocol._fit``, same
hyperparameters and validation-selected ``best_epoch`` as
``output/protocol/{name}.json``) and scores both the baseline (trunk +
reconciliation) and the method (trunk blended with the prototype cosine and
MLCL-KNN scores of an H-MLCL projector) on that same trunk.  Blend weights,
depth offsets and the DepthCal on/off choice come from validation only; test
metrics are written as measured.  A random-prototype control replaces the
SR-GAE embeddings.  Results go to ``output/sr_gae_paired/{name}.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(__file__))

from run_protocol import (  # noqa: E402
    FUNCAT,
    GO,
    OTHERS,
    TEXT,
    _fit,
    auprc,
    hyperparams,
    load_dataset,
    predict,
)
from run_sr_gae_depth import (  # noqa: E402
    eval_with_depth,
    fit_depth_offsets,
)

from hmc.models.hierarchical.graph_autoencoder import (  # noqa: E402
    HierarchicalMultiLabelContrastiveLoss,
    contrastive_knn_predict,
    pretrain_label_gae,
)

OUT_DIR = "output/sr_gae_paired"
SEEDS = (0, 42, 123)
GROUPS = {"funcat": FUNCAT, "others": OTHERS, "go": GO, "text": TEXT}


class ContrastiveProjector(nn.Module):
    """2-layer residual projector mapping features to unit-norm H-MLCL space."""

    def __init__(self, d_in: int, hidden: int, dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        h = min(hidden, 512)
        self.net = nn.Sequential(
            nn.Linear(d_in, h),
            nn.LayerNorm(h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(x), p=2, dim=-1)


def _best_protocol_objective(name: str) -> str:
    """Return the SOTA objective reported in the main paper tables for ``name``."""
    if name in TEXT or name in OTHERS:
        return "mc" if name == "enron_others" else "bce"
    return "mc"


def _train_hmlcl_projector(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    z_sr: np.ndarray,
    edges: np.ndarray,
    hidden: int,
    seed: int,
    device: str,
    epochs: int = 35,
) -> tuple[ContrastiveProjector, torch.Tensor]:
    """Train the H-MLCL contrastive projector over SR-GAE label prototypes."""
    torch.manual_seed(seed)
    z_t = torch.from_numpy(z_sr).float()
    z_norm = nn.functional.normalize(z_t, p=2, dim=-1).to(device)
    proj = ContrastiveProjector(x_tr.shape[1], hidden, z_sr.shape[1]).to(device)
    cl_mod = HierarchicalMultiLabelContrastiveLoss(
        z_norm, edges, proto_weight=1.0, inst_weight=1.0
    ).to(device)
    opt = torch.optim.AdamW(proj.parameters(), lr=1e-3, weight_decay=1e-4)
    if len(x_tr) > 16_000:
        rng = np.random.default_rng(seed)
        sub = rng.choice(len(x_tr), size=16_000, replace=False)
        x_tr, y_tr = x_tr[sub], y_tr[sub]
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    proj.train()
    cl_mod.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            u = proj(xb)
            loss = cl_mod(u, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return proj, z_norm


@torch.no_grad()
def _encode_proj(proj: ContrastiveProjector, x: np.ndarray, device: str, batch: int = 1024) -> np.ndarray:
    proj.eval()
    chunks = [
        proj(torch.from_numpy(x[i : i + batch]).to(device)).cpu()
        for i in range(0, len(x), batch)
    ]
    return torch.cat(chunks, dim=0).numpy()


def _fit_proto_and_knn_blend(
    data: dict,
    s_val_raw: np.ndarray,
    s_test_raw: np.ndarray,
    u_tr: np.ndarray,
    u_val: np.ndarray,
    u_test: np.ndarray,
    z_norm: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Select (gamma_proto, beta_knn, k, temp) on validation split only."""
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
    s_val_t = torch.from_numpy(s_val_g).to(device)
    base_rec_t = sparse_r.reconcile(s_val_t)[:, eval_cols]
    best_auprc = auprc(y_val_flat, base_rec_t)
    best_cfg = {"gamma_proto": 0.0, "beta": 0.0, "k": 25, "temp": 0.1, "val_auprc": best_auprc}

    # 1. Prototype cosine probability p_proto = sigmoid(4.0 * (u @ z_norm.T))
    proto_val_g = 1.0 / (1.0 + np.exp(-4.0 * (u_val_g @ z_norm.T)))
    proto_val_t = torch.from_numpy(proto_val_g).to(device)

    for k, temp in ((15, 0.08), (25, 0.10), (40, 0.15)):
        knn_val = contrastive_knn_predict(u_tr, y_tr, u_val_g, k=k, temp=temp)
        knn_val_t = torch.from_numpy(knn_val).to(device)
        for gamma in (0.0, 0.02, 0.05):
            for beta in (0.0, 0.02, 0.04, 0.06, 0.10, 0.15):
                if gamma == 0.0 and beta == 0.0:
                    continue
                w0 = 1.0 - gamma - beta
                blend_t = w0 * s_val_t + gamma * proto_val_t + beta * knn_val_t
                rec_t = sparse_r.reconcile(blend_t)[:, eval_cols]
                sc = auprc(y_val_flat, rec_t)
                if sc > best_auprc + 1e-4:
                    best_auprc = sc
                    best_cfg = {
                        "gamma_proto": float(gamma),
                        "beta": float(beta),
                        "k": int(k),
                        "temp": float(temp),
                        "val_auprc": float(sc),
                    }

    gamma = best_cfg["gamma_proto"]
    beta = best_cfg["beta"]
    w0 = 1.0 - gamma - beta
    best_val_raw = w0 * s_val_raw
    best_test_raw = w0 * s_test_raw
    if gamma > 0.0:
        proto_val_full = 1.0 / (1.0 + np.exp(-4.0 * (u_val @ z_norm.T)))
        proto_test_full = 1.0 / (1.0 + np.exp(-4.0 * (u_test @ z_norm.T)))
        best_val_raw = best_val_raw + gamma * proto_val_full
        best_test_raw = best_test_raw + gamma * proto_test_full
    if beta > 0.0:
        knn_val_full = contrastive_knn_predict(
            u_tr, y_tr, u_val, k=int(best_cfg["k"]), temp=float(best_cfg["temp"])
        )
        knn_test_full = contrastive_knn_predict(
            u_tr, y_tr, u_test, k=int(best_cfg["k"]), temp=float(best_cfg["temp"])
        )
        best_val_raw = best_val_raw + beta * knn_val_full
        best_test_raw = best_test_raw + beta * knn_test_full
    return best_val_raw, best_test_raw, best_cfg


def _protocol_best_epochs(name: str, obj: str) -> dict[int, dict]:
    """Per-seed protocol runs (``best_epoch`` and reference metrics) for ``obj``, if available."""
    p = os.path.join("output/protocol", f"{name}.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        proto = json.load(f)
    return {int(r["seed"]): r for r in proto["runs"] if r["objective"] == obj}


def _random_prototypes(n: int, dim: int, seed: int) -> np.ndarray:
    """Control: random Gaussian prototypes with no information about the hierarchy."""
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def _variant(data, sv_raw, st_raw, x, y_tr, z, edges, hidden, seed, device) -> dict:
    """Train the projector on prototypes ``z`` and evaluate the validation-selected blend."""
    proj, z_norm_t = _train_hmlcl_projector(x[0], y_tr, z, edges, hidden, seed, device)
    u_tr, u_val, u_test = (_encode_proj(proj, xs, device) for xs in x)
    sv_blend, st_blend, cfg = _fit_proto_and_knn_blend(
        data, sv_raw, st_raw, u_tr, u_val, u_test, z_norm_t.cpu().numpy(), device
    )
    offsets = fit_depth_offsets(data, sv_blend, device)
    res = eval_with_depth(data, sv_blend, st_blend, offsets, device)
    # DepthCal on/off chosen on validation F1 only; the test split never enters a choice.
    use_depth = res["depth_rec"]["val_f1"] > res["rec"]["val_f1"]
    res["selected"] = "depth_rec" if use_depth else "rec"
    res["blend_config"] = cfg
    return res


def run_sota_one(name: str, device: str, out_dir: str, dim: int = 64, force: bool = False) -> dict:
    """Paired 3-seed evaluation of SR-GAE + H-MLCL+KNN (+ DepthCal) on ``name``.

    The baseline and the method share one trunk per seed, trained with the
    protocol's own ``_fit`` and hyperparameters, so the comparison is paired.
    Every number is written exactly as measured: nothing is clamped to the
    baseline, no result is kept from an earlier execution, and every choice
    (blend weights, depth offsets, DepthCal on/off) is made on validation.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.json")
    if os.path.exists(out_path) and not force:
        print(f"[cached] {name:<16} loaded from {out_path}", flush=True)
        with open(out_path, encoding="utf-8") as f:
            return json.load(f)

    t0 = time.time()
    data = load_dataset(name)
    hp = hyperparams(name)
    obj = _best_protocol_objective(name)
    proto_by_seed = _protocol_best_epochs(name, obj)
    edges = data["edges"]
    n = data["y"][0].shape[1]
    x_tr, x_val, _ = data["x"]
    y_tr, y_val = data["y"][0], data["y"][1]

    seed_runs = []
    for seed in SEEDS:
        proto = proto_by_seed.get(seed)
        if proto is not None:
            best_ep = int(proto["best_epoch"])
            mod_val, _, _ = _fit(data, obj, seed, device, hp, x_tr, y_tr, best_ep)
        else:
            mod_val, best_ep, _ = _fit(
                data, obj, seed, device, hp, x_tr, y_tr, hp["max_epochs"], x_val=x_val
            )
        sv_raw = predict(mod_val, x_val, device)
        if hp["refit"]:
            mod_te, _, _ = _fit(
                data, obj, seed, device, hp,
                np.concatenate([x_tr, x_val]), np.concatenate([y_tr, y_val]), best_ep,
            )
        else:
            mod_te = mod_val
        st_raw = predict(mod_te, data["x"][2], device)

        zero = {int(lvl): 0.0 for lvl in set(data["depth"][data["eval_mask"]].tolist())}
        base = eval_with_depth(data, sv_raw, st_raw, zero, device)["rec"]

        z_sr = pretrain_label_gae(
            edges, n, dim=dim, hidden_dim=max(2 * dim, 64), epochs=120 if n > 2000 else 200,
            lr=1e-2, loss="sibling_repulsive", seed=seed, device=device,
        )
        args = (data, sv_raw, st_raw, data["x"], y_tr)
        ours = _variant(*args, z_sr, edges, hp["hidden"], seed, device)
        ctrl = _variant(*args, _random_prototypes(n, dim, seed), edges, hp["hidden"], seed, device)

        seed_runs.append(
            {
                "seed": seed,
                "best_epoch": best_ep,
                "baseline_rec": base,
                "protocol_rec": (
                    {k: float(proto["eval"]["rec"][k]) for k in ("auprc", "f1")} if proto else None
                ),
                "sr_gae_mlcl": ours,
                "random_proto_mlcl": ctrl,
            }
        )
        print(
            f"  {name:<16} seed {seed:>3} ep {best_ep:>3} | base AUPRC {base['auprc']:.4f} "
            f"F1 {base['f1']:.4f} | ours {ours['rec']['auprc']:.4f} "
            f"{ours[ours['selected']]['f1']:.4f} | random {ctrl['rec']['auprc']:.4f} "
            f"{ctrl[ctrl['selected']]['f1']:.4f}",
            flush=True,
        )

    def _col(fn):
        vals = [fn(r) for r in seed_runs]
        return float(np.mean(vals)), float(np.std(vals))

    summary = {}
    for key, fn in {
        "base_auprc": lambda r: r["baseline_rec"]["auprc"],
        "base_f1": lambda r: r["baseline_rec"]["f1"],
        "base_deep_f1": lambda r: r["baseline_rec"]["deep_f1"],
        "mlcl_auprc": lambda r: r["sr_gae_mlcl"]["rec"]["auprc"],
        "mlcl_f1": lambda r: r["sr_gae_mlcl"]["rec"]["f1"],
        "mlcl_depth_auprc": lambda r: r["sr_gae_mlcl"]["depth_rec"]["auprc"],
        "mlcl_depth_f1": lambda r: r["sr_gae_mlcl"]["depth_rec"]["f1"],
        "mlcl_depth_deep_f1": lambda r: r["sr_gae_mlcl"]["depth_rec"]["deep_f1"],
        "ours_f1": lambda r: r["sr_gae_mlcl"][r["sr_gae_mlcl"]["selected"]]["f1"],
        "random_auprc": lambda r: r["random_proto_mlcl"]["rec"]["auprc"],
        "random_f1": lambda r: r["random_proto_mlcl"][r["random_proto_mlcl"]["selected"]]["f1"],
    }.items():
        summary[f"{key}_mean"], summary[f"{key}_std"] = _col(fn)

    record = {
        "dataset": name,
        "objective": obj,
        "hyperparams": hp,
        "protocol": "paired trunk; all selection on validation; test metrics unmodified",
        "summary": summary,
        "runs": seed_runs,
        "wall_time_s": time.time() - t0,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, default=float)
    s = summary
    print(
        f"{name:<16} ({obj}) | AUPRC base {s['base_auprc_mean']:.4f} -> ours {s['mlcl_auprc_mean']:.4f} "
        f"({s['mlcl_auprc_mean'] - s['base_auprc_mean']:+.4f}) random {s['random_auprc_mean']:.4f} | "
        f"F1 base {s['base_f1_mean']:.4f} -> ours {s['ours_f1_mean']:.4f} "
        f"({s['ours_f1_mean'] - s['base_f1_mean']:+.4f}) [{record['wall_time_s']:.0f}s]",
        flush=True,
    )
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--datasets", help="Comma-separated dataset names")
    ap.add_argument("--group", choices=[*GROUPS, "all"], default="all")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    if args.datasets:
        names = args.datasets.split(",")
    elif args.group == "all":
        names = OTHERS + FUNCAT + TEXT + GO
    else:
        names = GROUPS[args.group]

    for name in names:
        run_sota_one(name, device, args.out_dir, args.dim, args.force)


if __name__ == "__main__":
    main()
