#!/usr/bin/env python3
"""Autoencoders in our global model, under the validation-only protocol of the paper.

Every variant keeps the paper's MLP trunk, MCLoss through the sparse R and reconciled
outputs (``run_protocol.py``); only the network changes:

  * ``gae_hybrid@D`` -- our free output layer plus Surj's branch (projection to D, cosine with
    frozen graph-autoencoder label embeddings, FC(V, V)), added to the logits through a
    learnable scale initialised at 0: the model starts as the baseline MLP;
  * ``feat_dae`` / ``feat_dae+aux`` -- the two hidden layers pretrained as a denoising
    autoencoder of the (training) features, then fine-tuned; ``+aux`` keeps the
    reconstruction as an auxiliary loss (weight 0.1);
  * ``label_ae@λ`` -- a C2AE-style label autoencoder: features and label vectors are mapped to
    a shared latent space, decoded to the labels; λ weighs label reconstruction + latent
    matching.

Stages:
  * ``screen`` -- the FunCat datasets, seed 0, early stopping on validation only (no refit,
    the test split is not scored).  The baseline ``mlp`` is rerun in the same code path.
  * ``final``  -- one variant with ``train_mlp`` (refit, threshold on validation, test once),
    seeds 0, 42, 123.

Usage:
    PYTHONPATH=src python experiments/run_autoencoder.py --stage screen --datasets cellcycle_FUN
    PYTHONPATH=src python experiments/run_autoencoder.py --stage final --variant feat_dae --group funcat
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

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(__file__))

from run_protocol import (  # noqa: E402  pylint: disable=wrong-import-position
    FUNCAT,
    GO,
    OTHERS,
    MLP,
    _fit,
    hyperparams,
    load_dataset,
    train_mlp,
)

from hmc.models.hierarchical.graph_autoencoder import (  # noqa: E402  pylint: disable=wrong-import-position
    pretrain_label_gae,
)

OUT_DIR = "output/autoencoder"
GROUPS = {"funcat": FUNCAT, "go": GO, "others": OTHERS}
SCREEN_VARIANTS = ["mlp", "gae_hybrid@16", "gae_hybrid@64", "feat_dae", "feat_dae+aux",
                   "label_ae@0.1", "label_ae@1"]
SEEDS = (0, 42, 123)
DAE_MASK, DAE_EPOCHS, DAE_AUX = 0.2, 100, 0.1


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


class GaeHybrid(MLP):
    """The paper's MLP plus a Surj branch over frozen label embeddings, gated by a scale at 0."""

    def __init__(self, d_in, hidden, d_out, dropout, z):
        super().__init__(d_in, hidden, d_out, dropout)  # same layers, same init as MLP
        z = torch.as_tensor(z, dtype=torch.float32)
        self.register_buffer("z", nn.functional.normalize(z, dim=1))
        self.proj = nn.Linear(hidden, z.shape[1])
        self.fc = nn.Linear(d_out, d_out)
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        h = self.net[:-1](x)
        cos = nn.functional.normalize(self.proj(h), dim=1) @ self.z.T
        return torch.sigmoid(self.net[-1](h) + self.scale * self.fc(cos))


class FeatDAE(MLP):
    """The paper's MLP whose hidden layers are pretrained as a denoising feature autoencoder."""

    def __init__(self, d_in, hidden, d_out, dropout, aux_weight=0.0):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        super().__init__(d_in, hidden, d_out, dropout)
        self.decoder = nn.Linear(hidden, d_in)
        self.aux_weight = aux_weight

    def _reconstruction(self, x):
        keep = (torch.rand_like(x) >= DAE_MASK).float()  # features are standardized: 0 = mean
        return nn.functional.mse_loss(self.decoder(self.net[:-1](x * keep)), x)

    def pretrain(self, x: np.ndarray, device, epochs=DAE_EPOCHS, batch=64, lr=1e-3):
        """Denoising reconstruction of ``x`` (the training features) for ``epochs`` epochs."""
        self.to(device).train()
        params = list(self.net[:-1].parameters()) + list(self.decoder.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        xt = torch.from_numpy(x).to(device)
        losses = []
        for _ in range(epochs):
            perm = torch.randperm(len(xt), device=device)
            total = 0.0
            for i in range(0, len(perm), batch):
                loss = self._reconstruction(xt[perm[i:i + batch]])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(perm[i:i + batch])
            losses.append(total / len(xt))
        self.pretrain_losses = losses  # pylint: disable=attribute-defined-outside-init
        return self

    def aux_loss(self, xb, yb):  # pylint: disable=unused-argument
        if not self.aux_weight:
            return xb.new_zeros(())
        return self.aux_weight * self._reconstruction(xb)


class LabelAE(nn.Module):
    """C2AE-style: F(x) -> latent <- E(y); D(latent) -> labels.  Prediction = D(F(x))."""

    def __init__(self, d_in, hidden, d_out, dropout, eval_mask, weight):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        super().__init__()
        latent = max(hidden // 2, 32)
        self.features = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, latent))
        self.labels = nn.Sequential(nn.Linear(d_out, latent), nn.ReLU(), nn.Linear(latent, latent))
        self.decoder = nn.Linear(latent, d_out)
        self.register_buffer("eval_cols", torch.from_numpy(np.nonzero(eval_mask)[0]))
        self.weight = weight

    def forward(self, x):
        return torch.sigmoid(self.decoder(self.features(x)))

    def aux_loss(self, xb, yb):
        lat_y = self.labels(yb)
        rec = torch.sigmoid(self.decoder(lat_y))[:, self.eval_cols]
        bce = nn.functional.binary_cross_entropy(rec.clamp(1e-7, 1 - 1e-7), yb[:, self.eval_cols])
        return self.weight * (bce + nn.functional.mse_loss(self.features(xb), lat_y))


def model_fn_for(variant: str, data: dict, hp: dict, device, cache: dict):
    """``model_fn(d_in, d_out)`` for ``_fit``; ``None`` is the baseline MLP."""
    kind, _, arg = variant.partition("@")
    if kind == "mlp":
        return None
    hidden, dropout = hp["hidden"], hp.get("dropout", 0.3)
    if kind == "gae_hybrid":
        dim = int(arg)
        if dim not in cache:
            n = data["y"][0].shape[1]
            cache[dim] = pretrain_label_gae(data["edges"], n, dim=dim, hidden_dim=32, epochs=200,
                                            lr=1e-2, loss="full", seed=0, device=device)
        return lambda d_in, d_out: GaeHybrid(d_in, hidden, d_out, dropout, cache[dim])
    if kind in ("feat_dae", "feat_dae+aux"):
        aux = DAE_AUX if kind.endswith("+aux") else 0.0
        # Pretrained on the training features only, for the early-stopped fit and the refit.
        return lambda d_in, d_out: FeatDAE(d_in, hidden, d_out, dropout, aux).pretrain(
            data["x"][0], device)
    if kind == "label_ae":
        return lambda d_in, d_out: LabelAE(d_in, hidden, d_out, dropout, data["eval_mask"],
                                           float(arg))
    raise ValueError(f"unknown variant {variant!r}")


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)


def screen(names, variants, device, out_dir, max_epochs=None):
    for name in names:
        todo = [v for v in variants
                if not os.path.exists(os.path.join(out_dir, "screen", v, f"{name}.json"))]
        if not todo:
            continue
        data, cache = load_dataset(name), {}
        hp = hyperparams(name) | ({"max_epochs": max_epochs} if max_epochs else {})
        for variant in todo:
            t0 = time.time()
            _, epoch, val = _fit(data, "mc", 0, device, hp, data["x"][0], data["y"][0],
                                 hp["max_epochs"], x_val=data["x"][1],
                                 model_fn=model_fn_for(variant, data, hp, device, cache))
            rec = {"dataset": name, "variant": variant, "seed": 0, "hyperparams": hp,
                   "val_auprc": val, "best_epoch": epoch, "time_s": time.time() - t0}
            _dump(rec, os.path.join(out_dir, "screen", variant, f"{name}.json"))
            print(f"{name:<15} {variant:<14} val AUPRC {val:.4f}  ep {epoch:>3}  "
                  f"{rec['time_s']:.0f}s", flush=True)


def final(names, variant, device, out_dir, seeds, max_epochs=None):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    for name in names:
        path = os.path.join(out_dir, "final", variant, f"{name}.json")
        if os.path.exists(path):
            continue
        data, cache = load_dataset(name), {}
        hp = hyperparams(name) | ({"max_epochs": max_epochs} if max_epochs else {})
        record = {"dataset": name, "variant": variant, "hyperparams": hp, "runs": []}
        for seed in seeds:
            run = train_mlp(data, "mc", seed, device, hp,
                            model_fn=model_fn_for(variant, data, hp, device, cache))
            record["runs"].append(run)
            print(f"{name:<15} {variant:<14} seed {seed:>3}  rec AUPRC "
                  f"{run['eval']['rec']['auprc']:.4f}  ep {run['best_epoch']}", flush=True)
        _dump(record, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--stage", choices=("screen", "final"), required=True)
    ap.add_argument("--variant", help="final stage: the variant to run")
    ap.add_argument("--variants", default=",".join(SCREEN_VARIANTS), help="screen stage")
    ap.add_argument("--datasets")
    ap.add_argument("--group", choices=GROUPS, default="funcat")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--max_epochs", type=int, help="smoke tests only")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    names = args.datasets.split(",") if args.datasets else GROUPS[args.group]
    if args.stage == "screen":
        screen(names, args.variants.split(","), device, args.out_dir, args.max_epochs)
    else:
        if not args.variant:
            ap.error("--stage final needs --variant")
        final(names, args.variant, device, args.out_dir,
              [int(s) for s in args.seeds.split(",")], args.max_epochs)


if __name__ == "__main__":
    main()
