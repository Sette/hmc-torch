#!/usr/bin/env python3
"""Does knowledge about the labels close the gap the constraint does not?

Runs :class:`hmc.models.hierarchical.label_embedding.LabelEmbeddingHead` on
FunCat and Gene Ontology under exactly the protocol of ``run_protocol.py``
(early stopping and threshold on validation, refit on train + validation,
three seeds), training with ``--objective`` (BCE or MCLoss through the sparse
R) and reconciling at inference.

Heads: ``free`` (per-class vectors, the baseline), ``graph`` (GCN over the
hierarchy), ``gae`` / ``gae+free`` (Surj-style: frozen graph-autoencoder label
embeddings scored by cosine similarity) on every dataset; ``gae-init`` /
``gae-reg`` (the free head with the autoencoder embeddings as an initialization
or a penalty; ``gae-reg@0.1`` sets the penalty weight) and ``gae-2stage``
(Surj's two stages: regress the label-set embedding, then freeze the encoder
and fit only the scale and biases) on request with ``--modes``; on GO also ``text``
(frozen SPECTER2 embeddings of each term's name and definition from
``go.obo``) and ``text+graph``.  FunCat category names are no longer
published, so it gets no text heads.

``--train_frac`` keeps a fixed random fraction of the training split
(validation and test unchanged, no refit) to test the low-data regime, where
Surj claims its gains.  Every run also reports the test AUPRC on the rare
classes (fewer than ``RARE_MAX`` positives in the full training split).

Writes ``output/label_embeddings/<objective>/<dataset>[_frac<f>].json``.

Usage:
    PYTHONPATH=src python experiments/run_label_embeddings.py --group go
    PYTHONPATH=src python experiments/run_label_embeddings.py --datasets cellcycle_FUN --objective mc
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(__file__))

from hmc.datasets.arxiv.manager import (  # noqa: E402  pylint: disable=wrong-import-position
    compute_transformer_embeddings,
)
from hmc.models.hierarchical.graph_autoencoder import (  # noqa: E402  pylint: disable=wrong-import-position
    pretrain_label_gae,
)
from hmc.models.hierarchical.label_embedding import (  # noqa: E402  pylint: disable=wrong-import-position
    LabelEmbeddingHead,
    go_term_texts,
    normalized_adjacency,
    pretrain_two_stage,
)
from run_protocol import (  # noqa: E402  pylint: disable=wrong-import-position
    FUNCAT,
    GO,
    auprc,
    hyperparams,
    load_dataset,
    train_mlp,
)

OBO_PATH = "./data/ontologies/go.obo"
RARE_MAX = 10


def go_text_embeddings(data, cache_dir="./data/ontologies/.cache"):
    """Frozen SPECTER2 embeddings of each GO class's name and definition."""
    names = [n or "" for n in data["node_names"]]
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{data['name']}_specter2.npy")
    if os.path.exists(path):
        return np.load(path)
    texts = go_term_texts(names, OBO_PATH)
    emb = compute_transformer_embeddings(texts, "allenai/specter2_base", model_cache_dir="./models")
    np.save(path, emb)
    return emb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["funcat", "go", "all"], default="all")
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--seeds", default="0,42,123")
    parser.add_argument("--out_dir", default="./output/label_embeddings")
    parser.add_argument("--objective", choices=["bce", "mc"], default="bce")
    parser.add_argument("--modes", default=None,
                        help="Comma-separated heads (default: free,graph,gae,gae+free, + text heads on GO)")
    parser.add_argument("--max_epochs", type=int, default=None, help="Override (smoke tests)")
    parser.add_argument("--train_frac", type=float, default=1.0,
                        help="Fraction of the training split to keep (seed 0; disables the refit)")
    args = parser.parse_args()
    out_dir = os.path.join(args.out_dir, args.objective)

    names = args.datasets.split(",") if args.datasets else (
        FUNCAT if args.group == "funcat" else GO if args.group == "go" else FUNCAT + GO)
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    for name in names:
        suffix = "" if args.train_frac >= 1 else f"_frac{args.train_frac:g}"
        path = os.path.join(out_dir, f"{name}{suffix}.json")
        if os.path.exists(path):
            print(f"[skip] {path} exists")
            continue
        data = load_dataset(name)
        hp = hyperparams(name)
        if args.max_epochs:
            hp["max_epochs"] = args.max_epochs
        rare = data["eval_mask"] & (data["y"][0].sum(0) < RARE_MAX)
        if args.train_frac < 1:
            n_train = len(data["y"][0])
            keep = np.sort(np.random.default_rng(0).choice(
                n_train, max(1, int(round(args.train_frac * n_train))), replace=False))
            data["x"] = [data["x"][0][keep], *data["x"][1:]]
            data["y"] = [data["y"][0][keep], *data["y"][1:]]
            hp["refit"] = False
        n_classes = data["y"][0].shape[1]
        adjacency = normalized_adjacency(data["edges"], n_classes).to(device)
        modes = args.modes.split(",") if args.modes else (
            ["free", "graph", "gae", "gae+free"] + (["text", "text+graph"] if name.endswith("_GO") else []))
        text = None
        if any("text" in m for m in modes):
            text = torch.from_numpy(go_text_embeddings(data)).to(device)
        gae = None
        if any("gae" in m for m in modes):
            gae = torch.from_numpy(pretrain_label_gae(
                data["edges"], n_classes, depth=data["depth"], device=device)).to(device)
        record = {"dataset": name, "objective": args.objective, "hyperparams": hp,
                  "train_frac": args.train_frac, "n_train": len(data["y"][0]),
                  "n_rare": int(rare.sum()), "runs": []}
        print(f"\n=== {name}: {n_classes} classes ({int(rare.sum())} rare), "
              f"{len(data['y'][0])} train, heads {modes}", flush=True)
        for seed in seeds:
            for spec in modes:
                mode, _, weight = spec.partition("@")
                stage1 = {}

                def model_fn(d_in, d_out, mode=mode, weight=weight, stage1=stage1):
                    head = LabelEmbeddingHead(
                        d_in, hp["hidden"], d_out, mode=mode,
                        adjacency=adjacency if "graph" in mode else None,
                        text_embeddings=text if "text" in mode else None,
                        label_features=gae if "gae" in mode else None,
                        dropout=hp.get("dropout", 0.3),
                        prior_weight=float(weight) if weight else 1e-3,
                    ).to(device)
                    if mode == "gae-2stage":
                        # Stage 1 runs once per seed on the training split; the
                        # refit reuses the same frozen encoder.
                        if not stage1:
                            stage1["epochs"] = pretrain_two_stage(
                                head, data["x"][0], data["y"][0], data["x"][1], data["y"][1])
                            stage1["state"] = {k: v.clone() for k, v in head.state_dict().items()}
                        else:
                            head.load_state_dict(stage1["state"])
                            head.freeze_encoder()
                    return head
                t0 = time.time()
                run = train_mlp(data, args.objective, seed, device, hp, model_fn=model_fn,
                                keep_scores=True)
                scores = run.pop("rec_test_scores")
                run.pop("raw_test_scores")
                run["rare_auprc"] = auprc(data["y"][2][:, rare], scores[:, rare])
                run["head"] = spec
                if stage1:
                    run["stage1_epochs"] = stage1["epochs"]
                record["runs"].append(run)
                ev = run["eval"]["rec"]
                print(f"  seed {seed:>3} {spec:<12} ep {run['best_epoch']:>3}  "
                      f"rec F1 {ev['f1']:.4f} AUPRC {ev['auprc']:.4f} rare {run['rare_auprc']:.4f}  "
                      f"({time.time() - t0:.0f}s)",
                      flush=True)
        with open(path, "w") as f:
            json.dump(record, f, indent=1, default=float)


if __name__ == "__main__":
    main()
