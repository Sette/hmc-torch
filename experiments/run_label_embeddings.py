#!/usr/bin/env python3
"""Does knowledge about the labels close the gap the constraint does not?

Runs :class:`hmc.models.hierarchical.label_embedding.LabelEmbeddingHead` on
FunCat and Gene Ontology under exactly the protocol of ``run_protocol.py``
(early stopping and threshold on validation, refit on train + validation,
three seeds), training with BCE and reconciling at inference.

Heads: ``free`` (per-class vectors, the baseline), ``graph`` (GCN over the
hierarchy) on every dataset; on GO also ``text`` (frozen SPECTER2 embeddings
of each term's name and definition from ``go.obo``) and ``text+graph``.
FunCat category names are no longer published, so it gets no text heads.

Writes ``output/label_embeddings/<dataset>.json``.

Usage:
    PYTHONPATH=src python experiments/run_label_embeddings.py --group go
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
from hmc.models.hierarchical.label_embedding import (  # noqa: E402  pylint: disable=wrong-import-position
    LabelEmbeddingHead,
    go_term_texts,
    normalized_adjacency,
)
from run_protocol import (  # noqa: E402  pylint: disable=wrong-import-position
    FUNCAT,
    GO,
    hyperparams,
    load_dataset,
    train_mlp,
)

OBO_PATH = "./data/ontologies/go.obo"


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
    args = parser.parse_args()

    names = args.datasets.split(",") if args.datasets else (
        FUNCAT if args.group == "funcat" else GO if args.group == "go" else FUNCAT + GO)
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    for name in names:
        path = os.path.join(args.out_dir, f"{name}.json")
        if os.path.exists(path):
            print(f"[skip] {path} exists")
            continue
        data = load_dataset(name)
        hp = hyperparams(name)
        n_classes = data["y"][0].shape[1]
        adjacency = normalized_adjacency(data["edges"], n_classes).to(device)
        modes = ["free", "graph"]
        text = None
        if name.endswith("_GO"):
            text = torch.from_numpy(go_text_embeddings(data)).to(device)
            modes += ["text", "text+graph"]
        record = {"dataset": name, "hyperparams": hp, "runs": []}
        print(f"\n=== {name}: {n_classes} classes, heads {modes}", flush=True)
        for seed in seeds:
            for mode in modes:
                def model_fn(d_in, d_out, mode=mode):
                    return LabelEmbeddingHead(
                        d_in, hp["hidden"], d_out, mode=mode,
                        adjacency=adjacency if "graph" in mode else None,
                        text_embeddings=text if "text" in mode else None,
                    )
                t0 = time.time()
                run = train_mlp(data, "bce", seed, device, hp, model_fn=model_fn)
                run["head"] = mode
                record["runs"].append(run)
                ev = run["eval"]["rec"]
                print(f"  seed {seed:>3} {mode:<10} ep {run['best_epoch']:>3}  "
                      f"rec F1 {ev['f1']:.4f} AUPRC {ev['auprc']:.4f}  ({time.time() - t0:.0f}s)",
                      flush=True)
        with open(path, "w") as f:
            json.dump(record, f, indent=1, default=float)


if __name__ == "__main__":
    main()
