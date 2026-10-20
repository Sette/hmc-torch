"""C-HMCNN's data pipeline, which Surj adopted (Yang & Howe, WWW'22, Sec. 4.4).

The ARFF files are parsed by C-HMCNN's own ``utils/parser.py`` (imported unmodified through
the shim in ``experiments/compare_chmcnn_loader.py``) and preprocessed as ``C-HMCNN/main.py``
does: mean imputation and a StandardScaler fit on train + validation (train only for the
``_others`` datasets, which ship without a validation file).  Surj's Table 1 gives those
datasets a 70/30 train/validation split of the training file; we draw it with seed 0.

Run every script from the repository root (C-HMCNN's parser reads ``./C-HMCNN/HMC_data``).
"""

from __future__ import annotations

import os
import sys

import networkx as nx
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

from compare_chmcnn_loader import (  # noqa: E402  pylint: disable=wrong-import-position
    EPOCHS,
    FUNCAT,
    GO,
    HIDDEN,
    OTHERS,
    PUBLISHED as CHMCNN_PUBLISHED,
    THEIR_DATASETS,
    _preprocess,
    load_theirs,
)

OTHERS_VAL_FRAC = 0.3


def arff_files(name: str) -> list[str]:
    """The ARFF files of a benchmark (for the checksums stored with each result)."""
    return [p for p in THEIR_DATASETS[name][1:] if isinstance(p, str)]


def published_epochs(name: str) -> int:
    """C-HMCNN's hard-coded number of epochs (found on validation, used on train + val)."""
    data_name, ontology = name.split("_")
    return EPOCHS[ontology][data_name]


def ancestor_pairs(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(ancestor, descendant) columns of every pair; ``a[i, j] = 1`` means j is i's parent."""
    g = nx.DiGraph(a)
    anc, desc = [], []
    for node in g.nodes():
        for x in nx.descendants(g, node):  # edges point child -> parent
            anc.append(x)
            desc.append(node)
    return np.asarray(anc, dtype=np.int64), np.asarray(desc, dtype=np.int64)


def load(name: str) -> dict:
    """Features, labels and hierarchy of one benchmark, exactly as C-HMCNN feeds them."""
    train, val, test = load_theirs(name)
    if val is None:  # _others: preprocess on the whole training file, then hold out 30%
        x_tr, _, x_te = _preprocess("chmcnn", train.X.astype(float), None, test.X.astype(float))
        y_tr_full = np.asarray(train.Y)
        perm = np.random.default_rng(0).permutation(len(x_tr))
        n_val = int(round(OTHERS_VAL_FRAC * len(x_tr)))
        v, t = perm[:n_val], perm[n_val:]
        x = (x_tr[t], x_tr[v], x_te)
        y = (y_tr_full[t], y_tr_full[v], np.asarray(test.Y))
    else:
        x = _preprocess("chmcnn", train.X.astype(float), val.X.astype(float),
                        test.X.astype(float))
        y = (np.asarray(train.Y), np.asarray(val.Y), np.asarray(test.Y))
    a = np.asarray(train.A)
    anc, desc = ancestor_pairs(a)
    data_name, ontology = name.split("_")
    return {
        "name": name,
        "x": tuple(xi.astype(np.float32) for xi in x),
        "y": tuple(yi.astype(np.float32) for yi in y),
        "a": a,
        "to_eval": np.asarray(train.to_eval, dtype=bool),
        "terms": list(train.terms),
        "anc": anc,
        "desc": desc,
        "hidden_dim": HIDDEN[ontology][data_name],
        "is_go": ontology == "GO",
    }


def leaves(a: np.ndarray) -> np.ndarray:
    """Columns nobody points to as a parent."""
    return np.nonzero(np.asarray(a).sum(axis=0) == 0)[0]


def subset_labels(data: dict, keep: np.ndarray) -> dict:
    """``data`` restricted to the label columns ``keep`` (sorted), hierarchy included."""
    keep = np.sort(keep)
    a = data["a"][np.ix_(keep, keep)]
    anc, desc = ancestor_pairs(a)
    return data | {
        "y": tuple(y[:, keep] for y in data["y"]),
        "a": a,
        "to_eval": data["to_eval"][keep],
        "terms": [data["terms"][i] for i in keep],
        "anc": anc,
        "desc": desc,
    }


def subsample(data: dict, frac: float, seed: int = 0) -> dict:
    """Keep ``frac`` of the training and validation examples (Surj Fig. 5)."""
    rng = np.random.default_rng(seed)
    x, y = list(data["x"]), list(data["y"])
    for split in (0, 1):
        idx = np.sort(rng.choice(len(x[split]), int(round(frac * len(x[split]))), replace=False))
        x[split], y[split] = x[split][idx], y[split][idx]
    return data | {"x": tuple(x), "y": tuple(y)}


__all__ = ["CHMCNN_PUBLISHED", "FUNCAT", "GO", "OTHERS", "ancestor_pairs", "leaves", "load",
           "subsample", "subset_labels"]
