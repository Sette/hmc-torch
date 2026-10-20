"""Training recipes: every knob of a run, named and stored in ``recipes.json``.

A run is fully determined by (recipe, dataset, seed, code commit, data files).  Each result
JSON stores all of them, so any number in ``results.md`` can be traced back and rerun.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RECIPES_FILE = os.path.join(HERE, "recipes.json")


@dataclasses.dataclass(frozen=True)
class Recipe:  # pylint: disable=too-many-instance-attributes
    """One training configuration.

    net: ``chmcnn`` (C-HMCNN's 3 FC + sigmoid), ``onehot`` (Surj's head with Z = I) or
        ``surj`` (Surj's head with the GAE embeddings).
    hidden: ``"chmcnn"`` for C-HMCNN's per-dataset size, or an integer.
    mcm / mcloss: C-HMCNN's max-constraint module at inference / MCLoss in training.
    epochs: ``"published"`` (C-HMCNN's hard-coded counts, trained on train + validation),
        ``"select"`` (early stopping on validation, then retrain on train + validation for
        that many epochs) or ``"select_norefit"`` (keep the train-only model).
    """

    net: str = "surj"
    lr: float = 1e-3
    batch: int = 4
    dropout: float = 0.7
    weight_decay: float = 1e-5
    hidden: str | int = "chmcnn"
    gae_dim: int = 16
    gae_epochs: int = 200
    mcm: bool = False
    mcloss: bool = False
    epochs: str = "select"
    max_epochs: int = 300
    patience: int = 20

    def with_overrides(self, overrides: dict) -> "Recipe":
        fields = {f.name: f.type for f in dataclasses.fields(self)}
        clean = {}
        for k, v in overrides.items():
            if k not in fields:
                raise KeyError(f"unknown recipe field {k!r}")
            clean[k] = _parse(k, v) if isinstance(v, str) else v
        return dataclasses.replace(self, **clean)


def _parse(key: str, value: str):
    if key in ("net", "epochs"):
        return value
    if key == "hidden":
        return value if value == "chmcnn" else int(value)
    if key in ("mcm", "mcloss"):
        return value.lower() in ("1", "true", "yes")
    if key in ("batch", "gae_dim", "gae_epochs", "max_epochs", "patience"):
        return int(value)
    return float(value)


def load_recipe(name: str, overrides: dict | None = None) -> Recipe:
    with open(RECIPES_FILE, encoding="utf-8") as f:
        table = json.load(f)
    if name not in table:
        raise KeyError(f"recipe {name!r} not in recipes.json ({', '.join(table)})")
    return Recipe().with_overrides(table[name]["set"]).with_overrides(overrides or {})


def parse_sets(items: list[str] | None) -> dict:
    """``["lr=1e-4", "batch=32"]`` -> ``{"lr": "1e-4", "batch": "32"}``."""
    return dict(item.split("=", 1) for item in items or [])


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(files: list[str]) -> dict:
    """Code commit, environment and data checksums of a run."""
    root = os.path.dirname(HERE)

    def git(*args):
        try:
            return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--", "surj_repro", "src",
                          "experiments/compare_chmcnn_loader.py")),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "data_md5": {os.path.basename(p): _md5(p) for p in files},
    }
