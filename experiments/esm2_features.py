"""ESM-2 features for the ``seq_FUN`` / ``seq_GO`` benchmarks, plugged into ``run_protocol``.

Every experiment script loads its data through ``run_protocol.load_dataset(name)``, which returns
the benchmark's hand-crafted features.  :func:`install` replaces that function by one that swaps
``data["x"]`` for the frozen ESM-2 matrices of :func:`hmc.features.protein.orf_embedding_features`
(same ORF mapping, embedding cache and standardization as ``run_protein_lm.py``), so the scripts
run unchanged.  Hierarchy, labels and splits are not touched.

:func:`write_protocol_esm2` writes ``output/protocol_esm2/<dataset>.json`` in the format of
``output/protocol/`` from the ESM-2 runs of ``run_protein_lm.py``: scripts that read the best
epochs of the protocol (``run_sr_gae_sota.py``) must read these, not the epochs of the benchmark
features (``HMC_PROTOCOL_DIR``).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(__file__))

ESM_DATASETS = ("seq_FUN", "seq_GO")
PROTEOME = "data/yeast/raw/orf_trans_all.fasta.gz"
IDS_DIR = "data/yeast"
PROTEIN_LM_DIR = "output/protein_lm"
PROTOCOL_ESM2_DIR = "output/protocol_esm2"


def install(model: str | None = None) -> None:
    """Make ``run_protocol.load_dataset`` return ESM-2 features for ``seq_FUN`` / ``seq_GO``."""
    import run_protocol  # pylint: disable=import-outside-toplevel
    from hmc.features.protein import (  # pylint: disable=import-outside-toplevel
        DEFAULT_PROTEIN_MODEL,
        ProteinFeatureEncoder,
        orf_embedding_features,
    )
    from hmc.features.yeast_orf import (  # pylint: disable=import-outside-toplevel
        read_orf_ids,
        read_proteome,
    )

    original = run_protocol.load_dataset
    state: dict = {}

    def load_dataset(name: str) -> dict:
        data = original(name)
        if name in ESM_DATASETS:
            if "encoder" not in state:
                state["encoder"] = ProteinFeatureEncoder(model or DEFAULT_PROTEIN_MODEL)
                state["proteome"] = read_proteome(PROTEOME)
            ids = read_orf_ids(os.path.join(IDS_DIR, f"{name}_ids.tsv"))
            data["x"] = orf_embedding_features(data["x"], ids, state["encoder"], state["proteome"])
            data["features"] = "esm2"
        return data

    run_protocol.load_dataset = load_dataset


def write_protocol_esm2(model_tag: str = "esm2_t33_650M_UR50D", out_dir: str = PROTOCOL_ESM2_DIR,
                        src_dir: str = PROTEIN_LM_DIR) -> list[str]:
    """Write the ESM-only runs of ``run_protein_lm.py`` as ``<out_dir>/<dataset>.json``."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name in ESM_DATASETS:
        path = os.path.join(src_dir, model_tag, f"{name}.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        rec["runs"] = [r for r in rec["runs"] if r["features"] == "esm"]
        rec["protocol"] = "ESM-2 only: " + rec.get("protocol", "")
        dest = os.path.join(out_dir, f"{name}.json")
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1)
        written.append(dest)
    return written
