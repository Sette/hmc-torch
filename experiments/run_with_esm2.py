#!/usr/bin/env python3
"""Run an experiment script with ESM-2 features on ``seq_FUN`` / ``seq_GO``, unchanged.

Installs :func:`esm2_features.install` (``run_protocol.load_dataset`` returns the ESM-2 matrices),
points the protocol-epoch lookups to ``output/protocol_esm2`` (``HMC_PROTOCOL_DIR``) and runs the
script as ``__main__``.  Always pass ``--out_dir`` under ``output/protein_lm_methods/`` so the
benchmark results are never overwritten.

Usage:
    PYTHONPATH=src python experiments/run_with_esm2.py experiments/run_sr_gae_sota.py \
        --datasets seq_FUN,seq_GO --out_dir output/protein_lm_methods/sr_gae_paired
"""

from __future__ import annotations

import os
import runpy
import sys

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(__file__))

import esm2_features  # noqa: E402  pylint: disable=wrong-import-position


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    script, argv = sys.argv[1], sys.argv[2:]
    if "--out_dir" not in argv:
        sys.exit("--out_dir is required (under output/protein_lm*): never write over benchmark results")
    out_dir = os.path.normpath(argv[argv.index("--out_dir") + 1])
    if not out_dir.startswith("output" + os.sep + "protein_lm"):
        sys.exit(f"--out_dir {out_dir!r} must be under output/protein_lm*")
    esm2_features.write_protocol_esm2()
    os.environ["HMC_PROTOCOL_DIR"] = esm2_features.PROTOCOL_ESM2_DIR
    esm2_features.install()
    sys.argv = [script, *argv]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
