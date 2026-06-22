#!/usr/bin/env python3
"""Download all datasets for hmc-torch.

Usage:
    python -m hmc.datasets.download_all
    python -m hmc.datasets.download_all --groups arxiv,wos
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

AVAILABLE = {
    "arxiv": {
        "module": "hmc.datasets.arxiv.download_arxiv",
        "description": "ArXiv metadata snapshot (Kaggle)",
        "size": "~1 GB",
    },
    "wos": {
        "module": "hmc.datasets.wos.download_wos",
        "description": "Web of Science abstracts (HTC benchmark)",
        "size": "~50 MB",
    },
}


def _run_module(module_name: str, output_dir: str) -> bool:
    import subprocess

    cmd = [sys.executable, "-m", module_name, "--output_dir", output_dir]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Download hmc-torch datasets")
    parser.add_argument("--output_dir", default="./data")
    parser.add_argument("--groups", default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    groups = list(AVAILABLE) if args.groups == "all" else args.groups.split(",")

    for g in groups:
        logger.info("  %-8s %s  [%s]", g, AVAILABLE[g]["description"], AVAILABLE[g]["size"])

    if args.dry_run:
        return

    success = fail = 0
    for g in groups:
        ok = _run_module(AVAILABLE[g]["module"], args.output_dir)
        if ok:
            success += 1
        else:
            fail += 1
            if not args.continue_on_error:
                sys.exit(1)

    logger.info("Done. %d succeeded, %d failed.", success, fail)


if __name__ == "__main__":
    main()
