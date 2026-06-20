#!/usr/bin/env python3
"""
Download all datasets for hmc-torch.

Usage:
    # Download everything:
    python -m hmc.datasets.download_all

    # Download specific groups:
    python -m hmc.datasets.download_all --groups arxiv,wos,FUN

    # Show what would be downloaded (dry run):
    python -m hmc.datasets.download_all --dry-run

    # Custom data directory:
    python -m hmc.datasets.download_all --output_dir ./my_data
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

AVAILABLE_GROUPS = {
    "arxiv": {
        "module": "hmc.datasets.arxiv.download_arxiv",
        "description": "ArXiv metadata snapshot (50k papers, Kaggle)",
        "size": "~1 GB",
    },
    "wos": {
        "module": "hmc.datasets.wos.download_wos",
        "description": "Web of Science abstracts (47k docs, HTC benchmark)",
        "size": "~50 MB",
    },
    "FUN": {
        "module": "hmc.datasets.gofun.download_arff",
        "description": "Yeast functional hierarchy — 8 datasets (cellcycle, derisi, "
                       "eisen, expr, gasch1, gasch2, seq, spo)",
        "size": "~10 MB",
    },
    "GO": {
        "module": "hmc.datasets.gofun.download_arff",
        "description": "Yeast Gene Ontology hierarchy — 8 datasets",
        "size": "~200 MB",
    },
    "others": {
        "module": "hmc.datasets.gofun.download_arff",
        "description": "Enron, Diatoms, ImCLEF07a, ImCLEF07d",
        "size": "~50 MB",
    },
}


def _run_module(module_name: str, output_dir: str, extra_args: list[str]) -> bool:
    """Run a download module as a subprocess and return success status."""
    import subprocess  # pylint: disable=import-outside-toplevel

    cmd = [sys.executable, "-m", module_name, "--output_dir", output_dir]
    cmd.extend(extra_args)
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Download all hmc-torch datasets")
    parser.add_argument(
        "--output_dir", type=str, default="./data",
        help="Root data directory (default: ./data)",
    )
    parser.add_argument(
        "--groups", type=str, default="all",
        help="Comma-separated list of groups to download: "
             "arxiv, wos, FUN, GO, others (default: all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be downloaded.",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="Continue downloading remaining groups even if one fails.",
    )
    args = parser.parse_args()

    # Resolve which groups to download
    if args.groups == "all":
        groups = list(AVAILABLE_GROUPS)
    else:
        groups = [g.strip() for g in args.groups.split(",")]
        for g in groups:
            if g not in AVAILABLE_GROUPS:
                logger.error(
                    "Unknown group '%s'. Available: %s",
                    g, ", ".join(AVAILABLE_GROUPS),
                )
                sys.exit(1)

    # Show plan
    logger.info("Groups to download: %s", ", ".join(groups))
    total_size_hint = sum(
        int(AVAILABLE_GROUPS[g]["size"].replace("~", "").split()[0])
        for g in groups
        if "MB" in AVAILABLE_GROUPS[g]["size"]
    ) + sum(
        int(AVAILABLE_GROUPS[g]["size"].replace("~", "").split()[0]) * 1000
        for g in groups
        if "GB" in AVAILABLE_GROUPS[g]["size"]
    )
    logger.info("Estimated total: ~%d MB", total_size_hint)

    for g in groups:
        info = AVAILABLE_GROUPS[g]
        logger.info("  %-8s %s  [%s]", g, info["description"], info["size"])

    if args.dry_run:
        logger.info("Dry run — nothing downloaded.")
        return

    output_root = str(Path(args.output_dir).resolve())
    logger.info("Output directory: %s", output_root)

    # Track FUN/GO/others — they share a download module
    processed_modules: set[str] = set()
    success_count = 0
    fail_count = 0

    for g in groups:
        info = AVAILABLE_GROUPS[g]
        module = info["module"]

        extra = []
        if module == "hmc.datasets.gofun.download_arff":
            # Collect all ARFF subsets requested
            arff_subsets = [s for s in groups if s in ("FUN", "GO", "others")]
            if module in processed_modules:
                continue  # already handled in a previous iteration
            # Batch them into a single call
            for subset in arff_subsets:
                logger.info("--- Downloading ARFF/%s ---", subset)
                ok = _run_module(module, output_root, ["--subset", subset])
                if ok:
                    success_count += 1
                    logger.info("ARFF/%s: OK", subset)
                else:
                    fail_count += 1
                    logger.error("ARFF/%s: FAILED", subset)
                    if not args.continue_on_error:
                        sys.exit(1)
            processed_modules.add(module)
            continue

        # Single-call modules (arxiv, wos)
        if g == "wos":
            extra = []
        else:
            extra = []

        logger.info("--- Downloading %s ---", g)
        subdir = output_root if g == "wos" else output_root
        ok = _run_module(module, subdir, extra)
        if ok:
            success_count += 1
            logger.info("%s: OK", g)
        else:
            fail_count += 1
            logger.error("%s: FAILED", g)
            if not args.continue_on_error:
                sys.exit(1)

    logger.info(
        "Done. %d succeeded, %d failed.", success_count, fail_count
    )
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
