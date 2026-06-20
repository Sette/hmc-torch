#!/usr/bin/env python3
"""
Download and prepare ARFF HMC datasets (GO, FUN, others) for hmc-torch.

These are the standard HMC benchmarks from Vens et al. (2008) and
Giunchiglia & Lukasiewicz (2020), distributed via the C-HMCNN repository.

Usage:
    # Download all ARFF datasets:
    python -m hmc.datasets.gofun.download_arff --output_dir ./data

    # Download only FUN datasets:
    python -m hmc.datasets.gofun.download_arff --output_dir ./data --subset FUN

    # Download only GO datasets:
    python -m hmc.datasets.gofun.download_arff --output_dir ./data --subset GO

    # Download only "others" datasets (enron, diatoms, imclef):
    python -m hmc.datasets.gofun.download_arff --output_dir ./data --subset others
"""

import argparse
import logging
import os
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# C-HMCNN repository raw data base
_BASE_URL = (
    "https://raw.githubusercontent.com/egivumpy/C-HMCNN/main/data/"
    "HMC_data_arff"
)

# FUN datasets: {name: [train, valid, test]}
_FUN_DATASETS: dict[str, tuple[str, str, str]] = {
    "cellcycle_FUN": (
        "cellcycle_FUN.train.arff", "cellcycle_FUN.valid.arff",
        "cellcycle_FUN.test.arff",
    ),
    "derisi_FUN": (
        "derisi_FUN.train.arff", "derisi_FUN.valid.arff",
        "derisi_FUN.test.arff",
    ),
    "eisen_FUN": (
        "eisen_FUN.train.arff", "eisen_FUN.valid.arff",
        "eisen_FUN.test.arff",
    ),
    "expr_FUN": (
        "expr_FUN.train.arff", "expr_FUN.valid.arff",
        "expr_FUN.test.arff",
    ),
    "gasch1_FUN": (
        "gasch1_FUN.train.arff", "gasch1_FUN.valid.arff",
        "gasch1_FUN.test.arff",
    ),
    "gasch2_FUN": (
        "gasch2_FUN.train.arff", "gasch2_FUN.valid.arff",
        "gasch2_FUN.test.arff",
    ),
    "seq_FUN": (
        "seq_FUN.train.arff", "seq_FUN.valid.arff",
        "seq_FUN.test.arff",
    ),
    "spo_FUN": (
        "spo_FUN.train.arff", "spo_FUN.valid.arff",
        "spo_FUN.test.arff",
    ),
}

# GO datasets
_GO_DATASETS: dict[str, tuple[str, str, str]] = {
    "cellcycle_GO": (
        "cellcycle_GO.train.arff", "cellcycle_GO.valid.arff",
        "cellcycle_GO.test.arff",
    ),
    "derisi_GO": (
        "derisi_GO.train.arff", "derisi_GO.valid.arff",
        "derisi_GO.test.arff",
    ),
    "eisen_GO": (
        "eisen_GO.train.arff", "eisen_GO.valid.arff",
        "eisen_GO.test.arff",
    ),
    "expr_GO": (
        "expr_GO.train.arff", "expr_GO.valid.arff",
        "expr_GO.test.arff",
    ),
    "gasch1_GO": (
        "gasch1_GO.train.arff", "gasch1_GO.valid.arff",
        "gasch1_GO.test.arff",
    ),
    "gasch2_GO": (
        "gasch2_GO.train.arff", "gasch2_GO.valid.arff",
        "gasch2_GO.test.arff",
    ),
    "seq_GO": (
        "seq_GO.train.arff", "seq_GO.valid.arff",
        "seq_GO.test.arff",
    ),
    "spo_GO": (
        "spo_GO.train.arff", "spo_GO.valid.arff",
        "spo_GO.test.arff",
    ),
}

# "Others" datasets: {name: [train, test]} (no validation split)
_OTHERS_DATASETS: dict[str, tuple[str, str]] = {
    "diatoms": ("Diatoms_train.arff", "Diatoms_test.arff"),
    "enron": ("Enron_corr_trainvalid.arff", "Enron_corr_test.arff"),
    "imclef07a": ("ImCLEF07A_Train.arff", "ImCLEF07A_Test.arff"),
    "imclef07d": ("ImCLEF07D_Train.arff", "ImCLEF07D_Test.arff"),
}


def _download_file(url: str, dest: Path) -> None:
    """Download a single file with progress display."""
    import urllib.request  # pylint: disable=import-outside-toplevel

    if dest.exists():
        logger.info("Already exists: %s", dest)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s ...", url)
    try:
        urllib.request.urlretrieve(url, str(dest))
    except Exception as exc:
        logger.error("Failed to download %s: %s", url, exc)
        if dest.exists():
            dest.unlink()
        raise


def _download_datasets(
    datasets: dict,
    base_dir: Path,
    repo_path: str,
    is_two_file: bool = False,
) -> None:
    """Download a group of datasets from C-HMCNN repo."""
    for name, files in datasets.items():
        dst_dir = base_dir / name
        if all((dst_dir / f).exists() for f in files):
            logger.info("Dataset %s: all files present", name)
            continue

        for fname in files:
            url = f"{_BASE_URL}/{repo_path}/{name}/{fname}"
            dest = dst_dir / fname
            _download_file(url, dest)

        logger.info("Dataset %s: done (%d files)", name, len(files))


def main():
    parser = argparse.ArgumentParser(description="Download ARFF HMC datasets")
    parser.add_argument(
        "--output_dir", type=str, default="./data",
        help="Output root directory (default: ./data). Files go to "
             "data/HMC_data_arff/ under this root.",
    )
    parser.add_argument(
        "--subset", type=str, choices=["FUN", "GO", "others", "all"],
        default="all",
        help="Which dataset group to download (default: all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be downloaded without doing it.",
    )
    args = parser.parse_args()

    arff_root = Path(args.output_dir) / "HMC_data_arff"

    subsets = {
        "FUN": (_FUN_DATASETS, "datasets_FUN", False),
        "GO": (_GO_DATASETS, "datasets_GO", False),
        "others": (_OTHERS_DATASETS, "others", True),
    }

    to_download = (
        list(subsets.items())
        if args.subset == "all"
        else [(args.subset, subsets[args.subset])]
    )

    total_files = 0
    for label, (datasets, repo_path, is_two) in to_download:
        n = sum(len(files) for files in datasets.values())
        total_files += n
        logger.info("%s: %d datasets, %d files", label, len(datasets), n)

    if args.dry_run:
        logger.info("Dry run — %d files would be downloaded.", total_files)
        return

    for label, (datasets, repo_path, is_two) in to_download:
        logger.info("Downloading %s datasets ...", label)
        _download_datasets(datasets, arff_root, repo_path, is_two)

    logger.info("All ARFF datasets ready at %s", arff_root)


if __name__ == "__main__":
    main()
