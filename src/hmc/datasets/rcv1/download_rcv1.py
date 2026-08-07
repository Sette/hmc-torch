#!/usr/bin/env python3
"""Download the RCV1-V2 dataset in HiAGM-compatible format.

The RCV1-V2 dataset is the standard HTC benchmark. This script downloads
the preprocessed version used by HiAGM/HiMatch papers (publicly available).

Usage:
    python -m hmc.datasets.rcv1.download_rcv1 --output_dir ./data/rcv1
"""

import argparse
import json
import logging
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# HiAGM RCV1 data (public Google Drive mirror, processed JSON format)
HIAGM_RCV1_URL = (
    "https://github.com/Alibaba-NLP/HiAGM/raw/master/data/rcv1/"
)


def _download_hiagm_format(output_dir: Path) -> None:
    """Download individual JSON files from HiAGM repo.

    The HiAGM repo contains:
      - rcv1_train.json (train split, token+label format)
      - rcv1_test.json  (test split)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "train.json": f"{HIAGM_RCV1_URL}rcv1_train.json",
        "test.json": f"{HIAGM_RCV1_URL}rcv1_test.json",
    }

    for local_name, url in files.items():
        output_path = output_dir / local_name
        if output_path.exists():
            logger.info(
                "%s already exists (%d bytes). Skipping.",
                local_name,
                output_path.stat().st_size,
            )
            continue

        logger.info("Downloading %s …", local_name)
        try:
            urlretrieve(url, str(output_path))
            logger.info(
                "Downloaded %s (%d bytes).",
                local_name,
                output_path.stat().st_size,
            )
        except Exception as exc:
            logger.error("Failed to download %s: %s", local_name, exc)
            _print_manual_instructions(output_dir)
            sys.exit(1)


def _print_manual_instructions(output_dir: Path) -> None:
    """Print manual download instructions."""
    logger.error(
        "Automatic download failed. Please download RCV1-V2 manually:\n"
        "\n"
        "Option 1 — HiAGM repo (recommended):\n"
        "  git clone https://github.com/Alibaba-NLP/HiAGM.git /tmp/HiAGM\n"
        "  cp /tmp/HiAGM/data/rcv1/rcv1_train.json %s/\n"
        "  cp /tmp/HiAGM/data/rcv1/rcv1_test.json %s/\n"
        "\n"
        "Option 2 — HiMatch repo:\n"
        "  git clone https://github.com/qianlima-lab/HiMatch.git /tmp/HiMatch\n"
        "  cp /tmp/HiMatch/data/rcv1/*.json %s/\n"
        "\n"
        "Option 3 — NIST original (requires license):\n"
        "  https://trec.nist.gov/data/reuters/reuters.html\n"
        "  After obtaining the XML, use the preprocessing script from HiAGM.\n",
        output_dir,
        output_dir,
        output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download RCV1-V2 dataset (HiAGM format)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/rcv1",
        help="Output directory (default: ./data/rcv1)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Check if data already exists
    train_path = output_dir / "train.json"
    test_path = output_dir / "test.json"
    if (
        train_path.exists()
        and test_path.exists()
        and train_path.stat().st_size > 1024
        and test_path.stat().st_size > 1024
    ):
        logger.info(
            "RCV1 files already exist. train=%d bytes, test=%d bytes. Skipping.",
            train_path.stat().st_size,
            test_path.stat().st_size,
        )
        return

    _download_hiagm_format(output_dir)

    # Verify
    if train_path.exists() and test_path.exists():
        logger.info("Done! RCV1-V2 dataset ready in %s", output_dir)
    else:
        logger.error("Download did not produce expected files.")
        sys.exit(1)


if __name__ == "__main__":
    main()
