#!/usr/bin/env python3
"""
Download and prepare the ArXiv dataset for hmc-torch.

Downloads the arxiv-metadata-oai-snapshot.json from Kaggle (via kagglehub)
or uses a local file if already present.

Usage:
    python -m hmc.datasets.arxiv.download_arxiv --output_dir ./data/arxiv
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

EXPECTED_FILE = "arxiv-metadata-oai-snapshot.json"
EXPECTED_SIZE_HINT = 1_000_000_000  # ~1 GB minimum


def _check_existing(data_dir: Path) -> Path | None:
    """Check if the dataset already exists locally."""
    dest = data_dir / EXPECTED_FILE
    if dest.exists() and dest.stat().st_size > EXPECTED_SIZE_HINT:
        logger.info(
            "ArXiv dataset already at %s (%.1f GB)", dest, dest.stat().st_size / 1e9
        )
        return dest
    if dest.exists():
        logger.warning(
            "%s exists but seems too small (%d bytes)", dest, dest.stat().st_size
        )
    return None


def download_kagglehub(data_dir: Path) -> Path:
    """Download via kagglehub (recommended, no API key needed)."""
    try:
        import kagglehub  # pylint: disable=import-outside-toplevel
    except ImportError:
        logger.error("kagglehub not installed. Run: pip install kagglehub")
        sys.exit(1)

    logger.info("Downloading ArXiv dataset via kagglehub ...")
    path = kagglehub.dataset_download(
        "Cornell-University/arxiv", path="arxiv-metadata-oai-snapshot.json"
    )
    src = Path(path)
    dest = data_dir / EXPECTED_FILE
    data_dir.mkdir(parents=True, exist_ok=True)
    os.rename(str(src), str(dest))
    logger.info("Downloaded to %s (%.1f GB)", dest, dest.stat().st_size / 1e9)
    return dest


def download_kaggle_api(data_dir: Path) -> Path:
    """Download via kaggle CLI (requires API key setup)."""
    import subprocess  # pylint: disable=import-outside-toplevel

    dest = data_dir / EXPECTED_FILE
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading ArXiv dataset via kaggle CLI ...")
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "Cornell-University/arxiv",
            "-f",
            "arxiv-metadata-oai-snapshot.json",
            "-p",
            str(data_dir),
            "--unzip",
        ],
        check=True,
    )
    logger.info("Downloaded to %s (%.1f GB)", dest, dest.stat().st_size / 1e9)
    return dest


def main():
    parser = argparse.ArgumentParser(description="Download ArXiv dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/arxiv",
        help="Output directory (default: ./data/arxiv)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["auto", "kagglehub", "kaggle", "manual"],
        default="auto",
        help="Download method (default: auto — try kagglehub first). "
        "Use 'manual' for instructions if auto fails.",
    )
    args = parser.parse_args()

    data_dir = Path(args.output_dir)

    # Check existing
    existing = _check_existing(data_dir)
    if existing:
        logger.info("Dataset ready. No download needed.")
        return

    # Try download
    if args.method in ("auto", "kagglehub"):
        try:
            download_kagglehub(data_dir)
            return
        except Exception as exc:
            logger.warning("kagglehub failed: %s", exc)

    if args.method in ("auto", "kaggle"):
        try:
            download_kaggle_api(data_dir)
            return
        except Exception as exc:
            logger.warning("kaggle API failed: %s", exc)

    # Manual instructions
    logger.error(
        "Automatic download failed.\n\n"
        "Manual steps:\n"
        "  1. Go to https://www.kaggle.com/datasets/Cornell-University/arxiv\n"
        "  2. Download arxiv-metadata-oai-snapshot.json\n"
        "  3. Place it at: %s\n"
        "  4. Re-run this script to verify.",
        data_dir / EXPECTED_FILE,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
