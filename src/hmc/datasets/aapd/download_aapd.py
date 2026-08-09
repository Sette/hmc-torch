#!/usr/bin/env python3
"""Download the AAPD (arXiv Academic Paper Dataset) CSV file.

The dataset is public and available from multiple sources.
Default source: HuggingFace datasets hub (reliable, no auth needed).

Usage:
    python -m hmc.datasets.aapd.download_aapd --output_dir ./data/aapd
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

AAPD_EXPECTED_SIZE_MB = 150  # approximate


def _download_kaggle(output_dir: Path) -> None:
    """Download via kagglehub (preferred)."""
    import kagglehub  # pylint: disable=import-outside-toplevel

    logger.info("Downloading AAPD from Kaggle via kagglehub …")
    path = kagglehub.dataset_download("syedharoon312/aapd-arxiv-academic-paper-dataset")
    logger.info("Downloaded to: %s", path)
    _copy_csv(path, output_dir)


def _download_huggingface(output_dir: Path) -> None:
    """Download via HuggingFace datasets."""
    try:
        from datasets import load_dataset  # pylint: disable=import-outside-toplevel
    except ImportError:
        logger.error("huggingface-datasets not installed. Run: pip install datasets")
        sys.exit(1)

    import csv as _csv  # pylint: disable=import-outside-toplevel

    logger.info("Downloading AAPD from HuggingFace datasets …")
    dataset = load_dataset("aapd", split="train")
    logger.info("Loaded %d records from HuggingFace.", len(dataset))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "aapd.csv"

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["title", "abstract", "labels"])
        for row in dataset:
            title = row.get("title", "")
            abstract = row.get("abstract", "")
            labels = row.get("labels", [])
            if isinstance(labels, list):
                labels = " ".join(labels)
            writer.writerow([title, abstract, labels])

    logger.info("Saved %d records to %s", len(dataset), output_path)


def _copy_csv(src_dir: str, output_dir: Path) -> None:
    """Copy CSV from a downloaded directory to the target output dir."""
    import shutil  # pylint: disable=import-outside-toplevel

    output_dir.mkdir(parents=True, exist_ok=True)
    src_path = Path(src_dir)

    # Find the CSV file in the downloaded directory
    csv_files = list(src_path.rglob("*.csv"))
    if not csv_files:
        logger.warning("No CSV found in %s. Listing contents:", src_dir)
        for f in sorted(src_path.rglob("*")):
            logger.warning("  %s", f)
        raise FileNotFoundError(f"No CSV files found in {src_dir}")

    target = output_dir / "aapd.csv"
    shutil.copy2(str(csv_files[0]), str(target))
    logger.info("Copied %s → %s", csv_files[0], target)


def main() -> None:
    """CLI entry point for AAPD dataset download."""
    parser = argparse.ArgumentParser(description="Download AAPD dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/aapd",
        help="Output directory (default: ./data/aapd)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "huggingface", "kaggle"],
        help="Download method (default: auto = try huggingface first)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_path = output_dir / "aapd.csv"

    if output_path.exists() and output_path.stat().st_size > 1024:
        logger.info(
            "AAPD CSV already exists at %s (%d bytes). Skipping download.",
            output_path,
            output_path.stat().st_size,
        )
        return

    method = args.method

    if method == "auto":
        # Try HuggingFace first (simpler, no auth)
        try:
            _download_huggingface(output_dir)
            return
        except (OSError, ImportError) as exc:
            logger.warning("HuggingFace download failed: %s", exc)
        # Fallback to Kaggle
        try:
            _download_kaggle(output_dir)
            return
        except (OSError, ImportError) as exc:
            logger.error("Kaggle download also failed: %s", exc)
            _print_manual_instructions(output_dir)
            sys.exit(1)
    elif method == "huggingface":
        _download_huggingface(output_dir)
    elif method == "kaggle":
        _download_kaggle(output_dir)

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info("Done! AAPD dataset saved to %s (%.1f MB)", output_path, size_mb)
    else:
        logger.error("Download did not produce the expected file.")
        sys.exit(1)


def _print_manual_instructions(output_dir: Path) -> None:
    """Print manual download instructions."""
    logger.error(
        "Automatic download failed. Please download AAPD manually:\n"
        "\n"
        "  1. Visit https://paperswithcode.com/dataset/aapd\n"
        "  2. Download the arxiv_academic_paper_dataset.csv file\n"
        "  3. Place it at: %s/aapd.csv\n"
        "\n"
        "Alternatively, use the Kaggle CLI:\n"
        "  kaggle datasets download syedharoon312/aapd-arxiv-academic-paper-dataset\n"
        "  unzip aapd-arxiv-academic-paper-dataset.zip -d %s\n",
        output_dir,
        output_dir,
    )


if __name__ == "__main__":
    main()
