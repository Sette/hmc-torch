#!/usr/bin/env python3
"""Download the EUR-Lex 57K dataset.

EUR-Lex 57K contains 57,000 EU legislative documents annotated with
EUROVOC concepts. It is a standard benchmark for large-scale multi-label
text classification with hierarchical labels.

Sources:
  - HuggingFace datasets: ``NLP-AUEB/eurlex`` (recommended)
  - Direct download: archive.org/details/EURLEX57K

Usage:
    python -m hmc.datasets.eurlex.download_eurlex --output_dir ./data/eurlex
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import urllib.error
from urllib.request import urlretrieve

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

ARCHIVE_BASE = "https://archive.org/download/EURLEX57K"
FILES = {
    "train.json": f"{ARCHIVE_BASE}/train.json",
    "dev.json": f"{ARCHIVE_BASE}/dev.json",
    "test.json": f"{ARCHIVE_BASE}/test.json",
    "eurovoc_concepts.jsonl": f"{ARCHIVE_BASE}/eurovoc_concepts.jsonl",
}


def _download_archive(output_dir: Path) -> None:
    """Download individual JSON files from archive.org."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for local_name, url in FILES.items():
        output_path = output_dir / local_name
        if output_path.exists() and output_path.stat().st_size > 1024:
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
        except (urllib.error.URLError, OSError) as exc:
            logger.error("Failed to download %s: %s", local_name, exc)
            if local_name == "eurovoc_concepts.jsonl":
                logger.warning(
                    "Concept file not available — will use flat hierarchy. "
                    "Training still works, but R-matrix benefit may be reduced."
                )
            else:
                raise


def _download_huggingface(output_dir: Path) -> None:
    """Download via HuggingFace datasets library."""
    try:
        from datasets import load_dataset  # pylint: disable=import-outside-toplevel
    except ImportError:
        logger.error("huggingface-datasets not installed. Run: pip install datasets")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "dev", "test"]:
        output_path = output_dir / f"{split}.json"
        if output_path.exists() and output_path.stat().st_size > 1024:
            logger.info("%s already exists. Skipping.", output_path.name)
            continue

        logger.info("Loading EURLEX57K/%s from HuggingFace …", split)
        dataset = load_dataset("NLP-AUEB/eurlex", split=split)

        records = []
        for row in dataset:
            records.append(
                {
                    "celex_id": row.get("celex_id", ""),
                    "title": row.get("title", ""),
                    "text": row.get("text", ""),
                    "eurovoc_concepts": (
                        row["eurovoc_concepts"]
                        if isinstance(row.get("eurovoc_concepts"), list)
                        else []
                    ),
                }
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)

        logger.info("Saved %d records to %s", len(records), output_path)


def main() -> None:
    """CLI entry point for EUR-Lex 57K dataset download."""
    parser = argparse.ArgumentParser(description="Download EUR-Lex 57K dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/eurlex",
        help="Output directory (default: ./data/eurlex)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "archive", "huggingface"],
        help="Download method (default: auto)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Check if data already exists
    train_path = output_dir / "train.json"
    if train_path.exists() and train_path.stat().st_size > 1024:
        logger.info(
            "EUR-Lex data already exists at %s. Skipping download.",
            output_dir,
        )
        return

    method = args.method

    if method == "auto":
        # Try archive.org first (direct download, no deps)
        try:
            _download_archive(output_dir)
        except (urllib.error.URLError, OSError) as exc:
            logger.warning("archive.org download failed: %s", exc)
            try:
                _download_huggingface(output_dir)
            except (OSError, ImportError) as exc2:
                logger.error("HuggingFace download also failed: %s", exc2)
                sys.exit(1)
    elif method == "archive":
        _download_archive(output_dir)
    elif method == "huggingface":
        _download_huggingface(output_dir)

    # Verify
    if train_path.exists():
        logger.info("Done! EUR-Lex 57K dataset ready in %s", output_dir)
    else:
        logger.error("Download did not produce expected files.")
        sys.exit(1)


if __name__ == "__main__":
    main()
