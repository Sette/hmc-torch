#!/usr/bin/env python3
"""Download the EUR-Lex 57K dataset.

EUR-Lex 57K contains 57,000 EU legislative documents annotated with
EUROVOC concepts. It is a standard benchmark for large-scale multi-label
text classification with hierarchical labels.

Source:
  - The pinned Parquet export of EURLEX57K on Hugging Face, loaded through the
    generic Parquet reader (no repository dataset script is executed).

Usage:
    python -m hmc.datasets.eurlex.download_eurlex --output_dir ./data/eurlex
"""

import argparse
import json
import logging
from pathlib import Path
import urllib.error
from urllib.request import urlretrieve

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Pin the Parquet export of the original EUR-Lex 57K data. Loading these files
# through the generic Parquet builder avoids the removed eurlex.py dataset script.
HF_DATASET_REVISION = "04b1573bfbb926f9c5c9e2c149468c65ab6e604f"
HF_PARQUET_BASE = (
    "https://huggingface.co/datasets/jonathanli/eurlex/resolve/"
    f"{HF_DATASET_REVISION}/eurlex57k/"
)
HF_SPLITS = {
    "train": ("train.json", "eurlex-train.parquet"),
    "validation": ("dev.json", "eurlex-validation.parquet"),
    "test": ("test.json", "eurlex-test.parquet"),
}
EUROVOC_CONCEPTS_URL = "https://archive.org/download/EURLEX57K/eurovoc_concepts.jsonl"


def _download_huggingface(output_dir: Path) -> None:
    """Load the public Parquet splits and save the manager's JSON format."""
    try:
        from datasets import load_dataset  # pylint: disable=import-outside-toplevel
    except ImportError:
        raise ImportError(
            "huggingface-datasets is required. Install it with: pip install datasets"
        ) from None

    output_dir.mkdir(parents=True, exist_ok=True)

    for hf_split, (local_name, parquet_name) in HF_SPLITS.items():
        output_path = output_dir / local_name
        if output_path.exists() and output_path.stat().st_size > 1024:
            logger.info("%s already exists. Skipping.", output_path.name)
            continue

        parquet_url = HF_PARQUET_BASE + parquet_name
        logger.info("Loading EURLEX57K/%s from Hugging Face Parquet …", hf_split)
        dataset = load_dataset(
            "parquet",
            data_files={hf_split: parquet_url},
            split=hf_split,
        )

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

    concept_path = output_dir / "eurovoc_concepts.jsonl"
    if not concept_path.exists():
        try:
            logger.info("Downloading EUROVOC concept hierarchy …")
            urlretrieve(EUROVOC_CONCEPTS_URL, str(concept_path))
            if concept_path.stat().st_size < 1024:
                concept_path.unlink()
                raise ValueError("Downloaded concept hierarchy is unexpectedly small")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            logger.warning(
                "Could not download EUROVOC concept hierarchy (%s); the manager "
                "will use a flat hierarchy.",
                exc,
            )


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
        choices=["auto", "huggingface"],
        help="Download method (the stable Parquet source is used by default)",
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

    if method in {"auto", "huggingface"}:
        try:
            _download_huggingface(output_dir)
        except (OSError, ImportError, RuntimeError, ValueError, urllib.error.URLError) as exc:
            logger.error("EUR-Lex download failed: %s", exc)
            raise SystemExit(1) from exc

    # Verify
    if train_path.exists():
        logger.info("Done! EUR-Lex 57K dataset ready in %s", output_dir)
    else:
        logger.error("Download did not produce expected files.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
