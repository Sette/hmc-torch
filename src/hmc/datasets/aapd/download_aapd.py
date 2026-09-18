#!/usr/bin/env python3
"""Download the AAPD dataset and convert it to the CSV the manager reads.

Source: Kaggle ``xiaojuanwang9/aapd-dataset`` -- a third-party re-packaging of
the canonical AAPD (Yang et al., SGM): 55,840 arXiv abstracts over 54 labels
across 9 areas, shipped as ``train.txt`` / ``val.txt`` / ``test.txt`` (two
lines per document: the text, then its space-separated label codes) plus
``label_to_index.json``.

The manager reads a single CSV with ``title``, ``abstract`` and ``labels``
columns and re-splits it itself, so this script concatenates the three splits
into ``aapd.csv``.  The label set is the canonical one -- 54 labels over the
older arXiv taxonomy -- not the 97-label variant some re-uploads carry.

Usage:
    python -m hmc.datasets.aapd.download_aapd --output_dir ./data/aapd
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

KAGGLE_DATASET = "xiaojuanwang9/aapd-dataset"
SPLIT_FILES = ("train.txt", "val.txt", "test.txt")
LABELS_FILE = "label_to_index.json"
CSV_NAME = "aapd.csv"


def _download_kagglehub() -> Path:
    """Fetch the dataset via kagglehub and return its directory."""
    import kagglehub  # pylint: disable=import-outside-toplevel

    logger.info("Downloading %s via kagglehub …", KAGGLE_DATASET)
    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def _find_source_dir(root: Path) -> Path:
    """Return the directory holding the SGM files (the zip's layout varies)."""
    for candidate in (root, *sorted(p for p in root.rglob("*") if p.is_dir())):
        if all((candidate / name).is_file() for name in SPLIT_FILES):
            return candidate
    raise FileNotFoundError(
        f"None of {SPLIT_FILES} found under {root}. Downloaded: "
        f"{sorted(p.name for p in root.rglob('*'))[:10]}"
    )


def _load_label_codes(source: Path) -> set:
    """Read ``label_to_index.json`` (a name→index mapping)."""
    data = json.loads((source / LABELS_FILE).read_text(encoding="utf-8"))
    # Works for both a name→index mapping (keys) and a plain list of codes
    codes = set(data)
    if not codes:
        raise ValueError(f"{LABELS_FILE} lists no labels")
    return codes


def _parse_splits(source: Path, codes: set) -> tuple[list, list]:
    """Read the two-lines-per-document files into (texts, labels) lists."""
    texts, labels = [], []
    for name in SPLIT_FILES:
        seen = len(texts)
        lines = (source / name).read_text(encoding="utf-8").splitlines()
        if len(lines) % 2 != 0:
            raise ValueError(f"{name}: {len(lines)} lines — expected pairs")
        for index in range(0, len(lines), 2):
            text = lines[index].strip()
            codes_line = lines[index + 1].strip()
            if not text or not codes_line:
                raise ValueError(f"{name}: empty text/labels at line {index + 1}")
            unknown = [tok for tok in codes_line.split() if tok not in codes]
            if unknown:
                raise ValueError(
                    f"{name} line {index + 2}: labels not in {LABELS_FILE}: "
                    f"{unknown[:5]}"
                )
            texts.append(text)
            labels.append(codes_line)
        logger.info("  %s: %d documents", name, len(texts) - seen)
    return texts, labels


def _write_csv(output_dir: Path, texts: list, labels: list, codes: set) -> Path:
    """Write the manager's CSV: title, abstract, labels."""
    path = output_dir / CSV_NAME
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["title", "abstract", "labels"])
        for text, label in zip(texts, labels):
            # The SGM release already joins title and abstract into one field.
            writer.writerow(["", text, label])
    areas = sorted({code.split(".")[0] for code in codes})
    logger.info(
        "Wrote %s: %d documents, %d labels in %d areas (%s)",
        path,
        len(texts),
        len(codes),
        len(areas),
        ", ".join(areas),
    )
    return path


def _print_manual_instructions(output_dir: Path) -> None:
    logger.error(
        "Automatic download failed. Fetch the dataset manually:\n"
        "\n"
        "  1. Download %s from Kaggle:\n"
        "       kaggle datasets download -d %s\n"
        "  2. Point this script at the unpacked directory:\n"
        "       python -m hmc.datasets.aapd.download_aapd --output_dir %s \\\n"
        "           --source_dir /path/to/unpacked\n",
        KAGGLE_DATASET,
        KAGGLE_DATASET,
        output_dir,
    )


def main() -> None:
    """CLI entry point for the AAPD dataset download."""
    parser = argparse.ArgumentParser(description="Download the AAPD dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/aapd",
        help="Output directory (default: ./data/aapd)",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default=None,
        help="Use an already-unpacked copy of the Kaggle dataset",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / CSV_NAME

    if csv_path.exists() and csv_path.stat().st_size > 1024:
        logger.info(
            "%s already exists (%d bytes). Skipping.", csv_path, csv_path.stat().st_size
        )
        return

    try:
        source = (
            _find_source_dir(Path(args.source_dir))
            if args.source_dir
            else _find_source_dir(_download_kagglehub())
        )
        codes = _load_label_codes(source)
        texts, labels = _parse_splits(source, codes)
        _write_csv(output_dir, texts, labels, codes)
    except (OSError, ImportError, ValueError, KeyError) as exc:
        logger.error("Download failed: %s", exc)
        _print_manual_instructions(output_dir)
        sys.exit(1)

    logger.info("Done! AAPD ready at %s", csv_path)


if __name__ == "__main__":
    main()
