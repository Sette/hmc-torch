#!/usr/bin/env python3
"""Obtain the RCV1-V2 dataset in the JSON format this project expects.

RCV1-V2 is gated: NIST distributes the corpus under a license agreement and the
preprocessed ``rcv1_train.json`` / ``rcv1_test.json`` files are *not* publicly
redistributed.  The similarly named files inside HiAGM/HiMatch are ~20 KB
samples, not the corpus (standard split: 23,149 train / 781,265 test documents),
so this script never downloads from there.

Supported paths:

1. ``--source_dir DIR`` — copy ``rcv1_train.json`` / ``rcv1_test.json`` (any of
   the name variants the manager accepts) that you produced yourself.
2. ``--sample`` — install the two ~10-document HiAGM files, for smoke-testing
   the pipeline only.
3. No ``--source_dir`` — run the upstream preprocessing scripts. Put the NIST
   source files in ``--raw_dir`` (defaults to ``--output_dir``).

Usage:
    python -m hmc.datasets.rcv1.download_rcv1 --output_dir ./data/rcv1 \
        --source_dir /tmp/HBGL/data/rcv1
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import urllib.error
from pathlib import Path
from urllib.request import urlretrieve

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

TRAIN_NAMES = ("rcv1_train.json", "train.json", "lyrl2004_tokens_train.json")
TEST_NAMES = ("rcv1_test.json", "test.json", "lyrl2004_tokens_test.json")
VAL_NAMES = ("rcv1_val.json", "val.json")

# HiAGM ships two tiny smoke-test files under data/ (9 and 10 documents, JSONL).
# They are *not* the corpus -- see the module docstring.
HIAGM_SAMPLE_URL = "https://raw.githubusercontent.com/Alibaba-NLP/HiAGM/master/data/"
HIAGM_SAMPLE_FILES = ("rcv1_train.json", "rcv1_test.json")

# 1 MiB: anything smaller cannot be a split of the corpus
MIN_BYTES = 1 << 20
# Below this, the file is almost certainly a demo sample rather than the corpus
SAMPLE_BYTES = 10 << 20


def _find(source_dir: Path, names: tuple) -> Path | None:
    """Return the first existing candidate file, or None."""
    for name in names:
        candidate = source_dir / name
        if candidate.is_file():
            return candidate
    return None


def _looks_like_records(path: Path) -> tuple[bool, str]:
    """Peek at the head of the file and check it holds token/label records."""
    with open(path, "r", encoding="utf-8") as handle:
        head = handle.read(4096)

    if not head.strip():
        return False, "empty file"
    if '"token"' in head and '"label"' in head:
        return True, "JSON objects with token/label"
    first_line = head.splitlines()[0] if head.splitlines() else ""
    try:
        record = json.loads(first_line)
    except json.JSONDecodeError:
        return False, "not JSON and not JSONL"
    if isinstance(record, dict) and "token" in record and "label" in record:
        return True, "JSONL records with token/label"
    return False, "no token/label fields"


def _validate(path: Path) -> None:
    """Raise ValueError when the file cannot be a corpus split."""
    size = path.stat().st_size
    if size < MIN_BYTES:
        raise ValueError(
            f"{path.name} is only {size / 1e6:.2f} MB — that is a sample file, "
            "not an RCV1-V2 split (expected hundreds of MB)"
        )
    if size < SAMPLE_BYTES:
        logger.warning(
            "%s is only %.1f MB — verify it is the full split, not a sample.",
            path.name,
            size / 1e6,
        )
    ok, why = _looks_like_records(path)
    if not ok:
        raise ValueError(f"{path.name}: {why}")


def _count_records(path: Path) -> int:
    """Number of records in an array or JSONL file."""
    with open(path, "r", encoding="utf-8") as handle:
        if handle.read(1) == "[":
            handle.seek(0)
            return len(json.load(handle))
        handle.seek(0)
        return sum(1 for line in handle if line.strip())


def _install_hiagm_sample(output_dir: Path) -> list[Path]:
    """Install the tiny HiAGM smoke-test files, loudly marked as not the corpus."""
    logger.warning(
        "Installing the HiAGM sample files.  They hold ~10 documents each, not "
        "the RCV1-V2 corpus (23,149 / 781,265 documents): use them to exercise "
        "the pipeline, never to report numbers."
    )
    installed = []
    for name in HIAGM_SAMPLE_FILES:
        destination = output_dir / name
        urlretrieve(HIAGM_SAMPLE_URL + name, str(destination))
        ok, why = _looks_like_records(destination)
        if not ok:
            raise ValueError(f"{name}: {why}")
        installed.append(destination)
        logger.warning("  %s: %d records", name, _count_records(destination))
    return installed


def _install(source: Path, output_dir: Path) -> Path:
    """Validate *source* and copy it into *output_dir* under the same name."""
    _validate(source)
    destination = output_dir / source.name
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def _already_present(output_dir: Path) -> tuple[Path, Path] | None:
    """Return the (train, test) pair already in place, when valid."""
    train = _find(output_dir, TRAIN_NAMES)
    test = _find(output_dir, TEST_NAMES)
    if train is None or test is None:
        return None
    _validate(train)
    _validate(test)
    return train, test


def _print_instructions(output_dir: Path, raw_dir: Path) -> None:
    """Explain how to produce the files from the gated original."""
    logger.error(
        "RCV1-V2 is not publicly redistributable, so there is nothing to\n"
        "download automatically.  To produce the JSON files:\n"
        "\n"
        "  1. Obtain the corpus (rcv1.tar.xz + lyrl2004_tokens_train.dat) from\n"
        "     https://trec.nist.gov/data/reuters/reuters.html — license required.\n"
        "     Place both files in %s (or pass that directory with --raw_dir).\n"
        "\n"
        "  2. Run the default command; it clones HBGL if needed, links the two\n"
        "     source files into its data/rcv1 directory, and runs both scripts.\n"
        "\n"
        "     python -m hmc.datasets.rcv1.download_rcv1 --output_dir %s\n"
        "\n"
        "     To use a different source directory: add --raw_dir DIR. To use\n"
        "     already-preprocessed JSON, pass --source_dir DIR instead.\n",
        raw_dir,
        output_dir,
    )


def _run_upstream_preprocessing(
    upstream_dir: Path, raw_dir: Path, output_dir: Path
) -> None:
    """Run HBGL preprocessing using NIST source files from *raw_dir*."""
    repo_url = "https://github.com/kongds/hbgl"
    if not (upstream_dir / ".git").is_dir():
        if upstream_dir.exists() and any(upstream_dir.iterdir()):
            raise ValueError(
                f"{upstream_dir} exists but is not an HBGL checkout; choose "
                "another --upstream_dir or remove that directory"
            )
        upstream_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(upstream_dir)],
            check=True,
        )

    data_dir = upstream_dir / "data" / "rcv1"
    preprocess = data_dir / "preprocess_rcv1.py"
    convert = data_dir / "data_rcv1.py"
    if not preprocess.is_file() or not convert.is_file():
        raise FileNotFoundError(
            f"HBGL RCV1 preprocessing scripts missing in {data_dir}"
        )
    for filename in ("rcv1.tar.xz", "lyrl2004_tokens_train.dat"):
        source = raw_dir / filename
        if not source.is_file():
            raise FileNotFoundError(
                f"Missing {source}. Obtain {filename} from NIST and place it in "
                f"{raw_dir}, or pass its directory with --raw_dir."
            )
        target = data_dir / filename
        if target.exists() or target.is_symlink():
            if target.resolve() != source.resolve():
                raise FileExistsError(
                    f"{target} already exists and differs from {source}; remove or "
                    "move the existing HBGL input, or choose another --upstream_dir."
                )
        else:
            target.symlink_to(source.resolve())
    logger.info("Running upstream RCV1 preprocessing in %s", data_dir)
    subprocess.run([sys.executable, str(preprocess), "."], cwd=data_dir, check=True)
    subprocess.run([sys.executable, str(convert)], cwd=data_dir, check=True)

    train = _find(data_dir, TRAIN_NAMES)
    test = _find(data_dir, TEST_NAMES)
    if train is None or test is None:
        raise FileNotFoundError(
            f"HBGL preprocessing completed, but train/test JSON files were not found in {data_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _install(train, output_dir)
    _install(test, output_dir)
    logger.info("RCV1 JSON splits installed in %s", output_dir)


def main() -> None:
    """CLI entry point for RCV1-V2 dataset preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare RCV1-V2 (HiAGM JSON format) from files you own"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/rcv1",
        help="Output directory (default: ./data/rcv1)",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default=None,
        help="Directory holding the preprocessed rcv1_train.json/rcv1_test.json",
    )
    parser.add_argument(
        "--upstream_dir",
        type=str,
        default="/tmp/HBGL",
        help="HBGL checkout directory used by the default preprocessing path",
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=None,
        help=(
            "Directory containing NIST rcv1.tar.xz and "
            "lyrl2004_tokens_train.dat (default: --output_dir)"
        ),
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help=(
            "Install the ~10-document HiAGM smoke-test files instead of the "
            "corpus (exercise the pipeline; never for reported results)"
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir) if args.raw_dir else output_dir

    if args.sample:
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            installed = _install_hiagm_sample(output_dir)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            logger.error("Could not install the HiAGM sample: %s", exc)
            sys.exit(1)
        for path in installed:
            logger.warning("Installed: %s", path)
        return

    if args.source_dir is None:
        try:
            present = _already_present(output_dir)
        except ValueError as exc:
            logger.error("Existing files are unusable: %s", exc)
            logger.error(
                "If these are the HiAGM smoke-test samples, re-run with "
                "--sample to keep them knowingly."
            )
            sys.exit(1)
        if present:
            logger.info(
                "RCV1 files already present and valid: %s, %s",
                present[0].name,
                present[1].name,
            )
            return
        try:
            _run_upstream_preprocessing(Path(args.upstream_dir), raw_dir, output_dir)
        except (OSError, subprocess.CalledProcessError, ValueError) as exc:
            logger.error("Upstream preprocessing failed: %s", exc)
            _print_instructions(output_dir, raw_dir)
            sys.exit(1)
        return

    source_dir = Path(args.source_dir)
    train = _find(source_dir, TRAIN_NAMES)
    test = _find(source_dir, TEST_NAMES)
    if train is None or test is None:
        logger.error(
            "Could not find rcv1_train.json / rcv1_test.json (or train.json / "
            "test.json) in %s. Found: %s",
            source_dir,
            sorted(p.name for p in source_dir.glob("*.json")) or "no JSON files",
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        installed = [_install(train, output_dir), _install(test, output_dir)]
    except ValueError as exc:
        logger.error("Refusing to use %s: %s", source_dir, exc)
        sys.exit(1)

    val = _find(source_dir, VAL_NAMES)
    if val is not None:
        try:
            installed.append(_install(val, output_dir))
        except ValueError as exc:
            logger.warning("Skipping validation split: %s", exc)

    for path in installed:
        logger.info("Ready: %s (%.1f MB)", path, path.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
