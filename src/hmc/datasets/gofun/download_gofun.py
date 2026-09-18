#!/usr/bin/env python3
"""Download the FunCat / Gene Ontology ARFF benchmarks (GoFun family).

The ARFF files are redistributed in the C-HMCNN repository
(``EGiunchiglia/C-HMCNN``, ``HMC_data/``), the same distribution used by the
HMCN and C-HMCNN papers.  They land in ``<output_dir>/HMC_data_arff/`` — the
layout ``hmc.utils.datasets.paths`` expects — covering the 10 ``*_FUN``
datasets, the 9 ``*_GO`` datasets and the ``*_others`` cross-domain ones.

Usage:
    python -m hmc.datasets.gofun.download_gofun --output_dir ./data
"""

import argparse
import logging
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

REPO_TARBALL = (
    "https://codeload.github.com/EGiunchiglia/C-HMCNN/tar.gz/refs/heads/master"
)
SOURCE_NOTE = "EGiunchiglia/C-HMCNN (HMC_data)"
ARFF_SUBDIRS = ("datasets_FUN", "datasets_GO", "others")


def _has_datasets(target: Path) -> bool:
    """True when every family is already present with ARFF files inside."""
    return all(
        (target / sub).is_dir() and any((target / sub).rglob("*.arff"))
        for sub in ARFF_SUBDIRS
    )


def _download_tarball(dest: Path) -> Path:
    logger.info("Downloading the ARFF bundle from %s …", SOURCE_NOTE)
    tarball = dest / "c-hmcnn.tar.gz"
    urlretrieve(REPO_TARBALL, str(tarball))
    return tarball


def _extract(tarball: Path, target_root: Path) -> int:
    """Extract ``HMC_data/<family>/…`` from the tarball into ``target_root``."""
    members = []
    with tarfile.open(tarball, "r:gz") as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            # <repo-root>/HMC_data/<family>/... → <family>/...
            if len(parts) >= 3 and parts[1] == "HMC_data" and parts[2] in ARFF_SUBDIRS:
                member.name = str(Path(*parts[2:]))
                members.append(member)
        # filter="data" refuses absolute paths and traversal outside target_root
        tar.extractall(path=target_root, members=members, filter="data")
    return len(members)


def _summarise(target: Path) -> None:
    """Log how many datasets of each family are available."""
    for sub in ARFF_SUBDIRS:
        arffs = sorted((target / sub).rglob("*.arff"))
        # FUN/GO ship one directory per dataset; `others` is a flat directory
        datasets = sorted({p.parent.name for p in arffs if p.parent != target / sub})
        if datasets:
            logger.info(
                "  %-12s %2d datasets: %s", sub, len(datasets), ", ".join(datasets)
            )
        else:
            logger.info(
                "  %-12s %2d ARFF files: %s",
                sub,
                len(arffs),
                ", ".join(p.name for p in arffs),
            )


def _print_manual_instructions(output_dir: Path) -> None:
    logger.error(
        "Automatic download failed. Fetch the ARFF files manually:\n"
        "\n"
        "  git clone --depth 1 https://github.com/EGiunchiglia/C-HMCNN.git /tmp/C-HMCNN\n"
        "  mkdir -p %s/HMC_data_arff\n"
        "  cp -r /tmp/C-HMCNN/HMC_data/datasets_FUN \\\n"
        "        /tmp/C-HMCNN/HMC_data/datasets_GO \\\n"
        "        /tmp/C-HMCNN/HMC_data/others %s/HMC_data_arff/\n",
        output_dir,
        output_dir,
    )


def main() -> None:
    """CLI entry point for the FunCat/GO ARFF download."""
    parser = argparse.ArgumentParser(
        description="Download the FunCat/GO ARFF benchmark datasets"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Directory that will contain HMC_data_arff/ (default: ./data)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when the datasets are already present",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    target = output_dir / "HMC_data_arff"

    if not args.force and _has_datasets(target):
        logger.info("ARFF datasets already present in %s. Skipping.", target)
        _summarise(target)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            tarball = _download_tarball(Path(tmp))
            count = _extract(tarball, target)
        except (OSError, tarfile.TarError) as exc:
            logger.error("Download failed: %s", exc)
            _print_manual_instructions(output_dir)
            sys.exit(1)

    logger.info("Extracted %d entries to %s", count, target)
    _summarise(target)


if __name__ == "__main__":
    main()
