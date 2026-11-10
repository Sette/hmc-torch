#!/usr/bin/env python3
"""Obtain the RCV1-V2 dataset in the JSON format this project expects.

The raw Reuters CD-ROMs (``rcv1.tar.xz``) are gated by a NIST license
agreement, and the preprocessed ``rcv1_train.json`` / ``rcv1_test.json`` files
are *not* publicly redistributed.  The similarly named files inside
HiAGM/HiMatch are ~20 KB samples, not the corpus (standard split: 23,149 train /
781,265 test documents), so this script never downloads from there.

Supported paths:

1. ``--lyrl`` — download the official RCV1-v2/LYRL2004 distribution from the
   JMLR/MIT mirror (tokenized text, topic qrels, topic hierarchy) and build the
   JSON splits.  That mirror is openly downloadable and needs no NIST
   agreement; only the raw XML corpus does.  Its text is the SMART stemmed
   token bag (tokens repeated by frequency, not running text), so it suits
   classic TF-IDF models but degrades transformer embeddings such as SPECTER2.
2. ``--source_dir DIR`` — copy ``rcv1_train.json`` / ``rcv1_test.json`` (any of
   the name variants the manager accepts) that you produced yourself.
3. ``--sample`` — install the two ~10-document HiAGM files, for smoke-testing
   the pipeline only.
4. No flag — run the upstream preprocessing scripts. Put the NIST source files
   in ``--raw_dir`` (defaults to ``--output_dir``).

Usage:
    python -m hmc.datasets.rcv1.download_rcv1 --output_dir ./data/rcv1 --lyrl
    python -m hmc.datasets.rcv1.download_rcv1 --output_dir ./data/rcv1 \
        --source_dir /tmp/HBGL/data/rcv1
"""

import argparse
import gzip
import json
import logging
import re
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

# Official RCV1-v2/LYRL2004 distribution (Lewis et al., JMLR 2004).  The MIT
# mirror is openly downloadable and ships the tokenized text, the topic qrels
# and the topic hierarchy -- everything the pipeline needs.  Only the raw
# Reuters CD-ROMs (rcv1.tar.xz) require the NIST agreement.
LYRL_BASE_URL = "http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a"
LYRL_TOKEN_URL = LYRL_BASE_URL + "/a12-token-files"
LYRL_TEST_PARTS = ("pt0", "pt1", "pt2", "pt3")
LYRL_QRELS_URL = LYRL_BASE_URL + "/a08-topic-qrels/rcv1-v2.topics.qrels.gz"
LYRL_HIERARCHY_URL = LYRL_BASE_URL + "/a02-orig-topics-hierarchy/rcv1.topics.hier.orig"

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


def _download(url: str, destination: Path) -> Path:
    """Download *url* to *destination* unless a complete file is already there."""
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s …", url)
    partial = destination.with_suffix(destination.suffix + ".part")
    urlretrieve(url, str(partial))
    partial.replace(destination)
    return destination


def _parse_smart_tokens(path: Path) -> dict[str, str]:
    """Parse a LYRL2004 ``.dat.gz`` file into ``{doc_id: text}``.

    Records are ``.I <id>`` followed by ``.W`` and the token lines.
    """
    docs: dict[str, str] = {}
    doc_id: str | None = None
    tokens: list[str] = []
    with gzip.open(path, "rt", encoding="latin-1") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(".I "):
                if doc_id is not None:
                    docs[doc_id] = " ".join(tokens)
                doc_id = line[3:].strip()
                tokens = []
            elif line.startswith(".W"):
                continue
            elif doc_id is not None and line.strip():
                tokens.extend(line.split())
    if doc_id is not None:
        docs[doc_id] = " ".join(tokens)
    return docs


def _parse_qrels(path: Path) -> dict[str, list[str]]:
    """Parse ``rcv1-v2.topics.qrels`` into ``{doc_id: [topic, ...]}``."""
    topics: dict[str, list[str]] = {}
    with gzip.open(path, "rt", encoding="latin-1") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 3:
                topics.setdefault(parts[1], []).append(parts[0])
    return topics


def _parse_topic_paths(path: Path) -> dict[str, str]:
    """Map each RCV1 topic code to its full path (e.g. ``CCAT/C15/C151``)."""
    parents: dict[str, str | None] = {}
    with open(path, "r", encoding="latin-1") as handle:
        for line in handle:
            match = re.match(r"parent:\s*(\S+)\s+child:\s*(\S+)", line)
            if not match:
                continue
            parent, child = match.groups()
            if child == "Root":
                continue
            parents[child] = None if parent == "Root" else parent

    def path_of(code: str) -> str:
        parts = [code]
        seen = {code}
        while parents.get(parts[-1]):
            parent = parents[parts[-1]]
            if parent in seen:
                break
            seen.add(parent)
            parts.append(parent)
        return "/".join(reversed(parts))

    return {code: path_of(code) for code in parents}


def _write_lyrl_split(
    destination: Path,
    docs: dict[str, str],
    qrels: dict[str, list[str]],
    topic_paths: dict[str, str],
) -> int:
    """Write one split as JSONL in the HiAGM format the manager reads."""
    with open(destination, "w", encoding="utf-8") as out:
        for doc_id, text in docs.items():
            labels = sorted(
                {topic_paths.get(topic, topic) for topic in qrels.get(doc_id, [])}
            )
            out.write(
                json.dumps({"token": text, "label": labels}, ensure_ascii=False) + "\n"
            )
    return len(docs)


def _download_lyrl(output_dir: Path) -> tuple[Path, Path]:
    """Build the RCV1-v2 JSON splits from the open LYRL2004 distribution."""
    logger.warning(
        "LYRL2004 text is stemmed, frequency-ordered tokens (a bag of words), "
        "not running text: fine for TF-IDF models, degraded for SPECTER2."
    )
    cache = output_dir / ".lyrl2004"
    cache.mkdir(parents=True, exist_ok=True)

    hierarchy = _download(LYRL_HIERARCHY_URL, cache / "rcv1.topics.hier.orig")
    qrels_path = _download(LYRL_QRELS_URL, cache / "rcv1-v2.topics.qrels.gz")
    topic_paths = _parse_topic_paths(hierarchy)
    qrels = _parse_qrels(qrels_path)
    logger.info(
        "LYRL2004: %d topics in the hierarchy, %d documents in the qrels.",
        len(topic_paths),
        len(qrels),
    )

    train_tokens = _download(
        f"{LYRL_TOKEN_URL}/lyrl2004_tokens_train.dat.gz",
        cache / "lyrl2004_tokens_train.dat.gz",
    )
    logger.info("Parsing training tokens …")
    train_docs = _parse_smart_tokens(train_tokens)

    test_docs: dict[str, str] = {}
    for part in LYRL_TEST_PARTS:
        test_tokens = _download(
            f"{LYRL_TOKEN_URL}/lyrl2004_tokens_test_{part}.dat.gz",
            cache / f"lyrl2004_tokens_test_{part}.dat.gz",
        )
        logger.info("Parsing test tokens (%s) …", part)
        test_docs.update(_parse_smart_tokens(test_tokens))

    train_out = output_dir / "rcv1_train.json"
    test_out = output_dir / "rcv1_test.json"
    n_train = _write_lyrl_split(train_out, train_docs, qrels, topic_paths)
    n_test = _write_lyrl_split(test_out, test_docs, qrels, topic_paths)
    logger.info(
        "RCV1-v2 built from LYRL2004: %d train / %d test documents.",
        n_train,
        n_test,
    )
    if (n_train, n_test) != (23149, 781265):
        logger.warning(
            "Expected the standard 23,149 / 781,265 split; got %d / %d. "
            "Check the downloaded LYRL2004 files.",
            n_train,
            n_test,
        )

    (output_dir / "LYRL2004_SOURCE.txt").write_text(
        "RCV1-v2 text and labels from the LYRL2004 distribution:\n"
        f"{LYRL_BASE_URL}/\n\n"
        "Cite: D. D. Lewis, Y. Yang, T. G. Rose, and F. Li, 'RCV1: A New\n"
        "Benchmark Collection for Text Categorization Research', JMLR 5,\n"
        "2004, pp. 361-397.  Source data: Reuters Ltd.\n",
        encoding="utf-8",
    )
    return train_out, test_out


def _print_instructions(output_dir: Path, raw_dir: Path) -> None:
    """Explain how to produce the files from the gated original."""
    logger.error(
        "RCV1-V2 is not publicly redistributable, so there is nothing to\n"
        "download automatically.  Two ways to produce the JSON files:\n"
        "\n"
        "  A. Open route (no NIST agreement): use the official RCV1-v2/LYRL2004\n"
        "     distribution from the JMLR/MIT mirror.\n"
        "\n"
        "     python -m hmc.datasets.rcv1.download_rcv1 --output_dir %s --lyrl\n"
        "\n"
        "  B. Raw-corpus route (NIST license required):\n"
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
        output_dir,
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


def _run_sample(output_dir: Path) -> None:
    """Install the tiny HiAGM smoke-test files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        installed = _install_hiagm_sample(output_dir)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.error("Could not install the HiAGM sample: %s", exc)
        sys.exit(1)
    for path in installed:
        logger.warning("Installed: %s", path)


def _run_lyrl(output_dir: Path) -> None:
    """Build the splits from the open LYRL2004 distribution."""
    try:
        present = _already_present(output_dir)
    except ValueError as exc:
        logger.error("Existing files are unusable: %s", exc)
        sys.exit(1)
    if present:
        logger.info(
            "RCV1 files already present and valid: %s, %s",
            present[0].name,
            present[1].name,
        )
        return
    try:
        _download_lyrl(output_dir)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.error("LYRL2004 preparation failed: %s", exc)
        sys.exit(1)


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
        "--lyrl",
        action="store_true",
        help=(
            "build the splits from the open RCV1-v2/LYRL2004 distribution "
            "(tokenized text, qrels, hierarchy; no NIST agreement needed)"
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
        _run_sample(output_dir)
        return

    if args.lyrl:
        _run_lyrl(output_dir)
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
