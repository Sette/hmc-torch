"""Recover the yeast ORF behind each row of the ``seq_`` ARFF datasets.

The Clare/Vens ARFF files carry no gene identifier, but ``seq_GO`` / ``seq_FUN`` keep the
protein length (``seq_len``), the amino-acid composition in percent with one decimal
(``aa_rat_*``) and the chromosome.  Recomputing those for the S288C proteome (SGD
``orf_trans_all.fasta``) pins most rows to a single ORF, which gives the sequence a protein
language model needs (see :mod:`hmc.features.protein`).

A row is matched to the ORF of the same chromosome whose length is within ``len_tol`` residues
and whose composition is closest in L1; the match is ``unique`` when that distance is at most
``max_dist`` and the runner-up is at least ``margin`` further away.  ORFs re-annotated since
2007 explain the tolerance.

Only the *features* of the ARFF (length, composition, chromosome) are read: no label column is
ever touched, so the mapping cannot leak annotations.  The proteome FASTA is not shipped; the
caller downloads it from SGD.
"""

from __future__ import annotations

import csv
import gzip
import os
import re

import numpy as np

AA = "acdefghiklmnpqrstvwy"
ROMAN = {
    r: str(i)
    for i, r in enumerate(
        [
            "I",
            "II",
            "III",
            "IV",
            "V",
            "VI",
            "VII",
            "VIII",
            "IX",
            "X",
            "XI",
            "XII",
            "XIII",
            "XIV",
            "XV",
            "XVI",
        ],
        start=1,
    )
}
ROMAN["Mito"] = "mit"
SPLITS = ("train", "valid", "test")

LEN_TOL, MAX_DIST, MARGIN = 5, 2.0, 1.0


def read_proteome(path: str) -> list[tuple[str, str, str]]:
    """Return ``[(orf, chromosome, sequence)]`` from an SGD protein FASTA (``.gz`` allowed)."""
    opener = gzip.open if path.endswith(".gz") else open
    records, header, chunks = [], None, []
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header, chunks = line[1:], []
            else:
                chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks)))

    out = []
    for head, seq in records:
        chrom = re.search(r"Chr (\S+?)[ ,]", head)
        out.append(
            (
                head.split()[0],
                ROMAN.get(chrom.group(1), "?") if chrom else "?",
                seq.rstrip("*").upper(),
            )
        )
    return out


def composition(seq: str) -> np.ndarray:
    """Percent composition over ``AA`` rounded to one decimal, as in the ARFF."""
    n = max(len(seq), 1)
    return np.array([round(100.0 * seq.count(a.upper()) / n, 1) for a in AA])


def read_arff(path: str) -> tuple[list[str], list[list[str]]]:
    """Return ``(attribute names, rows)`` of an ARFF file (rows as raw strings)."""
    names, rows, in_data = [], [], False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if in_data:
                rows.append(next(csv.reader([line])))
            elif line.lower().startswith("@attribute"):
                names.append(line.split()[1])
            elif line.lower().startswith("@data"):
                in_data = True
    return names, rows


def _num(value: str) -> float:
    return np.nan if value == "?" else float(value)


def match_split(  # pylint: disable=too-many-locals
    names: list[str],
    rows: list[list[str]],
    proteome: list[tuple[str, str, str]],
    len_tol: int = LEN_TOL,
    max_dist: float = MAX_DIST,
    margin: float = MARGIN,
) -> list[dict]:
    """Match every row of one split; return one result dict per row.

    Each dict has ``row``, ``orf``, ``dist``, ``second`` (distance of the best rival with a
    different sequence) and ``unique``.
    """
    col = {n: i for i, n in enumerate(names)}
    aa_cols = [col[f"aa_rat_{a}"] for a in AA]
    lengths = np.array([len(s) for _, _, s in proteome])
    comps = np.stack([composition(s) for _, _, s in proteome])
    chroms = np.array([c for _, c, _ in proteome])

    results = []
    for r, row in enumerate(rows):
        seq_len = _num(row[col["seq_len"]])
        comp = np.array([_num(row[i]) for i in aa_cols])
        chrom = row[col["chromosome"]] if "chromosome" in col else "?"
        cand = np.abs(lengths - seq_len) <= len_tol
        if chrom != "?":
            cand &= chroms == chrom
        idx = np.nonzero(cand)[0]
        if len(idx) == 0:
            results.append(
                {"row": r, "orf": "", "dist": np.inf, "second": np.inf, "unique": False}
            )
            continue
        dist = np.nansum(np.abs(comps[idx] - comp), axis=1) + np.abs(
            lengths[idx] - seq_len
        )
        order = np.argsort(dist)
        best = float(dist[order[0]])
        # Paralogs with identical proteins tie at the top; any of them gives the same sequence.
        best_seq = proteome[idx[order[0]]][2]
        rivals = [k for k in order[1:] if proteome[idx[k]][2] != best_seq]
        second = float(dist[rivals[0]]) if rivals else np.inf
        results.append(
            {
                "row": r,
                "orf": proteome[idx[order[0]]][0],
                "dist": best,
                "second": second,
                "unique": best <= max_dist and second - best >= margin,
            }
        )
    return results


def recover_orf_ids(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    arff_dir: str,
    dataset: str,
    proteome: list[tuple[str, str, str]],
    len_tol: int = LEN_TOL,
    max_dist: float = MAX_DIST,
    margin: float = MARGIN,
) -> dict[str, list[dict]]:
    """Match the three splits of ``<arff_dir>/<dataset>.{train,valid,test}.arff``."""
    out = {}
    for split in SPLITS:
        names, rows = read_arff(os.path.join(arff_dir, f"{dataset}.{split}.arff"))
        out[split] = match_split(names, rows, proteome, len_tol, max_dist, margin)
    return out


def write_orf_table(path: str, results: dict[str, list[dict]]) -> None:
    """Write the ``split / row / orf / dist / second / unique`` table read by :func:`read_orf_ids`."""
    with open(path, "w", encoding="utf-8") as out:
        out.write("split\trow\torf\tdist\tsecond\tunique\n")
        for split in SPLITS:
            out.writelines(
                f"{split}\t{x['row']}\t{x['orf']}\t{x['dist']:.2f}\t{x['second']:.2f}\t{int(x['unique'])}\n"
                for x in results[split]
            )


def read_orf_ids(path: str) -> dict[str, list[str | None]]:
    """``{split: [orf or None per row]}`` from a table written by :func:`write_orf_table`.

    Rows without a unique match are ``None``.
    """
    ids: dict[str, list[str | None]] = {s: [] for s in SPLITS}
    with open(path, encoding="utf-8") as fh:
        for rec in csv.DictReader(fh, delimiter="\t"):
            ids[rec["split"]].append(rec["orf"] if rec["unique"] == "1" else None)
    return ids
