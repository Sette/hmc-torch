#!/usr/bin/env python3
"""Recover the yeast ORF behind each row of the ``seq_`` ARFF datasets.

The Clare/Vens ARFF files carry no gene identifier, but ``seq_GO`` / ``seq_FUN``
keep the protein length (``seq_len``), the amino-acid composition in percent
with one decimal (``aa_rat_*``) and the chromosome.  Recomputing those for the
S288C proteome (SGD ``orf_trans_all.fasta``) pins most rows to a single ORF,
which gives the sequence a protein language model needs.

A row is matched to the ORF of the same chromosome whose length is within
``--len_tol`` residues and whose composition is closest in L1; the match is
``unique`` when that distance is at most ``--max_dist`` and the runner-up is
at least ``--margin`` further away.  ORFs re-annotated since 2007 explain the
tolerance.

Writes ``data/yeast/<dataset>_ids.tsv`` and prints the unique-match rate.

Usage:
    PYTHONPATH=src python experiments/recover_seq_ids.py --dataset seq_GO
"""

import argparse
import csv
import gzip
import os
import re

import numpy as np

AA = "acdefghiklmnpqrstvwy"
ROMAN = {
    r: str(i)
    for i, r in enumerate(
        "I II III IV V VI VII VIII IX X XI XII XIII XIV XV XVI".split(), start=1
    )
}
ROMAN["Mito"] = "mit"


def read_proteome(path):
    """Return ``[(orf, chromosome, sequence)]`` from an SGD protein FASTA."""
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


def composition(seq):
    """Percent composition over ``AA`` rounded to one decimal, as in the ARFF."""
    n = max(len(seq), 1)
    return np.array([round(100.0 * seq.count(a.upper()) / n, 1) for a in AA])


def read_arff(path):
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


def _num(value):
    return np.nan if value == "?" else float(value)


def match_split(names, rows, proteome, args):
    """Match every row of one split; return a list of result dicts."""
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
        cand = np.abs(lengths - seq_len) <= args.len_tol
        if chrom != "?":
            cand &= chroms == chrom
        idx = np.nonzero(cand)[0]
        if len(idx) == 0:
            results.append({"row": r, "orf": "", "dist": np.inf, "second": np.inf, "unique": False})
            continue
        dist = np.nansum(np.abs(comps[idx] - comp), axis=1) + np.abs(lengths[idx] - seq_len)
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
                "unique": best <= args.max_dist and second - best >= args.margin,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--dataset", default="seq_GO")
    parser.add_argument("--arff_dir", default=None)
    parser.add_argument("--proteome", default="data/yeast/raw/orf_trans_all.fasta.gz")
    parser.add_argument("--out_dir", default="data/yeast")
    parser.add_argument("--len_tol", type=int, default=5)
    parser.add_argument("--max_dist", type=float, default=2.0)
    parser.add_argument("--margin", type=float, default=1.0)
    args = parser.parse_args()

    family = "datasets_GO" if args.dataset.endswith("_GO") else "datasets_FUN"
    arff_dir = args.arff_dir or os.path.join("data/HMC_data_arff", family, args.dataset)
    proteome = read_proteome(args.proteome)
    print(f"proteome: {len(proteome)} ORFs")

    out_path = os.path.join(args.out_dir, f"{args.dataset}_ids.tsv")
    total, unique, seen = 0, 0, {}
    with open(out_path, "w", encoding="utf-8") as out:
        out.write("split\trow\torf\tdist\tsecond\tunique\n")
        for split in ("train", "valid", "test"):
            names, rows = read_arff(os.path.join(arff_dir, f"{args.dataset}.{split}.arff"))
            res = match_split(names, rows, proteome, args)
            n_unique = sum(x["unique"] for x in res)
            total += len(res)
            unique += n_unique
            for x in res:
                if x["unique"]:
                    seen.setdefault(x["orf"], []).append(split)
                out.write(
                    f"{split}\t{x['row']}\t{x['orf']}\t{x['dist']:.2f}\t{x['second']:.2f}\t{int(x['unique'])}\n"
                )
            print(f"{split}: {n_unique}/{len(res)} unique ({100 * n_unique / len(res):.1f}%)")
    dup = sum(len(v) > 1 for v in seen.values())
    print(f"total: {unique}/{total} unique ({100 * unique / total:.1f}%); ORFs hit by >1 row: {dup}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
