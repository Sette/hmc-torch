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

The mapping itself lives in :mod:`hmc.features.yeast_orf`; this script is its command line.

Writes ``data/yeast/<dataset>_ids.tsv`` and prints the unique-match rate.

Usage:
    PYTHONPATH=src python experiments/recover_seq_ids.py --dataset seq_GO
"""

import argparse
import os

from hmc.features.yeast_orf import (
    LEN_TOL,
    MARGIN,
    MAX_DIST,
    SPLITS,
    match_split,
    read_arff,
    read_proteome,
    write_orf_table,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--dataset", default="seq_GO")
    parser.add_argument("--arff_dir", default=None)
    parser.add_argument("--proteome", default="data/yeast/raw/orf_trans_all.fasta.gz")
    parser.add_argument("--out_dir", default="data/yeast")
    parser.add_argument("--len_tol", type=int, default=LEN_TOL)
    parser.add_argument("--max_dist", type=float, default=MAX_DIST)
    parser.add_argument("--margin", type=float, default=MARGIN)
    args = parser.parse_args()

    family = "datasets_GO" if args.dataset.endswith("_GO") else "datasets_FUN"
    arff_dir = args.arff_dir or os.path.join("data/HMC_data_arff", family, args.dataset)
    proteome = read_proteome(args.proteome)
    print(f"proteome: {len(proteome)} ORFs")

    results, total, unique, seen = {}, 0, 0, {}
    for split in SPLITS:
        names, rows = read_arff(os.path.join(arff_dir, f"{args.dataset}.{split}.arff"))
        res = match_split(names, rows, proteome, args.len_tol, args.max_dist, args.margin)
        results[split] = res
        n_unique = sum(x["unique"] for x in res)
        total += len(res)
        unique += n_unique
        for x in res:
            if x["unique"]:
                seen.setdefault(x["orf"], []).append(split)
        print(f"{split}: {n_unique}/{len(res)} unique ({100 * n_unique / len(res):.1f}%)")
    out_path = os.path.join(args.out_dir, f"{args.dataset}_ids.tsv")
    write_orf_table(out_path, results)
    dup = sum(len(v) > 1 for v in seen.values())
    print(f"total: {unique}/{total} unique ({100 * unique / total:.1f}%); ORFs hit by >1 row: {dup}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
