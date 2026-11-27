"""Tests for the seq_ ARFF → yeast ORF recovery and the ORF-based protein features."""

import numpy as np
import pytest

from hmc.features.protein import orf_embedding_features
from hmc.features.yeast_orf import (
    AA,
    composition,
    match_split,
    read_arff,
    read_orf_ids,
    recover_orf_ids,
    write_orf_table,
)

PROTEOME = [
    ("YA", "1", "MKTAYIAKQR"),
    ("YB", "2", "MKTAYIAKQR"),
    ("YC", "1", "MGGGGGGGGW"),
]


def _row(seq, chrom):
    names = [f"aa_rat_{a}" for a in AA] + ["seq_len", "chromosome"]
    return names, [f"{v:.1f}" for v in composition(seq)] + [str(len(seq)), chrom]


def test_match_split_finds_composition_and_breaks_ties_by_chromosome():
    names, row = _row("MKTAYIAKQR", "2")
    (res,) = match_split(names, [row], PROTEOME)
    assert res["orf"] == "YB" and res["unique"] and res["dist"] == 0


def test_identical_paralogs_count_as_unique():
    proteome = [("YA", "1", "MKTAYIAKQR"), ("YB", "1", "MKTAYIAKQR")]
    names, row = _row("MKTAYIAKQR", "1")
    (res,) = match_split(names, [row], proteome)
    assert res["unique"]


def test_no_candidate_is_not_unique():
    names, row = _row("MKTAYIAKQR", "3")  # chromosome 3 has no ORF
    (res,) = match_split(names, [row], PROTEOME)
    assert res["orf"] == "" and not res["unique"]


def test_ambiguous_match_is_not_unique():
    proteome = [
        ("YA", "1", "MKTAYIAKQR"),
        ("YB", "1", "MKTAYIAKQK"),
    ]  # nearly identical, different protein
    names, row = _row("MKTAYIAKQR", "1")
    (res,) = match_split(
        names, [row], proteome, margin=30.0
    )  # rival is 20 away: too close
    assert not res["unique"]


def _write_arff(path, rows):
    names, _ = _row("M", "1")
    with open(path, "w", encoding="utf-8") as f:
        f.write("@relation toy\n")
        for n in names:
            f.write(f"@attribute {n} numeric\n")
        f.write(
            "@attribute class_labels string\n@data\n"
        )  # a label column the mapping must ignore
        for r in rows:
            f.write(",".join(r) + ",1/2@1/3\n")


def test_recover_orf_ids_reads_only_features_and_roundtrips(tmp_path):
    for split in ("train", "valid", "test"):
        _write_arff(
            tmp_path / f"toy.{split}.arff",
            [_row("MKTAYIAKQR", "2")[1], _row("MGGGGGGGGW", "1")[1]],
        )
    names, rows = read_arff(str(tmp_path / "toy.train.arff"))
    assert names[-1] == "class_labels" and len(rows) == 2

    results = recover_orf_ids(str(tmp_path), "toy", PROTEOME)
    assert [x["orf"] for x in results["train"]] == ["YB", "YC"]

    table = tmp_path / "toy_ids.tsv"
    write_orf_table(str(table), results)
    ids = read_orf_ids(str(table))
    assert ids == {"train": ["YB", "YC"], "valid": ["YB", "YC"], "test": ["YB", "YC"]}


def test_read_orf_ids_marks_non_unique_rows_as_none(tmp_path):
    table = tmp_path / "t.tsv"
    table.write_text(
        "split\trow\torf\tdist\tsecond\tunique\n"
        "train\t0\tYA\t0.00\t9.00\t1\ntrain\t1\tYB\t1.00\t1.50\t0\n"
        "valid\t0\t\tinf\tinf\t0\ntest\t0\tYC\t0.00\t9.00\t1\n"
    )
    ids = read_orf_ids(str(table))
    assert ids == {"train": ["YA", None], "valid": [None], "test": ["YC"]}


class _FakeEncoder:
    """Deterministic stand-in for the protein encoder (embedding = length, A-count, 1)."""

    def encode(self, sequences):
        return np.array(
            [[len(s), s.count("A"), 1.0] for s in sequences], dtype=np.float32
        )


def test_orf_embedding_features_shapes_flag_and_standardization():
    ids = {"train": ["YA", "YC", None, "YA"], "valid": ["YC", None], "test": ["YA"]}
    x_splits = [np.zeros((4, 7)), np.zeros((2, 7)), np.zeros((1, 7))]
    train, valid, test = orf_embedding_features(x_splits, ids, _FakeEncoder(), PROTEOME)
    assert [m.shape for m in (train, valid, test)] == [(4, 4), (2, 4), (1, 4)]
    np.testing.assert_array_equal(train[:, -1], [0, 0, 1, 0])  # missing-ORF flag
    np.testing.assert_array_equal(valid[:, -1], [0, 1])
    # z-scored on the training rows: the train embedding columns have mean 0 (imputed row included)
    np.testing.assert_allclose(train[:, :2].mean(axis=0), 0.0, atol=1e-5)
    assert train.dtype == np.float32


def test_orf_embedding_features_rejects_length_mismatch():
    ids = {"train": ["YA"], "valid": ["YA"], "test": ["YA"]}
    with pytest.raises(ValueError, match="train"):
        orf_embedding_features(
            [np.zeros((2, 1)), np.zeros((1, 1)), np.zeros((1, 1))],
            ids,
            _FakeEncoder(),
            PROTEOME,
        )
