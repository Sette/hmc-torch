"""Tests for the protein encoder and the seq_ ARFF → ORF recovery."""

import os
import sys
from argparse import Namespace

import numpy as np
import pytest

from hmc.features.protein import ProteinFeatureEncoder, sequences_hash

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from recover_seq_ids import (  # pylint: disable=wrong-import-position
    AA,
    composition,
    match_split,
)

TINY = "facebook/esm2_t6_8M_UR50D"
SEQS = ["MKTAYIAKQRQISFVKSHFSRQ", "MIVNNTHVLT", "ACDE"]


def test_sequences_hash_depends_on_model_and_order():
    h = sequences_hash(TINY, SEQS)
    assert h == sequences_hash(TINY, list(SEQS))
    assert h != sequences_hash(TINY, SEQS[::-1])
    assert h != sequences_hash("other", SEQS)


@pytest.fixture(scope="module")
def encoder(tmp_path_factory):
    enc = ProteinFeatureEncoder(TINY, cache_dir=str(tmp_path_factory.mktemp("cache")),
                                device="cpu")
    try:
        enc._ensure_loaded()  # pylint: disable=protected-access
    except OSError as exc:  # offline and not cached
        pytest.skip(f"{TINY} unavailable: {exc}")
    return enc


def test_encode_shape_and_cache(encoder):
    x = encoder.encode(SEQS)
    assert x.shape == (3, 320) and x.dtype == np.float32
    encoder._model = None  # pylint: disable=protected-access
    np.testing.assert_array_equal(encoder.encode(SEQS), x)  # served from cache
    assert encoder._model is None  # pylint: disable=protected-access


def test_short_sequences_ignore_window(encoder):
    windowed = ProteinFeatureEncoder(TINY, cache_dir=None, device="cpu", window=10)
    windowed._model, windowed._tokenizer = (  # pylint: disable=protected-access
        encoder._model,  # pylint: disable=protected-access
        encoder._tokenizer,  # pylint: disable=protected-access
    )
    full = ProteinFeatureEncoder(TINY, cache_dir=None, device="cpu")
    full._model, full._tokenizer = windowed._model, windowed._tokenizer  # pylint: disable=protected-access
    a, b = windowed.encode(SEQS), full.encode(SEQS)
    np.testing.assert_allclose(a[1:], b[1:], atol=1e-5)  # fit in one window
    assert not np.allclose(a[0], b[0])  # split into three windows


def _row(seq, chrom):
    names = [f"aa_rat_{a}" for a in AA] + ["seq_len", "chromosome"]
    return names, [f"{v:.1f}" for v in composition(seq)] + [str(len(seq)), chrom]


def test_match_split_finds_composition_and_breaks_ties_by_chromosome():
    proteome = [("YA", "1", "MKTAYIAKQR"), ("YB", "2", "MKTAYIAKQR"), ("YC", "1", "MGGGGGGGGW")]
    names, row = _row("MKTAYIAKQR", "2")
    args = Namespace(len_tol=5, max_dist=2.0, margin=1.0)
    (res,) = match_split(names, [row], proteome, args)
    assert res["orf"] == "YB" and res["unique"] and res["dist"] == 0


def test_identical_paralogs_count_as_unique():
    proteome = [("YA", "1", "MKTAYIAKQR"), ("YB", "1", "MKTAYIAKQR")]
    names, row = _row("MKTAYIAKQR", "1")
    (res,) = match_split(names, [row], proteome, Namespace(len_tol=5, max_dist=2.0, margin=1.0))
    assert res["unique"]
