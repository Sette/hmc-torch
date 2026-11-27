"""Tests for the protein encoder (the ARFF → ORF recovery is in ``test_yeast_orf.py``)."""

import numpy as np
import pytest

from hmc.features.protein import ProteinFeatureEncoder, sequences_hash

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
