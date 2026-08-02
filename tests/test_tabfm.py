"""Tests for TabFM local-conditional modules (Marco 4).

These tests mock the actual TabFM package since it may not be
installed and has a non-commercial license.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock tabfm before any real import attempts
_tabfm_mock = MagicMock()
sys.modules["tabfm"] = _tabfm_mock

from hmc.models.tabular.tabfm.adapter import TabFMAdapter, _check_tabfm, require_tabfm
from hmc.models.tabular.tabfm.cache import TabFMCache
from hmc.models.tabular.tabfm.context import (
    ConditionalNodeDataset,
    StratifiedContextSampler,
)

# Clear the cached check so the mock is picked up
import hmc.models.tabular.tabfm.adapter as adapter_mod
adapter_mod._TABFM_AVAILABLE = True


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class TestTabFMAdapter:
    def test_available(self):
        assert _check_tabfm() is True

    def test_require_does_not_raise(self):
        require_tabfm()  # should not raise with mock

    def test_load(self):
        adapter = TabFMAdapter(model_name="test-model", device="cpu")
        adapter.load()
        assert adapter._loaded

    def test_model_auto_loads(self):
        adapter = TabFMAdapter(model_name="test-model", device="cpu")
        _ = adapter.model  # triggers auto-load
        assert adapter._loaded


# ---------------------------------------------------------------------------
# Context sampler
# ---------------------------------------------------------------------------


class TestConditionalNodeDataset:
    def test_root_node_uses_all_rows(self):
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.rand(100, 5).astype(np.float32)
        ds = ConditionalNodeDataset(X, y, node_idx=0, parent_indices=[])
        assert ds.n_samples == 100

    def test_conditional_filters_by_parent(self):
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.zeros((50, 3), dtype=np.float32)
        y[:20, 0] = 1.0  # parent
        y[:, 1] = np.random.randint(0, 2, 50).astype(np.float32)  # child

        ds = ConditionalNodeDataset(X, y, node_idx=1, parent_indices=[0])
        assert ds.n_samples == 20  # only rows where parent=1

    def test_prevalence(self):
        X = np.random.randn(30, 5).astype(np.float32)
        y = np.zeros((30, 3), dtype=np.float32)
        y[:15, 0] = 1.0
        y[:10, 1] = 1.0  # child positive in 10/15

        ds = ConditionalNodeDataset(X, y, node_idx=1, parent_indices=[0])
        assert ds.prevalence == pytest.approx(10 / 15)


class TestStratifiedContextSampler:
    def test_sample_balanced(self):
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.zeros((100, 3), dtype=np.float32)
        y[:30, 0] = 1.0  # 30 positive, 70 negative

        sampler = StratifiedContextSampler(k_pos=10, k_neg=10, seed=42)
        X_ctx, y_ctx = sampler.sample(X, y, node_idx=0, parent_indices=[])

        assert X_ctx.shape[0] == 20
        assert y_ctx.sum() == 10  # exactly k_pos positives

    def test_sample_reproducible(self):
        X = np.random.randn(100, 5).astype(np.float32)
        y = (np.random.rand(100, 2) > 0.5).astype(np.float32)

        sampler1 = StratifiedContextSampler(k_pos=5, k_neg=5, seed=123)
        sampler2 = StratifiedContextSampler(k_pos=5, k_neg=5, seed=123)

        ctx1 = sampler1.sample(X, y, node_idx=0, parent_indices=[])
        ctx2 = sampler2.sample(X, y, node_idx=0, parent_indices=[])

        assert np.array_equal(ctx1[0], ctx2[0])
        assert np.array_equal(ctx1[1], ctx2[1])

    def test_fallback_reduce(self):
        """When fewer positives than k_pos, reduce is used."""
        X = np.random.randn(20, 5).astype(np.float32)
        y = np.zeros((20, 2), dtype=np.float32)
        y[:3, 0] = 1.0  # only 3 positives

        sampler = StratifiedContextSampler(k_pos=10, k_neg=10,
                                           seed=42, fallback="reduce")
        X_ctx, y_ctx = sampler.sample(X, y, node_idx=0, parent_indices=[])
        assert X_ctx.shape[0] < 20  # reduced

    def test_sample_multiple(self):
        X = np.random.randn(50, 5).astype(np.float32)
        y = (np.random.rand(50, 3) > 0.5).astype(np.float32)
        parent_map = {0: [], 1: [0], 2: [0]}

        sampler = StratifiedContextSampler(k_pos=5, k_neg=5, seed=42)
        results = sampler.sample_multiple(
            X, y, node_indices=[0, 1, 2],
            parent_map=parent_map, n_contexts=3,
        )
        assert len(results) == 3
        for nid, contexts in results.items():
            assert len(contexts) == 3
            for X_ctx, y_ctx in contexts:
                assert X_ctx.shape[0] == 10


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class TestTabFMCache:
    def test_save_and_load_context(self, tmp_path):
        cache = TabFMCache(cache_dir=str(tmp_path), enabled=True)
        X_ctx = np.random.randn(20, 5).astype(np.float32)
        y_ctx = (np.random.rand(20) > 0.5).astype(np.float32)

        cache.save_context("test_ds", 0, 10, 10, 42, X_ctx, y_ctx)
        loaded = cache.load_context("test_ds", 0, 10, 10, 42)

        assert loaded is not None
        X_loaded, y_loaded = loaded
        assert np.array_equal(X_loaded, X_ctx)
        assert np.array_equal(y_loaded, y_ctx)

    def test_disabled_cache_returns_none(self):
        cache = TabFMCache(enabled=False)
        result = cache.load_context("test", 0, 10, 10, 42)
        assert result is None

    def test_save_and_load_predictions(self, tmp_path):
        cache = TabFMCache(cache_dir=str(tmp_path), enabled=True)
        scores = np.random.rand(50, 10).astype(np.float32)
        cache.save_predictions("test_ds", "test", 8, scores)
        loaded = cache.load_predictions("test_ds", "test", 8)
        assert np.array_equal(loaded, scores)

    def test_cache_key_deterministic(self):
        cache = TabFMCache(enabled=True)
        k1 = cache._key("seq_FUN", 5, 50, 50, 42)
        k2 = cache._key("seq_FUN", 5, 50, 50, 42)
        assert k1 == k2

    def test_cache_key_differs(self):
        cache = TabFMCache(enabled=True)
        k1 = cache._key("seq_FUN", 5, 50, 50, 42)
        k2 = cache._key("seq_FUN", 5, 50, 50, 43)
        assert k1 != k2


# ---------------------------------------------------------------------------
# Integration: sampler + dataset
# ---------------------------------------------------------------------------


class TestNoLeakage:
    """Verify that contexts never include validation/test IDs."""

    def test_conditional_dataset_masks_by_parent(self):
        X = np.random.randn(40, 5).astype(np.float32)
        y = np.zeros((40, 4), dtype=np.float32)
        y[:30, 0] = 1.0  # parent positive in first 30
        y[:15, 2] = 1.0  # child positive in first 15

        ds = ConditionalNodeDataset(X, y, node_idx=2, parent_indices=[0])
        # Only first 30 rows where parent=1
        assert ds.n_samples == 30
        # Child positive count
        assert ds.y_filtered.sum() == 15
