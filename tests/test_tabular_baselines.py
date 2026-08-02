"""Unit tests for tabular baselines: preprocessing, GBDT, MLP (Marco 3)."""

import numpy as np
import pytest
import torch

from hmc.models.tabular.gbdt_ovr import GBDTOvRClassifier
from hmc.models.tabular.mlp import ResidualBlock, ResidualMLPEncoder, TabularMLPModel
from hmc.models.tabular.preprocessing import TabularPreprocessor


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TestTabularPreprocessor:
    def test_fit_transform_no_selection(self):
        X = np.random.randn(100, 20).astype(np.float32)
        y = (np.random.rand(100, 5) > 0.5).astype(np.float32)

        pp = TabularPreprocessor(with_imputation=True, with_scaling=True)
        X_out = pp.fit_transform(X, y)
        assert X_out.shape == (100, 20)
        assert pp.fitted_

    def test_imputation(self):
        X = np.random.randn(50, 10).astype(np.float32)
        X[0, 0] = np.nan
        X[1, 1] = np.nan

        pp = TabularPreprocessor(with_imputation=True, with_scaling=False)
        X_out = pp.fit_transform(X)
        assert not np.isnan(X_out).any()

    def test_scaling(self):
        X = np.random.randn(100, 5).astype(np.float32) * 10 + 5

        pp = TabularPreprocessor(with_imputation=False, with_scaling=True)
        X_out = pp.fit_transform(X)
        assert abs(X_out.mean()) < 1e-5
        assert abs(X_out.std() - 1.0) < 0.1

    def test_variance_selection(self):
        X = np.random.randn(100, 50).astype(np.float32)
        # Make some columns constant
        X[:, 10:15] = 0.0
        y = (np.random.rand(100, 5) > 0.5).astype(np.float32)

        pp = TabularPreprocessor(
            with_scaling=False,
            feature_selector="variance",
            selector_kwargs={"threshold": 1e-6},
        )
        X_out = pp.fit_transform(X, y)
        assert X_out.shape[1] < 50  # constant columns removed

    def test_mutual_info_selection(self):
        X = np.random.randn(100, 30).astype(np.float32)
        y = (np.random.rand(100, 3) > 0.5).astype(np.float32)

        pp = TabularPreprocessor(
            with_scaling=False,
            feature_selector="mutual_info",
            selector_kwargs={"k": 10},
        )
        X_out = pp.fit_transform(X, y)
        assert X_out.shape[1] == 10

    def test_get_params(self):
        pp = TabularPreprocessor(feature_selector="variance")
        X = np.random.randn(50, 10).astype(np.float32)
        pp.fit(X)
        params = pp.get_params()
        assert "n_features_selected" in params
        assert params["feature_selector"] == "variance"

    def test_transform_before_fit_raises(self):
        pp = TabularPreprocessor()
        X = np.random.randn(10, 5).astype(np.float32)
        with pytest.raises(RuntimeError, match="fit"):
            pp.transform(X)


# ---------------------------------------------------------------------------
# GBDT
# ---------------------------------------------------------------------------


class TestGBDTOvR:
    def test_fit_predict_proba(self):
        X = np.random.randn(100, 10).astype(np.float32)
        y = (np.random.rand(100, 3) > 0.5).astype(np.float32)
        # Ensure at least some positive/negative per class
        y[:10, 0] = 1
        y[-10:, 0] = 0

        model = GBDTOvRClassifier(backend="histgb")
        model.fit(X, y)
        assert model.fitted_
        assert model.n_estimators_trained > 0

        proba = model.predict_proba(X)
        assert proba.shape == (100, 3)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_predict_binary(self):
        X = np.random.randn(50, 5).astype(np.float32)
        y = np.zeros((50, 2), dtype=np.float32)
        y[:25, 0] = 1
        y[25:, 1] = 1

        model = GBDTOvRClassifier(backend="histgb")
        model.fit(X, y)
        pred = model.predict(X, threshold=0.5)
        assert pred.shape == (50, 2)

    def test_eval_mask(self):
        X = np.random.randn(60, 5).astype(np.float32)
        y = np.zeros((60, 4), dtype=np.float32)
        y[:20, 0] = 1
        y[20:40, 1] = 1
        y[40:, 2] = 1
        # Node 3 is all negative (single class)

        eval_mask = np.array([True, True, True, False])
        model = GBDTOvRClassifier(backend="histgb")
        model.fit(X, y, eval_mask=eval_mask)
        assert model.n_estimators_trained == 3  # skip node 3

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            GBDTOvRClassifier(backend="xgboost")


# ---------------------------------------------------------------------------
# Residual MLP
# ---------------------------------------------------------------------------


class TestResidualMLP:
    def test_residual_block(self):
        block = ResidualBlock(dim=64, dropout=0.1)
        x = torch.randn(4, 64)
        out = block(x)
        assert out.shape == (4, 64)
        assert not torch.isnan(out).any()

    def test_encoder_output_dim(self):
        encoder = ResidualMLPEncoder(
            input_dim=50, hidden_dim=128, n_blocks=3, dropout=0.1
        )
        x = torch.randn(8, 50)
        out = encoder(x)
        assert out.shape == (8, 128)

    def test_full_model(self):
        model = TabularMLPModel(
            input_dim=30, n_nodes=10, hidden_dim=64,
            n_blocks=2, head_layers=1, dropout=0.1,
        )
        x = torch.randn(16, 30)
        out = model(x)
        assert out.shape == (16, 10)
        assert (out >= 0).all() and (out <= 1).all()

    def test_get_embeddings(self):
        model = TabularMLPModel(
            input_dim=20, n_nodes=5, hidden_dim=64, n_blocks=2,
        )
        x = torch.randn(4, 20)
        emb = model.get_embeddings(x)
        assert emb.shape == (4, 64)

    def test_training_step(self):
        """One training step should reduce loss."""
        model = TabularMLPModel(
            input_dim=10, n_nodes=3, hidden_dim=32,
            n_blocks=1, head_layers=1, dropout=0.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.BCELoss()

        x = torch.randn(32, 10)
        y = (torch.rand(32, 3) > 0.5).float()

        # Forward before training
        with torch.no_grad():
            loss_before = criterion(model(x), y).item()

        # One step
        model.train()
        preds = model(x)
        loss = criterion(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss should decrease
        with torch.no_grad():
            loss_after = criterion(model(x), y).item()

        assert loss_after < loss_before, \
            f"Training step should reduce loss: {loss_before:.4f} -> {loss_after:.4f}"
