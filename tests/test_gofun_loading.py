"""Tests for GoFun ARFF dataset loading and local classifier training."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# We patch paths at import time so the fixture paths are resolved correctly
_PATCHED = False


def _patch_gofun_paths(temp_dir: str):
    """Redirect GoFun dataset paths to the synthetic fixture directory."""
    global _PATCHED
    import hmc.utils.datasets.paths as path_mod

    # Store originals so we can restore
    if not _PATCHED:
        path_mod._original_get_dataset_paths = path_mod.get_dataset_paths
        _PATCHED = True

    def _fixture_paths(dataset_path="./data"):
        """Return paths pointing into the temp fixture directory."""
        # Ignore dataset_path, always use temp_dir
        base = Path(temp_dir) / "HMC_data_arff"
        return {
            "seq_FUN": (
                False,
                str(base / "datasets_FUN/seq_FUN/seq_FUN.train.arff"),
                str(base / "datasets_FUN/seq_FUN/seq_FUN.valid.arff"),
                str(base / "datasets_FUN/seq_FUN/seq_FUN.test.arff"),
            ),
            "seq_GO": (
                True,
                str(base / "datasets_GO/seq_GO/seq_GO.train.arff"),
                str(base / "datasets_GO/seq_GO/seq_GO.valid.arff"),
                str(base / "datasets_GO/seq_GO/seq_GO.test.arff"),
            ),
        }

    path_mod.get_dataset_paths = _fixture_paths


def _restore_gofun_paths():
    import hmc.utils.datasets.paths as path_mod
    if hasattr(path_mod, "_original_get_dataset_paths"):
        path_mod.get_dataset_paths = path_mod._original_get_dataset_paths


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

from tests.fixtures.arff import SyntheticARFFFixture


@pytest.fixture(scope="module")
def seq_fun_arff_dir():
    """Create a minimal seq_FUN ARFF dataset on disk."""
    with SyntheticARFFFixture(
        name="seq_FUN",
        hierarchy="root.CC.CC01,root.CC.CC02,root.MF.MF01,root.BP.BP01",
        num_features=10,
        num_train=30,
        num_valid=6,
        num_test=6,
        is_go=False,
    ) as f:
        _patch_gofun_paths(f._tmpdir.name)
        yield f._tmpdir.name
        _restore_gofun_paths()


# ---------------------------------------------------------------------------
# Marco 0 — acceptance tests
# ---------------------------------------------------------------------------

class TestGoFunDataLoading:
    """Verify that GoFun ARFF datasets can be loaded."""

    def test_load_seq_fun_returns_manager(self, seq_fun_arff_dir):
        from hmc.datasets.dataset_manager import initialize_dataset_experiments

        mgr = initialize_dataset_experiments(
            "seq_FUN", device="cpu",
            dataset_path=seq_fun_arff_dir, is_global=False,
        )
        assert mgr is not None
        assert mgr.input_dim == 10
        assert mgr.output_dim > 0
        assert mgr.max_depth > 0

    def test_load_seq_fun_splits(self, seq_fun_arff_dir):
        from hmc.datasets.dataset_manager import initialize_dataset_experiments

        mgr = initialize_dataset_experiments(
            "seq_FUN", device="cpu",
            dataset_path=seq_fun_arff_dir, is_global=False,
        )
        train, valid, test = mgr.get_datasets()

        assert train.x.shape[0] == 30
        assert train.x.shape[1] == 10
        assert valid.x.shape[0] == 6
        assert test.x.shape[0] == 6

        # Labels should be binary vectors of length n_nodes
        n_nodes = mgr.output_dim
        assert train.y.shape == (30, n_nodes)
        assert valid.y.shape == (6, n_nodes)
        assert test.y.shape == (6, n_nodes)

    def test_label_closure_fun(self, seq_fun_arff_dir):
        """Every positive leaf must have its ancestors also positive."""
        from hmc.datasets.dataset_manager import initialize_dataset_experiments

        mgr = initialize_dataset_experiments(
            "seq_FUN", device="cpu",
            dataset_path=seq_fun_arff_dir, is_global=False,
        )
        train, _, _ = mgr.get_datasets()

        # Use the hierarchy graph from the parsed dataset (parent→child
        # direction via g_t), which is the correct graph for finding
        # ancestors of a node via nx.ancestors.
        g_anc = train.g_t

        import networkx as nx
        for i in range(len(train.y)):
            for node, idx in mgr.nodes_idx.items():
                if train.y[i, idx] == 1:
                    for ancestor in nx.ancestors(g_anc, node):
                        anc_idx = mgr.nodes_idx[ancestor]
                        assert train.y[i, anc_idx] == 1, (
                            f"Sample {i}: node '{node}' is positive but ancestor "
                            f"'{ancestor}' is not (idx={idx}, anc_idx={anc_idx})"
                        )

    def test_to_eval_excludes_root(self, seq_fun_arff_dir):
        from hmc.datasets.dataset_manager import initialize_dataset_experiments

        mgr = initialize_dataset_experiments(
            "seq_FUN", device="cpu",
            dataset_path=seq_fun_arff_dir, is_global=False,
        )
        # "root" node should be excluded from evaluation
        root_idx = mgr.nodes_idx.get("root")
        assert root_idx is not None, "root node should exist"
        assert not mgr.to_eval[root_idx], "root node should be excluded from eval"

    def test_unknown_dataset_raises(self):
        from hmc.datasets.dataset_manager import initialize_dataset_experiments

        with pytest.raises(ValueError):
            initialize_dataset_experiments(
                "nonexistent_dataset", device="cpu",
                dataset_path="/tmp", is_global=False,
            )


class TestLocalClassifierWithGoFun:
    """Verify one epoch of local classifier training on synthetic GoFun data."""

    def test_one_epoch_seq_fun_local(self, seq_fun_arff_dir):
        import torch
        from hmc.arguments import Args, DatasetConfig, TrainingConfig
        from hmc.datasets.dataset_manager import initialize_dataset_experiments
        from hmc.datasets.registry import DatasetRegistry
        from hmc.models.local_classifier.model import LocalModel
        from hmc.utils.train.job import create_job_id_name
        from torch.utils.data import DataLoader

        device = torch.device("cpu")

        # 1. Load dataset
        mgr = initialize_dataset_experiments(
            "seq_FUN", device="cpu",
            dataset_path=seq_fun_arff_dir, is_global=False,
        )
        train, valid, test = mgr.get_datasets()
        registry = DatasetRegistry()

        levels_size = mgr.levels_size
        max_depth = mgr.max_depth
        to_eval = torch.as_tensor(mgr.to_eval, dtype=torch.bool)

        # 2. Convert to tensors
        for split in (train, valid, test):
            split.samples.x = torch.tensor(split.x).float()
            split.samples.y = torch.tensor(split.y).float()

        # 3. DataLoaders
        train_dataset = list(zip(train.x, train.y, train.y_local))
        for x, y, yl in zip(valid.x, valid.y, valid.y_local):
            train_dataset.append((x, y, yl))
        test_dataset = list(zip(test.x, test.y, test.y_local))

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        # 4. Model
        model = LocalModel(
            input_dim=mgr.input_dim,
            levels_size=levels_size,
            hidden_dim=32,
            num_layers=2,
            dropout=0.1,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.BCELoss()

        # 5. Train one epoch
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            x, _, y_local = batch
            preds = model(x)
            loss = torch.tensor(0.0)
            for lvl in sorted(levels_size):
                y_true = y_local[lvl].float()
                loss = loss + criterion(preds[str(lvl)], y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        assert total_loss > 0, "Training loss should be positive"
        assert not torch.isnan(torch.tensor(total_loss)), "Loss should not be NaN"

        # 6. Quick eval
        import numpy as np
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y_global, _y_local in test_loader:
                preds = model(x)
                for i in range(len(x)):
                    global_pred = np.zeros(mgr.output_dim, dtype=np.float32)
                    global_label = y_global[i].cpu().numpy()
                    for lvl in sorted(levels_size):
                        local_pred = preds[str(lvl)][i].cpu().numpy()
                        local_idx = mgr.local_nodes_idx[lvl]
                        for name, lidx in local_idx.items():
                            gidx = mgr.nodes_idx[name]
                            global_pred[gidx] = local_pred[lidx]
                    all_preds.append(global_pred)
                    all_labels.append(global_label)

        y_pred = np.stack(all_preds)
        y_true = np.stack(all_labels)
        y_bin = (y_pred > 0.5).astype(np.float32)
        tp = (y_bin[:, to_eval.numpy()] * y_true[:, to_eval.numpy()]).sum()
        assert tp >= 0, "True positives should be non-negative"

    def test_dataset_registry_has_gofun_defaults(self):
        from hmc.datasets.registry import DatasetRegistry

        r = DatasetRegistry()
        assert "hidden_dim" in r.gofun_defaults
        assert r.gofun_defaults["epochs"] == 100
