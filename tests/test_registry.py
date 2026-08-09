"""Tests for the dataset registry and plugin system."""

from __future__ import annotations

import os

import pytest


# ---------------------------------------------------------------------------
# Helpers — skip integration tests when data files are missing (CI/CD)
# ---------------------------------------------------------------------------

_DATA_CHECKS: dict[str, str] = {
    "arxiv": "./data/arxiv/arxiv-metadata-oai-snapshot.json",
    "wos": "./data/wos",
}


def _data_available(dataset_name: str) -> bool:
    """Return True if the required data files/directories exist."""
    path = _DATA_CHECKS.get(dataset_name)
    if path is None:
        return False
    return os.path.exists(path)


class TestDatasetRegistry:
    """Unit tests for DatasetRegistry register / get / list."""

    def test_register_and_get(self):
        """Custom dataset can be registered and retrieved."""
        from hmc.data import DatasetRegistry

        name = "test_registry_ds"

        # Clean up from previous test runs
        DatasetRegistry.unregister(name)

        # Register a mock factory
        DatasetRegistry.register(name, lambda **kw: {"name": name, **kw})

        assert name in DatasetRegistry.list_available()

        result = DatasetRegistry.get(name, extra=42)
        assert result == {"name": name, "extra": 42}

        # Clean up
        DatasetRegistry.unregister(name)

    def test_register_duplicate_raises(self):
        """Registering the same name twice raises ValueError."""
        from hmc.data import DatasetRegistry

        name = "test_dup_ds"
        DatasetRegistry.unregister(name)

        DatasetRegistry.register(name, lambda **kw: None)
        with pytest.raises(ValueError, match="already registered"):
            DatasetRegistry.register(name, lambda **kw: None)

        DatasetRegistry.unregister(name)

    def test_get_unknown_raises(self):
        """Getting an unregistered dataset raises ValueError."""
        from hmc.data import DatasetRegistry

        with pytest.raises(ValueError, match="not found in registry"):
            DatasetRegistry.get("definitely_not_a_dataset_xyz")

    def test_list_available_includes_builtins(self):
        """Built-in datasets appear in list_available after import."""
        # Trigger builtin registration
        from hmc.data import DatasetRegistry

        available = DatasetRegistry.list_available()
        # At least the 5 text datasets should be registered
        for expected in ["arxiv", "wos", "aapd", "rcv1", "eurlex"]:
            assert expected in available, f"{expected} missing from {available}"

    def test_unregister(self):
        """Unregister removes a dataset from the registry."""
        from hmc.data import DatasetRegistry

        name = "test_unreg_ds"
        DatasetRegistry.unregister(name)

        DatasetRegistry.register(name, lambda **kw: None)
        assert name in DatasetRegistry.list_available()

        DatasetRegistry.unregister(name)
        assert name not in DatasetRegistry.list_available()

    def test_defaults_stored_and_retrieved(self):
        """Registered defaults are returned by get_defaults."""
        from hmc.data import DatasetRegistry

        name = "test_defaults_ds"
        DatasetRegistry.unregister(name)

        defaults = {"hidden_dim": 128, "lr": 0.001, "epochs": 10}
        DatasetRegistry.register(name, lambda **kw: None, defaults=defaults)

        assert DatasetRegistry.get_defaults(name) == defaults

        DatasetRegistry.unregister(name)

    def test_get_defaults_unknown_returns_empty(self):
        """get_defaults for unknown dataset returns empty dict."""
        from hmc.data import DatasetRegistry

        assert DatasetRegistry.get_defaults("no_such_dataset") == {}


class TestProtocolCompliance:
    """Verify that built-in managers satisfy DatasetManagerProtocol."""

    @pytest.mark.parametrize(
        "dataset_name",
        [
            "arxiv",
            "wos",
        ],
    )
    def test_manager_satisfies_protocol(self, dataset_name):
        """Built-in manager instances pass protocol check."""
        if not _data_available(dataset_name):
            pytest.skip(
                f"Data files for '{dataset_name}' not found "
                f"({_DATA_CHECKS[dataset_name]}). Run download first."
            )

        from hmc.datasets.dataset_manager import initialize_dataset_experiments

        manager = initialize_dataset_experiments(
            dataset_name,
            device="cpu",
            dataset_path="./data",
            arxiv_load_features=False,
            arxiv_max_records=100,
        )

        # Structural checks — all required attrs must exist after _fit()
        required_attrs = [
            "input_dim",
            "output_dim",
            "levels_size",
            "max_depth",
            "a",
            "edge_index",
            "nodes_idx",
            "local_nodes_idx",
            "to_eval",
            "hierarchy_map",
            "get_datasets",
        ]
        for attr in required_attrs:
            assert hasattr(manager, attr), f"Missing attribute: {attr}"

        # get_datasets returns (train, valid, test)
        splits = manager.get_datasets()
        assert len(splits) == 3


class TestTrainAPI:
    """Smoke tests for the high-level hmc.train() API."""

    def test_train_accepts_args(self):
        """train() can be called with minimal arguments."""
        # Just verify the function signature — don't actually train.
        import inspect

        import hmc

        sig = inspect.signature(hmc.train)
        params = list(sig.parameters.keys())
        assert "dataset_name" in params
        assert "method" in params
        assert "device" in params
        assert "epochs" in params
        assert "batch_size" in params
