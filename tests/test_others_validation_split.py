"""Regression tests: releases without a validation file must not alias test.

The ``*_others`` datasets (enron, diatoms, imclef07a/d) ship only train and
test ARFFs.  Both GoFun managers used to point ``valid_file`` at the test
file, so every table script (which merges train+valid before fitting) trained
on the test split.  These tests pin the fixed behaviour: the validation slice
is carved out of the training file, never out of the test file.
"""

import importlib
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tests.fixtures.arff import _make_arff_content

NAME = "synthetic_others"  # the "_others" suffix selects the GoFun loader
HIERARCHY = "root.A.A1,root.A.A2,root.B.B1"
NUM_FEATURES = 6
NUM_TRAIN = 40
NUM_TEST = 15

LOADERS = [
    "hmc.datasets.manager.dataset_manager",  # used by experiments/
    "hmc.datasets.dataset_manager",  # public API
]


@pytest.fixture
def no_valid_dataset():
    """A dataset directory holding train/test ARFFs only (3-tuple style)."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "HMC_data_arff" / "others"
        base.mkdir(parents=True)
        # Distinct seeds per split, so a row identifies its file: the shared
        # fixture generator would otherwise repeat rows across train and test.
        for split, n_samples, seed in (
            ("train", NUM_TRAIN, 1),
            ("test", NUM_TEST, 2),
        ):
            content = _make_arff_content(HIERARCHY, NUM_FEATURES, n_samples, seed=seed)
            (base / f"{NAME}_{split}.arff").write_text(content, encoding="utf-8")

        def _fixture_paths(dataset_path="./data"):
            return {
                NAME: (
                    False,
                    str(base / f"{NAME}_train.arff"),
                    str(base / f"{NAME}_test.arff"),
                )
            }

        with _patched_paths(_fixture_paths):
            yield tmp


def _load(loader, dataset_path):
    module = importlib.import_module(loader)
    return module.initialize_dataset_experiments(
        NAME,
        device="cpu",
        dataset_path=dataset_path,
        dataset_type="arff",
        is_global=True,
    )


def _fingerprints(split):
    """Row identities of a split (features *and* labels)."""
    features = np.ascontiguousarray(split.x)
    labels = np.ascontiguousarray(split.y)
    return {
        features[i].tobytes() + labels[i].tobytes() for i in range(features.shape[0])
    }


@contextmanager
def _patched_paths(path_table):
    """Patch every module-level binding of the dataset path table.

    The project keeps two copies of the table (``hmc.utils.datasets.paths``
    and ``hmc.datasets.gofun``) and the managers bind them at import time
    (``hmc.datasets.manager.dataset_manager``) or lazily
    (``hmc.datasets.dataset_manager``), so each binding is replaced.
    """
    import hmc.datasets.gofun as gofun_mod
    import hmc.datasets.manager.dataset_manager as manager_mod
    import hmc.utils.datasets.paths as paths_mod

    modules = [gofun_mod, paths_mod, manager_mod]
    originals = [
        (module, module.get_dataset_paths)
        for module in modules
        if hasattr(module, "get_dataset_paths")
    ]
    for module, _ in originals:
        module.get_dataset_paths = path_table
    try:
        yield
    finally:
        for module, original in originals:
            module.get_dataset_paths = original


@pytest.mark.parametrize("loader", LOADERS)
def test_validation_never_overlaps_test(no_valid_dataset, loader):
    train, valid, test = _load(loader, no_valid_dataset).get_datasets()

    assert valid is not test, "validation still aliases the test split"
    assert _fingerprints(valid).isdisjoint(_fingerprints(test))
    # train + validation is exactly the shipped training file, so the test
    # split cannot reach the optimizer through the merged training set.
    assert train.x.shape[0] + valid.x.shape[0] == NUM_TRAIN
    assert test.x.shape[0] == NUM_TEST


@pytest.mark.parametrize("loader", LOADERS)
def test_split_is_deterministic(no_valid_dataset, loader):
    first_valid = _load(loader, no_valid_dataset).get_datasets()[1]
    second_valid = _load(loader, no_valid_dataset).get_datasets()[1]

    assert np.array_equal(first_valid.x, second_valid.x)
    assert np.array_equal(np.asarray(first_valid.y), np.asarray(second_valid.y))


@pytest.mark.parametrize("loader", LOADERS)
def test_held_out_slice_keeps_hierarchy(no_valid_dataset, loader):
    manager = _load(loader, no_valid_dataset)
    train, valid, _test = manager.get_datasets()

    assert valid.x.shape[1] == train.x.shape[1] == NUM_FEATURES
    assert valid.y.shape[1] == train.y.shape[1]
    assert set(manager.nodes_idx) == set(train.nodes_idx)
    assert valid.levels_size == train.levels_size


def test_explicit_validation_file_is_untouched():
    """A 4-tuple release (train/valid/test on disk) keeps its own splits."""
    from tests.fixtures.arff import SyntheticARFFFixture

    with SyntheticARFFFixture(
        name="synthetic_FUN",
        hierarchy=HIERARCHY,
        num_features=NUM_FEATURES,
        num_train=NUM_TRAIN,
        num_valid=8,
        num_test=NUM_TEST,
    ) as fixture:
        base = Path(fixture._tmpdir.name) / "HMC_data_arff"

        def _fixture_paths(dataset_path="./data"):
            return {
                "synthetic_FUN": (
                    False,
                    str(base / "datasets_FUN/synthetic_FUN/synthetic_FUN.train.arff"),
                    str(base / "datasets_FUN/synthetic_FUN/synthetic_FUN.valid.arff"),
                    str(base / "datasets_FUN/synthetic_FUN/synthetic_FUN.test.arff"),
                )
            }

        with _patched_paths(_fixture_paths):
            from hmc.datasets.manager.dataset_manager import (
                initialize_dataset_experiments,
            )

            train, valid, test = initialize_dataset_experiments(
                "synthetic_FUN",
                device="cpu",
                dataset_path=fixture._tmpdir.name,
                dataset_type="arff",
                is_global=True,
            ).get_datasets()

    assert (train.x.shape[0], valid.x.shape[0], test.x.shape[0]) == (
        NUM_TRAIN,
        8,
        NUM_TEST,
    )
