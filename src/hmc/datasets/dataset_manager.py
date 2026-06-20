"""Dataset manager — entry point for loading all supported HMC datasets."""

import os

from hmc.datasets.gofun.manager import HMCDatasetManager
from hmc.utils.datasets.paths import get_dataset_paths


def initialize_dataset_experiments(
    name: str,
    device: str = "cpu",
    dataset_path: str = "data/",
    dataset_type="torch",
    is_global: bool = False,
    arxiv_model_name: str = "allenai/specter2_base",
    arxiv_max_records: int = 50_000,
    arxiv_cache_dir: str = None,
    arxiv_load_features: bool = True,
) -> HMCDatasetManager:
    """
    Initialize and return a dataset manager for the specified dataset.

    Parameters:
    - name (str): Name of the dataset to load.
    - device (str, optional): Device to be used ('cpu' or 'cuda'). Default 'cpu'.
    - dataset_path (str): Root directory for dataset files.
    - dataset_type (str): Type hint for ARFF datasets (ignored for arxiv).
    - is_global (bool): Whether to load in global-classifier mode.

    Returns:
    - HMCDatasetManager or ArXivManager instance.
    """
    if name == "arxiv":
        from hmc.datasets.arxiv.manager import (  # pylint: disable=import-outside-toplevel
            ArXivManager,
        )

        jsonl_path = os.path.join(
            dataset_path, "arxiv", "arxiv-metadata-oai-snapshot.json"
        )
        return ArXivManager(
            jsonl_path=jsonl_path,
            model_name=arxiv_model_name,
            max_records=arxiv_max_records if arxiv_max_records > 0 else None,
            cache_dir=arxiv_cache_dir,
            load_features=arxiv_load_features,
        )

    if name == "wos":
        from hmc.datasets.wos.manager import (  # pylint: disable=import-outside-toplevel
            WOSManager,
        )

        data_dir = os.path.join(dataset_path, "wos")
        return WOSManager(
            data_dir=data_dir,
            model_name=arxiv_model_name,
            cache_dir=arxiv_cache_dir,
            load_features=arxiv_load_features,
        )

    # Load dataset paths
    datasets = get_dataset_paths(dataset_path=dataset_path)

    if name not in datasets:
        raise ValueError(
            f"Dataset '{name}' not found in experiments datasets. "
            f"Available datasets: {list(datasets.keys())}"
        )

    kwargs = {
        "dataset": datasets[name],
        "dataset_type": dataset_type,
        "device": device,
        "is_global": is_global,
    }

    return HMCDatasetManager(**kwargs)
