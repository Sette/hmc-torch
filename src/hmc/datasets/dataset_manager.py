"""Dataset manager — entry point for loading all supported HMC datasets.

Supported families:
  - ``arxiv``, ``wos`` — text transformer-based (SPECTER2)
  - ``seq_FUN``, ``cellcycle_FUN``, ..., ``spo_GO`` — ARFF tabular (GoFun)
"""

import os


def _load_gofun_dataset(name: str, device: str, dataset_path: str,
                        is_global: bool):
    """Load a GoFun ARFF dataset using HMCDatasetManager."""
    from hmc.datasets.gofun.manager import HMCDatasetManager
    from hmc.utils.datasets.paths import get_dataset_paths

    datasets = get_dataset_paths(dataset_path=dataset_path)

    if name not in datasets:
        raise ValueError(
            f"Dataset '{name}' not found in GoFun paths. "
            f"Available: {[k for k in datasets if '_FUN' in k or '_GO' in k or '_others' in k]}"
        )

    kwargs = {
        "dataset": datasets[name],
        "dataset_type": "arff",
        "device": device,
        "is_global": is_global,
    }

    return HMCDatasetManager(**kwargs)


def initialize_dataset_experiments(
    name: str,
    device: str = "cpu",
    dataset_path: str = "data/",
    is_global: bool = False,
    arxiv_model_name: str = "allenai/specter2_base",
    arxiv_max_records: int = 50_000,
    arxiv_cache_dir: str = None,
    arxiv_load_features: bool = True,
    model_cache_dir: str = "./models",
):
    """Initialize and return a dataset manager for the specified dataset.

    Supported datasets: ``"arxiv"``, ``"wos"``, and all GoFun ARFF datasets
    (e.g. ``"seq_FUN"``, ``"cellcycle_FUN"``, ``"eisen_GO"``, ``"enron_others"``).

    Returns an ``ArXivManager``, ``WOSManager``, or ``HMCDatasetManager`` instance.
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
            model_cache_dir=model_cache_dir,
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
            model_cache_dir=model_cache_dir,
        )

    # GoFun ARFF datasets: seq_FUN, cellcycle_FUN, eisen_GO, enron_others, etc.
    return _load_gofun_dataset(
        name, device=device, dataset_path=dataset_path, is_global=is_global,
    )
