"""Dataset manager — entry point for loading ArXiv and WOS HMC datasets."""

import os


def initialize_dataset_experiments(
    name: str,
    device: str = "cpu",
    dataset_path: str = "data/",
    dataset_type: str = "arxiv",
    is_global: bool = False,
    arxiv_model_name: str = "allenai/specter2_base",
    arxiv_max_records: int = 50_000,
    arxiv_cache_dir: str = None,
    arxiv_load_features: bool = True,
    model_cache_dir: str = "./models",
):
    """Initialize and return a dataset manager for the specified dataset.

    Supported datasets: ``"arxiv"``, ``"wos"``.

    Returns an ``ArXivManager`` or ``WOSManager`` instance.
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

    raise ValueError(
        f"Dataset '{name}' not supported. Available: arxiv, wos"
    )
