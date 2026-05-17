from hmc.datasets.arxiv.dataset_arxiv import ArXivHierarchyManager
from hmc.datasets.arxiv.dataset_arxiv import ArXivPyTorchDataset
from hmc.datasets.gofun.manager import HMCDatasetManager
from hmc.utils.datasets.paths import get_dataset_paths

def initialize_dataset_experiments(
    name: str,
    device: str = "cpu",
    dataset_path: str = "data/",
    dataset_type="torch",
    is_global: bool = False,
) -> HMCDatasetManager:
    """
    Initialize and return an HMCDatasetManager for the specified dataset.

    Parameters:
    - name (str): Name of the dataset to load.
    - output_path (str): Path to store output files.
    - device (str, optional): Device to be used ('cpu' or 'cuda'). \
        Default is 'cpu'.
    - is_local (bool, optional): Whether to use local_classifier hierarchy. \
        Default is False.
    - is_global (bool, optional): Whether to use global hierarchy. \
        Default is False.

    Returns:
    - HMCDatasetManager: Initialized dataset manager.
    """
    if name == "arxiv":
        
        # 1. Preparar a Taxonomia
        manager = ArXivHierarchyManager()
        manager.fit_from_jsonl("arxiv_downloaded_subset.jsonl")

        # 2. Inicializar Dataset
        dataset = ArXivPyTorchDataset(
            jsonl_path="arxiv_downloaded_subset.jsonl",
            hierarchy_manager=manager,
            tokenizer=tokenizer,
        )

        return dataset
    else:
        # Load dataset paths
        datasets = get_dataset_paths(dataset_path=dataset_path)

        # Validate if the dataset exists
        if name not in datasets:
            raise ValueError(
                f"Dataset '{name}' not found in experiments datasets. \
                Available datasets: {list(datasets.keys())}"
            )

        # Initialize dataset manager
        kwargs = {
            "dataset": datasets[name],
            "dataset_type": dataset_type,
            "device": device,
            "is_global": is_global,
        }

        return HMCDatasetManager(**kwargs)


