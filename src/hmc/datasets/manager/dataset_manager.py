"""
This module provides functionality to manage hierarchical multi-label datasets.
"""

import logging
from collections import defaultdict

import networkx as nx
import numpy as np
import torch

from hmc.datasets.gofun import to_skip
from hmc.datasets.gofun.dataset_arff import HMCDatasetArff
from hmc.utils.path.files import __load_json__
from hmc.datasets.gofun import get_dataset_paths

# Set a logger config
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


class HMCDatasetManager:
    """
    Manages hierarchical multi-label datasets, \
    including loading features (X), labels (Y),
    and optionally applying input scaling and hierarchical structure.

    Parameters:
    - dataset (tuple): Tuple containing paths to \
        (train_csv, valid_csv, test_csv, labels_json, _).
    - output_path (str, optional): Path to store processed outputs. Default is 'data'.
    - device (str, optional): Computation device ('cpu' or 'cuda'). \
        Default is 'cpu'.
    - is_local (bool, optional): Whether to use local_classifier \
        hierarchy. Default is False.
    - is_global (bool, optional): Whether to use global hierarchy. \
        Default is False.
    - input_scaler (bool, optional): Whether to apply input scaling
    (imputation + standardization). Default is True.

    """

    def __init__(self, **kwargs):
        self.dataset_values = {
            "test": None,
            "train": None,
            "valid": None,
            "to_eval": None,
            "max_depth": None,
            "r_matrix": None,
            "g_t": None,
            "is_go": None,
            "train_file": None,
            "valid_file": None,
            "test_file": None,
            "levels": {},
            "levels_size": {},
            "nodes_idx": {},
            "local_nodes_idx": {},
            "edge_index": {},
            "all_matrix_r": {},
            "hierarchy_map": {},
            "labels": [],
            "nodes": [],
            "a": [],
            "g": nx.DiGraph(),
            # Initialize attributes
            "to_skip": to_skip,
            "device": kwargs["device"],
            "dataset": kwargs["dataset"],
            "dataset_type": kwargs["dataset_type"],
            "is_global": kwargs["is_global"],
        }

        if kwargs["dataset_type"] == "arff":
            (
                self.dataset_values["is_go"],
                self.dataset_values["train_file"],
                self.dataset_values["valid_file"],
                self.dataset_values["test_file"],
            ) = kwargs["dataset"]
            self.load_arff_data()

    def load_structure_from_json(self, labels_json):
        """
        Load the hierarchy structure from a JSON file.
        Args:
            labels_json (str): Path to the JSON file containing the hierarchy structure.
        """
        # Load labels JSON
        self.dataset_values["labels"] = __load_json__(labels_json)
        for cat in self.dataset_values["labels"]:
            terms = cat.split("/")
            if self.dataset_values["is_global"]:
                self.dataset_values["g"].add_edge(terms[1], terms[0])
            else:
                if len(terms) == 1:
                    self.dataset_values["g"].add_edge(terms[0], "root")
                else:
                    for i in range(2, len(terms) + 1):
                        self.dataset_values["g"].add_edge(
                            ".".join(terms[:i]), ".".join(terms[: i - 1])
                        )

        self.dataset_values["nodes"] = sorted(
            self.dataset_values["g"].nodes(),
            key=lambda x: (
                (nx.shortest_path_length(self.dataset_values["g"], x, "root"), x)
                if self.dataset_values["is_global"]
                else (len(x.split(".")), x)
            ),
        )
        self.dataset_values["nodes_idx"] = dict(
            zip(self.dataset_values["nodes"], range(len(self.dataset_values["nodes"])))
        )
        self.dataset_values["g_t"] = self.dataset_values["g"].reverse()

        self.dataset_values["a"] = nx.to_numpy_array(
            self.dataset_values["g"], nodelist=self.dataset_values["nodes"]
        )

    def get_hierarchy_levels(self):
        """
        Returns a dictionary with nodes grouped by level in the hierarchy.
        """
        self.dataset_values["levels_size"] = defaultdict(int)
        self.dataset_values["levels"] = defaultdict(list)
        for label in self.dataset_values["nodes"]:
            level = label.count(".")
            self.dataset_values["levels"][level].append(label)
            self.dataset_values["levels_size"][level] += 1

        self.dataset_values["max_depth"] = len(self.dataset_values["levels_size"])
        for idx, level_nodes in self.dataset_values["levels"].items():
            self.dataset_values["local_nodes_idx"][idx] = {
                node: i for i, node in enumerate(level_nodes)
            }

    def compute_r_matrix(self, edges_matrix):
        """
        Compute matrix of ancestors R, named matrix_r
        Given n classes, R is an (n x n) matrix where R_ij = 1 \
        if class i is ancestor of class j
        Args:
            edges_matrix (np.ndarray): Matrix of edges.
        Returns:
            np.ndarray: Matrix of ancestors.
        """
        matrix_r = np.zeros(edges_matrix.shape)
        np.fill_diagonal(matrix_r, 1)
        g = nx.DiGraph(edges_matrix)
        for i in range(len(edges_matrix)):
            descendants = list(nx.descendants(g, i))
            if descendants:
                matrix_r[i, descendants] = 1
        matrix_r = torch.tensor(matrix_r)
        # Transpose to get the ancestors for each node
        matrix_r = matrix_r.transpose(1, 0)
        matrix_r = matrix_r.unsqueeze(0)
        return matrix_r

    def compute_r_matrix_local(self):
        """
        Compute the list with local matrix of ancestors R, named matrix_r
        Given n classes, R is an (n x n) matrix where R_ij = 1 \
        if class i is ancestor of class j
        """
        for idx, edges_matrix in self.dataset_values["edge_index"].items():
            self.dataset_values["all_matrix_r"][idx] = self.compute_r_matrix(
                edges_matrix
            )
            logging.info(
                "Computed matrix R for level %d with shape %s",
                idx,
                self.dataset_values["all_matrix_r"][idx].shape,
            )

    def transform_labels(self, dataset_labels):
        """
        Transform labels to binary vectors.
        Args:
            dataset_labels (list): List of labels.
        Returns:
            list: List of binary vectors.
        """
        y_ = []
        y = []
        for labels in dataset_labels:
            if self.dataset_values["is_global"]:
                y_ = np.zeros(len(self.dataset_values["nodes"]))
            else:
                sorted_keys = sorted(self.dataset_values["levels_size"].keys())
                y_ = [
                    np.zeros(self.dataset_values["levels_size"].get(key))
                    for key in sorted_keys
                ]
            for node in labels.split("@"):
                if self.dataset_values["is_global"]:
                    y_[
                        [
                            self.dataset_values["nodes_idx"].get(a)
                            for a in self.dataset_values["g_t"].ancestors(node)
                        ]
                    ] = 1
                    y_[self.dataset_values["nodes_idx"][node]] = 1

                if not self.dataset_values["is_global"]:
                    depth = nx.shortest_path_length(
                        self.dataset_values["g_t"], "root"
                    ).get(node)
                    y_[depth][
                        self.dataset_values["local_nodes_idx"][depth].get(node)
                    ] = 1
                    for ancestor in self.dataset_values["g_t"].ancestors(node):
                        if ancestor != "root":
                            depth = nx.shortest_path_length(
                                self.dataset_values["g_t"], "root"
                            ).get(ancestor)
                            y_[depth][
                                self.dataset_values["local_nodes_idx"][depth].get(
                                    ancestor
                                )
                            ] = 1

            if self.dataset_values["is_global"]:
                y.append(y_)
            else:
                y.append([np.stack(y) for y in y_])
        if self.dataset_values["is_global"]:
            y = np.stack(y)
        return y

    def load_arff_data(self):
        """
        Load features and labels from ARFF, and optionally a hierarchy graph from JSON.
        Args:
            arff_file (str): Path to the ARFF file.
            is_go (bool): Whether the ARFF file contains Gene Ontology data.
        """
        logging.info("Loading dataset from %s", self.dataset_values["train_file"])
        self.dataset_values["train"] = HMCDatasetArff(
            self.dataset_values["train_file"], is_go=self.dataset_values["is_go"]
        )
        logging.info("Loading dataset from %s", self.dataset_values["valid_file"])
        self.dataset_values["valid"] = HMCDatasetArff(
            self.dataset_values["valid_file"], is_go=self.dataset_values["is_go"]
        )
        logging.info("Loading dataset from %s", self.dataset_values["test_file"])
        self.dataset_values["test"] = HMCDatasetArff(
            self.dataset_values["test_file"], is_go=self.dataset_values["is_go"]
        )
        self.a = self.dataset_values["train"].a
        self.edge_index = self.dataset_values["train"].edge_index
        # self.r_matrix = self.compute_r_matrix(self.a)
        # self.compute_r_matrix_local()
        self.build_hierarchy_map()
        self.to_eval = self.dataset_values["train"].to_eval
        self.dataset_values["nodes"] = self.dataset_values["train"].g.nodes()
        self.dataset_values["g_t"] = self.dataset_values["train"].g.copy()
        self.dataset_values["nodes_idx"] = self.dataset_values["train"].nodes_idx
        self.dataset_values["local_nodes_idx"] = self.dataset_values[
            "train"
        ].local_nodes_idx
        self.dataset_values["max_depth"] = self.dataset_values["train"].max_depth
        self.dataset_values["levels"] = self.dataset_values["train"].levels
        self.dataset_values["levels_size"] = self.dataset_values["train"].levels_size

    def build_hierarchy_map(self):
        """
        Builds the hierarchy map.
        """
        for parent in self.dataset_values["train"].g.nodes():
            children = list(self.dataset_values["train"].g.successors(parent))
            if children:
                self.dataset_values["hierarchy_map"][parent] = children

    def get_datasets(self):
        """
        Return the datasets.
        Returns:
            tuple: (train, valid, test)
        """
        return (
            self.dataset_values["train"],
            self.dataset_values["valid"],
            self.dataset_values["test"],
        )


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
