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

    def __init__(self, dataset, dataset_type="arff", device="cpu", is_global=False):
        # Extract dataset paths
        (
            self.test,
            self.train,
            self.valid,
            self.to_eval,
            self.max_depth,
            self.r_matrix,
        ) = (
            None,
            None,
            None,
            None,
            None,
            None,
        )

        (
            self.levels,
            self.levels_size,
            self.nodes_idx,
            self.local_nodes_idx,
            self.edge_index,
            self.all_matrix_r,
        ) = (
            {},
            {},
            {},
            {},
            {},
            {},
        )

        (
            self.labels,
            self.r_matrixoots,
            self.nodes,
            self.g_t,
            self.a,
        ) = (
            [],
            [],
            [],
            [],
            [],
        )

        self.to_skip = to_skip
        # Initialize attributes
        self.is_global = is_global
        self.device = device

        # Construct graph path
        self.g = nx.DiGraph()
        self.hierarchy_map = {}

        if dataset_type == "arff":
            self.is_go, self.train_file, self.valid_file, self.test_file = dataset
            self.load_arff_data()

    def load_structure_from_json(self, labels_json):
        """
        Load the hierarchy structure from a JSON file.
        Args:
            labels_json (str): Path to the JSON file containing the hierarchy structure.
        """
        # Load labels JSON
        self.labels = __load_json__(labels_json)
        for cat in self.labels["labels"]:
            terms = cat.split("/")
            if self.is_go:
                self.g.add_edge(terms[1], terms[0])
            else:
                if len(terms) == 1:
                    self.g.add_edge(terms[0], "root")
                else:
                    for i in range(2, len(terms) + 1):
                        self.g.add_edge(".".join(terms[:i]), ".".join(terms[: i - 1]))

        self.nodes = sorted(
            self.g.nodes(),
            key=lambda x: (
                (nx.shortest_path_length(self.g, x, "root"), x)
                if self.is_go
                else (len(x.split(".")), x)
            ),
        )
        self.nodes_idx = dict(zip(self.nodes, range(len(self.nodes))))
        self.g_t = self.g.reverse()

        self.a = nx.to_numpy_array(self.g, nodelist=self.nodes)

    def get_hierarchy_levels(self):
        """
        Retorna um dicionário com os nós agrupados por nível na hierarquia.
        """
        self.levels_size = defaultdict(int)
        self.levels = defaultdict(list)
        for label in self.nodes:
            level = label.count(".")
            self.levels[level].append(label)
            self.levels_size[level] += 1

        self.max_depth = len(self.levels_size)
        print(self.levels_size)
        self.local_nodes_idx = {}
        for idx, level_nodes in self.levels.items():
            self.local_nodes_idx[idx] = {node: i for i, node in enumerate(level_nodes)}

    def compute_r_matrix(self, edges_matrix):
        """
        Compute matrix of ancestors R, named matrix_r
        Given n classes, R is an (n x n) matrix where R_ij = 1 i\
        f class i is ancestor of class j
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
        for idx, edges_matrix in self.edge_index.items():
            matrix_r = self.compute_r_matrix(edges_matrix)
            logger.info(
                "Computed matrix R for level %d with shape %s", idx, matrix_r.shape
            )
            self.all_matrix_r[idx] = matrix_r

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
            if self.is_global:
                y_ = np.zeros(len(self.nodes))
            else:
                sorted_keys = sorted(self.levels_size.keys())
                y_ = [np.zeros(self.levels_size.get(key)) for key in sorted_keys]
            for node in labels.split("@"):
                if self.is_global:
                    y_[
                        [self.nodes_idx.get(a) for a in nx.ancestors(self.g_t, node)]
                    ] = 1
                    y_[self.nodes_idx[node]] = 1

                if not self.is_global:
                    depth = nx.shortest_path_length(self.g_t, "root").get(node)
                    y_[depth][self.local_nodes_idx[depth].get(node)] = 1
                    for ancestor in nx.ancestors(self.g_t, node):
                        if ancestor != "root":
                            depth = nx.shortest_path_length(self.g_t, "root").get(
                                ancestor
                            )
                            y_[depth][self.local_nodes_idx[depth].get(ancestor)] = 1

            if self.is_global:
                y.append(y_)
            else:
                y.append([np.stack(y) for y in y_])
        if self.is_global:
            y = np.stack(y)
        return y

    def load_arff_data(self):
        """
        Load features and labels from ARFF, and optionally a hierarchy graph from JSON.
        Args:
            arff_file (str): Path to the ARFF file.
            is_go (bool): Whether the ARFF file contains Gene Ontology data.
        """
        logger.info("Loading dataset from %s", self.train_file)
        self.train = HMCDatasetArff(self.train_file, is_go=self.is_go)
        logger.info("Loading dataset from %s", self.valid_file)
        self.valid = HMCDatasetArff(self.valid_file, is_go=self.is_go)
        logger.info("Loading dataset from %s", self.test_file)
        self.test = HMCDatasetArff(self.test_file, is_go=self.is_go)
        self.a = self.train.a
        self.edge_index = self.train.edge_index
        # self.r_matrix = self.compute_r_matrix(self.a)
        # self.compute_r_matrix_local()
        self.build_hierarchy_map()
        self.to_eval = self.train.to_eval
        self.nodes = self.train.g.nodes()
        self.nodes_idx = self.train.nodes_idx
        self.local_nodes_idx = self.train.local_nodes_idx
        self.max_depth = self.train.max_depth
        self.levels = self.train.levels
        self.levels_size = self.train.levels_size

    def build_hierarchy_map(self):
        """
        Builds the hierarchy map.
        """
        for parent in self.g.nodes():
            children = list(self.g.successors(parent))
            if children:
                self.hierarchy_map[parent] = children

    def get_datasets(self):
        """
        Return the datasets.
        Returns:
            tuple: (train, valid, test)
        """
        return self.train, self.valid, self.test


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
    return HMCDatasetManager(
        datasets[name],
        dataset_type=dataset_type,
        device=device,
        is_global=is_global,
    )
