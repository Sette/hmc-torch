import logging
from collections import defaultdict

import networkx as nx
import numpy as np
import torch

from hmc.datasets.datasets.gofun import get_dataset_paths, to_skip
from hmc.datasets.datasets.gofun.dataset_arff import HMCDatasetArff
from hmc.datasets.datasets.gofun.dataset_csv import HMCDatasetCsv
from hmc.datasets.datasets.gofun.dataset_torch import HMCDatasetTorch
from hmc.utils.dir import __load_json__

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
        self.test, self.train, self.valid, self.to_eval, self.max_depth, self.R = (
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
            self.edges_matrix_dict,
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
            self.roots,
            self.nodes,
            self.g_t,
            self.A,
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
        # train_csv_name = Path(self.train_file).name
        # self.graph_path = os.path.join(output_path, \
        # train_csv_name.replace('-.csv', '.graphml'))

        if dataset_type == "arff":
            self.is_go, self.train_file, self.valid_file, self.test_file = dataset
        else:
            self.train_file, self.valid_file, self.test_file, self.labels_file = dataset
            # Infer dataset type
            self.is_go = any(keyword in self.train_file for keyword in ["GO", "go"])
            self.is_fma = any(keyword in self.train_file for keyword in ["fma", "FMA"])
            # Load hierarchical structure
            self.load_structure_from_json(self.labels_file)

        logger.info("Loading dataset from %s", self.train_file)

        if dataset_type == "csv":
            self.load_csv_data()
            self.to_eval = [t not in self.to_skip for t in self.nodes]
        elif dataset_type == "torch":
            self.load_torch_data()
            self.to_eval = [t not in self.to_skip for t in self.nodes]
        elif dataset_type == "arff":
            self.load_arff_data()

    def load_structure_from_json(self, labels_json):
        """
        Loads a hierarchical structure from a JSON file containing label information and constructs a directed graph representation.

        Args:
            labels_json (str): Path to the JSON file containing label hierarchy information.

        Side Effects:
            - Sets `self.labels` to the loaded JSON data.
            - Builds a directed graph (`self.g`) representing the label hierarchy.
            - Populates `self.nodes` with sorted node names based on hierarchy depth or shortest path length.
            - Creates a mapping `self.nodes_idx` from node names to their indices.
            - Generates the reversed graph `self.g_t`.
            - Computes the adjacency matrix `self.A` for the graph.

        Notes:
            - If `self.is_go` is True, edges are added according to Gene Ontology conventions.
            - Otherwise, edges are added based on dot-separated label hierarchy.
            - Assumes existence of a helper function `__load_json__` for loading JSON files.
        """
        # Load labels JSON
        self.labels = __load_json__(labels_json)
        for cat in self.labels["labels"]:
            terms = cat.split(".")
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

        self.A = nx.to_numpy_array(self.g, nodelist=self.nodes)

    def get_hierarchy_levels(self):
        """
        Returns a dictionary with nodes grouped by level in the hierarchy.
        """
        self.levels_size = defaultdict(int)
        self.levels = defaultdict(list)

        if self.is_go:
            for label in self.nodes:
                level = nx.shortest_path_length(self.g_t, "root").get(label)
                self.levels[level].append(label)
                self.levels_size[level] += 1
            self.max_depth = len(self.levels_size)
            print(self.levels_size)
            self.local_nodes_idx = {}
            for idx, level_nodes in self.levels.items():
                self.local_nodes_idx[idx] = {
                    node: i for i, node in enumerate(level_nodes)
                }
        else:
            for label in self.nodes:
                level = label.count(".")
                self.levels[level].append(label)
                self.levels_size[level] += 1

        self.max_depth = len(self.levels_size)
        print(self.levels_size)
        self.local_nodes_idx = {}
        for idx, level_nodes in self.levels.items():
            self.local_nodes_idx[idx] = {node: i for i, node in enumerate(level_nodes)}

    def _matrix_r(self, edges_matrix):
        """
        Computes the ancestor matrix R for a directed graph represented by the given edges matrix.

        Given an adjacency matrix `edges_matrix` for a graph with n classes, this method constructs
        an (n x n) matrix R where R_ij = 1 if class i is an ancestor of class j (including itself),
        and 0 otherwise. The resulting matrix is returned as a PyTorch tensor with shape (1, n, n),
        where the first dimension is a batch dimension.

        Args:
            edges_matrix (np.ndarray): An (n x n) adjacency matrix representing the directed edges
                between classes in the graph.

        Returns:
            torch.Tensor: A tensor of shape (1, n, n) where each entry indicates ancestor relationships
                between classes.
        """
        # Compute matrix of ancestors R, named matrix_r
        # Given n classes, R is an (n x n) matrix where R_ij = 1 i\
        # f class i is ancestor of class j
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

    def _matrix_r_local(self):
        """
        Computes the matrix R for each level in the edges_matrix_dict and stores the results in all_matrix_r.

        Iterates over all items in edges_matrix_dict, computes the corresponding matrix R using the _matrix_r method,
        logs the shape of each computed matrix, and saves it in the all_matrix_r dictionary with the associated index.

        Returns:
            None
        """
        for idx, edges_matrix in self.edges_matrix_dict.items():
            matrix_r = self._matrix_r(edges_matrix)
            logger.info(
                "Computed matrix R for level %d with shape %s", idx, matrix_r.shape
            )
            self.all_matrix_r[idx] = matrix_r

    def transform_labels(self, dataset_labels):
        """
        Transforms hierarchical dataset labels into binary indicator arrays for global or local node representations.

        Depending on the `is_global` attribute:
        - If `is_global` is True, each label is converted into a binary array indicating the presence of nodes and their ancestors in the global node list.
        - If `is_global` is False, each label is converted into a list of binary arrays for each hierarchy level, indicating the presence of nodes and their ancestors at each level.

        Args:
            dataset_labels (Iterable[str]): An iterable of label strings, where each label may contain multiple node identifiers separated by "@".

        Returns:
            np.ndarray or List[List[np.ndarray]]:
                - If `is_global` is True, returns a stacked numpy array of binary indicator arrays for each label.
                - If `is_global` is False, returns a list of lists of stacked numpy arrays, one for each hierarchy level per label.

        Notes:
            - Assumes existence of attributes: `is_global`, `nodes`, `nodes_idx`, `g_t`, `levels_size`, `local_nodes_idx`.
            - Uses NetworkX for graph operations.
            - The function expects the graph to have a "root" node for local transformations.
        """
        y_local_ = []
        y_ = []
        Y = []
        Y_local = []
        for labels in dataset_labels:
            y_ = np.zeros(len(self.nodes))

            sorted_keys = sorted(self.levels_size.keys())
            max_depth = len(self.levels_size)
            y_local_ = [np.zeros(self.levels_size.get(key)) for key in sorted_keys]
            for node in labels.split("@"):

                y_[[self.nodes_idx.get(a) for a in nx.ancestors(self.g_t, node)]] = 1
                y_[self.nodes_idx[node]] = 1

                if self.is_go:
                    depth = nx.shortest_path_length(self.g_t, "root").get(node)
                    y_local_[depth][self.local_nodes_idx[depth].get(node)] = 1
                    for ancestor in nx.ancestors(self.g_t, node):
                        if ancestor != "root":
                            depth = nx.shortest_path_length(self.g_t, "root").get(ancestor)
                            y_local_[depth][self.local_nodes_idx[depth].get(ancestor)] = 1

                else:
                    depth = len(node.split('.')) + 1
                    for index in range(depth, 0, -1):
                        local_terms = node.split(".")[:index]
                        local_label = "/".join(local_terms)
                        local_depth = local_label.count("/")

                        y_local_[local_depth][
                            self.local_nodes_idx.get(local_depth).get(local_label)
                        ] = 1
                    
                
            Y.append(y_)
            Y_local.append([np.stack(y) for y in y_local_])

        Y = np.stack(Y)
        return Y, Y_local

    def load_csv_data(self):
        """
        Load features and labels from CSV, and optionally a hierarchy graph from
        JSON.

        This method performs the following steps:
        1. Loads the train, validation, and test datasets from their respective file paths.
        2. Retrieves the labels from each dataset.
        3. Transforms the labels using the `transform_labels` method.
        4. Sets the transformed labels back to each dataset.

        """
        self.train = HMCDatasetCsv(self.train_file, is_go=self.is_go)
        self.valid = HMCDatasetCsv(self.valid_file, is_go=self.is_go)
        self.test = HMCDatasetCsv(self.test_file, is_go=self.is_go)

        self.get_hierarchy_levels()

        dataset_labels = self.valid.df.labels.values
        logger.info("Transforming valid labels")
        self.valid.Y, self.valid.Y_local = self.transform_labels(dataset_labels)

        dataset_labels = self.test.df.labels.values
        logger.info("Transforming test labels")
        self.test.Y, self.test.Y_local = self.transform_labels(dataset_labels)

        dataset_labels = self.train.df.labels.values
        logger.info("Transforming train labels")
        self.train.Y, self.train.Y_local = self.transform_labels(dataset_labels)

    def load_torch_data(self):
        """
        Loads training, validation, and test datasets using the HMCDatasetTorch class,
        and applies label transformation to each dataset.

        This method performs the following steps:
        1. Loads the train, validation, and test datasets from their respective file paths.
        2. Retrieves the labels from each dataset.
        3. Transforms the labels using the `transform_labels` method.
        4. Sets the transformed labels back to each dataset.

        """
        self.train = HMCDatasetTorch(self.train_file)
        self.valid = HMCDatasetTorch(self.valid_file)
        self.test = HMCDatasetTorch(self.test_file)

        dataset_labels = self.train.Y
        logger.info("Transforming train labels")
        self.train.set_y(self.transform_labels(dataset_labels))

        dataset_labels = self.valid.Y
        logger.info("Transforming valid labels")
        self.valid.set_y(self.transform_labels(dataset_labels))

        dataset_labels = self.test.Y
        logger.info("Transforming test labels")
        self.test.set_y(self.transform_labels(dataset_labels))

    def load_arff_data(self):
        """
        Loads training, validation, and test datasets using the HMCDatasetArff class,
        and sets up hierarchical structures and matrices for ARFF datasets.

        This method performs the following steps:
        1. Loads the train, validation, and test datasets from their respective ARFF file paths.
        2. Sets the adjacency matrix, edge matrices, and ancestor matrices from the train dataset.
        3. Computes local ancestor matrices for each hierarchy level.
        4. Copies hierarchy-related attributes (nodes, indices, levels, etc.) from the train dataset.

        Side Effects:
            - Sets attributes: train, valid, test, A, edges_matrix_dict, R, all_matrix_r,
              to_eval, nodes, nodes_idx, local_nodes_idx, max_depth, levels, levels_size.
        """
        self.train = HMCDatasetArff(self.train_file, is_go=self.is_go)
        self.valid = HMCDatasetArff(self.valid_file, is_go=self.is_go)
        self.test = HMCDatasetArff(self.test_file, is_go=self.is_go)
        self.A = self.train.A
        self.edges_matrix_dict = self.train.edges_matrix_dict
        self.R = self._matrix_r(self.A)
        self._matrix_r_local()
        self.to_eval = self.train.to_eval
        self.nodes = self.train.g.nodes()
        self.nodes_idx = self.train.nodes_idx
        self.local_nodes_idx = self.train.local_nodes_idx
        self.max_depth = self.train.max_depth
        self.levels = self.train.levels
        self.levels_size = self.train.levels_size

    def get_datasets(self):
        """
        Returns the training, validation, and test datasets.

        Returns:
            tuple: A tuple containing the train, valid, and test datasets.
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
    datasets = get_dataset_paths(dataset_path=dataset_path, dataset_type=dataset_type)

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
