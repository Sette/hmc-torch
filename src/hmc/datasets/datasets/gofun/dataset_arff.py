import logging
from collections import defaultdict
from itertools import chain
import keras
import networkx as nx
import numpy as np
import torch

from hmc.datasets.datasets.gofun import to_skip

# Set a logger config
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


def get_depth_by_root(g_t, t, roots):
    for root in roots:
        depth = nx.shortest_path_length(g_t, t, root)
        if depth is not None:
            return depth
    return None


class HMCDatasetArff:
    def __init__(self, arff_file, is_go, read_data=True, use_sample=False):
        self.arff_file = arff_file
        (
            self.X,
            self.Y,
            self.Y_local,
            self.A,
            self.edges_matrix_dict,
            self.terms,
            self.g,
            self.levels,
            self.levels_size,
            self.nodes_idx,
            self.local_nodes_idx,
            self.max_depth,
        ) = parse_arff(
            arff_file=arff_file, is_go=is_go, read_data=read_data, use_sample=use_sample
        )
        self.to_eval = [t not in to_skip for t in self.terms]
        if read_data:
            r_, c_ = np.where(np.isnan(self.X))
            m = np.nanmean(self.X, axis=0)
            for i, j in zip(r_, c_):
                self.X[i, j] = m[j]


def parse_arff(arff_file, is_go=False, read_data=True, use_sample=False):
    reading_data = False
    print("Use sample %s" % use_sample)
    with open(arff_file, "r", encoding="utf-8") as f:
        X = []
        Y = []
        Y_local = []
        levels_size = defaultdict(int)
        levels = defaultdict(list)
        g = nx.DiGraph()
        feature_types = []
        d = []
        cats_lens = []
        all_terms = []
        max_depth = 0
        local_nodes_idx = {}
        nodes_idx = {}
        nodes = []
        count = 0
        for l in f:
            if l.startswith("@ATTRIBUTE"):
                if l.startswith("@ATTRIBUTE class"):
                    h = l.split("hierarchical")[1].strip()
                    for branch in h.split(","):

                        branch = branch.replace("/", ".")

                        terms = branch.split(".")
                        all_terms.append(branch)
                        if is_go:
                            g.add_edge(terms[1], terms[0])
                        else:
                            level = branch.count(
                                "."
                            )  # Count the number of '.' to determine the level
                            levels[level].append(branch)
                            if len(terms) == 1:
                                g.add_edge(terms[0], "root")
                            else:
                                for i in range(2, len(terms) + 1):
                                    g.add_edge(
                                        ".".join(terms[:i]), ".".join(terms[: i - 1])
                                    )
                    nodes = sorted(
                        g.nodes(),
                        key=lambda x: (
                            (nx.shortest_path_length(g, x, "root"), x)
                            if is_go
                            else (len(x.split(".")), x)
                        ),
                    )
                    nodes_idx = dict(zip(nodes, range(len(nodes))))
                    g_t = g.reverse()

                    if is_go:
                        for label in nodes:
                            if label != "root":
                                level = (
                                    nx.shortest_path_length(g_t, "root").get(label) - 1
                                )
                                # print(f"Label {label} level {level}")
                                levels[level].append(label)

                        levels_size = {
                            key: len(set(value)) for key, value in levels.items()
                        }
                        max_depth = len(levels_size)
                        print(f"Levels size go dataset: {levels_size}")
                        print(f"Max depth: {max_depth}")
                    else:
                        levels_size = {
                            key: len(set(value)) for key, value in levels.items()
                        }
                        max_depth = len(levels_size)
                        print(f"Levels size: {levels_size}")

                    local_nodes_idx = {
                        idx: dict(zip(level_nodes, range(len(level_nodes))))
                        for idx, level_nodes in levels.items()
                    }
                else:
                    _, _, f_type = l.split()

                    if f_type == "numeric" or f_type == "NUMERIC":
                        d.append([])
                        cats_lens.append(1)
                        feature_types.append(
                            lambda x, i: [float(x)] if x != "?" else [np.nan]
                        )

                    else:
                        cats = f_type[1:-1].split(",")
                        cats_lens.append(len(cats))
                        d.append(
                            {
                                key: keras.utils.to_categorical(i, len(cats)).tolist()
                                for i, key in enumerate(cats)
                            }
                        )
                        feature_types.append(
                            lambda x, i: d[i].get(x, [0.0] * cats_lens[i])
                        )
            elif l.startswith("@DATA"):
                if read_data:
                    reading_data = True
            elif reading_data:
                y_ = np.zeros(len(nodes))
                sorted_keys = sorted(levels_size.keys())
                y_local_ = [np.zeros(levels_size.get(key)) for key in sorted_keys]
                d_line = l.split("%")[0].strip().split(",")
                lab = d_line[len(feature_types)].strip()

                X.append(
                    list(
                        chain(
                            *[
                                feature_types[i](x, i)
                                for i, x in enumerate(d_line[: len(feature_types)])
                            ]
                        )
                    )
                )
                count += 1
                for t in lab.split("@"):
                    y_node = t.replace("/", ".")
                    y_[[nodes_idx.get(a) for a in nx.ancestors(g_t, y_node)]] = 1
                    y_[nodes_idx[y_node]] = 1

                    if is_go:
                        depth = nx.shortest_path_length(g_t, "root").get(y_node) - 1
                        y_local_[depth][local_nodes_idx[depth].get(y_node)] = 1
                        for ancestor in nx.ancestors(g_t, y_node):
                            if ancestor != "root":
                                depth = (
                                    nx.shortest_path_length(g_t, "root").get(ancestor)
                                    - 1
                                )
                                y_local_[depth][
                                    local_nodes_idx[depth].get(ancestor)
                                ] = 1
                    else:

                        depth = y_node.count(".") + 1

                        assert depth is not None

                        for index in range(depth, 0, -1):
                            local_terms = y_node.split(".")[:index]
                            local_label = ".".join(local_terms)
                            local_depth = local_label.count(".")

                            y_local_[local_depth][
                                local_nodes_idx.get(local_depth).get(local_label)
                            ] = 1

                Y.append(y_)
                Y_local.append([np.stack(y) for y in y_local_])
            if use_sample and count != 0 and count % 20 == 0:
                print("fazendo parada")
                break
        X = np.array(X)
        Y = np.stack(Y)
        # Dictionary to store the adjacency matrix by level (N_previous x N_current)
        edges_matrix_dict = {}
        # Assuming 'levels' is ordered (e.g., levels.values() returns L0, L1, L2, ...)
        level_nodes_list = list(levels.values())
        for idx, current_level_nodes in enumerate(level_nodes_list):
            if idx == 0:
                # Level 0 has no ancestor; the Constraint Layer starts from level 1.
                continue
            # 1. Identify Previous Level (Ancestor) and Current Level (Child)
            prev_level_nodes = level_nodes_list[idx - 1]
            # 2. Format node names (adjust for your case with 'replace')
            # The R_sub matrix should contain all nodes from the previous level (rows)
            # and all nodes from the current level (columns).
            # Replace '/' with '.' if necessary (this depends on how your graph 'g' is labeled)
            ancestral_nodelist = [node.replace("/", ".") for node in prev_level_nodes]
            child_nodelist = [node.replace("/", ".") for node in current_level_nodes]
            # 3. Build the N_previous x N_current adjacency matrix
            # row_order (nodelist): defines the ROWS (Ancestors / Previous Level)
            # column_order: defines the COLUMNS (Children / Current Level)
            # nx.to_numpy_array will create the matrix A[i, j] where:
            # A[i, j] = 1 if there is an edge from row_order[i] to column_order[j]
            R_sub_numpy = nx.to_numpy_array(
                g,
                nodelist=ancestral_nodelist
                + child_nodelist,  # Rows (Previous Level / Ancestors)
                dtype=np.float32,  # Use float32 for consistency with PyTorch
            )
            # NetworkX returns 0 or 1 for edges, which is perfect for a binary mask.
            # 4. Store the matrix in the dictionary
            # We store as a PyTorch Tensor to avoid repeated conversion during training.
            edges_matrix_dict[idx] = torch.from_numpy(R_sub_numpy)
        # R_sub_matrix_dict[level] now contains the PyTorch Tensor (N_previous x N_current)
        # Example: R_sub_matrix_dict[1] maps Level 0 -> Level 1

        # logger.info(
        #     "Shape of edges matrix: %s",
        #     {k: v.shape for k, v in edges_matrix_dict.items()},
        # )
        logger.info("Parsed ARFF file: %s", arff_file)
        # logger.info("Number of matrix: %d", len(edges_matrix_dict))

        return (
            X,
            Y,
            Y_local,
            np.array(nx.to_numpy_array(g, nodelist=nodes)),
            edges_matrix_dict,
            nodes,
            g,
            levels,
            levels_size,
            nodes_idx,
            local_nodes_idx,
            max_depth,
        )
