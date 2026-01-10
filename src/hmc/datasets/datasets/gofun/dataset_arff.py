from collections import defaultdict
from itertools import chain
import torch
import keras
import networkx as nx
import numpy as np
import logging
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
    def __init__(self, arff_file, is_go):
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
        ) = parse_arff(arff_file=arff_file, is_go=is_go)
        self.to_eval = [t not in to_skip for t in self.terms]
        r_, c_ = np.where(np.isnan(self.X))
        m = np.nanmean(self.X, axis=0)
        for i, j in zip(r_, c_):
            self.X[i, j] = m[j]


def parse_arff(arff_file, is_go=False):
    with open(arff_file, "r", encoding="utf-8") as f:
        read_data = False
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
        for _, l in enumerate(f):
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
                            level = len(terms) - 1
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
                    else:
                        levels_size = {
                            key: len(set(value)) for key, value in levels.items()
                        }
                        max_depth = len(levels_size)

                    max_depth = len(levels_size)
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
                read_data = True
            elif read_data:
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
                            local_depth = len(local_terms) - 1
                            y_local_[local_depth][
                                local_nodes_idx.get(local_depth).get(local_label)
                            ] = 1

                Y.append(y_)
                Y_local.append([np.stack(y) for y in y_local_])
        X = np.array(X)
        Y = np.stack(Y)
        level_nodes_list = list(levels.values())
        edges_matrix_dict = {}
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
            ancestral_nodelist = prev_level_nodes
            child_nodelist = current_level_nodes
            # 3. Build the N_previous x N_current adjacency matrix
            # row_order (nodelist): defines the ROWS (Ancestors / Previous Level)
            # column_order: defines the COLUMNS (Children / Current Level)
            # nx.to_numpy_array will create the matrix A[i, j] where:
            # A[i, j] = 1 if there is an edge from row_order[i] to column_order[j]
            # 2. Criar matriz vazia com o shape CORRETO: (N_Pais, N_Filhos)
            # Exemplo: Se tem 18 pais e 80 filhos -> Shape (18, 80)

            shape = (len(ancestral_nodelist), len(child_nodelist))
            matrix = np.zeros(shape, dtype=np.float32)
            
            # 3. Preencher a matriz de forma eficiente
            # Criamos um mapa para achar o índice da coluna (filho) rapidamente
            child_map = {node: i for i, node in enumerate(child_nodelist)}
            parent_map = {node: i for i, node in enumerate(ancestral_nodelist)}

            for c_node in child_nodelist:
                if g.has_node(c_node):
                    p_node = list(g.successors(c_node))  # Get the parent node
                    for p in p_node:
                        a = parent_map.get(p, None)
                        b = child_map.get(c_node, None)
                        if a is not None and b is not None:
                            matrix[a, b] = 1.0          
            edges_matrix_dict[idx] = matrix

        logger.info(
            "Shape of edges matrix: %s",
            {k: v.shape for k, v in edges_matrix_dict.items()},
        )
        logger.info("Parsed ARFF file: %s", arff_file)
        logger.info("Number of matrix: %d", len(edges_matrix_dict))

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
