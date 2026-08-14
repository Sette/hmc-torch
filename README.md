<p align="center">
  <h1 align="center">🔥 HMC-Torch</h1>
  <p align="center"><strong>A Modular Platform for Hierarchical Multi-Label Classification with R-Matrix Constraints</strong></p>
</p>

<p align="center">
  <a href="https://pypi.org/project/hmc-torch/"><img src="https://img.shields.io/pypi/v/hmc-torch?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/hmc-torch/"><img src="https://img.shields.io/pypi/pyversions/hmc-torch" alt="Python"></a>
  <a href="https://github.com/Sette/hmc-torch/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="https://pypi.org/project/hmc-torch/"><img src="https://img.shields.io/pypi/dm/hmc-torch" alt="Downloads"></a>
</p>

---

**HMC-Torch** is a modular, extensible platform for Hierarchical Multi-Label Classification (HMC)
that integrates the **R-matrix constraint** — originally proposed by
[Giunchiglia & Lukasiewicz (2018, NeurIPS)](https://papers.nips.cc/paper_files/paper/2018/hash/08aacd96e9e26e79b77e4f65b9c0aa33-Abstract.html) —
as a reusable, first-class architectural component.

> 📄 **Paper**: *HMC-Torch: A Modular Platform for Hierarchical Multi-Label Classification with R-Matrix Constraints* (Bruno Sette, UFSCar, 2026)

---

## ✨ Key Features

- 🔌 **Modular pipeline**: `DatasetAdapter → FeatureEncoder → HierarchicalHead → Calibrator → Reconciler`
- 🧱 **R-Matrix as infrastructure**: reusable ancestor-closure constraint for training + inference
- 🌳 **Explicit hierarchy modeling**: Tree (FunCat) and DAG (Gene Ontology) with type-specific reconciliation
- 📊 **25+ datasets across 5 domains**: scientific text, genomics, email, microscopy, medical imaging
- 🧬 **Multi-modal**: tabular, text (transformers), protein sequences, images
- ⚡ **GPU-accelerated**: 14× training speedup
- 📦 **Plugin system**: register your own datasets without modifying the package
- 🔁 **Reproducible**: experiment manifests with git SHA, seeds, and dependency versions

---

## 📦 Installation

```bash
pip install hmc-torch
```

For optional features:

```bash
pip install "hmc-torch[vision]"      # Image models (timm)
pip install "hmc-torch[protein]"     # Protein models (fair-esm)
pip install "hmc-torch[expression]"  # Expression autoencoders
```

### From source

```bash
git clone https://github.com/Sette/hmc-torch.git
cd hmc-torch
uv sync --all-groups
export PYTHONPATH=src
```

---

## 🚀 Quick Start

### Python API

```python
import hmc

# Train a global classifier with R-matrix constraint
results = hmc.train("wos", method="globalE2E", device="cuda", epochs=5)

# Train a tabular baseline
results = hmc.train("cellcycle_FUN", method="tabular_mlp", device="cuda", epochs=100)
```

### CLI

```bash
# Frozen embeddings baseline
python -m hmc.main --dataset_name wos --method global --device cuda \
  --dataset_path ./data --output_path ./output

# Fine-tuned transformer (SOTA)
python -m hmc.main --dataset_name arxiv --method globalE2E --device cuda \
  --dataset_path ./data --epochs 50 --batch_size 32 --output_path ./output

# Tabular baseline
python -m hmc.main --dataset_name cellcycle_FUN --method tabular_mlp --device cuda \
  --dataset_path ./data --output_path ./output
```

---

## 🗂️ Supported Datasets

### Built-in datasets (25+)

| Domain | Datasets | Modality | Hierarchy | Classes |
|---|---|---|---|---|
| **Scientific Text** | ArXiv, WOS | Text (SPECTER2) | Tree | 141–156 |
| **Genomics (FunCat)** | cellcycle, church, derisi, eisen, expr, gasch1, gasch2, pheno, seq, spo | Tabular | Tree | ~499 |
| **Genomics (GO)** | cellcycle, derisi, eisen, expr, gasch1, gasch2, pheno, seq, spo | Tabular | DAG | 3,570–4,130 |
| **Email** | Enron | Tabular | Tree | 56 |
| **Microscopy** | Diatoms | Tabular | Tree | 398 |
| **Medical Imaging** | ImCLEF07a, ImCLEF07d | Tabular | Tree | 46–96 |
| **Multi-label Text** | AAPD, RCV1, EURLex | Text | Tree | 54–3,993 |

### Register your own dataset

The framework derives everything it needs — adjacency matrix, level sizes,
edge indices, input/output dimensions — **from your labels**.  Your only
job is to supply the raw data and make sure the labels encode the hierarchy
in a way the framework can parse.

**Minimal example — just data + labels:**

```python
from hmc.utils import build_digraph_from_labels
from hmc.data.hierarchy import TreeHierarchy

# 1. Your data
texts = ["paper about deep learning", "paper about transformers", ...]
label_strs = ["cs.AI cs.LG", "cs.CL", ...]   # one string per sample (space-separated)

# 2. Build hierarchy from labels (one-liner)
h = TreeHierarchy.from_graph(build_digraph_from_labels(label_strs))

# 3. Encode labels (ancestors auto-activated)
Y_global, Y_local = h.encode_labels(label_strs)
Y_local = Y_local[1:]  # drop root level

# 4. Compute features, create splits, and train — see full example below.
```

The rest of this section walks through each step in detail and shows how
to package everything into a reusable manager.

**1. Pick a label format**

Labels must follow a **dot-separated path notation** where each segment is
one level of the hierarchy.  For example, a paper tagged as ``cs.AI`` and
``stat.ML`` belongs to both *Artificial Intelligence* (under *Computer
Science*) and *Machine Learning* (under *Statistics*):

```
root
├── cs          (level 1 — area)
│   └── cs.AI   (level 2 — subcategory)
└── stat        (level 1 — area)
    └── stat.ML (level 2 — subcategory)
```

The same convention works for any tree-shaped taxonomy —
``A.A1.B1``, ``medicine.cardiology``, ``physics.optics.lasers``, etc.

**2. Build the hierarchy** — just collect your unique labels and pass them to
:func:`~hmc.utils.build_digraph_from_labels`:

```python
from hmc.utils import build_digraph_from_labels
from hmc.data.hierarchy import TreeHierarchy

# Collect all unique labels from your data
all_labels = ["cs", "cs.AI", "cs.LG", "stat", "stat.ML"]

# One-liner: build the child→parent graph from label strings
g = build_digraph_from_labels(all_labels)

hierarchy = TreeHierarchy.from_graph(g)
# → hierarchy.a, hierarchy.edge_index, hierarchy.to_eval, hierarchy.local_nodes_idx
#   are all computed automatically — nothing else to do.
```

The function infers intermediate nodes, handles duplicates, and supports
custom separators (``sep="/"``) and root names.  That's it — **you only
need your data and your labels**; the framework derives everything else.

**3. Encode labels** with a single call:

```python
Y_global, Y_local = hierarchy.encode_labels(
    ["cs.AI", "stat.ML cs.LG"]       # one string per sample (space-separated)
)
# Y_global: (n_samples, n_nodes) — ancestors automatically activated
# Y_local:  list of (n_samples, n_level_nodes) per depth level
```

**4. Wire everything into your manager** (the full ``__init__`` pattern):

```python
class MyManager:
    def __init__(self, data_path, ...):
        # (a) Load raw data and collect unique labels
        texts, label_strings = load_my_data(data_path)

        # (b) Build hierarchy
        g = build_digraph_from_labels(label_strings)
        h = TreeHierarchy.from_graph(g)

        # (c) Compute features
        X = compute_features(texts)

        # (d) Encode labels
        Y_global, Y_local_all = h.encode_labels(label_strings)
        Y_local = Y_local_all[1:]   # drop root level (pipeline convention)

        # (e) Create splits
        X_train, X_val, X_test, Yg_train, ... = split_data(X, Y_global, Y_local)

        # (f) Expose — copy from hierarchy + splits
        self.input_dim = X.shape[1]
        self.output_dim = h.n_nodes
        self.levels_size = {k-1: v for k, v in h.level_sizes.items() if k > 0}
        self.max_depth = len(self.levels_size)
        self.a = h.adjacency
        self.edge_index = h.edge_index       # ← now provided by TreeHierarchy
        self.nodes_idx = h.node_index
        self.local_nodes_idx = h.local_nodes_idx  # ← now provided
        self.to_eval = h.to_eval             # ← now provided
        self.hierarchy_map = {}

        self._train = Split(X_train, Yg_train, Y_local_train)
        self._valid = Split(X_val, Yg_val, Y_local_val)
        self._test = Split(X_test, Yg_test, Y_local_test)

    def get_datasets(self):
        return self._train, self._valid, self._test
```

The built-in managers follow this exact same pattern — they're a good
reference if you want to see a production version:

| Manager | File |
|---|---|
| `ArXivManager` | `src/hmc/datasets/arxiv/manager.py` |
| `WOSManager` | `src/hmc/datasets/wos/manager.py` |
| `HMCDatasetManager` | `src/hmc/datasets/gofun/manager.py` |

**5. Register** your manager — no need to modify any package source:

```python
from hmc.data import DatasetRegistry

DatasetRegistry.register(
    "my_data",
    lambda **kw: MyManager(**kw),
    defaults={"hidden_dim": 256, "lr": 1e-4, "epochs": 50, "dropout": 0.3},
)

# Use it anywhere
manager = DatasetRegistry.get("my_data", device="cuda", dataset_path="./data")
train, valid, test = manager.get_datasets()

import hmc
hmc.train(dataset_name="my_data", method="globalE2E", device="cuda", epochs=50)
```

If you ship your manager as a pip-installable package, register it
automatically via ``pyproject.toml``:

```toml
[project.entry-points."hmc_torch.datasets"]
my_data = "my_package.manager:create_manager"
```

The entry-point target must be a callable that accepts ``**kwargs`` and
returns a manager instance.

---

## 🧠 Methods

| Method | Description |
|---|---|
| `global` | Frozen embeddings + MLP + R-matrix constraint |
| `globalE2E` | End-to-end fine-tuned transformer + MLP + R-matrix |
| `globalSOTA` | E2E + GCN label-graph encoder (HiAGM-style) |
| `local` | Frozen embeddings, one MLP per hierarchy level |
| `localE2E` | Fine-tuned transformer + per-level MLPs |
| `tabular_gbdt` | Gradient Boosting One-vs-Rest baseline |
| `tabular_mlp` | Residual MLP baseline for tabular data |

---

## 📊 Results

### Text Benchmarks (Micro-F1)

| Method | ArXiv | WOS |
|---|---|---|
| HiAGM (Zhou+, ACL'20) | 0.5950 | 0.8604 |
| HTC-infoMAX | — | 0.8720 |
| **HMC-Torch (frozen)** | **0.7295** | 0.7515 |
| **HMC-Torch (E2E)** | — | **0.8743** |

> 🏆 **New SOTA on ArXiv** (+13.5 pts over HiAGM) and **WOS** (+0.2 pts over HTC-infoMAX)

### FunCat Genomic Benchmarks (AUPRC)

| Dataset | HMCN-F | C-HMCNN | HMC-Torch (GBDT) | HMC-Torch (MLP) |
|---|---|---|---|---|
| cellcycle_FUN | 0.235 | 0.248 | 0.253 | 0.255 |
| seq_FUN | 0.291 | 0.299 | 0.306 | **0.307** |
| spo_FUN | 0.228 | 0.228 | 0.229 | 0.229 |

> 🏆 **Beats HMCN-F on seq_FUN** (0.307 vs 0.291)

### Gene Ontology (Block-Diagonal R-Matrix)

AUPRC improves on **all 9 GO datasets** (avg +0.0055) with zero hierarchy violations, using
$O(N)$ memory instead of $O(N^2)$ (>1,400× reduction).

### GPU Speedup

| Configuration | CPU | GPU | Speedup |
|---|---|---|---|
| global (FunCat avg) | 16.7s | 1.2s | **14×** |
| tabular_mlp (FunCat avg) | 5.3s | 2.0s | 2.7× |

---

## 🧬 Architecture

```
DatasetAdapter  →  FeatureEncoder  →  HierarchicalHead  →  Calibrator  →  Reconciler
    ↓                    ↓                    ↓                ↓              ↓
DatasetBundle     fit/transform       GlobalSigmoidHead   Platt Scaling   Bottom-up
  + Hierarchy     (tabular, text,     LocalLevelHead      Temperature     max-prop
  + Splits         protein, vision)   TreePathHead
```

### R-Matrix Constraint

The ancestor closure matrix $R_{ij} = 1$ iff class $i$ is an ancestor of $j$. Applied at:

1. **Training**: $\mathcal{L}_{\text{hier}} = \max(0, p_j - p_i + \gamma)$ for all ancestor pairs
2. **Inference**: $p_i^{\text{rec}} = \max(p_i, \max_{j: \text{child}(i,j)} p_j^{\text{rec}})$

### Block-Diagonal R-Matrix (Sparse)

For large DAGs (4,000+ classes), the dense $R$ matrix alone costs $>65$ MB
($C^2 \times 4$ bytes) to store — and applying it to a batch of logits
materializes a `batch × C × C` float64 tensor (≈8.7 GB at batch 64, exceeding
GPU memory). Our sparse approximation uses graph traversal ($O(N+E)$ memory)
with zero hierarchy violations.

---

## 📁 Project Structure

```
src/hmc/
├── data/              # Data contracts (DatasetBundle, Split, Hierarchy)
├── datasets/          # Built-in dataset implementations
│   ├── arxiv/         # ArXiv (JSONL + SPECTER2)
│   ├── wos/           # WOS (Web of Science)
│   ├── gofun/         # FunCat + GO (ARFF tabular)
│   ├── aapd/          # Arxiv Academic Paper Dataset
│   ├── rcv1/          # Reuters Corpus Volume 1
│   └── eurlex/        # EUR-Lex documents
├── features/          # Feature encoders (text, tabular, vision, protein)
├── models/            # HMC model components
│   ├── global_classifier/  # Global heads + R-matrix
│   ├── local_classifier/   # Per-level local heads
│   ├── hierarchical/       # Sparse R-matrix, label GCN
│   └── tabular/            # GBDT + MLP baselines
├── pipeline/          # Training pipelines
└── utils/             # Metrics, manifests, caching
```

---

## 🤝 Contributing

Contributions welcome! Areas we'd love help with:

- New dataset adapters
- Additional feature encoders (genomics, graphs)
- New hierarchical heads and reconciliation strategies
- Documentation and tutorials

```bash
git clone https://github.com/Sette/hmc-torch.git
cd hmc-torch
uv sync --all-groups
make test
make lint
```

---

## 📚 Citation

```bibtex
@article{sette2026hmctorch,
  title   = {HMC-Torch: A Modular Platform for Hierarchical Multi-Label
             Classification with R-Matrix Constraints},
  author  = {Bruno Sette},
  journal = {arXiv preprint},
  year    = {2026},
}
```

---

## 📄 License

MIT © [Bruno Sette](https://github.com/Sette)
