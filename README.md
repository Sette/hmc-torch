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

> 📄 **Paper**: *HMC-Torch: A Modular Platform and an Empirical Study of When Hierarchical Constraints Help* (Bruno Sette, UFSCar, 2026)

---

## ✨ Key Features

- 🔌 **Modular pipeline**: `DatasetAdapter → FeatureEncoder → HierarchicalHead → Calibrator → Reconciler`
- 🧱 **R-Matrix as infrastructure**: reusable ancestor-closure constraint for training + inference
- 🌳 **Explicit hierarchy modeling**: Tree (FunCat) and DAG (Gene Ontology) with type-specific reconciliation
- 📊 **27 datasets across 5 domains**: scientific text, genomics, email, microscopy, medical imaging
- 🧬 **Multiple modalities**: production paths for tabular and text; experimental encoders for expression, protein sequences, and images
- ⚡ **GPU-ready**: exact sparse reconciliation that scales to 4,000-class Gene Ontology hierarchies
- 📦 **Plugin system**: register your own datasets without modifying the package
- 🔁 **Reproducible**: one evaluation protocol and scripts that regenerate every number in the paper

---

## 📦 Installation

```bash
pip install hmc-torch
```

For optional features:

```bash
pip install "hmc-torch[download]"    # Dataset downloaders (kagglehub, HF datasets)
pip install "hmc-torch[vision]"      # Image models (timm)
pip install "hmc-torch[protein]"     # Protein models (fair-esm)
pip install "hmc-torch[expression]"  # Expression autoencoders
```

### From source

```bash
git clone https://github.com/Sette/hmc-torch.git
cd hmc-torch
uv sync --all-groups --extra download
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

# Fine-tuned transformer (end-to-end)
python -m hmc.main --dataset_name arxiv --method globalE2E --device cuda \
  --dataset_path ./data --epochs 50 --batch_size 32 --output_path ./output

# Tabular baseline
python -m hmc.main --dataset_name cellcycle_FUN --method tabular_mlp --device cuda \
  --dataset_path ./data --output_path ./output

# Reproduce the paper's recipe: plain BCE + inference-time reconciliation
python -m hmc.main --dataset_name wos --method global --consistency_loss none \
  --dataset_path ./data --output_path ./output
```

**Training-time hierarchy handling** (`--consistency_loss`, for `global` and `globalE2E`):

| Value | Training loss | Constraint applied |
|---|---|---|
| `mc` (default) | constrained outputs mixed into the loss (C-HMCNN style) | training + inference |
| `hinge` | paper's Eq. 2 added with weight `--lambda_hier` | training + inference |
| `none` | plain BCE | inference only |

The constraint is always applied at inference (`get_constr_out`); the flag only
changes what happens during training. The numbers reported in the paper use
`none`, so pass it to reproduce them.

### Docker

Build the image from the repository root:

```bash
docker build -t hmc-torch .
```

The GitHub Actions workflow publishes release images to GitHub Container
Registry (GHCR) whenever a `v*` tag is pushed. For example, tag `v0.0.10`
publishes `ghcr.io/sette/hmc-torch:0.0.10` and the `latest` tag. You can also
run the workflow manually from the Actions tab; that publishes an image tagged
with the commit SHA.

Download (pull) a published image from GHCR before running it. Use a version tag
for a reproducible release, or `latest` for the most recently published
release:

```bash
docker pull ghcr.io/sette/hmc-torch:0.0.10
# Or pull the latest release
docker pull ghcr.io/sette/hmc-torch:latest
```

The image runs `python -m hmc.main`; without arguments it displays the CLI help.
Pass the same options used by the Python CLI to start an experiment. Mount the
dataset and output directories so they are available inside the container and
results persist after it exits:

```bash
docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/output:/app/output" \
  ghcr.io/sette/hmc-torch:0.0.10 \
  --dataset_name wos --method global --device cpu \
  --dataset_path /app/data --output_path /app/output
```

For GPU training, use Docker with NVIDIA Container Toolkit configured and add
`--gpus all`; set `--device cuda` in the application arguments:

```bash
docker run --rm --gpus all \
  -v "$PWD/data:/app/data" \
  -v "$PWD/output:/app/output" \
  ghcr.io/sette/hmc-torch:0.0.10 \
  --dataset_name wos --method global --device cuda \
  --dataset_path /app/data --output_path /app/output
```

### Data

```bash
make download-all      # every dataset
make download-arff     # FunCat + Gene Ontology ARFF benchmarks (GoFun family)
make download-arxiv    # ArXiv metadata snapshot (Kaggle)
make download-wos      # Web of Science abstracts
make download-aapd     # AAPD
make download-rcv1     # RCV1-V2
make download-eurlex   # EUR-Lex 57K
```

The tabular benchmarks land in `data/HMC_data_arff/` (the layout
`hmc.utils.datasets.paths` expects, covering the 10 `*_FUN`, the 9 `*_GO` and
the `*_others` datasets); the text datasets each get their own `data/<name>/`.

RCV1-V2 is gated (NIST license), so obtain `rcv1.tar.xz` and
`lyrl2004_tokens_train.dat` from NIST first and place them in
`data/rcv1/` (the output directory by default). Then `make download-rcv1` clones
HBGL when needed, links the source files into its checkout, and runs upstream
preprocessing. Pass `--raw_dir` if the licensed files are stored elsewhere, or
`--upstream_dir` to select another HBGL checkout. The other targets download
from public mirrors. For a pipeline smoke test only, `python -m
hmc.datasets.rcv1.download_rcv1 --sample` installs the ~10-document HiAGM files.

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
| **Multi-label Text** | AAPD, RCV1, EURLex | Text | Tree | 97–3,993 |

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

All numbers come from `experiments/run_protocol.py` and `experiments/run_wos_e2e.py`
under one protocol: training on the training split, epoch and decision threshold
chosen on the validation split, the test split used once, mean over three seeds.
`experiments/make_paper_artifacts.py` regenerates every table and figure of the paper
(`docs/hmc-paper`).

### Text (Micro-F1)

| Method | ArXiv | WOS | AAPD | EUR-Lex 57K |
|---|---|---|---|---|
| BERT+HiAGM (reimpl. by HGCLR, ACL'22) | — | 0.8604 | — | — |
| BERT+HTCInfoMax (reimpl. by HGCLR, ACL'22) | — | 0.8630 | — | — |
| BERT+HiMatch (Chen+, ACL'21) | — | 0.8670 | — | — |
| HGCLR (Wang+, ACL'22) | — | 0.8711 | — | — |
| HPT (Wang+, EMNLP'22) | — | **0.8716** | — | — |
| HMC-Torch, frozen SPECTER2 + MLP + reconciliation | 0.723 | 0.746 | 0.823 | 0.786 |
| HMC-Torch, fine-tuned SPECTER2 + reconciliation | — | 0.863 ± 0.006 | — | — |

WOS baselines are from Table 1 of HPT (EMNLP'22), which trains on the training split
only, as we do. ArXiv and AAPD use our own 64/16/20 splits and EUR-Lex uses the
EUROVOC hierarchy, so there are no comparable published numbers for them.

### FunCat and Gene Ontology (AUPRC, mean over the 8 datasets with published results)

| Method | FunCat | GO |
|---|---|---|
| Surj (Yang & Howe, WWW'22) | **0.312** | **0.467** |
| C-HMCNN (Giunchiglia & Lukasiewicz, NeurIPS'20) | 0.264 | 0.420 |
| HMCN-F (Wehrmann et al., ICML'18) | 0.261 | 0.422 |
| HMC-Torch, MLP (BCE) + reconciliation | 0.253 | 0.411 |
| HMC-Torch, MLP (MCLoss) + reconciliation | 0.255 | 0.411 |
| HMC-Torch, GBDT + reconciliation | 0.226 | — |

### What the constraint contributes

Reconciliation at inference raises test AUPRC on 25 of 27 datasets (median +0.002).
Adding the constraint to training — a hinge penalty or C-HMCNN's MCLoss — gives no
consistent gain over plain BCE followed by reconciliation. The sparse
(edge-list) reconciliation is exact and differentiable, so MCLoss trains at batch
size 64 on the 4,000-class GO hierarchies.

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

For large DAGs (4,000+ classes), the dense $R$ matrix requires $>65$ MB.
Our sparse approximation uses graph traversal ($O(N+E)$ memory) with zero hierarchy violations.

---

## 📁 Project Structure

The package has two related data layers: `hmc.datasets` contains built-in
loaders, managers, downloaders, and compatibility adapters; `hmc.data` defines
the shared split, hierarchy, metadata, and registry contracts. The code map and
current capability boundaries are documented in [docs/project-structure.md](docs/project-structure.md).

```
src/hmc/
├── arguments.py, main.py, train.py  # CLI, dispatch e API Python
├── data/              # Contratos, hierarquia e registro de datasets
├── datasets/          # Managers, adapters e downloaders por dataset
├── features/          # Encoders tabular, text, expression, vision e protein
├── models/            # Modelos globais, locais, hierárquicos e tabulares
├── pipeline/          # Loops de treino por família de modelo
└── utils/             # Métricas, caminhos, cache, manifestos e treino
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
uv sync --all-groups --extra download
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
