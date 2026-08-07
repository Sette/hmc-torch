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

> 📄 **Paper**: *HMC-Torch: A Modular Platform for Hierarchical Multi-Label Classification with R-Matrix Constraints* (Bruno Sette, UFMG, 2026)

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
pip install "hmc-torch[tabfm]"       # TabFM foundation model
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

# Train with LLM reranking
results = hmc.train(
    "arxiv", method="globalLLMLite", device="cuda",
    llm_model="qwen3:14b", llm_cache=True
)
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

```python
from hmc.data import DatasetRegistry, Split, DatasetBundle, Hierarchy

# Option 1: Programmatic registration
class MyDataset:
    def get_datasets(self):
        train = Split(features=X_train, labels=Y_train)
        test = Split(features=X_test, labels=Y_test)
        return train, None, test

    @property
    def hierarchy(self):
        return TreeHierarchy.from_edges([("root", "child1"), ...])

DatasetRegistry.register("my_data", lambda **kw: MyDataset(**kw))

# Option 2: Entry points (for packages)
# In your setup.cfg or pyproject.toml:
# [project.entry-points."hmc_torch.datasets"]
# my_data = "my_package:create_manager"
```

---

## 🧠 Methods

| Method | Description |
|---|---|
| `global` | Frozen embeddings + MLP + R-matrix constraint |
| `globalE2E` | End-to-end fine-tuned transformer + MLP + R-matrix |
| `globalSOTA` | E2E + GCN label-graph encoder (HiAGM-style) |
| `globalLLM` | E2E + Ollama reranking over candidate labels |
| `globalLLMLite` | E2E + cheaper uncertainty-gated Ollama reranking with cache/fallback |
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

For large DAGs (4,000+ classes), the dense $R$ matrix requires $>65$ MB.
Our sparse approximation uses graph traversal ($O(N+E)$ memory) with zero hierarchy violations.

---

## 🔬 LLM Reranking

Use local LLMs to refine predictions:

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Pull model & train
ollama pull qwen3:14b
python -m hmc.main --dataset_name arxiv --method globalLLMLite --device cuda \
  --dataset_path ./data --output_path ./output \
  --llm_model qwen3:14b --llm_cache true --llm_fallback_on_error true
```

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
├── llm/               # Ollama LLM agent
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
