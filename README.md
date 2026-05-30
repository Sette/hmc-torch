# HMC Torch

Hierarchical Multi-Label Classification (HMC) network implemented in PyTorch.

Supports two classification strategies — **global** (single model for all labels) and **local** (one model per hierarchy level) — with optional hyperparameter optimization via Optuna.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Datasets](#datasets)
- [Running](#running)
- [Configuration](#configuration)
- [Testing](#testing)

---

## Overview

HMC problems involve predicting labels organized in a hierarchy (e.g., Gene Ontology, ArXiv taxonomy). This project benchmarks several approaches:

| Method | Description |
|---|---|
| `global` | C-HMCNN: single MLP constrained by a hierarchy matrix (R-matrix) |
| `global_baseline` | Same model without hierarchical constraint enforcement |
| `globalGNN` | MLP document encoder + 2-layer GCN on the label hierarchy graph |
| `globalE2E` | HuggingFace transformer fine-tuned end-to-end with MLP head |
| `globalSOTA` | Transformer (fine-tuned) + GCN label encoder — HiAGM-style SOTA |
| `globalLM` | Lightning-wrapped version of `global` |
| `local` | One MLP per hierarchy level, trained jointly with early stopping |
| `local_test` | Local model in inference-only mode |

### Method comparison (ArXiv, Micro-F1 approximate)

| Method | Features | Architecture | Micro-F1 |
|---|---|---|---|
| `global` | TF-IDF + SVD(256) | MLP + R-matrix | ~70% |
| `globalGNN` | TF-IDF or SPECTER | MLP + GCN label graph + R-matrix | ~75–80% |
| `globalE2E` | SPECTER fine-tuned | Transformer + MLP head + R-matrix | ~85–87% |
| `globalSOTA` | SPECTER fine-tuned | Transformer + GCN label graph + R-matrix | ~87–90% |

---

## Project Structure

```
hmc-torch/
├── src/hmc/
│   ├── arguments.py                   # Args dataclass + argparse CLI (parse_args)
│   ├── main.py                        # Entry point — routes to global or local pipeline
│   ├── env.py                         # Logging and environment configuration
│   │
│   ├── datasets/
│   │   ├── dataset_manager.py         # initialize_dataset_experiments — main loader entry
│   │   ├── dataset_torch.py           # PyTorch dataset wrapper for .pt tensor files
│   │   ├── registry.py                # DatasetRegistry — dimensions, lr, epochs per dataset
│   │   ├── arxiv/
│   │   │   ├── dataset_arxiv.py       # ArXivHierarchyManager, ArXivSplit, ArXivPyTorchDataset
│   │   │   └── manager.py             # ArXivManager — TF-IDF/SVD or transformer embedding features
│   │   └── gofun/
│   │       ├── dataset_arff.py        # HMCDatasetArff — ARFF parser, HierarchyData, SampleData
│   │       └── manager.py             # HMCDatasetManager — scaling, splits, adjacency matrices
│   │
│   ├── models/
│   │   ├── base.py                    # HierarchicalModel — abstract base for all classifiers
│   │   ├── global_classifier/
│   │   │   ├── constraint/
│   │   │   │   ├── model.py           # ConstrainedModel, ConstrainedGNNModel, ConstrainedLightningModel
│   │   │   │   └── utils.py           # get_constr_out — applies R-matrix constraint
│   │   │   └── e2e/
│   │   │       └── model.py           # E2EConstrainedModel (globalE2E), E2EGNNModel (globalSOTA)
│   │   └── local_classifier/
│   │       ├── baseline/
│   │       │   └── model.py           # HMCLocalModel — per-level MLP ensemble
│   │       └── networks.py            # ClassificationNetwork, BuildClassification (MLP/GCN/GAT)
│   │
│   ├── pipeline/
│   │   ├── global_classifier/
│   │   │   ├── main.py                # train_global, train_global_e2e, train_global_sota
│   │   │   ├── e2e_train.py           # train_e2e_step — training loop for E2E and SOTA methods
│   │   │   └── core/
│   │   │       └── train.py           # _run_training_loop, _collect_test_outputs, _compute_local_scores
│   │   └── local_classifier/
│   │       ├── main.py                # main_local(), train_local(), test_local(), get_train_methods()
│   │       ├── core/
│   │       │   ├── train.py           # train_step — progressive level training
│   │       │   ├── validate.py        # validate_step — per-level metrics + early stopping
│   │       │   └── predict.py         # test_step — threshold search + final scores
│   │       └── hpo/
│   │           └── hpo_local.py       # optimize_hyperparameters — Optuna study per level
│   │
│   └── utils/
│       ├── parser.py                  # create_example — feature/label tuple to dict
│       ├── datasets/
│       │   ├── labels.py              # Label conversion (local↔global), group_labels_by_level
│       │   ├── paths.py               # get_dataset_paths — file paths for all datasets
│       │   └── convert_hpo_json.py    # HPO result consolidation to YAML
│       ├── metrics/
│       │   └── calculate_metrics.py   # precision, recall, f1, avg precision
│       ├── path/
│       │   ├── files.py               # create_dir, join_path, __load_json__
│       │   └── output.py              # save_dict_to_json
│       ├── predict/
│       │   └── metrics.py             # find_best_threshold(s), create_report_metrics
│       └── train/
│           ├── early_stopping.py      # check_early_stopping_normalized, check_loss, check_metric
│           ├── job.py                 # create_job_id_name, timers, GPU logging, threshold search
│           └── losses.py              # compute_loss, calculate_local_loss, global_contrastive_loss
│
├── tests/
│   ├── conftest.py                    # Adds src/ to Python path
│   └── train_global_test.py           # Integration test for global pipeline (seq_FUN)
│
├── docs/
│   └── arxiv-hmc-sota.md              # SOTA analysis and improvement roadmap for ArXiv
│
├── config.yaml                        # HPO-tuned hyperparameters per dataset
├── run.sh                             # Main training script (reads config.yaml via yq)
├── install.sh                         # Dependency installation helper
├── deploy_kaggle.sh                   # Kaggle dataset upload helper
├── Makefile                           # lint, test, build, run targets
├── pyproject.toml                     # Project metadata and dependencies (uv)
└── uv.lock                            # Locked dependency tree
```

---

## Installation

The project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv (if not already installed)
pip install uv

# Install all dependencies including dev tools
uv sync --all-groups
```

Set the Python path before running:

```bash
export PYTHONPATH=src
```

---

## Datasets

Datasets follow the naming convention `{data}_{ontology}`, e.g. `seq_FUN`, `expr_GO`.

**Supported datasets:**

| Group | Datasets |
|---|---|
| FUN / GO | `cellcycle`, `derisi`, `eisen`, `expr`, `gasch1`, `gasch2`, `seq`, `spo` |
| Others | `diatoms`, `enron`, `imclef07a`, `imclef07d` |
| ArXiv | `arxiv` (category hierarchy: cs.AI, cs.LG, …) |

With kaggle python package:

```bash
pip install kaggle
```

**Download FUN / GO from Kaggle:**

```bash
kaggle datasets download brunosette/gene-ontology-original --unzip -p data/
```

**Download ArXiv from Kaggle:**

```bash
kaggle datasets download -d Cornell-University/arxiv --unzip -p data/arxiv/
```

---

## Running

### Using `run.sh`

The script reads per-dataset hyperparameters from `config.yaml` automatically.

```bash
chmod +x run.sh

# Train local classifier on a single dataset (CPU)
./run.sh --dataset_name seq_FUN --method local --device cpu

# Train on all datasets
./run.sh --dataset_name all --method local --device cuda

# Train global classifier
./run.sh --dataset_name seq_FUN --method global --device cuda
```

### ArXiv dataset

The ArXiv dataset uses a two-level category hierarchy (e.g. `cs` → `cs.AI`, `cs.LG`) and supports two text feature extraction modes controlled by `--arxiv_feature_type`:

| Mode | Flag | Description | `input_dim` |
|---|---|---|---|
| `tfidf` (default) | `--arxiv_feature_type tfidf` | TF-IDF (50 K terms) + TruncatedSVD to 256 dims — fast, no GPU needed | 256 |
| `embedding` | `--arxiv_feature_type embedding` | Transformer embeddings via HuggingFace — more semantic, GPU recommended | model-dependent |

Place the dataset file at `data/arxiv/arxiv-metadata-oai-snapshot.json` (see [Datasets](#datasets)), then:

```bash
# Default: TF-IDF + SVD features, MLP + R-matrix
./run.sh --dataset_name arxiv --method global --device cpu --epochs 50

# GNN on label graph + SPECTER embeddings
./run.sh --dataset_name arxiv --method globalGNN --device cuda --epochs 50 \
  --arxiv_feature_type embedding \
  --arxiv_model_name allenai/specter2_base

# GNN + hierarchical contrastive loss
./run.sh --dataset_name arxiv --method globalGNN --device cuda --epochs 50 \
  --arxiv_feature_type embedding \
  --use_contrastive_loss true \
  --lambda_contrastive 0.1

# E2E fine-tuning (transformer + MLP head)
python -m hmc.main \
  --dataset_name arxiv --method globalE2E --device cuda \
  --arxiv_model_name allenai/specter2_base \
  --dataset_path ./data --output_path ./output

# SOTA: E2E transformer + GCN label graph (HiAGM-style, best expected F1)
python -m hmc.main \
  --dataset_name arxiv --method globalSOTA --device cuda \
  --arxiv_model_name allenai/specter2_base \
  --dataset_path ./data --output_path ./output
```

The global pipeline uses the hyperparameters from `DatasetRegistry.arxiv_defaults` (hidden_dim=512, lr=1e-4, 3 layers, dropout=0.3). The local pipeline reads them from `config.yaml` under the `arxiv` key.

> **Feature cache:** both modes cache the computed feature matrix to `data/arxiv/.feature_cache/<hash>.npy`. The hash encodes the JSONL path, file mtime/size, `feature_type`, `model_name`, `n_components`, and number of loaded records — the cache is invalidated automatically if any of those change.
>
> **Note on embedding mode:** the first run downloads model weights from the HuggingFace Hub (~500 MB for SPECTER2). Encoding 50 K documents takes a few minutes on CPU; use `--device cuda` to speed it up.

### globalSOTA architecture

`globalSOTA` implements the core idea from [HiAGM (Zhou et al., ACL 2020)](https://aclanthology.org/2020.acl-main.104/): a text encoder and a graph-aware label encoder whose representations are combined via dot-product scoring.

```
Input text
    │
    ▼
HuggingFace transformer (fine-tuned, discriminative LR)
    │  CLS / mean-pool
    ▼
doc_emb  (hidden_dim)
    │
    ├──────────────────────────────────────────┐
    │                                          │
    ▼                                          ▼
label_emb = GCN(label_hierarchy)     sigmoid(doc_emb @ label_emb.T)
              (N, hidden_dim)                  │
                                      get_constr_out (R-matrix)
                                               │
                                          predictions
```

The transformer is optimized at a smaller learning rate (`--lr_transformer`, default `2e-5`) while the GCN label encoder uses the standard `--lr`.

**Recommended models for `--arxiv_model_name`:**

| Model | Dims | Notes |
|---|---|---|
| `allenai/specter2_base` (default) | 768 | Pre-trained on scientific paper retrieval — best for ArXiv |
| `allenai/scibert_scivocab_uncased` | 768 | Scientific language model |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Strong general-purpose baseline |

**Common options:**

| Option | Default | Description |
|---|---|---|
| `--dataset_name` | `seq_FUN` | Dataset name or `all` |
| `--method` | `local` | See method table above |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--epochs` | `2000` | Training epochs |
| `--hpo` | `false` | Enable Optuna HPO (`true`/`false`) |
| `--n_trials` | `30` | HPO trials per level |
| `--output_path` | `./results` | Where to save models and scores |
| `--epochs_to_evaluate` | `20` | Validation frequency |
| `--warmup` | `false` | Progressive level activation |
| `--arxiv_feature_type` | `tfidf` | ArXiv features: `tfidf` or `embedding` |
| `--arxiv_model_name` | `allenai/specter2_base` | HuggingFace model for `embedding` / E2E modes |
| `--arxiv_max_records` | `50000` | Max ArXiv records to load (0 = all) |
| `--use_contrastive_loss` | `false` | Hierarchical contrastive loss for `globalGNN` |
| `--lambda_contrastive` | `0.1` | Weight of the contrastive loss term |
| `--lr_transformer` | `2e-5` | Transformer learning rate for `globalE2E` / `globalSOTA` |

### Direct Python invocation

```bash
python -m hmc.main \
  --dataset_path ./data \
  --output_path ./results \
  --dataset_name seq_FUN \
  --method local \
  --device cpu \
  --epochs 2000 \
  --lr_values 0.001 0.0001 0.0005 0.001 0.0002 0.00005 \
  --dropout_values 0.3 0.4 0.5 0.3 0.4 0.5 \
  --hidden_dims "[[512],[256],[128],[256],[128],[64]]" \
  --num_layers_values 1 1 1 1 1 2 \
  --weight_decay_values 1e-4 1e-4 1e-4 1e-4 1e-4 1e-4
```

### Using `make`

```bash
make run    # runs run.sh with default settings (seq_FUN, local, cuda)
make test   # runs pytest with coverage
make lint   # autopep8 + black + ruff + isort + pylint
```

---

## Configuration

`config.yaml` stores HPO-tuned hyperparameters for each dataset. `run.sh` reads these values using `yq` and passes them to the CLI.

To add a new dataset, register its dimensions in `datasets/registry.py:DatasetRegistry`, then append a new entry to `config.yaml`:

```yaml
datasets_params:
  my_dataset_FUN:
    hidden_dims: [[256], [128], [64]]
    lr_values: [0.001, 0.0005, 0.0002]
    dropout_values: [0.3, 0.4, 0.5]
    num_layers_values: [1, 1, 1]
    weight_decay_values: [1e-4, 1e-4, 1e-4]
```

---

## Testing

```bash
pytest --verbose --cov=. tests/
```

The integration test in `tests/train_global_test.py` runs the full global pipeline on `seq_FUN` with mocked `sys.argv` and validates output metrics.
