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

HMC problems involve predicting labels that are organized in a hierarchy (e.g., Gene Ontology). This project benchmarks two approaches:

| Method | Description |
|---|---|
| `global` | C-HMCNN: single MLP constrained by a hierarchy matrix (R-matrix) |
| `global_baseline` | Same model without hierarchical constraint enforcement |
| `local` | One MLP per hierarchy level, trained jointly with early stopping |
| `local_test` | Local model in inference-only mode |

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
│   │   │   └── manager.py             # ArXivManager — TF-IDF/SVD or transformer mean-pool features
│   │   └── gofun/
│   │       ├── dataset_arff.py        # HMCDatasetArff — ARFF parser, HierarchyData, SampleData
│   │       └── manager.py             # HMCDatasetManager — scaling, splits, adjacency matrices
│   │
│   ├── models/
│   │   ├── base.py                    # HierarchicalModel — abstract base for all classifiers
│   │   ├── global_classifier/
│   │   │   └── constraint/
│   │   │       ├── model.py           # ConstrainedModel, ConstrainedLightningModel
│   │   │       └── utils.py           # get_constr_out — applies R-matrix constraint
│   │   └── local_classifier/
│   │       ├── baseline/
│   │       │   └── model.py           # HMCLocalModel — per-level MLP ensemble
│   │       └── networks.py            # ClassificationNetwork, BuildClassification (MLP/GCN/GAT)
│   │
│   ├── pipeline/
│   │   ├── global_classifier/
│   │   │   ├── main.py                # train_global() — setup, R-matrix, data loading, fit
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
│           └── losses.py              # compute_loss, calculate_local_loss
│
├── tests/
│   ├── conftest.py                    # Adds src/ to Python path
│   └── train_global_test.py           # Integration test for global pipeline (seq_FUN)
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
| `embedding` | `--arxiv_feature_type embedding` | Mean-pooled transformer embeddings via HuggingFace `transformers` — more semantic | model-dependent |

Place the dataset file at `data/arxiv/arxiv-metadata-oai-snapshot.json` (see [Datasets](#datasets) for the download command), then:

```bash
# Default: TF-IDF + SVD features
./run.sh --dataset_name arxiv --method global --device cpu --epochs 50 --dataset_type arxiv

# Semantic embeddings with the default model (all-MiniLM-L6-v2, 384-dim)
./run.sh --dataset_name arxiv --method global --device cpu --epochs 50 --dataset_type arxiv \
  --arxiv_feature_type embedding

# Different model (e.g. bert-base-uncased, 768-dim)
./run.sh --dataset_name arxiv --method global --device cpu --epochs 50 --dataset_type arxiv \
  --arxiv_feature_type embedding \
  --arxiv_model_name "bert-base-uncased"

# Local classifier with embeddings
./run.sh --dataset_name arxiv --method local --device cuda --epochs 100 --dataset_type arxiv \
  --arxiv_feature_type embedding

# Hyperparameter search
./run.sh --dataset_name arxiv --method local --hpo true --n_trials 30 --device cuda \
  --dataset_type arxiv
```

The global pipeline uses the hyperparameters from `DatasetRegistry.arxiv_defaults` (hidden_dim=512, lr=1e-4, 3 layers, dropout=0.3). The local pipeline reads them from `config.yaml` under the `arxiv` key.

> **Feature cache:** both modes cache the computed feature matrix to `data/arxiv/.feature_cache/<hash>.npy`. The hash encodes the JSONL path, file mtime/size, `feature_type`, `model_name`, `n_components`, and number of loaded records — so the cache is invalidated automatically if any of those change. Subsequent runs skip feature computation entirely.
>
> **Note on embedding mode:** the first run downloads model weights from the HuggingFace Hub (~80 MB for MiniLM). Encoding 50 K documents takes a few minutes on CPU; use `--device cuda` to speed it up.

**Common options:**

| Option | Default | Description |
|---|---|---|
| `--dataset_name` | `seq_FUN` | Dataset name or `all` |
| `--method` | `local` | `local`, `local_test`, `global`, `global_baseline` |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--epochs` | `4000` | Training epochs |
| `--hpo` | `false` | Enable Optuna HPO (`true`/`false`) |
| `--n_trials` | `30` | HPO trials per level |
| `--output_path` | `./results` | Where to save models and scores |
| `--epochs_to_evaluate` | `20` | Validation frequency |
| `--warmup` | `false` | Progressive level activation |
| `--arxiv_feature_type` | `tfidf` | ArXiv features: `tfidf` or `embedding` |
| `--arxiv_model_name` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model for `embedding` mode |

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
