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
│   ├── env.py                         # Environment variables
│   │
│   ├── datasets/
│   │   ├── dataset_torch.py           # PyTorch dataset wrapper
│   │   ├── gofun/
│   │   │   └── dataset_arff.py        # ARFF file loader (Gene Ontology / FUN)
│   │   └── manager/
│   │       └── dataset_manager.py     # initialize_dataset_experiments — main loader entry
│   │
│   ├── models/
│   │   ├── base.py                    # Base model class
│   │   ├── global_classifier/
│   │   │   └── constraint/
│   │   │       ├── model.py           # ConstrainedModel, ConstrainedLightningModel
│   │   │       └── utils.py           # get_constr_out — applies R-matrix constraint
│   │   └── local_classifier/
│   │       ├── baseline/
│   │       │   └── model.py           # HMCLocalModel — per-level MLP ensemble
│   │       └── networks.py            # Shared network building blocks
│   │
│   ├── pipeline/
│   │   ├── global_classifier/
│   │   │   ├── main.py                # train_global() — setup, data loading, fit
│   │   │   └── core/
│   │   │       └── train.py           # train_step, test_step, get_local_scores
│   │   └── local_classifier/
│   │       ├── main.py                # main_local(), train_local(), test_local()
│   │       ├── core/
│   │       │   ├── train.py           # train_step — progressive level training
│   │       │   ├── validate.py        # validate_step — per-level metrics + early stopping
│   │       │   └── test.py            # test_step — threshold search + final scores
│   │       └── hpo/
│   │           └── hpo_local.py       # optimize_hyperparameters — Optuna study per level
│   │
│   └── utils/
│       ├── dataset/
│       │   ├── labels.py              # Label conversion (local↔global), binarization
│       │   └── convert_hpo_json.py    # HPO result format conversion
│       ├── metrics/
│       │   └── calculate_metrics.py   # precision, recall, f1, avg precision
│       ├── path/
│       │   ├── files.py               # create_dir
│       │   └── output.py              # save_dict_to_json
│       ├── predict/
│       │   └── metrics.py
│       └── train/
│           ├── early_stopping.py      # check_early_stopping_normalized, check_loss
│           ├── job.py                 # create_job_id_name, timers, threshold search
│           └── losses.py              # compute_loss, focal loss, hierarchical loss
│
├── tests/
│   ├── conftest.py
│   └── train_global_test.py           # Integration test for global pipeline
│
├── config.yaml                        # HPO-tuned hyperparameters per dataset
├── run.sh                             # Main training script (reads config.yaml via yq)
├── run.ps1                            # Windows equivalent
├── Makefile                           # lint, test, build targets
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

Make sure do you have data dir:


**Supported datasets:**

| Group | Datasets |
|---|---|
| FUN / GO | `cellcycle`, `derisi`, `eisen`, `expr`, `gasch1`, `gasch2`, `seq`, `spo` |
| Others | `diatoms`, `enron`, `imclef07a`, `imclef07d` |


With kaggle python package:

```bash
pip install kaggle
```

**Download FUN / GO from Kaggle:**


```bash
kaggle datasets download brunosette/gene-ontology-original --unzip -p data/
```

**Download arxiv from Kaggle:**

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

To add a new dataset, append a new entry to `config.yaml`:

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
