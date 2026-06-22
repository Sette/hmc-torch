# HMC Torch

Hierarchical Multi-Label Classification with transformer embeddings and R-matrix constraints.

**Datasets**: ArXiv (scientific papers) and WOS — Web of Science (HTC benchmark).

## Quick start

```bash
# Install
uv sync --all-groups
export PYTHONPATH=src

# Download data
make download-all

# Train (frozen embeddings baseline)
python -m hmc.main --dataset_name wos --method global --device cuda \
  --dataset_path ./data --output_path ./output

# Train (fine-tuned transformer — state-of-the-art)
python -m hmc.main --dataset_name wos --method globalE2E --device cuda \
  --dataset_path ./data --epochs 5 --batch_size 4 --output_path ./output

# Train (fine-tuned + label-graph GCN)
python -m hmc.main --dataset_name arxiv --method globalSOTA --device cuda \
  --dataset_path ./data --epochs 5 --batch_size 4 --output_path ./output
```

## Methods

| Method | Description |
|---|---|
| `global` | Frozen SPECTER2 embeddings + MLP + R-matrix constraint |
| `globalE2E` | End-to-end fine-tuned transformer + MLP + R-matrix |
| `globalSOTA` | E2E + GCN label-graph encoder (HiAGM-style) |

## Results (WOS, 64/16/20 HPT split, 5 epochs)

| Configuration | Micro-F1 |
|---|---|
| Baseline (frozen SPECTER2) | 74.1% |
| + Fine-tuning E2E | **87.4%** |
| + Label GNN (SOTA) | 86.6% |

Literature: HiAGM 85.8%, HGCLR 87.1%, HPT 87.2%.

## Results (ArXiv, 64/16/20 HPT split)

| Configuration | Micro-F1 |
|---|---|
| Baseline (frozen SPECTER2, 50k) | 72.1% |
| SciBERT E2E (100k, 10ep) | **72.6%** |
