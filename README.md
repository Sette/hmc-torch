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

## Local LLM reranking

Install ollama:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Run Ollama locally in one terminal:

```bash
ollama serve
```

Pull the default reranker model once:

```bash
ollama pull qwen3:14b
```

Train with the optimized LLM reranker:

```bash
PYTHONPATH=src .venv/bin/python -m hmc.main \
  --dataset_name arxiv \
  --method globalLLMLite \
  --device cuda \
  --dataset_path ./data \
  --output_path ./output \
  --epochs 5 \
  --batch_size 4 \
  --llm_model qwen3:14b \
  --llm_cache true \
  --llm_fallback_on_error true \
  --llm_preserve_scores true \
  --llm_max_document_chars 3000
```

Use `--dataset_name wos` for WOS. The reranker uses `http://localhost:11434` by default.

## Methods

| Method | Description |
|---|---|
| `global` | Frozen SPECTER2 embeddings + MLP + R-matrix constraint |
| `globalE2E` | End-to-end fine-tuned transformer + MLP + R-matrix |
| `globalSOTA` | E2E + GCN label-graph encoder (HiAGM-style) |
| `globalLLM` | E2E + Ollama reranking over candidate labels |
| `globalLLMLite` | E2E + cheaper uncertainty-gated Ollama reranking with cache/fallback |

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
