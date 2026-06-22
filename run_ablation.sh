#!/bin/bash
# Ablation experiments: globalE2E + globalSOTA on WOS and ArXiv
# Runs sequentially on a single GPU.

set -e
export PYTHONPATH=src
VENV=.venv/bin/activate
DEVICE=cuda
DATA=./data
OUT=./output
BATCH=4
EPOCHS=5

[ -f "$VENV" ] && source "$VENV"

run_exp() {
    local ds=$1 method=$2
    local log="/tmp/${ds}_${method}.log"
    echo "=== $(date) Starting $method on $ds ===" | tee "$log"
    python -m hmc.main \
        --dataset_name "$ds" --method "$method" --device "$DEVICE" \
        --dataset_path "$DATA" --epochs "$EPOCHS" --batch_size "$BATCH" \
        --output_path "$OUT" >> "$log" 2>&1
    echo "=== $(date) Done $method on $ds ===" | tee -a "$log"
    # Print key results
    grep -E "Local evaluation|Global evaluation|Precision|Recall|F1-score|micro|macro|best_score|Tempo" "$log" | tail -10
}

echo "Starting ablation at $(date)"
echo "WOS E2E → WOS SOTA → ArXiv E2E → ArXiv SOTA"

run_exp wos globalE2E
run_exp wos globalSOTA
run_exp arxiv globalE2E
run_exp arxiv globalSOTA

echo "All done at $(date)"
