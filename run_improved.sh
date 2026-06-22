#!/bin/bash
# Improved experiments: more epochs, better model for ArXiv
set -e
export PYTHONPATH=src
source .venv/bin/activate 2>/dev/null || true
DATA=./data OUT=./output DEVICE=cuda BATCH=4

run_exp() {
    local ds=$1 method=$2 epochs=$3 extra_args=$4
    local log="/tmp/${ds}_${method}_v2.log"
    echo "=== $(date) Starting $method on $ds ($epochs epochs) $extra_args ===" | tee "$log"
    python -m hmc.main \
        --dataset_name "$ds" --method "$method" --device "$DEVICE" \
        --dataset_path "$DATA" --epochs "$epochs" --batch_size "$BATCH" \
        --output_path "$OUT" $extra_args >> "$log" 2>&1
    echo "=== $(date) Done $method on $ds ===" | tee -a "$log"
    grep -E "Local evaluation|Global evaluation|Precision.*0\.|Recall.*0\.|F1-score" "$log" | tail -10
}

echo "Starting improved experiments at $(date)"
echo "1. WOS E2E 10 epochs → 2. ArXiv scibert 100k"

run_exp wos globalE2E 10 ""
run_exp arxiv globalE2E 10 "--arxiv_model_name allenai/scibert_scivocab_uncased --arxiv_max_records 100000"

echo "All done at $(date)"
