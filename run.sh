#!/bin/bash
# Quick launcher for hmc-torch experiments (ArXiv and WOS datasets).

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH=src

DATASET_PATH="./data"
BATCH_SIZE=32
DEVICE="cuda"
EPOCHS=50
OUTPUT_PATH="./output"
METHOD="global"
DATASET_NAME="wos"
ARXIV_MODEL_NAME="allenai/specter2_base"

usage() {
    echo "Usage: $0 [options]"
    echo "  --dataset_name <arxiv|wos>  (default: $DATASET_NAME)"
    echo "  --method <global|globalE2E|globalSOTA|local>  (default: $METHOD)"
    echo "  --device <cuda|cpu>         (default: $DEVICE)"
    echo "  --epochs <num>              (default: $EPOCHS)"
    echo "  --batch_size <num>          (default: $BATCH_SIZE)"
    echo "  --output_path <path>        (default: $OUTPUT_PATH)"
    echo "  --arxiv_model_name <name>   (default: $ARXIV_MODEL_NAME)"
    echo "  --help"
    exit 0
}

while [ "$#" -gt 0 ]; do
    case $1 in
        --dataset_name) DATASET_NAME="$2"; shift ;;
        --method) METHOD="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --output_path) OUTPUT_PATH="$2"; shift ;;
        --arxiv_model_name) ARXIV_MODEL_NAME="$2"; shift ;;
        --help) usage ;;
        *) echo "Invalid: $1"; usage ;;
    esac
    shift
done

cmd="python -m hmc.main \
    --dataset_name $DATASET_NAME \
    --method $METHOD \
    --device $DEVICE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --dataset_path $DATASET_PATH \
    --output_path $OUTPUT_PATH \
    --arxiv_model_name $ARXIV_MODEL_NAME"

echo "$cmd"
$cmd
