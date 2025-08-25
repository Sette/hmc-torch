#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

# Lista de datasets
datasets=('cellcycle_GO' 'derisi_GO' 'eisen_GO' 'expr_GO' 'gasch1_GO'
          'gasch2_GO' 'seq_GO' 'spo_GO' 'cellcycle_FUN' 'derisi_FUN'
          'eisen_FUN' 'expr_FUN' 'gasch1_FUN' 'gasch2_FUN' 'seq_FUN' 'spo_FUN')

# Definição de valores padrão para os parâmetros
DATASET="seq_FUN"
DATASET_PATH="./data"
BATCH_SIZE=4
NON_LIN="relu"
DEVICE="cpu"
EPOCHS=2000
EPOCHS_TO_EVALUATE=20
OUTPUT_PATH="results"
METHOD="local"
SEED=0
DATASET_TYPE="arff"
HPO="false"



HIDDEN_DIMS="127 84 70 222 83 84"
LR_VALUES="1.4640464735777067e-05 0.00010837097192333429 0.00019540574419398742 0.0003753240911181399 0.0005686655590971576 1.2869218027962399e-05"
DROPOUT_VALUES="0.5213189930776595 0.34321812846268623 0.6491134016969438 0.6682887953141635 0.3898928671880422 0.7047579081435292"
NUM_LAYERS_VALUES="1 1 2 2 2 2"
WEIGHT_DECAY_VALUES="1.5381868474682456e-05 4.0454234170148376e-05 3.4940476995010944e-06 3.5616632491932315e-05 1.1718710512206057e-05 7.627331701828602e-06"


export PYTHONPATH=src
export DATASET_PATH
export OUTPUT_PATH

# Function to display help
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Available options:"
    echo "  --dataset <name>          Dataset name (default: $DATASET)"
    echo "  --dataset_path <path>     Dataset path (default: $DATASET_PATH)"
    echo "  --seed <num>              Random seed (default: $SEED)"
    echo "  --dataset_type <type>     Dataset type (default: $DATASET_TYPE)"
    echo "  --batch_size <num>        Batch size (default: $BATCH_SIZE)"
    echo "  --lr_values <values>      Learning rates"
    echo "  --dropout <values>        Dropout rates (default: $DROPOUT_VALUES)"
    echo "  --hidden_dims <values>    Hidden dimensions (default: $HIDDEN_DIMS)"
    echo "  --dropout_values <values> Dropout rates"
    echo "  --hidden_dims <values>    Hidden dimensions"
    echo "  --num_layers_values <values> Number of layers"
    echo "  --weight_decay_values <values> Weight decay"
    echo "  --non_lin <function>      Activation function (default: $NON_LIN)"
    echo "  --device <type>           Device (cuda/cpu) (default: $DEVICE)"
    echo "  --epochs <num>            Number of epochs (default: $EPOCHS)"
    echo "  --output_path <path>      Output path for results (default: $OUTPUT_PATH)"
    echo "  --method <method>         Training method (default: $METHOD)"
    echo "  --hpo <true/false>        Hyperparameter optimization (default: $HPO)"
    echo "  --active_levels <num>     Number of active levels"
    echo "  --epochs_to_evaluate <num> Number of epochs to evaluate"
    echo "  --help                    Display this message and exit"
    exit 0
}

# Processamento dos argumentos
while [ "$#" -gt 0 ]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --dataset_path) DATASET_PATH="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --dataset_type) DATASET_TYPE="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --lr_values) LR_VALUES=($2); shift ;;
        --dropout_values) DROPOUT_VALUES=($2); shift ;;
        --hidden_dims) HIDDEN_DIMS=($2); shift ;;
        --num_layers_values) NUM_LAYERS_VALUES=($2); shift ;;
        --weight_decay_values) WEIGHT_DECAY_VALUES=($2); shift ;;
        --non_lin) NON_LIN="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --output_path) OUTPUT_PATH="$2"; shift ;;
        --method) METHOD="$2"; shift ;;
        --hpo) HPO="$2"; shift ;;
        --active_levels) ACTIVE_LEVELS=($2); shift ;;
        --epochs_to_evaluate) EPOCHS_TO_EVALUATE="$2"; shift ;;
        --help) usage ;;
        *) echo "Invalid option: $1"; usage ;;
    esac
    shift
done

 cmd="python -m hmc.trainers.main \
                --dataset_path $DATASET_PATH \
                --batch_size $BATCH_SIZE \
                --dataset_type $DATASET_TYPE \
                --non_lin $NON_LIN \
                --device $DEVICE \
                --epochs $EPOCHS \
                --seed $SEED \
                --output_path $OUTPUT_PATH \
                --method $METHOD \
                --epochs_to_evaluate $EPOCHS_TO_EVALUATE \
                --hpo $HPO"

if [ "$ACTIVE_LEVELS" ]; then
    cmd+=" --active_levels $ACTIVE_LEVELS"
fi


if [ "$HPO" = "false" ] && { [ "$METHOD" = "local" ] || [ "$METHOD" = "local_constrained" ] || [ "$METHOD" = "local_mask" ]; }; then
        cmd+=" \
            --lr_values ${LR_VALUES[@]} \
            --dropout_values ${DROPOUT_VALUES[@]} \
            --hidden_dims ${HIDDEN_DIMS[@]} \
            --num_layers_values ${NUM_LAYERS_VALUES[@]} \
            --weight_decay_values ${WEIGHT_DECAY_VALUES[@]}"
fi
if [ "$DATASET" = "all" ]; then
    MAX_JOBS=6
    current_jobs=0

    for dataset in "${datasets[@]}"; do
        cmd="$cmd --datasets $dataset"

        echo "Running: $cmd"
        $cmd &

        current_jobs=$((current_jobs + 1))

        if (( current_jobs >= MAX_JOBS )); then
            wait -n
            current_jobs=$((current_jobs - 1))
        fi
    done
else
    cmd="$cmd --datasets $DATASET"

    echo "Running: $cmd"
    $cmd &
fi

TRAIN_PID=$!
trap "kill $TRAIN_PID" SIGINT SIGTERM
wait

echo "All experiments completed!"
