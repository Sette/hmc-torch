#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

# Lista de datasets
# datasets=('cellcycle_GO' 'derisi_GO' 'eisen_GO' 'expr_GO' 'gasch1_GO'
#           'gasch2_GO' 'seq_GO' 'spo_GO' 'cellcycle_FUN' 'derisi_FUN'
#           'eisen_FUN' 'expr_FUN' 'gasch1_FUN' 'gasch2_FUN' 'seq_FUN' 'spo_FUN')


DATASETS="spo_GO cellcycle_FUN derisi_FUN eisen_FUN expr_FUN gasch1_FUN gasch2_FUN seq_FUN spo_FUN"


DATASET_PATH="./data"
BATCH_SIZE=64
NON_LIN="relu"
DEVICE="cpu"
EPOCHS=2000
EPOCHS_TO_EVALUATE=10
OUTPUT_PATH="results"
METHOD="local"
SEED=0
DATASET_TYPE="arff"
HPO="false"
REMOTE="false"
N_TRIALS=30



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
    echo "  --n_trials <num>          Numer of HPO trials (default: $N_TRIALS)"
    echo "  --method <method>         Training method (default: $METHOD)"
    echo "  --hpo <true/false>        Hyperparameter optimization (default: $HPO)"
    echo "  --remote <yes/no>         Execute on remote server (default: $REMOTE)"
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
        --n_trials) N_TRIALS="$2"; shift ;;
        --method) METHOD="$2"; shift ;;
        --hpo) HPO="$2"; shift ;;
        --remote) REMOTE="$2"; shift ;;
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
                --hpo $HPO \
                --n_trials $N_TRIALS" \



if [ "$DATASET" = "all" ]; then
    for dataset_local in $DATASETS; do
        HIDDEN_DIMS=$(yq '.datasets_params.'"$dataset_local"'.hidden_dims[]' config.yaml | xargs)
        LR_VALUES=$(yq '.datasets_params.'"$dataset_local"'.lr_values[]' config.yaml | xargs)
        DROPOUT_VALUES=$(yq '.datasets_params.'"$dataset_local"'.dropout_values[]' config.yaml | xargs)
        NUM_LAYERS_VALUES=$(yq '.datasets_params.'"$dataset_local"'.num_layers_values[]' config.yaml | xargs)
        WEIGHT_DECAY_VALUES=$(yq '.datasets_params.'"$dataset_local"'.weight_decay_values[]' config.yaml | xargs)

        echo "Using dataset: $dataset_local"
        echo "Using hidden dimensions: $HIDDEN_DIMS"
        cmd_dataset=$cmd
        if [ "$ACTIVE_LEVELS" ]; then
            cmd_dataset=cmd+" --active_levels $ACTIVE_LEVELS"
        fi

        if [ "$HPO" = "false" ] && { [ "$METHOD" = "local" ] || [ "$METHOD" = "local_constrained" ] || [ "$METHOD" = "local_mask" ]; }; then
            cmd_dataset+=" --lr_values ${LR_VALUES[@]} \
                --dropout_values ${DROPOUT_VALUES[@]} \
                --hidden_dims ${HIDDEN_DIMS[@]} \
                --num_layers_values ${NUM_LAYERS_VALUES[@]} \
                --weight_decay_values ${WEIGHT_DECAY_VALUES[@]}"
        fi


        echo "Starting experiment for dataset: $dataset_local"
        TRAIN_PID=$!
        cmd_dataset+=" --datasets $dataset_local"
        echo "Running: $cmd_dataset"
        $cmd_dataset
        trap "kill $TRAIN_PID" SIGINT SIGTERM
        wait

    done
else
    echo "Using specific dataset: $DATASET"
    HIDDEN_DIMS=$(yq '.datasets_params.'"$DATASET"'.hidden_dims[]' config.yaml | xargs)
    LR_VALUES=$(yq '.datasets_params.'"$DATASET"'.lr_values[]' config.yaml | xargs)
    DROPOUT_VALUES=$(yq '.datasets_params.'"$DATASET"'.dropout_values[]' config.yaml | xargs)
    NUM_LAYERS_VALUES=$(yq '.datasets_params.'"$DATASET"'.num_layers_values[]' config.yaml | xargs)
    WEIGHT_DECAY_VALUES=$(yq '.datasets_params.'"$DATASET"'.weight_decay_values[]' config.yaml | xargs)

    echo "Using hidden dimensions: $HIDDEN_DIMS"

    cmd="$cmd --datasets $DATASET"

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

    TRAIN_PID=$!
    echo "Running: $cmd"
    $cmd

    trap "kill $TRAIN_PID" SIGINT SIGTERM
    wait

    echo "All experiments completed!"

fi

