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
# ACTIVE_LEVELS="0"
# HIDDEN_DIMS="356 451 184 300 216 253"
# LR_VALUES="0.00044766011460352974 0.00014979270540666786 0.000979239810055003 0.00011945129570958105 1.0158886816263624e-06 0.0003857168137677134"
# DROPOUT_VALUES="0.5423049728240137 0.6259684313528631 0.5637537833386804 0.7472570730186732 0.4045542102539288 0.4248880393091625"
# NUM_LAYERS_VALUES="1 1 1 1 3 1"
# WEIGHT_DECAY_VALUES="2.5275292389739167e-06 0.008477855117302344 0.008931270081837467 8.049371645772185e-05 1.360066713089424e-05 0.004867224985511117"


HIDDEN_DIMS="494 475 85 174 311 466" 
LR_VALUES="4.687489776721649e-06 1.4351257097300674e-05 2.829538308724334e-05 0.00022043552720109737 0.00016655940276079522 0.0005482732851516332" 
DROPOUT_VALUES="0.4217063347487683 0.39093261149753145 0.30013555073078557 0.30640277029661966 0.7039046471246466 0.6954921515177706" 
NUM_LAYERS_VALUES="1 2 2 1 3 2" 
WEIGHT_DECAY_VALUES="0.00035758347745051426 7.941410135074069e-05 5.964375732677727e-06 7.137680057167304e-06 1.2610023807852354e-06 3.6516906700827284e-06"
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

 cmd="python -m hmc.train.main \
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
