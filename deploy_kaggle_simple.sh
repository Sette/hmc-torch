#!/bin/bash

# Definições do servidor remoto
REMOTE_USER="root"
REMOTE_HOST="Kaggle"
REMOTE_PATH="/root/git/hmc-torch"
SCRIPT_TO_RUN="train.sh"  # Nome do script a ser executado remotamente
TMUX_SESSION="train_session"  # Nome da sessão tmux
AUTO_YES=false
RUN_ONLY=false
PYTORCH_MODE=""

for arg in "$@"; do
    case "$arg" in
        --all)
            AUTO_YES=true
            ;;
        --run)
            RUN_ONLY=true
            ;;
        cuda|cpu)
            PYTORCH_MODE=$arg
            ;;
        *)
            echo "Argumento desconhecido: $arg"
            echo "Uso: $0 [cuda | cpu] [--all] [--run]"
            exit 1
            ;;
    esac
done

if [ "$PYTORCH_MODE" != "cuda" ] && [ "$PYTORCH_MODE" != "cpu" ]; then
    echo "Erro: O argumento deve ser 'cuda' ou 'cpu'."
    exit 1
fi

echo "Executando git clone no servidor..."
ssh "$REMOTE_HOST" "
    export GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' &&
    git clone -b feature/new-early-stopping --single-branch https://github.com/Sette/hmc-torch.git $REMOTE_PATH &&
    cd $REMOTE_PATH &&
    git pull &&
    source ~/.bashrc &&
            apt update &&
            apt install -y python3-venv &&
            cd $REMOTE_PATH &&
            python3 -m venv .venv &&
    source .venv/bin/activate &&
            pip install pip --upgrade &&
            pip install --upgrade pip setuptools wheel &&
            pip install poetry &&
    chmod +x $SCRIPT_TO_RUN && ./$SCRIPT_TO_RUN --device cuda \
        --dataset_path /kaggle/input/gene-ontology-parsed \
        --output_path  /kaggle/working/results \
        --method local \
        --epochs_to_evaluate 5 \
        --hpo true \
        --remote true \
        --dataset_type csv \
        --n_trials 30 \
        --dataset all \
    "