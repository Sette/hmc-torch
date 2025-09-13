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

if [ "$RUN_ONLY" = false ]; then


    if [ "$AUTO_YES" = true ]; then
        GIT_CLONE_CHOICE="y"
    else
        # Pergunta ao usuário se deseja executar git pull antes de rodar o treinamento
        echo -n "Do you want to clone the repository before starting training? (y/n): "
        read GIT_CLONE_CHOICE
        GIT_CLONE_CHOICE=$(echo "$GIT_CLONE_CHOICE" | tr '[:upper:]' '[:lower:]')  # Converte para minúsculas

    fi

    if [ "$GIT_CLONE_CHOICE" = "y" ]; then
        echo "Executando git clone no servidor..."
        ssh "$REMOTE_HOST" "
            export GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null'
            git clone -b feature/new-early-stopping --single-branch https://github.com/Sette/hmc-torch.git $REMOTE_PATH &&
            cd $REMOTE_PATH &&
            git config core.sshCommand 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' \
        "
    else
        echo "Pulando git clone."
    fi

    if [ "$AUTO_YES" = true ]; then
        GIT_PULL_CHOICE="y"
    else
        # Pergunta ao usuário se deseja executar git pull antes de rodar o treinamento
        echo -n "Do you want to run git pull before starting training? (y/n): "
        read GIT_PULL_CHOICE
        GIT_PULL_CHOICE=$(echo "$GIT_PULL_CHOICE" | tr '[:upper:]' '[:lower:]')  # Converte para minúsculas
    fi


    if [ "$GIT_PULL_CHOICE" = "y" ]; then
        echo "Executando git pull no servidor..."
        ssh "$REMOTE_HOST" "
            export GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null'
            cd $REMOTE_PATH &&
            git pull \
        "
    else
        echo "Pulando git pull."
    fi

    if [ "$AUTO_YES" = true ]; then
        INSTALL_VENV_CHOICE="y"
    else
        # Pergunta ao usuário se deseja criar um ambiente virtual local
        echo -n "Do you want to create a local venv? (y/n): "
        read INSTALL_VENV_CHOICE
        INSTALL_VENV_CHOICE=$(echo "$INSTALL_VENV_CHOICE" | tr '[:upper:]' '[:lower:]')  # Converte para minúsculas
    fi

    if [ "$INSTALL_VENV_CHOICE" = "y" ]; then
        ssh "$REMOTE_HOST" "
            source ~/.bashrc &&
            apt update &&
            apt install -y python3-venv python3.11-venv &&
            cd $REMOTE_PATH &&
            pip install --upgrade virtualenv &&
            python -m venv .venv \
        "
    else
        echo "Skipping venv installation."
    fi

    if [ "$AUTO_YES" = true ]; then
        INSTALL_POETRY_CHOICE="y"
    else
        # Pergunta ao usuário se deseja instalar o Poetry
        echo -n "Do you want to install Poetry? (y/n): "
        read INSTALL_POETRY_CHOICE
        INSTALL_POETRY_CHOICE=$(echo "$INSTALL_POETRY_CHOICE" | tr '[:upper:]' '[:lower:]')  # Converte para minúsculas
    fi

    # Check if the user wants to install Poetry using pip
    if [ "$INSTALL_POETRY_CHOICE" = "y" ]; then
        echo "Installing Poetry with pip..."
        python3 -m pip install poetry
        ssh "$REMOTE_HOST" "
            source ~/.bashrc &&
            cd $REMOTE_PATH &&
            source .venv/bin/activate &&
            pip install pip --upgrade &&
            pip install --upgrade pip setuptools wheel &&
            pip install poetry \
        "
    else
        echo "Skipping Poetry installation."
    fi

    if [ "$AUTO_YES" = true ]; then
        INSTALL_CHOICE="y"
    else
        # Pergunta ao usuário se deseja instalar as dependências com Poetry
        echo -n "Do you want to install dependencies with Poetry? (y/n): "
        read INSTALL_CHOICE
        INSTALL_CHOICE=$(echo "$INSTALL_CHOICE" | tr '[:upper:]' '[:lower:]')
    fi

    if [ "$INSTALL_CHOICE" = "y" ]; then
        echo "Instalando dependências com Poetry..."
        ssh "$REMOTE_HOST" "
            source ~/.bashrc &&
            cd $REMOTE_PATH &&
            source .venv/bin/activate &&
            poetry source add pytorch --priority=explicit &&
            poetry source remove pytorch-cpu || true &&
            poetry add \
              https://storage.googleapis.com/libtpu-nightly-releases/wheels/libtpu-nightly/libtpu_nightly-0.1.dev20210615-py3-none-any.whl &&
            poetry install --no-root \
        "
    else
        echo "Pulando instalação de dependências."
    fi
fi

echo "Executando o script de treinamento diretamente..."
echo "Usando PyTorch com CUDA..."
ssh "$REMOTE_HOST" "
    add-apt-repository ppa:rmescandon/yq -y &&
    apt update && apt install yq -y &&
    cd $REMOTE_PATH &&
    source .venv/bin/activate &&
    chmod +x $SCRIPT_TO_RUN && ./$SCRIPT_TO_RUN --device cuda \
        --dataset_path /kaggle/input/gene-ontology-original \
        --output_path  /kaggle/working/results \
        --method local \
        --epochs_to_evaluate 10 \
        --hpo true \
        --remote true \
        --dataset_type arff \
        --n_trials 30 \
        --dataset all \
    "
