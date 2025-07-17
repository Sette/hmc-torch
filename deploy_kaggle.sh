#!/bin/bash

# Definições do servidor remoto
REMOTE_USER="root"
REMOTE_HOST="Kaggle"
REMOTE_PATH="/root/git/hmc-torch"
SCRIPT_TO_RUN="train.sh"  # Nome do script a ser executado remotamente
TMUX_SESSION="train_session"  # Nome da sessão tmux

# Verifica se foi passado um argumento (cuda ou cpu)
if [ "$#" -ne 1 ]; then
    echo "Uso: $0 [cuda | cpu]"
    exit 1
fi

PYTORCH_MODE=$1

if [ "$PYTORCH_MODE" != "cuda" ] && [ "$PYTORCH_MODE" != "cpu" ]; then
    echo "Erro: O argumento deve ser 'cuda' ou 'cpu'."
    exit 1
fi


# Pergunta ao usuário se deseja executar git pull antes de rodar o treinamento
echo -n "Do you want to clone the repository before starting training? (y/n): "
read GIT_CLONE_CHOICE
GIT_CLONE_CHOICE=$(echo "$GIT_CLONE_CHOICE" | tr '[:upper:]' '[:lower:]')  # Converte para minúsculas

if [ "$GIT_CLONE_CHOICE" = "y" ]; then
    echo "Executando git clone no servidor..."
    ssh "$REMOTE_HOST" "
        export GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null'
        git clone -b develop --single-branch https://github.com/Sette/hmc-torch.git $REMOTE_PATH &&
        cd $REMOTE_PATH &&
        git config core.sshCommand 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' \
    "
else
    echo "Pulando git clone."
fi



# Pergunta ao usuário se deseja executar git pull antes de rodar o treinamento
echo -n "Do you want to run git pull before starting training? (y/n): "
read GIT_PULL_CHOICE
GIT_PULL_CHOICE=$(echo "$GIT_PULL_CHOICE" | tr '[:upper:]' '[:lower:]')  # Converte para minúsculas

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


# Ask the user if they want to create a local virtual environment
echo -n "Do you want to create a local venv? (y/n): "
read INSTALL_VENV_CHOICE
INSTALL_VENV_CHOICE=$(echo "$INSTALL_VENV_CHOICE" | tr '[:upper:]' '[:lower:]')
# Check if the user wants to install Poetry using pip
if [ "$INSTALL_VENV_CHOICE" = "y" ]; then
    ssh "$REMOTE_HOST" "
        source ~/.bashrc &&
        apt update &&
        apt install -y python3-venv &&
        cd $REMOTE_PATH &&
        python3 -m venv .venv \
    "
else
    echo "Skipping venv installation."
fi



# Ask the user if they want to install packages with poetry
echo -n "Do you want to Poetry with pip? (y/n): "
read INSTALL_POETRY_CHOICE
INSTALL_POETRY_CHOICE=$(echo "$INSTALL_POETRY_CHOICE" | tr '[:upper:]' '[:lower:]')

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


echo -n "Do you want to install dependencies with Poetry? (y/n): "
read INSTALL_CHOICE
INSTALL_CHOICE=$(echo "$INSTALL_CHOICE" | tr '[:upper:]' '[:lower:]')

if [ "$INSTALL_CHOICE" = "y" ]; then
    echo "Instalando dependências com Poetry..."
    ssh "$REMOTE_HOST" "
        source ~/.bashrc &&
        cd $REMOTE_PATH &&
        source .venv/bin/activate &&
        poetry source add pytorch --priority=explicit &&
        poetry source remove pytorch-cpu || true &&
        poetry install --no-root \
    "
else
    echo "Pulando instalação de dependências."
fi



echo "Executando o script de treinamento diretamente..."
echo "Usando PyTorch com CUDA..."
ssh "$REMOTE_HOST" "
    cd $REMOTE_PATH &&
    source .venv/bin/activate &&
    chmod +x $SCRIPT_TO_RUN && ./$SCRIPT_TO_RUN --device cuda --dataset_path /kaggle/input/gene-ontology-original --method local \
    "
