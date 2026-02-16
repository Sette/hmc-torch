#!/bin/zsh

sudo apt-get install jq yq -y


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



# Ask the user if they want to create a local virtual environment
echo -n "Do you want to create a local venv? (y/n): "
read INSTALL_VENV_CHOICE
INSTALL_VENV_CHOICE=$(echo "$INSTALL_VENV_CHOICE" | tr '[:upper:]' '[:lower:]')
# Check if the user wants to install Poetry using pip
if [ "$INSTALL_VENV_CHOICE" = "y" ]; then
    echo "Creeating venv with python..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "Venv created and activated."
    echo "Installing pip..."
    python3 -m pip install --upgrade pip
    echo "Pip installed."
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
else
    echo "Skipping Poetry installation."
fi

echo -n "Do you want to install dependencies with Poetry? (y/n): "
read INSTALL_CHOICE
INSTALL_CHOICE=$(echo "$INSTALL_CHOICE" | tr '[:upper:]' '[:lower:]')


echo "Executando o script de treinamento diretamente..."
source ~/.zshrc
if [ "$INSTALL_CHOICE" = "y" ] && [ "$PYTORCH_MODE" = "cuda" ]; then
    poetry source add pytorch-gpu https://download.pytorch.org/whl/cu118 --priority=explicit &&
    poetry source remove pytorch-cpu || true &&
    poetry install --no-root &&
    chmod +x run.sh
fi

if [ "$INSTALL_CHOICE" = "y" ] && [ "$PYTORCH_MODE" = "cpu" ]; then
    poetry source add pytorch-cpu https://download.pytorch.org/whl/cpu --priority=explicit &&
    poetry source remove pytorch-gpu || true &&
    poetry install --no-root &&
    chmod +x run.sh
fi

