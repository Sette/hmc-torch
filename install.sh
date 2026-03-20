#!/bin/zsh



# Function to display help
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Available options:"
    echo "  --os <os>                 OS (default: ubuntu)"
    echo "  --device <device>         Device (cuda/cpu) (default: cpu)"
    echo "  --help                    Display this help message"
    echo ""
    echo "Example:"
    echo "  $0 --os fedora --device cuda"
    echo "  $0 --os ubuntu --device cpu"
    echo "  $0 --help"
    echo ""
    exit 1
}

function install_python_global() {
  uv python install 3.12.11
  uv python pin 3.12.11
}

# Processamento dos argumentos
while [ "$#" -gt 0 ]; do
    case $1 in
    --os) OS="$2"; shift ;;
    --device) DEVICE="$2"; shift ;;
    --help) usage ;;
        *) echo "Invalid option: $1"; usage ;;
    esac
    shift
done

if [ "$OS" != "fedora" ] && [ "$DEVICE" != "ubuntu" ]; then
    echo "Erro: O argumento --os deve ser 'fedora' ou 'ubuntu'."
    exit 1
fi

if [ "$DEVICE" != "cuda" ] && [ "$DEVICE" != "cpu" ]; then
    echo "Erro: O argumento --device deve ser 'cuda' ou 'cpu'."
    exit 1
fi

# Ask the user if they want to create a local virtual environment
echo -n "Do you want to install uv? (y/n): "
read INSTALL_UV_CHOICE
INSTALL_UV_CHOICE=$(echo "$INSTALL_UV_CHOICE" | tr '[:upper:]' '[:lower:]')
# Check if the user wants to install Poetry using pip
if [ "$INSTALL_UV_CHOICE" = "y" ]; then
    echo "Installing uv..."
    if [ "$OS" = "ubuntu"]; then
        #Instalando UV
        echo "Instalando requirements system packages..."
        sudo apt install -y make curl build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
        check_success "sudo apt install"
    fi

    if [ "$OS" = "fedora"]; then
        # Install required system packages on Fedora using dnf
        sudo dnf install -y make curl gcc gcc-c++ openssl-devel bzip2-devel readline-devel sqlite-devel wget llvm ncurses-devel xz tk-devel libxml2-devel xmlsec1-devel libffi-devel lzma-sdk-devel
        check_success "sudo dnf install"
    fi

    # Install or upgrade uv - preferred method is via rpm or curl installer
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    check_success "curl -LsSf https://astral.sh/uv/install.sh"
    install_python_global

    export PATH="$HOME/.local/bin:$PATH"

else
    echo "Skipping uv installation."
fi



# Ask the user if they want to create a local virtual environment
echo -n "Do you want to create a local venv? (y/n): "
read INSTALL_VENV_CHOICE
INSTALL_VENV_CHOICE=$(echo "$INSTALL_VENV_CHOICE" | tr '[:upper:]' '[:lower:]')
# Check if the user wants to install Poetry using pip
if [ "$INSTALL_VENV_CHOICE" = "y" ]; then
    echo "Creeating venv with python..."
    uv venv
    source .venv/bin/activate
    echo "Venv created and activated."
else
    echo "Skipping venv installation."
fi



# Ask the user if they want to install packages with poetry
echo -n "Do you want to Poetry with pipx? (y/n): "
read INSTALL_POETRY_CHOICE
INSTALL_POETRY_CHOICE=$(echo "$INSTALL_POETRY_CHOICE" | tr '[:upper:]' '[:lower:]')

# Check if the user wants to install Poetry using pip
if [ "$INSTALL_POETRY_CHOICE" = "y" ]; then
    echo "Installing Poetry with pip and pipx..."
    uv pip install pipx
    pipx install poetry
else
    echo "Skipping Poetry installation."
fi

echo -n "Do you want to install dependencies with Poetry? (y/n): "
read INSTALL_CHOICE
INSTALL_CHOICE=$(echo "$INSTALL_CHOICE" | tr '[:upper:]' '[:lower:]')


echo "Executando o script de treinamento diretamente..."
source ~/.zshrc
if [ "$INSTALL_CHOICE" = "y" ] && [ "$DEVICE" = "cuda" ]; then
    poetry source remove pytorch
    poetry source add pytorch https://download.pytorch.org/whl/cu130 --priority=explicit &&
    poetry install --no-root &&
    chmod +x run.sh
fi

if [ "$INSTALL_CHOICE" = "y" ] && [ "$DEVICE" = "cpu" ]; then
    poetry source remove pytorch
    poetry source add pytorch https://download.pytorch.org/whl/cpu --priority=explicit &&
    poetry install --no-root &&
    chmod +x run.sh
fi

