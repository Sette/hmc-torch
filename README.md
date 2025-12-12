
# HMC Torch Project Summary

## Overview
This document summarizes the HMC Torch project, a hierarchical multi-label classification network implemented in PyTorch. The project is currently at version 0.0.1 and includes various Jupyter notebooks, scripts, and configuration files.

## Key Updates
- **Version**: 0.0.1
- **Main Changes**:
  - Removed the main function from the training file to simplify the code structure.
  - Updated `.gitignore` to improve version control management.

## Project Structure
This section outlines the key components of the project to help users navigate the codebase.
The project contains the following key files and directories:
- **Notebooks**:
  - `Dataset.ipynb`: Handles dataset loading and preprocessing.
  - `Executer-model.ipynb`: Contains the model execution logic.
  - `Inference.ipynb`: Used for making predictions with the trained model.
- **Scripts**:
  - `executer.py`: Core execution script for the model.
- **Configuration**:
  - `pyproject.toml`: Project configuration file.
  - `poetry.lock`: Dependency lock file.
- **Documentation**:
  - `README.md`: Provides an overview and instructions for the project.
  - `LICENSE`: Licensing information for the project.

## Prerequisites Installation

Before setting up the project, ensure you have the following prerequisites installed and configured:

### 1. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies. Run the following command to create one:

```bash
python -m venv .venv
```

#### Activation Steps:
- **Command Prompt (Windows):**
  ```bash
  source .\.venv\Scripts\activate
  ```
- **PowerShell (Windows):**
  ```bash
  .\.venv\Scripts\Activate.ps1
  ```
- **Linux/MacOS:**
  ```bash
  source .venv/bin/activate
  ```

### 2. Install Poetry
Poetry is used for dependency management. Install it using pip or another method:

```bash
pip install poetry
```

### 3a. GPU Setup (Optional)
If you plan to use a GPU for training, configure the project with the following commands:

```bash
poetry source add pytorch-gpu https://download.pytorch.org/whl/cu118 --priority=explicit &&
poetry source remove pytorch-cpu || true
```

### 3b. CPU Setup (Optional)
If you plan to use a CPU for training, configure the project with the following commands:

```bash
poetry source add pytorch-cpu https://download.pytorch.org/whl/cpu --priority=explicit &&
poetry source remove pytorch-gpu || true

```


These steps will ensure that the project is ready for GPU-based execution.

### 4. Install Dependencies
Once the virtual environment is activated and Poetry is installed, run the following command to install all project dependencies:

```bash
poetry install --no-root --with dev
```

By completing these steps, your environment will be fully prepared to run the HMC Torch project.

### 5. Download dataset Using Kaggle (Optional)

##### 1. Download dataset via Kaggle CLI
```bash
pip install kaggle
```

```bash
kaggle datasets download brunosette/gene-ontology-original
```


##### 2. Download dataset via curl
```bash
# Export your Kaggle username and API key
# export KAGGLE_USERNAME=<YOUR USERNAME>
# export KAGGLE_KEY=<YOUR KAGGLE KEY>

curl -L -u $KAGGLE_USERNAME:$KAGGLE_KEY\
  -o ~/Downloads/gene-ontology-original.zip\
  https://www.kaggle.com/api/v1/datasets/download/brunosette/gene-ontology-original

```

```bash
mkdir data
```

```bash
unzip gene-ontology-original.zip -d data/
```

### 5. Running the Project

To execute the training process, follow these steps:

#### 5.1. Make the Script Executable
Before running the training script, ensure it has the necessary execution permissions:

```bash
chmod +x run.sh
```

#### 5.2. Run the Training Script
You can run the training script with the desired device configuration:

- **For CPU Execution**:
  ```bash
  ./run.sh --device cpu
  ```

- **For GPU Execution**:
  ```bash
  ./run.sh --device gpu
  ```

These commands will initiate the training process using the specified hardware.

#### 5.3. Deploy Locally
To deploy the project locally, you only need to run the deployment script, as it automates all the steps outlined above. Ensure the script has execution permissions and specify the desired hardware configuration:

1. Make the script executable:
  ```bash
  chmod +x deploy_local.sh
  ```

2. Run the deployment script:
  - **For GPU Deployment**:
    ```bash
    ./deploy_local cuda
    ```
  - **For CPU Deployment**:
    ```bash
    ./deploy_local cpu
    ```

By following these instructions, the deployment script will handle the setup process, including dependency installation and environment configuration, based on your hardware preferences.


## Commit History
The project has a total of 8 commits, with the latest updates made on August 25, 2024. Notable commits include:
- **Remove main func in train file**: Simplified the training process.
- **Update project**: General updates across various files.

## Conclusion
The HMC Torch project is structured to facilitate the development and implementation of hierarchical multi-label classification models using PyTorch. The recent updates have streamlined the codebase and improved project organization.
