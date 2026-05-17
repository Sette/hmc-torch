# CLAUDE.md

Guia de desenvolvimento para o projeto **hmc-torch** (Hierarchical Multi-Label Classification).

---

## Comandos essenciais

```bash
export PYTHONPATH=src          # obrigatório antes de qualquer execução

make test                      # pytest com coverage
make lint                      # autopep8 + black + ruff + isort + pylint
make run                       # ./run.sh com seq_FUN, method=local, device=cuda

uv sync --all-groups           # instalar/atualizar dependências
```

---

## Arquitetura

### Fluxo de execução

```
main.py → parse_args() → Args dataclass
            │
            ├── method=global*  → pipeline/global_classifier/main.py:train_global()
            │                       └── core/train.py: _run_training_loop → _collect_test_outputs → _compute_local_scores
            │
            └── method=local*   → pipeline/local_classifier/main.py:main_local()
                                    ├── train_local()
                                    │     ├── HPO: hpo/hpo_local.py:optimize_hyperparameters()
                                    │     └── sem HPO: core/train.py:train_step()
                                    │           └── core/validate.py:validate_step() a cada epochs_to_evaluate épocas
                                    └── test_local() → core/predict.py:test_step()
```

### Configuração (`Args` dataclass)

Definida em `src/hmc/arguments.py`. Organizada em grupos aninhados:

- `DatasetConfig` — paths, dataset_name, device
- `TrainingConfig` — epochs, batch_size, epochs_to_evaluate, warmup, early_metric
- `HyperparameterConfig` — lr_values, dropout_values, hidden_dims, num_layers_values, weight_decay_values
- `HpoConfig` — hpo, n_trials, output_path

`parse_args()` converte o namespace do argparse para o dataclass. Acesso transparente: `args.epochs` delega para `args.training.epochs` automaticamente. Todos os booleans são `bool` (não string).

O objeto `args` recebe atributos dinâmicos ao longo da execução (ex: `args.hmc_dataset`, `args.model`, `args.train_loader`). Isso é intencional — dataclasses Python permitem atributos extras sem `__slots__`.

### Datasets

Nome do dataset segue o padrão `{data}_{ontology}`, ex: `seq_FUN`, `expr_GO`.

- `data`: nome do experimento biológico (cellcycle, derisi, eisen, expr, gasch1, gasch2, seq, spo)
- `ontology`: `FUN` (função) ou `GO` (Gene Ontology)
- Datasets "others": diatoms, enron, imclef07a, imclef07d (sem split data/ontology para dimensões)
- Datasets ArXiv: `arxiv` — hierarquia de categorias cs.AI, cs.LG, etc.

O split `args.data, args.ontology = dataset_name.split("_")` é feito no início de cada pipeline (não se aplica a "others" nem arxiv).

Carregamento: `datasets/dataset_manager.py:initialize_dataset_experiments()`  
- Para ARFF (FUN/GO/others): usa `datasets/gofun/manager.py:HMCDatasetManager`  
- Para ArXiv: usa `datasets/arxiv/dataset_arxiv.py:ArXivHierarchyManager` + `ArXivPyTorchDataset`

Dimensões por dataset registradas em `datasets/registry.py:DatasetRegistry` (não em `main.py`).

### Modelos

| Classe | Arquivo | Uso |
|---|---|---|
| `HierarchicalModel` | `models/base.py` | Classe abstrata base para todos os modelos |
| `ConstrainedModel` | `models/global_classifier/constraint/model.py` | Global — MLP com R-matrix |
| `ConstrainedLightningModel` | idem | Global com PyTorch Lightning (método `globalLM`) |
| `HMCLocalModel` | `models/local_classifier/baseline/model.py` | Local — dict de MLPs por nível |
| `ClassificationNetwork` / `BuildClassification` | `models/local_classifier/networks.py` | Blocos de rede compartilhados (MLP, GCN, GAT) |

**R-matrix** (global): matriz de ancestralidade calculada com NetworkX. `r_matrix[i, j] = 1` se `j` é ancestral de `i`. Aplicada via `get_constr_out()` em `models/global_classifier/constraint/utils.py`.

### Hyperparâmetros

- **Com HPO** (`--hpo true`): Optuna roda por nível em `hpo/hpo_local.py`. Resultados salvos em `{output_path}/hpo/...`
- **Sem HPO**: lidos de `config.yaml` via `run.sh` (usando `yq`) e passados como `--lr_values`, `--dropout_values`, etc. Um valor por nível de hierarquia.

`config.yaml` tem os melhores hiperparâmetros já encontrados para cada dataset.

---

## Convenções do projeto

### Estrutura de diretórios de saída

```
{output_path}/
├── train/
│   └── {method}-{dataset_name}/
│       └── {job_id}/
│           ├── best_model_level_0.pth
│           ├── best_model_level_1.pth
│           └── {job_id}.json          ← scores finais
└── hpo/
    └── {method}/{dataset_name}/{job_id}/
        ├── best_params_{dataset_name}-{level}.json
        └── best_params_{dataset_name}.json
```

### Early stopping

Local: por nível, monitorando `f1-score` ou `avg-score` (configurável via `--early_metric`).  
Controle em `utils/train/early_stopping.py:check_early_stopping_normalized()`.  
Quando um nível para, seus parâmetros são congelados (`requires_grad = False`).

### Flags booleanas no CLI

O CLI aceita `"true"`/`"false"` como strings (compatibilidade com `run.sh`). A conversão para `bool` acontece dentro de `parse_args()` em `arguments.py`. **Não usar** `parse_str_flags` — foi removida.

### Logging e ambiente

Configurado em `env.py`. Suporta controle de log por fonte (TRAIN, MODEL, DATASET) via variáveis de ambiente.

### Testes

Testes de integração em `tests/train_global_test.py` usam `mock.patch.object(sys, "argv", ...)` para simular args de linha de comando. Não mocam datasets — precisam dos dados em `./data`.

---

## Dependências principais

| Pacote | Uso |
|---|---|
| `torch` | Modelos, tensores, DataLoaders |
| `lightning` | `ConstrainedLightningModel` (método `globalLM`) |
| `optuna` | HPO por nível |
| `scikit-learn` | Normalização, métricas, SimpleImputer |
| `networkx` | Construção da R-matrix de ancestralidade |
| `transformers` | Tokenização para datasets ArXiv |
| `torch-geometric` | Suporte a GCN/GAT em `BuildClassification` |
| `yq` / `jq` | Leitura de `config.yaml` no `run.sh` |

---

## Adicionando um novo dataset

1. Colocar os arquivos ARFF em `data/{dataset_name}/`
2. Registrar dimensões em `datasets/registry.py:DatasetRegistry` (input_dims, output_dims, hidden_dims, lrs, all_epochs)
3. Rodar HPO: `./run.sh --dataset_name {nome} --hpo true --n_trials 50`
4. Copiar os melhores hiperparâmetros para `config.yaml`

## Adicionando um novo método local

1. Criar (ou reusar) modelo em `models/local_classifier/`
2. Registrar em `pipeline/local_classifier/main.py:get_train_methods()`
3. Adicionar o nome ao argparse em `arguments.py` (campo `method`, choices)
4. Adicionar ao `run.sh` na condição de leitura de hiperparâmetros
