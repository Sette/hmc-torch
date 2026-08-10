# CLAUDE.md

Guia de desenvolvimento para o projeto **hmc-torch** (Hierarchical Multi-Label Classification).

Datasets suportados: **ArXiv** e **WOS** (Web of Science).

---

## Comandos essenciais

```bash
export PYTHONPATH=src          # obrigatório antes de qualquer execução

. .venv/bin/activate           # ativar virtualenv
uv sync --all-groups           # instalar/atualizar dependências

# Download datasets
make download-all              # arxiv + wos
make download-arxiv            # apenas ArXiv (via kagglehub)
make download-wos              # apenas WOS

# Treino rápido
./run.sh                       # wos, global, cuda
./run.sh --dataset_name arxiv --method globalE2E --epochs 5 --batch_size 4

# Python direto
python -m hmc.main --dataset_name wos --method global --device cuda \
  --dataset_path ./data --output_path ./output --epochs 50 --batch_size 32

make lint                      # autopep8 + black + ruff + isort + pylint
make test                      # pytest com coverage
```

---

## Arquitetura

### Fluxo de execução

```
main.py → parse_args() → Args dataclass
            │
            ├── method=global       → train_global()       — frozen embeddings + MLP + R-matrix
            ├── method=globalE2E    → train_global_e2e()   — fine-tuned transformer + MLP + R-matrix
            ├── method=globalSOTA   → train_global_sota()  — fine-tuned transformer + GCN label graph + R-matrix
            └── method=local        → train_local()        — frozen embeddings, one MLP per level
```

### Datasets

| Dataset | Manager | Features | Split | Hierarquia |
|---|---|---|---|---|
| `arxiv` | `ArXivManager` | SPECTER2 (768d, CLS-pool) | 64/16/20 (HPT) | 19 áreas + 137 subcats |
| `wos` | `WOSManager` | SPECTER2 (768d, CLS-pool) | 64/16/20 (HPT) | 7 áreas + 134 subcats |

Ambos usam embeddings de transformer (SPECTER2-base por padrão, configurável via `--arxiv_model_name`). Feature cache com hash MD5 invalida automaticamente.

### Modelos

| Classe | Arquivo | Uso |
|---|---|---|
| `ConstrainedModel` | `models/global_classifier/constraint/model.py` | MLP + R-matrix (frozen) |
| `ConstrainedGNNModel` | idem | MLP + R-matrix + GCN label graph |
| `E2EConstrainedModel` | `models/global_classifier/e2e/model.py` | Transformer fine-tuned + MLP + R-matrix |
| `E2EGNNModel` | idem | Transformer fine-tuned + GCN + R-matrix |
| `LocalModel` | `models/local_classifier/model.py` | One MLP per level, frozen embeddings |

**R-matrix**: matriz de ancestralidade (NetworkX). `r_matrix[i, j] = 1` se `j` é ancestral de `i`. Aplicada via `get_constr_out()`.

### Registry

`DatasetRegistry` em `datasets/registry.py` — hiperparâmetros padrão por dataset:
- `arxiv_defaults`: hidden=512, lr=1e-4, epochs=50, dropout=0.3
- `wos_defaults`: mesmos valores

---

## Adicionando um novo dataset

### Usando dados próprios (sem modificar o pacote)

Use `build_digraph_from_labels` + `TreeHierarchy.from_graph` — o framework
deriva tudo (adjacência, levels, edge_index, dimensões) das labels:

```python
from hmc.utils import build_digraph_from_labels
from hmc.data.hierarchy import TreeHierarchy
from hmc.data import DatasetRegistry, Split

# 1. Seus dados
texts, label_strs = load_seus_dados()  # label_strs: ["cs.AI", "stat.ML cs.LG", ...]

# 2. Hierarquia derivada das labels
h = TreeHierarchy.from_graph(build_digraph_from_labels(label_strs))

# 3. Features + encode
X = seu_encoder(texts)  # shape (n, d)
Y_global, Y_local_all = h.encode_labels(label_strs)
Y_local = Y_local_all[1:]  # drop root

# 4. Splits + manager (ver README.md "Register your own dataset")
```

### Adicionando um dataset built-in (texto + transformer)

1. Criar manager em `datasets/<nome>/manager.py` implementando:
   - `get_datasets()` → (train, valid, test) com `.x`, `.y`, `.y_local`, `.samples`
   - `levels_size`, `max_depth`, `a`, `edge_index`, `nodes_idx`, `to_eval`, `input_dim`, `output_dim`
2. Adicionar dispatch em `dataset_manager.py`
3. Adicionar defaults em `registry.py`
4. Adicionar ao `download_all.py`
