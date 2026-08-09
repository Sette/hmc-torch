# Plano: Remover disables do pylint um a um

## Contexto

O `pyproject.toml` tem 10 regras desabilitadas no pylint. Dessas, 3 são **permanentes**
(falsos positivos ou padrões intencionais do projeto) e 7 podem ser **corrigidas**.

---

## Regras que DEVEM ficar desabilitadas (sem solução prática)

### E1101 — no-member
**Motivo**: Falsos positivos do pylint com atributos dinâmicos de numpy/torch
(`.cuda()`, `.float()`, `.to(device)`, etc.). Estão em todo o código e não há
como "consertar" — é limitação do pylint com código de ML.
**Ação**: Manter no disable.

### R0801 — duplicate-code
**Motivo**: Os dataset managers (arxiv, wos, aapd, rcv1, eurlex) compartilham
deliberadamente a mesma lógica de cache de embeddings. O pattern é intencional.
**Ação**: Manter no disable.

### W0201 — attribute-defined-outside-init
**Motivo**: Padrão `_fit()` usado em todos os managers (atributos populados após
`__init__`, via lazy initialization). É o design pattern do projeto.
**Ação**: Manter no disable.

---

## Regras a CORRIGIR (uma por vez, do mais fácil ao mais difícil)

### Fase 1: W0718 — broad-exception-caught (~12 ocorrências)

Já corrigimos vários. Restam nos scripts de download e em `eurlex/manager.py`.

**Arquivos a corrigir**:
- `src/hmc/datasets/arxiv/download_arxiv.py`: `except Exception` em download
- `src/hmc/datasets/wos/download_wos.py`: `except Exception` em extração
- `src/hmc/datasets/aapd/download_aapd.py`: `except Exception` em download
- `src/hmc/datasets/rcv1/download_rcv1.py`: `except Exception`
- `src/hmc/datasets/eurlex/download_eurlex.py`: `except Exception` (3x)
- `src/hmc/datasets/eurlex/manager.py`: `except Exception` em JSON parse

**Padrão**: Trocar `except Exception` por exceções específicas (`OSError`,
`ImportError`, `json.JSONDecodeError`, `urllib.error.URLError`).

### Fase 2: W0613 — unused-argument (~3 ocorrências)

**Arquivos a corrigir**:
- `src/hmc/pipeline/global_classifier/main.py:130`: `model_name` em `_get_transformer_dataset`
- `src/hmc/models/global_classifier/e2e/model.py`: `**kwargs` em `forward()`

**Padrão**: Prefixar com `_` (ex: `_model_name`) ou usar `**kwargs` com `# pylint: disable=unused-argument` inline.

### Fase 3: W0404 — reimport (~3 ocorrências)

**Arquivos a corrigir**:
- `src/hmc/datasets/wos/dataset_wos.py`: `torch` reimportado nas linhas 240, 271; `numpy` na linha 277

**Padrão**: Remover reimports e usar os imports do topo do arquivo.

### Fase 4: R0912 — too-many-branches (1 ocorrência)

**Arquivo a corrigir**:
- `src/hmc/datasets/eurlex/dataset_eurlex.py:131`: 14 branches

**Padrão**: Extrair lógica de URL resolution para função auxiliar.

### Fase 5: C0415 — import-outside-toplevel (~10 ocorrências)

Já adicionamos `# pylint: disable=import-outside-toplevel` inline em vários
lugares. Restam alguns sem o comentário.

**Arquivos a verificar**: Fazer grep por `import` dentro de funções e garantir
que cada um tem o comentário inline.

**Padrão**: `from x import y  # pylint: disable=import-outside-toplevel`

### Fase 6: C0116 — missing-function-docstring (~20 ocorrências)

**Arquivos**: Distribuídos em vários módulos (`model.py`, `train.py`, etc.)

**Padrão**: Adicionar docstring curta (`"""Forward pass."""`) em métodos sem docstring.

### Fase 7: R0914 — too-many-locals (~3 ocorrências)

**Arquivos a corrigir**:
- `src/hmc/pipeline/tabular/main.py`: `train_tabular_mlp` (43 locals)
- `src/hmc/pipeline/local_classifier/main.py`: funções de treino

**Padrão**: Extrair grupos de variáveis relacionadas em funções auxiliares
(como feito em `train.py`).

---

## Ordem de execução

1. W0718 → remover do disable (fácil, ~12 arquivos)
2. W0613 → remover do disable (fácil, ~3 arquivos)
3. W0404 → remover do disable (fácil, ~3 arquivos)
4. R0912 → remover do disable (médio, 1 arquivo)
5. C0415 → remover do disable (fácil, verificar ~10 imports)
6. C0116 → remover do disable (médio, ~20 docstrings)
7. R0914 → remover do disable (médio, 3 refactors)

Ao final: 3 regras permanentes no disable (E1101, R0801, W0201).
