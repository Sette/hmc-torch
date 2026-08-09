# Expansão de Dados — Novos Datasets Textuais para HMC-Torch

**Data:** 2026-08-04 | **Branch:** `release/0.0.9`

---

## Objetivo

Adicionar **3 novos datasets de texto** ao HMC-Torch para fortalecer o claim de
generalização do R-matrix em classificação hierárquica textual. Atualmente temos
apenas 2 datasets de texto (ArXiv, WOS) entre 25 totais.

---

## Datasets Selecionados

| # | Dataset | Docs | Labels | Prof. | Domínio | Prioridade |
|---|---------|------|--------|-------|---------|------------|
| 1 | **AAPD** | 55,840 | 54 (6+48) | 2 níveis, tree | CS papers | ⭐ Fácil, já temos SPECTER2 |
| 2 | **RCV1-V2** | 804K → 50K | 103 | 4 níveis, tree | Notícias | ⭐⭐ Benchmark padrão |
| 3 | **EUR-Lex 57K** | 57K | 4,271 | Profunda, tree | Legislação EU | ⭐⭐⭐ Testa sparse R |

### Por que estes?

1. **AAPD** (arXiv Academic Paper Dataset): Papers do arXiv com 54 labels (6 áreas + 48 subáreas).
   Texto de título + abstract disponível publicamente. Pipeline idêntico ao ArXiv/WOS.
   Ideal como *aquecimento* — mesma família SPECTER2, hierarquia simples.

2. **RCV1-V2** (Reuters Corpus): **O benchmark padrão** de HMC textual. Todo paper
   (C-HMCNN, HMCN-F, HiAGM, HTCInfoMax, HBGL) reporta nele. 103 tópicos em 4 níveis.
   Comparação direta com SOTA publicado.

3. **EUR-Lex 57K**: Legislação europeia com ~4K labels EUROVOC em hierarquia profunda.
   Testa o **sparse R-matrix** em texto (não só GO tabular). Domínio legal, diferente
   de papers científicos.

---

## Passos de Implementação

### Passo 1: AAPD (dataset mais simples)

```
[ ] 1.1 Criar src/hmc/datasets/aapd/__init__.py
[ ] 1.2 Criar src/hmc/datasets/aapd/dataset_aapd.py — hierarchy manager + split class
[ ] 1.3 Criar src/hmc/datasets/aapd/manager.py — AAPDManager (seguir ArXivManager)
[ ] 1.4 Criar src/hmc/datasets/aapd/download_aapd.py — download script
[ ] 1.5 Adicionar dispatch "aapd" em src/hmc/datasets/dataset_manager.py
[ ] 1.6 Adicionar aapd_defaults em src/hmc/datasets/registry.py
[ ] 1.7 Adicionar download-aapd no Makefile
[ ] 1.8 Testar: python -m hmc.main --dataset_name aapd --method global --device cuda
```

### Passo 2: RCV1-V2 (benchmark padrão)

```
[ ] 2.1 Criar src/hmc/datasets/rcv1/__init__.py
[ ] 2.2 Criar src/hmc/datasets/rcv1/dataset_rcv1.py — hierarchy + split
[ ] 2.3 Criar src/hmc/datasets/rcv1/manager.py — RCV1Manager
[ ] 2.4 Criar src/hmc/datasets/rcv1/download_rcv1.py
[ ] 2.5 Adicionar dispatch "rcv1" em dataset_manager.py
[ ] 2.6 Adicionar rcv1_defaults em registry.py
[ ] 2.7 Adicionar download-rcv1 no Makefile
[ ] 2.8 Testar: python -m hmc.main --dataset_name rcv1 --method global --device cuda
```

### Passo 3: EUR-Lex 57K (larga escala)

```
[ ] 3.1 Criar src/hmc/datasets/eurlex/__init__.py
[ ] 3.2 Criar src/hmc/datasets/eurlex/dataset_eurlex.py
[ ] 3.3 Criar src/hmc/datasets/eurlex/manager.py — EURLexManager com sparse R
[ ] 3.4 Criar src/hmc/datasets/eurlex/download_eurlex.py
[ ] 3.5 Adicionar dispatch "eurlex" em dataset_manager.py
[ ] 3.6 Adicionar eurlex_defaults em registry.py
[ ] 3.7 Adicionar download-eurlex no Makefile
[ ] 3.8 Testar: python -m hmc.main --dataset_name eurlex --method global --device cuda
```

### Passo 4: Experimentos

```
[ ] 4.1 Criar run_new_datasets.py — benchmark completo nos 3 datasets
[ ] 4.2 Rodar com 3 seeds: global, globalE2E, globalSOTA, local
[ ] 4.3 Ablação R-matrix on/off
[ ] 4.4 Análise por nível hierárquico
[ ] 4.5 Atualizar paper com novos resultados
```

---

## Estrutura de Diretórios Esperada

```
data/
├── aapd/                   # NOVO
│   └── aapd.csv            # título + abstract + labels
├── rcv1/                   # NOVO
│   ├── rcv1_train.json     # textos + labels hierárquicos
│   └── rcv1_test.json
├── eurlex/                 # NOVO
│   └── eurlex_57k/
│       ├── train.json
│       ├── dev.json
│       └── test.json
```

---

## Grid de Experimentos

| Dataset | Métodos | Seeds | Tempo est. GPU | Output |
|---------|---------|-------|---------------|--------|
| AAPD | global, globalE2E, globalSOTA, local | 3 | ~6h | `output/new/aapd/` |
| RCV1-V2 | global, globalE2E, globalSOTA, local | 3 | ~15h | `output/new/rcv1/` |
| EUR-Lex | global (sparse R), globalE2E, local | 3 | ~30h | `output/new/eurlex/` |

---

## Métricas por Dataset

| Dataset | Micro-F1 | AUPRC | Per-level F1 | R-matrix Δ |
|---------|----------|-------|-------------|------------|
| AAPD | target > 0.70 | — | L1, L2 | +XX |
| RCV1-V2 | target > 0.80 | — | L1-L4 | +XX |
| EUR-Lex | target > 0.60 | — | L1-L8 | +XX |

---

## SOTAs Publicados (referência)

### RCV1-V2 (Micro-F1)
| Método | Ano | Micro-F1 |
|--------|-----|----------|
| HiAGM | 2019 | 0.834 |
| HTCInfoMax | 2020 | 0.847 |
| HiMatch | 2021 | 0.863 |
| C-HMCNN | 2021 | 0.836 |
| HBGL | 2023 | 0.876 |

### AAPD (Micro-F1)
| Método | Ano | Micro-F1 |
|--------|-----|----------|
| HiAGM | 2019 | 0.701 |
| HTCInfoMax | 2020 | 0.716 |
| HiMatch | 2021 | 0.723 |

---

## Progresso

- [x] Passo 1: AAPD — manager, hierarchy, download, dispatch, registry
- [x] Passo 2: RCV1-V2 — manager, hierarchy, download, dispatch, registry
- [x] Passo 3: EUR-Lex 57K — manager, hierarchy (EUROVOC tree), sparse R, download, dispatch, registry
- [ ] Passo 4: Experimentos com dados reais

### Arquivos criados

```
src/hmc/datasets/aapd/
├── __init__.py              # exports AAPDManager
├── dataset_aapd.py          # AAPDHierarchyManager + AAPDSplit
├── download_aapd.py         # download via HuggingFace/Kaggle
└── manager.py               # AAPDManager (SPECTER2 + splits)

src/hmc/datasets/rcv1/
├── __init__.py              # exports RCV1Manager
├── dataset_rcv1.py          # RCV1HierarchyManager + RCV1Split
├── download_rcv1.py         # download via HiAGM repo
└── manager.py               # RCV1Manager (SPECTER2 + splits)

src/hmc/datasets/eurlex/
├── __init__.py              # exports EURLexManager
├── dataset_eurlex.py        # EURLexHierarchyManager (EUROVOC tree) + EURLexSplit
├── download_eurlex.py       # download via archive.org / HuggingFace
└── manager.py               # EURLexManager (SPECTER2 + sparse R support)

run_new_datasets.py           # experiment runner (dense + sparse R)
```

```
src/hmc/datasets/aapd/
├── __init__.py              # exports AAPDManager
├── dataset_aapd.py          # AAPDHierarchyManager + AAPDSplit
├── download_aapd.py         # download via HuggingFace/Kaggle
└── manager.py               # AAPDManager (SPECTER2 + splits)

src/hmc/datasets/rcv1/
├── __init__.py              # exports RCV1Manager
├── dataset_rcv1.py          # RCV1HierarchyManager + RCV1Split
├── download_rcv1.py         # download via HiAGM repo
└── manager.py               # RCV1Manager (SPECTER2 + splits)

run_new_datasets.py           # experiment runner for new datasets
```

### Modificados

- `src/hmc/datasets/dataset_manager.py` — dispatch para `aapd` + `rcv1`
- `src/hmc/datasets/registry.py` — `aapd_defaults` + `rcv1_defaults`
- `src/hmc/datasets/download_all.py` — entradas aapd + rcv1
- `Makefile` — targets `download-aapd` + `download-rcv1`
- `docs/expansao_dados.md` — este arquivo
