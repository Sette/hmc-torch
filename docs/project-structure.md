# Estrutura do projeto e módulos

Este documento descreve a organização implementada em `src/hmc/`. Planos de
pesquisa e funcionalidades futuras ficam nos documentos de planejamento; a
presença de um módulo experimental ou encoder não significa que exista um
dataset integrado para essa modalidade.

## Visão geral

```text
CLI / API Python
    └── argumentos e configuração
        └── registry + dataset manager
            ├── adapters de datasets → splits e hierarquia
            └── features → representações
                └── pipeline de treino → modelos
                    └── métricas, manifestos e artefatos
```

O fluxo concreto depende do método. Os pipelines de texto usam managers em
`hmc.datasets`; os baselines ARFF também têm loaders em `hmc.data.gofun`. Os
contratos em `hmc.data` são a camada comum para integrações e APIs recentes,
mas nem todos os pipelines legados foram migrados para um único `DatasetBundle`.

## Mapa de módulos

| Caminho | Responsabilidade |
|---|---|
| `src/hmc/main.py`, `arguments.py` | Entrada de treino, argumentos e dispatch por método. |
| `src/hmc/train.py` | API de treino usada por código Python. |
| `src/hmc/data/base.py`, `protocols.py` | `Split`, `DatasetBundle`, metadados, modalidade e protocolo de manager. |
| `src/hmc/data/hierarchy.py` | Abstrações e operações comuns para hierarquias em árvore e DAG. |
| `src/hmc/data/registry.py` | Registro central de managers e defaults. |
| `src/hmc/data/gofun/` | Leitura e adaptação dos datasets ARFF FunCat/GO. |
| `src/hmc/datasets/<nome>/` | Implementações por dataset: manager, parsing e scripts de download. |
| `src/hmc/datasets/dataset_manager.py`, `registry.py` | Dispatch e configuração usados pelos pipelines atuais. |
| `src/hmc/datasets/download_all.py` | Orquestração dos downloaders. |
| `src/hmc/features/` | Encoders para texto, tabular, expressão, visão e proteína. |
| `src/hmc/models/global_classifier/` | Classificadores globais, constraint e variantes end-to-end/GNN. |
| `src/hmc/models/local_classifier/` | Classificação local por nível. |
| `src/hmc/models/hierarchical/` | Cabeças, perdas, pós-processamento, grafo de labels e R esparsa. |
| `src/hmc/models/tabular/` | Pré-processamento e baselines GBDT/MLP/TabFM. |
| `src/hmc/pipeline/` | Loops de treino global, local e tabular. |
| `src/hmc/utils/` | Métricas, grafos, caminhos, predição, cache e utilitários de treino. |
| `src/hmc/audit.py` | CLI para auditar um dataset e produzir um cartão JSON. |
| `experiments/` | Scripts de pesquisa e execução de matrizes experimentais. |
| `tests/` | Testes unitários e fixtures locais. |
| `configs/` | Configurações versionadas de execução. |

### Datasets integrados

Os adaptadores em `src/hmc/datasets/` abrangem ArXiv, WOS, AAPD, RCV1-V2,
EUR-Lex e GoFun/ARFF. A disponibilidade dos dados varia: RCV1-V2 depende do
corpus licenciado da NIST; os demais downloaders dependem de fontes externas
que podem mudar. Os dados baixados ficam em `data/`, fora do pacote e ignorados
pelo Git.

`src/hmc/data/gofun/` contém adicionalmente o parser e manager ARFF consumidos
por caminhos da API de dados. Ao alterar datasets ARFF, confira as tabelas de
caminhos em `src/hmc/utils/datasets/paths.py` e os consumidores em ambas as
camadas `data` e `datasets`.

### Modalidades e maturidade

- **Texto:** managers de ArXiv, WOS, AAPD, RCV1 e EUR-Lex; os pipelines atuais
  usam features de texto/transformers conforme cada integração.
- **Tabular:** loaders ARFF, pipelines GBDT/MLP e pré-processamento.
- **Expressão gênica:** há encoder e autoencoder para features tabulares.
- **Visão e proteína:** há módulos de encoder, mas a integração completa com
  datasets e pipelines não está disponível por padrão.
- **TabFM:** módulos de adapter/cache/contexto estão em desenvolvimento e
  dependem de instalação opcional.

## Métodos de treino

Os métodos aceitos pela CLI estão definidos em `src/hmc/arguments.py` e
despachados em `src/hmc/main.py`:

| Método CLI | Pipeline |
|---|---|
| `global` | `hmc.pipeline.global_classifier.main` |
| `globalE2E`, `globalSOTA` | Pipeline global com transformer ajustável; `globalSOTA` inclui GCN de labels. |
| `local`, `localE2E` | `hmc.pipeline.local_classifier.main` |
| `tabular_gbdt`, `tabular_mlp` | `hmc.pipeline.tabular.main` |

O dispatch também reconhece aliases internos `global_baseline` e `globalLM`,
mas eles não aparecem nas escolhas do parser CLI. Use os nomes listados acima.

## Diretórios de trabalho

- `data/`: datasets locais baixados ou preparados; não versionar dados brutos.
- `output/`: resultados e artefatos de treino.
- `models/`: cache de modelos/embeddings quando configurado.
- `docs/`: documentação, planos e material do paper.
- `docs/paper/`: fontes do paper mantidas neste repositório; alguns trabalhos
  anteriores podem referenciar o checkout aninhado `docs/hmc-paper`.

Comandos básicos de desenvolvimento e download estão no `README.md` e no
`CLAUDE.md`. Os arquivos em `docs/*plan*.md` descrevem objetivos e experimentos,
não necessariamente estado implementado.
