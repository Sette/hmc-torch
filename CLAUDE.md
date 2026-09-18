# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Guia de desenvolvimento para o projeto **hmc-torch** (Hierarchical Multi-Label Classification).
Datasets de texto: **ArXiv**, **WOS**, **AAPD**, **EUR-Lex**; tabulares: **FunCat** e **Gene Ontology**.

---

## Comandos essenciais

```bash
export PYTHONPATH=src          # necessário para `python -m hmc.*` e para os scripts de experiments/
uv sync --all-groups           # instala tudo (inclui tf-keras, exigido pelo parser ARFF)

make lint-check                # flake8 + ruff format --check + ruff check + pylint (apenas src/)
make lint                      # idem, aplicando a formatação
make test                      # pytest com coverage
uv run pytest tests/test_aapd_taxonomy.py -v                          # um arquivo
uv run pytest tests/test_hierarchy.py::TestTreeHierarchy::test_nodes  # um teste

# Dados — não vêm no repo (./data é gitignored)
make download-arff             # FunCat + GO + others → data/HMC_data_arff/ (tarball do repo C-HMCNN)
make download-arxiv | wos | aapd | eurlex
make download-rcv1             # gated pelo NIST: imprime o passo a passo (--source_dir / --sample)

# Treino
./run.sh                       # wos, global, cuda
python -m hmc.main --dataset_name wos --method global --device cuda \
  --dataset_path ./data --output_path ./output --consistency_loss none
```

O paper vive em `docs/hmc-paper` (repositório git **aninhado**, não rastreado por este):
`make -C docs/hmc-paper check` compila e falha se sobrar `\ref`/`\cite` sem resolver.

---

## Métodos

Escolhas reais de `--method` (ver `src/hmc/arguments.py`), despachadas em `src/hmc/main.py`:

| Método | Treino |
|---|---|
| `global` | features congeladas + MLP + R-matrix |
| `globalGNN` | idem + GCN sobre o grafo de labels |
| `globalE2E` | transformer fine-tuned + MLP + R-matrix |
| `globalSOTA` | transformer fine-tuned + GCN + R-matrix |
| `local` / `localE2E` | um MLP por nível da hierarquia |
| `tabular_gbdt` / `tabular_mlp` | baselines tabulares (baselines da tabela FunCat) |

Cada método tem sua função de treino em `src/hmc/pipeline/`; os modelos em `src/hmc/models/`.
Não há métodos LLM/Ollama neste repositório (as menções antigas a `globalLLM*` estão obsoletas).

`--consistency_loss {mc,hinge,none}` (para `global` e `globalE2E`): `mc` (default) mistura as
saídas constrangidas no loss, `hinge` soma a Eq. 2 do paper com peso `--lambda_hier`, e `none`
treina com BCE puro deixando a R-matrix só para a inferência — **é a receita dos números do
paper**. O constraint de inferência (`get_constr_out`) vale nos três casos.

---

## R-matrix: as duas convenções (leia antes de mexer)

Existem **duas** convenções, transpostas entre si; usar a errada inverte a penalidade:

- **Pipeline** (`src/hmc/pipeline/global_classifier/main.py`): `R[i,j] = 1` ⟺ *i é ancestral de j*
  (linha = ancestral). É a matriz que os modelos e `get_constr_out` recebem.
- **`Hierarchy.r_matrix`** (`src/hmc/data/hierarchy.py`): a **transposta** (linha = descendente).
  É o layout que `HierarchicalConsistencyLoss` espera — por isso `_resolve_consistency()`
  transpõe antes de instanciá-la.
- Os grafos de hierarquia (`a`, `g`) têm arestas **filho → pai** em todos os managers; logo
  `nx.descendants(g, n)` devolve os **ancestrais** de `n`.

`get_constr_out` (`models/global_classifier/constraint/utils.py`) faz
`out[i] = max(x[i], max dos descendentes de i)`: **folhas saem intactas** e só nós internos
podem mudar de score — em AAPD são 7 áreas entre 61 labels avaliadas; em ArXiv, 19 entre 156.

---

## Dados e avaliação

- **Layout**: tabulares em `data/HMC_data_arff/<família>/<dataset>/` (FunCat/GO/others), texto em
  `data/<nome>/`. O `paths.py` espera exatamente isso — o downloader usa `--output_dir ./data`
  justamente por causa dessa raiz compartilhada.
- **Duas tabelas de caminhos**: a viva é `src/hmc/utils/datasets/paths.py` (é o que
  `dataset_manager` importa); `src/hmc/datasets/gofun/__init__.py` guarda uma cópia que já
  divergiu. Ao adicionar um dataset ARFF, edite **as duas**.
- **Cap de registros**: `arxiv_max_records` (default 50.000, `0` = todos) vale para
  arxiv/aapd/rcv1/eurlex. O AAPD canônico tem 55.840 documentos — com o default o split vira
  40.000/10.000 em vez dos 44.672/11.168 do paper.
- **Splits**: ArXiv/AAPD usam 64/16/20; WOS/EUR-Lex/RCV1 seguem as partições publicadas; FunCat e
  GO usam os arquivos `.train/.valid/.test.arff` do benchmark. Features de texto são SPECTER2
  (768d, CLS-pool) com cache em disco invalidado por hash MD5.
- **Threshold**: todos os caminhos (`compute_metrics` dos scripts e `--best_threshold` do CLI)
  escolhem o melhor threshold varrendo o **test split**; AUPRC não é afetado por isso.
- **Fontes externas**: AAPD vem de um re-upload do Kaggle (`xiaojuanwang9/aapd-dataset`, 54
  labels); RCV1-V2 é gated (corpus do NIST + pré-processamento dos repos HBGL/HiAdv); RCV1
  também aceita `--sample` para smoke test com os ~10 documentos do HiAGM.

---

## Scripts de `experiments/`

Rodam **no import** (o loop fica no nível do módulo, não há `main()`), gravam em `./output/` e
exigem `PYTHONPATH=src`. Os principais:

| Script | O que produz |
|---|---|
| `run_new_datasets.py --dataset aapd --seeds 3 [--max_records 0]` | AAPD/EUR-Lex com e sem R (tabela de texto) |
| `run_sota_comparison.py [--datasets ... --device cpu\|cuda]` | `global` nos FunCat (reproduz a tabela do paper) |
| `run_experiment_matrix.py [--datasets ... --methods ... --device ...]` | `tabular_gbdt` e `tabular_mlp` nos FunCat |
| `run_go_experiments.py` / `run_all_funcat_go.py` | GO com raw vs sparse R / os datasets "restantes" |
| `statistical_tests.py` | Friedman + Nemenyi da tabela FunCat |

`json.dump` nesses scripts precisa de `default=float`: os resultados contêm `numpy.float32` e o
script inteiro morre na última linha sem isso (já custou horas de compute).

---

## Adicionando um novo dataset

### Com dados próprios (sem tocar no pacote)

`build_digraph_from_labels` + `TreeHierarchy.from_graph` derivam tudo (adjacência, levels,
edge_index, dimensões) a partir das labels:

```python
from hmc.utils import build_digraph_from_labels
from hmc.data.hierarchy import TreeHierarchy
from hmc.data import DatasetRegistry, Split

texts, label_strs = load_seus_dados()          # label_strs: ["cs.AI", "stat.ML cs.LG", ...]
h = TreeHierarchy.from_graph(build_digraph_from_labels(label_strs))
X = seu_encoder(texts)                          # shape (n, d)
Y_global, Y_local_all = h.encode_labels(label_strs)
Y_local = Y_local_all[1:]                       # drop root
```

Depois registre o manager (ver "Register your own dataset" no README.md).

### Dataset built-in (texto + transformer)

1. Manager em `src/hmc/datasets/<nome>/manager.py`: `get_datasets()` → (train, valid, test) com
   `.x`, `.y`, `.y_local`, `.samples`; expõe `levels_size`, `max_depth`, `a`, `edge_index`,
   `nodes_idx`, `to_eval`, `input_dim`, `output_dim`.
2. Registre os caminhos em `src/hmc/utils/datasets/paths.py` **e** na cópia de
   `src/hmc/datasets/gofun/__init__.py`.
3. Dispatch em `src/hmc/datasets/dataset_manager.py`, defaults em `src/hmc/datasets/registry.py`,
   entrada em `src/hmc/datasets/download_all.py` e alvo no `Makefile`.
