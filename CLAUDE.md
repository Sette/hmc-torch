# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Guia de desenvolvimento para o projeto **hmc-torch** (Hierarchical Multi-Label Classification).
Datasets de texto: **ArXiv**, **WOS**, **AAPD**, **EUR-Lex**; tabulares: **FunCat** e **Gene Ontology**.

---

## Comandos essenciais

```bash
export PYTHONPATH=src          # necessário para `python -m hmc.*` e para os scripts de experiments/
uv sync --all-groups --extra download   # dev + kagglehub/datasets (usados só pelos `make download-*`)

make lint-check                # flake8 + ruff format --check + ruff check + pylint (apenas src/)
make lint                      # idem, aplicando a formatação
make test                      # pytest com coverage
uv run pytest tests/test_aapd_taxonomy.py -v                          # um arquivo
uv run pytest tests/test_hierarchy.py::TestTreeHierarchy::test_nodes  # um teste

# Dados — não vêm no repo (./data é gitignored)
make download-arff             # FunCat + GO + others → data/HMC_data_arff/ (tarball do repo C-HMCNN)
make download-arxiv | wos | aapd | eurlex
make download-rcv1             # corpus NIST em data/rcv1; roda HBGL (--raw_dir / --source_dir / --sample)
make download-rcv1-lyrl        # rota aberta LYRL2004/JMLR (tokens com stem, sem texto bruto)

# Treino
./run.sh                       # wos, global, cuda
python -m hmc.main --dataset_name wos --method global --device cuda \
  --dataset_path ./data --output_path ./output --consistency_loss none
```

O manuscrito é `docs/hmc-paper/main.tex`, um repositório git próprio (fora do versionamento do
hmc-torch), com `make pdf` / `make check` no seu `Makefile`. Não há outra versão do paper no repo.

---

## Commits e release

Commits seguem Conventional Commits com o **tipo na posição 1**: `tipo(escopo): emoji descrição`
(ex.: `feat(hpt): ✨ add label-text encoder`). O emoji vem depois dos dois-pontos; no formato
antigo (`✨ feat: ...`) o `release-please` não parseia o commit e não abre o PR de release.

O `release-please` (`.github/workflows/release-please.yml`, config em `release-please-config.json`
e `.release-please-manifest.json`) lê os commits da `main` desde a última tag `hmc-torch-vX.Y.Z`,
atualiza `pyproject.toml`, `src/hmc/__init__.py`, `.bumpversion.toml` e `CHANGELOG.md` e abre um
PR de release. Títulos de squash/merge também precisam seguir o formato — um merge inteiro fica
invisível para ele se o subject não for parseável.

---

## Métodos

Escolhas reais de `--method` (ver `src/hmc/arguments.py`), despachadas em `src/hmc/main.py`:

| Método | Treino |
|---|---|
| `global` | features congeladas + MLP + R-matrix |
| `globalE2E` | transformer fine-tuned + MLP + R-matrix |
| `globalSOTA` | transformer fine-tuned + GCN + R-matrix |
| `local` / `localE2E` | um MLP por nível da hierarquia |
| `tabular_gbdt` / `tabular_mlp` | baselines tabulares (baselines da tabela FunCat) |
| `surj` | Surj (WWW'22): GAE sobre a hierarquia → embeddings congelados; MLP → cosine → FC(V→V) → sigmoid, BCE |
| `globalHPT` | HPT (EMNLP'22) + matriz R: prompt com um `[PRED]` por nível, tokens de rótulo via GAT, ZMLCE + MLM; `--consistency_loss` vira `none` (HPT puro) / `mc` (MCLoss nos logits) / `hinge` |

Cada método aceito pela CLI está em `src/hmc/arguments.py` e tem sua função de treino em
`src/hmc/pipeline/`; os modelos ficam em `src/hmc/models/`. Embora o dispatch em
`src/hmc/main.py` conserve alguns aliases internos, `globalGNN` e `global_baseline`
não são escolhas válidas no parser atual.

`--consistency_loss {mc,hinge,none}` (para `global`, `globalE2E` e `globalHPT`): `mc` (default) é o MCLoss
do C-HMCNN (BCE sobre as saídas constrangidas), `hinge` soma a Eq. 2 do paper com peso
`--lambda_hier`, e `none` treina com BCE puro. O constraint de inferência (`get_constr_out`) vale
nos três casos. O paper compara os três objetivos, cada um com e sem reconciliação.

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

`BlockDiagonalR` (`models/hierarchical/sparse_r.py`) faz a mesma coisa em O(B·E) e é
diferenciável (serve para o MCLoss no GO). Ele indexa colunas por `column_index`: **passe sempre
o `nodes_idx` do dataset** (ou `{i: i}` se o grafo tiver nós inteiros). O default é a ordem
interna (profundidade, nome), que no FunCat difere da ordem das colunas e reconcilia as colunas
erradas.

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
  GO usam os arquivos `.train/.valid/.test.arff` do benchmark. Os `_others` (enron, diatoms,
  imclef) não têm arquivo de validação: os managers ARFF separam **20% fixos do treino**
  (`_holdout_split`, seed 0). Antes eles reaproveitavam o test como validação — não volte a isso.
  Features de texto são SPECTER2 (768d, CLS-pool) com cache em disco invalidado por hash MD5.
- **Protocolo de avaliação**: treino só no split de treino; época e threshold escolhidos na
  **validação**; test usado uma vez. No CLI (`global`, `globalE2E`, `globalSOTA`, `local`,
  `localE2E`, `surj`) o `--best_threshold true` varre a validação. Os scripts antigos de `experiments/`
  (listados abaixo como legados) ainda escolhem o threshold no test e juntam validação ao treino.
- **Textos do GO**: `data/ontologies/go.obo` (baixado de `purl.obolibrary.org` por
  `label_embedding.load_go_obo` se faltar); cobre todos os termos dos 9 datasets GO, inclusive
  `alt_id`s e obsoletos. O catálogo de nomes do FunCat não está mais disponível online.
- **EUR-Lex**: a hierarquia EUROVOC (conceito → termo de topo → microtesauro → domínio) vem do
  SPARQL do EU Publications Office; `download_eurlex.add_eurovoc_relations` grava os `broader`
  no `eurovoc_concepts.jsonl`. Sem isso o manager cai numa hierarquia plana.
- **Fontes externas**: AAPD vem de um re-upload do Kaggle (`xiaojuanwang9/aapd-dataset`, 54
  labels); RCV1-V2 é gated (corpus do NIST + pré-processamento dos repos HBGL/HiAdv); RCV1
  também aceita `--sample` para smoke test com os ~10 documentos do HiAGM.

---

## Scripts de `experiments/`

Exigem `PYTHONPATH=src` e gravam em `./output/`. **Todos os números do paper vêm de:**

| Script | O que produz |
|---|---|
| `run_protocol.py --group {funcat,go,others,text,all}` | `output/protocol/<dataset>.json`: 3 seeds × {BCE, hinge, MCLoss} × {raw, reconciliado}, + GBDT/Platt no FunCat |
| `run_wos_e2e.py --seed N --epochs 10` | SPECTER2 fine-tuned no WOS com o protocolo do HPT (`output/protocol/wos_ep10/wos_e2e_seedN.json`; seed 0 = 0,8671, o número do paper) |
| `hierarchy_structure.py` | `output/protocol/structure.json`: fração de nós internos e quanto a reconciliação move os scores (análise da Seção 5.2) |
| `run_surj.py --experiment {main,literal,evolving,datasize}` | `output/surj/<exp>/`: reprodução do Surj (cabeças `surj` e `onehot`, raw/rec, GHV, tempos); `literal` = Adam 1e-3, 10 seeds, sem refit, com o teto diagnóstico `oracle_test_auprc` (época escolhida no test — nunca reportar como resultado) |
| `run_label_embeddings.py --group {funcat,go,all}` | `output/label_embeddings/`: heads com embeddings de rótulos (livre, GCN, texto do `go.obo`, texto+GCN) no mesmo protocolo |
| `run_autoencoder.py --stage {screen,final}` | `output/autoencoder/`: autoencoders no modelo global (GAE híbrido, DAE de features, AE de labels C2AE); triagem só na validação, vencedor com 3 seeds (Seção `sec:autoenc` do paper) |
| `run_hpt.py --dataset {wos,aapd,arxiv,eurlex} --stage {screen,final}` | `output/hpt/<stage>/<dataset>/`: HPT + R-matrix. `screen` = seed 0, ≤10 épocas, só validação, {bert, specter2} × {none, mc, hinge} + o HPT publicado (`bert/none/down`); `final` = 3 seeds, test uma vez (`--auto` pega o melhor do screen). Bateria: `experiments/run_hpt_all.sh` |
| `make_paper_artifacts.py --paper_dir docs/hmc-paper` | `data/protocol_summary.json`, `tables/*.tex`, figuras do paper e estatísticas (Friedman/Nemenyi, Wilcoxon, Spearman) |

`output/protocol/run_all.sh` roda a bateria inteira (~10–12 h numa RTX 3060, a maior parte no E2E).
`run_protocol.py` pula datasets cujo JSON já existe — apague o arquivo para refazer.

Os demais scripts (`run_new_datasets.py`, `run_sota_comparison.py`, `run_go_experiments.py`,
`run_all_datasets.py`, …) são **legados**: rodam no import, juntam validação ao treino e
escolhem o threshold no test. Não use os números deles.

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

---

## HPT + matriz R (`globalHPT`, `experiments/run_hpt.py`)

- Código: `models/global_classifier/hpt/` (modelo + losses) e `pipeline/hpt/main.py` (`fit_hpt` /
  `evaluate_hpt`, compartilhados pelo CLI e pelo script). Reimplementação sobre transformers 5 / PyG 2.7;
  `HPT/` (fora do versionamento) é só a referência.
- `--hpt_graph down` é o grafo do HPT: com arestas só pai→filho e nível→rótulo, os tokens de nível
  (os únicos que entram no prompt) não recebem mensagem nenhuma além do self-loop — a hierarquia não
  chega ao encoder. `both` (default) adiciona as arestas reversas.
- O cache de modelos (`ensure_transformer_model_cached`) salva via `AutoModel`, que **descarta a cabeça
  MLM**; o HPT usa `auto_class="AutoModelForMaskedLM"` (pasta `models/<nome>__AutoModelForMaskedLM`).
  O SPECTER2 não publica cabeça MLM: o transform começa aleatório.
- Métricas: micro/macro-F1 (definição do HPT: média sobre todos os rótulos, ausentes contam 0, em
  `utils/metrics/multilabel.py`), AUPRC e GHV; raw e reconciliado, no threshold fixo do HPT
  (logit 0 = 0,5) e no escolhido na validação. Splits de texto agora carregam `split.texts`.
- Custo (RTX 3060, bf16, batch 8 × accum 2): ~17 min por época no WOS.
- `--selection {rec_val_micro,hpt_raw}`: o default é o nosso protocolo (micro-F1 reconciliado na
  validação, threshold ajustado); `hpt_raw` reproduz o loop do HPT (macro-F1 **cru** na validação
  em logit 0, paciência 5) para a reexecução do baseline. `experiments/summarize_hpt.py` agrega
  os JSONs por seed em `summary.json`.

