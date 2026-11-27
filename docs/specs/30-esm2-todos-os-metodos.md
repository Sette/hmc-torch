# 30 — ESM-2: todos os nossos métodos e todos os baselines, reportados (P1)

## Objetivo

Reportar nos manuscritos (KBS, tese; IJCNN só se couber) **todos** os métodos do projeto sobre
as mesmas features ESM-2 (`seq_FUN`, `seq_GO`), inclusive os que deram resultado nulo ou
negativo. Continua o [spec 29](29-esm2-comparacao-metodos.md), que cobriu só o MLP global
(BCE/MCLoss), C-HMCNN, HMCN-F e GBDT. O resultado de lá (HMCN-F > nosso MLP) mostra que
falta saber se o pipeline completo (H-MLCL + H-KNN, DepthCal etc.) muda a ordem.

## Fatos a usar

- Spec 29: HMCN-F 0,5432 / 0,6154 (rec.), nosso MLP BCE + rec. 0,5279 / 0,6011, C-HMCNN
  0,5240 / 0,5938, GBDT 0,381 / 0,471 (`seq_FUN` / `seq_GO`). O grid de HP (rodando) não
  melhorou o nosso MLP em `seq_FUN` (~0,5275).
- Todos os scripts abaixo carregam as features por `run_protocol.load_dataset(name)`, que
  devolve as features do benchmark. Para o ESM-2, `data["x"]` precisa ser trocado por
  `run_protein_lm.esm_features(...)` (mesmo cache de embeddings, mesmo mapeamento de ORFs).
- **Armadilha:** `run_sr_gae_sota._protocol_best_epochs` lê `output/protocol/<nome>.json`
  (épocas e objetivo escolhidos **com as features do benchmark**). Como `seq_FUN` existe lá,
  rodar o script sobre ESM-2 sem tratar isso reaproveitaria épocas erradas **sem erro**.
- O hinge nunca rodou com ESM-2 (`output/protein_lm/` só tem `bce` e `mc`).

## Métodos (todos com ESM-2, `seq_FUN` e `seq_GO`, 3 seeds, protocolo de validação)

| # | Método | Script | Saída (nova) |
|---|---|---|---|
| 1 | MLP global: BCE, MCLoss (já), **hinge** (novo), cru e rec. | `run_protein_lm.py --objectives hinge` | `output/protein_lm/` |
| 2 | **H-MLCL + H-KNN** com protótipos SR-GAE e com protótipos aleatórios (controle) | `run_sr_gae_sota.py` | `output/protein_lm_methods/sr_gae_paired/` |
| 3 | **DepthCal** sobre o blend | `run_sr_gae_depth.py` | `output/protein_lm_methods/sr_gae_depth/` |
| 4 | Cabeças com embeddings de rótulos: livre, GCN, GAE, GAE+livre, texto do `go.obo` (só GO) | `run_label_embeddings.py` | `output/protein_lm_methods/label_embeddings/` |
| 5 | Autoencoders: GAE híbrido, DAE de features, AE de rótulos (C2AE) | `run_autoencoder.py --stage final` | `output/protein_lm_methods/autoencoder/` |
| 6 | HPT / E2E (ESM-2 afinado) | — | **fora**, ver abaixo |
| 7 | **Surj, reprodução nossa** (cabeças `surj` e `onehot`) | projeto próprio `~/git/surj-repro` | `surj-repro/results_esm2/` |

Mais os baselines do spec 29 (C-HMCNN, HMCN-F, GBDT). Fora de escopo:
- **6. Fine-tuning do ESM-2:** não existe pipeline de proteína E2E, e 650M de parâmetros não
  cabem na RTX 3060 (12 GB) sem LoRA/gradient checkpointing. Registrar como limitação/trabalho
  futuro; o texto de Limitações deve dizer "frozen ESM-2", não "end-to-end".

## Surj: reprodução em projeto próprio (decisão do usuário, 2026-11-26)

O Surj entra, rotulado **"reprodução nossa (a partir do paper)"**, nunca "Surj publicado": o
código nunca foi liberado e a reprodução não chega nos números reportados (0,232 contra 0,312).
As cabeças do item 4 (`gae`, `gae+free`) e o GAE híbrido do item 5 são implementações nossas
inspiradas nele e entram com o próprio rótulo.

- **Projeto:** já existe `~/git/surj-repro` (extraído do hmc-torch em 16/11; um commit,
  tag local `surj-repro-2026-11`, MIT, `surj_repro/` com GAE, modelo, treino, receitas e
  JSONs por seed). Estender esse, não criar outro.
- **Pendência de publicação (usuário):** o remote aponta para
  `https://github.com/Sette/surj-repro-.git` — **com hífen no fim**, provavelmente um engano
  de digitação; conferir antes de subir. A tag `surj-repro-2026-11` está só local (spec 22);
  empurrar com `git push origin surj-repro-2026-11`.
- **Entrada ESM-2 via pacote do PyPI (decisão do usuário):** o `surj-repro` passa a depender de
  `hmc-torch` publicado (`pip install hmc-torch`), sem copiar código. Verificado: o wheel
  **0.3.0** contém `hmc.features.protein.ProteinFeatureEncoder` e `hmc.utils.datasets.scaling`,
  byte a byte iguais aos de `src/` (conferido com diff). Portanto:
  - fixar a versão no `pyproject.toml` do `surj-repro` (`hmc-torch==<release>`) para a reprodução
    não mudar com um release novo;
  - o que **ainda não** está no pacote é o mapeamento linha → ORF (`recover_seq_ids.py`, 174
    linhas) e `run_protein_lm.esm_features`; hoje ficam em `experiments/`, fora do wheel.
    **Decisão do usuário: se o pacote não traz, passa a trazer.** Em vez de duplicar no
    `surj-repro`, migrar para o hmc-torch (cada coisa uma vez só):
    - `hmc.features.yeast_orf` (novo): `read_proteome`, `composition`, `read_arff` e
      `match_split` do `recover_seq_ids.py`, mais `load_orf_ids(path)`; o download do
      proteoma fica por conta do chamador (ou de uma função com cache), sem dado no wheel;
    - `hmc.features.protein.esm_features(x_splits, ids, encoder, proteome)` (a versão de
      `run_protein_lm.py`: imputação pela média do treino, z-score no treino, coluna
      indicadora de ORF ausente);
    - `experiments/recover_seq_ids.py` e `run_protein_lm.py` passam a **importar** do pacote
      (sem duplicar código); testes em `tests/` (ORF único/ambíguo/ausente, forma das saídas,
      sem rótulos na entrada — a auditoria "só features" vira teste);
    - commit `feat(features): ✨ ...` na `main` (formato do `CLAUDE.md`); o `release-please`
      abre o PR de release e define a versão — **não editar `project.version` nem o manifest
      à mão**; o `surj-repro` fixa `hmc-torch>=<versão do release>` só depois do release.
      Até lá, desenvolvimento com dependência por git (`hmc-torch @ git+https://github.com/Sette/hmc-torch@<sha>`),
      sem publicar nada que dependa dela.
    - As duas tabelas de ORFs (`seq_FUN_ids.tsv`, `seq_GO_ids.tsv`) são geradas pela função do
      pacote; o `surj-repro` as regenera ou leva cópias pequenas (dado derivado de fontes
      públicas).
  - o `surj-repro` ganha `--features esm2` que troca só o `x`; hierarquia, rótulos e splits
    continuam do loader dele.
  - **Cuidado de reprodutibilidade:** o cache de embeddings é por hash MD5 do modelo e das
    sequências (`models/protein_embeddings/`), então o `surj-repro` reutiliza o cache do
    hmc-torch se apontado para ele; as matrizes têm que bater (mesmas 1281 colunas) com as do
    spec 29 — teste: o MLP do spec 29 nas matrizes do `surj-repro` dá o mesmo AUPRC.
- **Protocolo:** o do `surj-repro` (as mesmas receitas já publicadas nele), 3 seeds,
  `results_esm2/` separado; nada de sobrescrever os JSONs do benchmark.
- **Anonimato:** o IJCNN é double-blind; **não citar** o repositório nem a tag no IJCNN.
  Só KBS e tese (com o DOI do Zenodo do spec 28).

## Passos

0. **Surj (projeto próprio):** (a) migrar o mapeamento de ORFs e `esm_features` para o pacote
   (acima) e esperar o release; (b) `surj-repro` com a dependência do release + `--features
   esm2` + teste de paridade das matrizes com as do spec 29; (c) rodar `seq_FUN` e `seq_GO`,
   3 seeds; commit lá, usuário faz o push.
> **Estado (2026-11-27):** passos 1 e 2 feitos — `experiments/esm2_features.py`,
> `experiments/run_with_esm2.py` (exige `--out_dir` em `output/protein_lm*`), `HMC_PROTOCOL_DIR` e
> `HMC_SOTA_OBJECTIVE` em `run_sr_gae_sota.py`, e `tests/test_esm2_adapter.py` (4 testes, incluindo
> o das épocas). Mapeamento de ORFs/`esm_features` migrado para o pacote (PR #97, na `dev`).
> Falta rodar os itens 1–5 (GPU ocupada pelo grid do spec 29) e o Surj (depende do release).

1. **Adaptador de features** `experiments/esm2_features.py`: função `load_dataset_esm2(name)`
   (`run_protocol.load_dataset` + troca de `data["x"]`) e um lançador
   `experiments/run_with_esm2.py <script> [args]` que aplica o patch **antes** de importar o
   script e executa com `runpy` (assim cada script roda sem alteração).
2. **Neutralizar a armadilha:** o lançador também redireciona `_protocol_best_epochs` para
   `output/protocol_esm2/<nome>.json`, gerado a partir de `output/protein_lm/<modelo>/<nome>.json`
   (runs com `features == "esm"`, no formato de `output/protocol/`). **Teste:** com o adaptador,
   as épocas lidas diferem das de `output/protocol/seq_FUN.json` e vêm dos runs ESM-2.
3. **Rodar o item 1 (hinge)** e depois os itens 2–5 com `--out_dir` em `output/protein_lm_methods/`
   (cada script pula arquivos existentes; nunca escrever nos diretórios do benchmark).
4. **Hiperparâmetros:** os do protocolo (`run_protocol.hyperparams`), iguais aos já usados. O
   grid do spec 29 mostrou que o MLP não ganha com ajuste; reportar isso, e que os baselines
   (C-HMCNN, HMCN-F) foram ajustados no mesmo grid.
5. **Tabela única** `tables/esm2_all_methods.tex` (via `make_paper_artifacts.py`, não à mão):
   linhas = todos os métodos acima + baselines; colunas = {`seq_FUN`, `seq_GO`} × {cru, rec.}
   (AUPRC; F1 no apêndice), média ± desvio (ddof = 1). Ordenar por seção (baselines, nossos
   métodos, nulos/negativos), **sem** ordenar por valor para destacar o melhor.
6. **Texto:** seção "Frozen protein language model" na KBS e na tese reportando **todas** as
   linhas, os resultados nulos/negativos incluídos; enunciado pelo que a tabela sustentar:
   o encoder explica o salto (0,29 → 0,53); ordem entre métodos pequena; reconciliação geral.
   Nada de "supera"/"SOTA" sem diferença > 1 desvio e mesmo orçamento de HP.
7. **IJCNN:** só uma frase em Trabalhos futuros, se houver espaço; o paper está congelado em
   24/01 e no limite de páginas.

## Saídas esperadas

- `output/protein_lm_methods/**` com JSONs por seed; `output/protocol_esm2/` (adaptador).
- `tables/esm2_all_methods.tex`; seção em KBS e tese; limitação ajustada.

## Critério de aceite

- [ ] Todos os itens 1–5 e 7 (Surj, rotulado como reprodução nossa) rodaram nos dois datasets, 3 seeds, ou a falta está justificada.
- [ ] O teste do passo 2 passa (épocas do ESM-2, não do benchmark).
- [ ] Nenhum arquivo de `output/protocol*/` do benchmark foi sobrescrito.
- [ ] Todos os números das tabelas vêm dos JSONs por script.
- [ ] O texto reporta os resultados nulos/negativos (GAE, autoencoders) sem os omitir.
- [ ] Nenhuma frase de superioridade sem diferença > 1 desvio e mesmo orçamento de HP.

## Riscos

- **Só 2 datasets** (única entrada com sequência): a conclusão é de caso.
- **Armadilha das épocas** (acima): silenciosa se o passo 2 for pulado.
- **Custo:** `run_sr_gae_sota` e `run_label_embeddings` em `seq_GO` (~4.100 rótulos, hidden
  9.000) podem passar de horas por dataset; GPU única, rodar em série depois do grid do spec 29.
- **Vazamento pelo mapeamento ORF:** auditado (só features); manter a nota.
- **ESM-2 pode ter visto as proteínas** no pré-treino; declarar como ameaça.

## Tempo/custo

~1 dia de código/teste + vários dias de GPU em série (estimar após o smoke test do passo 3).
