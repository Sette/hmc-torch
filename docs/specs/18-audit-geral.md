# 18 — Auditoria geral: tese, KBS, IJCNN e experimentos (2026-11-13)

Documento de apoio (não sequencial). Auditoria somente leitura de `docs/tese/`,
`docs/hmc-paper/main.tex`, `docs/hmc-paper-ijcnn/main.tex`, dos scripts de `experiments/` que
geram os números e de `output/`. Nenhum manuscrito, código ou resultado foi alterado; nenhum
experimento foi rodado. Os achados viram os specs 19–25.

Linhas de `main.tex` e dos capítulos são as de 13/11/2026 (KBS em `0b24288`, hmc-torch em
`db57381`).

## Veredito

| Manuscrito | Estado | Veredito |
|---|---|---|
| IJCNN (6 p.) | compila, 0 over/underfull | Não pode ser submetido como está: viola o duplo-cego (autores e URL do GitHub) e descreve um BCE ponderado que o código não usa. Os números conferem. |
| KBS (48 p.) | compila, 0 indefinidas | As tabelas principais conferem com `output/`. Seis passagens reportam números ou procedimentos que não vêm do protocolo (λ, BCE ponderado, Platt, F1 por nível, tempos, artefatos por run). Título e highlights anunciam componentes cujo resultado é negativo. |
| Tese (80 p.) | compila, 4 overfull | Herda os seis problemas da KBS. Além disso: título desalinhado do resultado, revisão bibliográfica curta (24 referências), RQ4 respondida com conteúdo que a pergunta não contém, e ameaças à validade incompletas. |
| Experimentos | — | Nenhum vazamento do test encontrado. Há três escolhas de protocolo não declaradas que afetam a comparação raw × rec, e dois resultados cujo código não está no repositório. |

## Pontos fortes (conferidos)

- **Sem vazamento do test** em `run_protocol.py`, `run_wos_e2e.py`, `run_sr_gae_sota.py`
  (versão em disco), `run_sr_gae_depth.py`, `run_surj.py` (o `oracle_test_auprc` é só
  diagnóstico) e `hierarchy_structure.py`. Padronização ajustada só no treino
  (`src/hmc/utils/datasets/scaling.py`).
- **Tabela fatorial** (8 linhas × 6 colunas), tabelas GO, cross-domain e texto recalculadas de
  `output/protocol/*.json`: todos os valores batem na 4ª casa; desvios são amostrais (ddof = 1).
- **Avaliação pareada** H-MLCL+H-KNN / protótipos aleatórios / DepthCal: os 20 números do
  texto reproduzem a partir de `output/sr_gae_paired/` (+0,0022; 25/27; p = 1,9e-5; γ = 0 em
  78/81; aleatório +0,0025; DepthCal +0,0013; F1 profundo menor em 21/27).
- **Paridade com o C-HMCNN** (`output/chmcnn_parity/summary.md`): o pipeline reproduz os 8
  números publicados de FunCat a ±0,002.
- **Ablação de receita**, **RCV1**, tabela de precisão/recall, parágrafo do Nemenyi: conferem.
- O resultado negativo (protótipos reparados não ajudam; ganho vem do k-NN) está reportado com
  controle e sem maquiagem nos três manuscritos.

## Achados

Severidade: **P0** errado ou eliminatório; **P1** objeção provável de revisor/banca; **P2**
polimento. Coluna “Onde”: I = IJCNN, K = KBS, T = tese.

### A. Experimento × texto

| # | Sev | Achado | Evidência | Onde | Spec |
|---|---|---|---|---|---|
| A1 | P0 | λ do hinge: o texto diz 0,1; o protocolo usa 1,0. | K `main.tex:551` “($\lambda = 0.1$ in our protocol)”; T `03-plataforma.tex:184` “\lambda = 0.1”; código `experiments/run_protocol.py:77` `LAMBDA_HIER = 1.0` | K T | 20 |
| A2 | P0 | BCE “ponderado por prevalência”; o protocolo usa `binary_cross_entropy` sem pesos. | I `main.tex:176` “prevalence-weighted binary cross-entropy”; K `:528` “optionally weighted by class prevalence”, `:782`; T `03-plataforma.tex:174-179` (Eq. com `w_i`, “inverse prevalence”) | I K T | 20 |
| A3 | P1 | Conjunto de pares do hinge: o texto diz arestas diretas na formulação esparsa; os experimentos usam todos os pares do fecho. | I `:187` “$\mathcal{P}=\mathcal{E}$ in the sparse formulation”; K `:406`; T `03-plataforma.tex:96`, `:160`; código `run_protocol.py:18` “hinge penalty over all ancestor pairs” | I K T | 20 |
| A4 | P1 | Early stopping usa AUPRC de validação **reconciliado** em todos os braços (`hp["select"]="rec"`); os braços raw são avaliados numa época escolhida para o rec. Não declarado; favorece o rec na comparação central (mediana +0,0016). | T `05-funcat-go.tex:27` “early stopping on validation Micro-AUPRC”; K seção de protocolo | I K T | 20 (texto), 23 (controle) |
| A5 | P1 | EUR-Lex: os 9 runs param em 96–100 de 100 épocas (BCE: 100, 100, 96) — teto atingido. A validação do early stopping usa as primeiras 1.500 linhas e roda a cada 4 épocas (EUR-Lex, RCV1). Nada disso é declarado (o caso análogo do `spo` é). | `output/protocol/eurlex.json` (`best_epoch`); `run_protocol.py` (`val_step`, subconjunto) | I K T | 20, 23 |
| A6 | P2 | O limiar é escolhido nos scores de validação do modelo treinado só no treino e aplicado ao modelo refeito em treino+validação. Não declarado. | `run_protocol.py` (refit) | K T | 20 |
| A7 | P0 | Platt: o texto diz que a calibração **reduz** o AUPRC do GBDT de 0,219 para 0,212 e recomenda não calibrar. O 0,219 é do modelo refeito (treino+validação); o Platt foi aplicado ao modelo só-treino, cujo AUPRC cru é 0,2023. Na comparação correta o Platt **sobe** 0,202 → 0,212. | K `:2050` “\textbf{decreased} raw GBDT AUPRC from 0.219 to 0.212”; T `06-cross-domain.tex:72-77`; fonte: `output/protocol/*_FUN.json` → `gbdt.trainonly_auprc` (média 8 = 0,2023), `gbdt.platt_auprc` (0,2125), `gbdt.raw.auprc` (0,2191) | K T | 21 |
| A8 | P1 | F1 por nível: o texto descreve a figura, mas a figura (18/10, anterior aos JSONs de 03–05/11) inclui a raiz como nível. ArXiv “0,93 → 0,59” e WOS “0,86 → 0,65” comparam o nó raiz (L0, n = 1, não avaliado no protocolo) com as folhas; entre os níveis avaliados o protocolo dá 0,814 → 0,575 e 0,837 → 0,645. No GO, o “single node near zero” é o L1 com n = 1 da figura; no JSON o `cellcycle_GO` começa no nível 2 (0,674) e os níveis 11–12 são 0,0. Conferido na figura renderizada. | K `:1839-1847`; T `04-r-matrix-texto.tex:166-167`; figura `docs/hmc-paper/figures/fig5_per_level_f1.pdf`; fonte `output/protocol/{arxiv,wos,cellcycle_GO}.json` → `eval.rec.per_level_f1` | K T | 21 |
| A9 | P0 | Tabela de tempos: “global (FunCat avg) 15,7 s CPU / 5,5 s GPU”. No protocolo o treino de um run FunCat leva em média 254 s (MCLoss; 99–518 s) e 103 s (BCE). Os números são de uma configuração legada. | K `:1257-1280` (`:1274`); T `06-cross-domain.tex:78-80`; fonte `train_time_s` em `output/protocol/*_FUN.json` | K T | 21 |
| A10 | P0 | “Cada run grava `scores_before_postprocess.npz`, `scores_final.npz`, `metrics.json`, `run-config.json`” e “kernels cuDNN determinísticos”. O `run_protocol.py` grava um JSON por dataset, não grava scores e só chama `torch.manual_seed`/`np.random.seed`. | K `:676-682`; T `03-plataforma.tex:329-333`, `appendices/d-reprodutibilidade.tex:17-20` (“storing both raw and reconciled test scores”) | K T | 20 |
| A11 | P1 | Coluna “Train = train+validation, the partition optimized in our runs” vale para ARFF; os datasets de texto não têm refit e treinam só no treino (WOS 30.070; AAPD 35.737; ArXiv 32.000; EUR-Lex 45.000). | K `:703-710`; T `04-r-matrix-texto.tex:16`, `:21`, `appendices/a-datasets.tex:11`; fonte `n_train` e `refit: False` nos JSONs de texto | K T | 20 |
| A12 | P1 | Componentes descritos como avaliados e sem resultado no manuscrito: `LocalLevelHead`, “Tabular MLP” (aparece só na tabela de tempos legada), filtro de variância do `TabularFeatureEncoder`. As abstrações `DatasetBundle`/`Split`/`FeatureMetadata` existem em `src/`, mas os scripts que geram os números não passam por elas. | K `:316-369`, `:778-782`; T `03-plataforma.tex:19-66` | K T | 20 |
| A13 | P2 | E2E no WOS usa `max_length=512` e dropout 0,3; o texto diz limite de 256 tokens para frozen **e** E2E, e o apêndice B da tese diz dropout 0,1. O GBDT aparece como “Train only” no apêndice B, mas as colunas raw/rec vêm do modelo refeito. | `experiments/run_wos_e2e.py:104` e tokenizer; I `:281`; K `:801`; T `04-r-matrix-texto.tex:31`, `appendices/b-hiperparametros.tex` (linhas “WOS E2E” e “Tabular GBDT”) | I K T | 20 |
| A14 | P1 | O braço citado nos resumos muda por família (MCLoss em FunCat/GO, BCE em cross-domain) sem ser nomeado; a escolha coincide com o melhor no test. O IJCNN já nomeia (spec 15). | K `:76-78`; T `frontmatter/abstract.tex` (“a multilayer perceptron is within 0.002 … and above it on cross-domain”), `01-introducao.tex:129-132`, `08-conclusao.tex:47-50` | K T | 24 |
| A15 | P1 | H-MLCL+H-KNN: subamostragens não declaradas — projetor treinado em ≤ 16.000 linhas, índice k-NN com ≤ 10.000, blend escolhido em ≤ 2.000 linhas de validação. | `experiments/run_sr_gae_sota.py`; T `03-plataforma.tex:255-313` | K T | 20 |
| A16 | P0 | A avaliação pareada (`output/sr_gae_paired/`, 11/11) foi gerada pela versão **não commitada** de `run_sr_gae_sota.py` (114+/196−). A versão do repositório grava em `output/sr_gae_sota` com outra lógica e não reproduz a tabela. O script também não está na tabela do `CLAUDE.md`. | `git status` (` M experiments/run_sr_gae_sota.py`) | repo | 22 |
| A17 | P0 | “`surj_repro/` in the repository”: o diretório foi removido em `42aadba` (02/11). Os números do texto (0,232 / 0,156 / 0,352) só existem em `git show 42aadba^:surj_repro/results.md`. O script que o `CLAUDE.md` indica (`run_surj.py --experiment literal`) dá outros números: Surj 0,202, one-hot 0,194. | K `:1471`; T `05-funcat-go.tex:188`, `appendices/c-surj.tex:6`, `appendices/d-reprodutibilidade.tex:26`; `output/surj/literal/*.json` | K T repo | 22 |
| A18 | P0 | “per-seed metrics, and test score archives … publicly available” / “open-source implementation with per-seed artifacts”: `output/` é gitignored (0 arquivos versionados) e os scores de test não são gravados. | K `:2261-2266`; I `:108`; `.gitignore` | I K | 22 |
| A19 | P2 | Figuras órfãs ou com valores fixos (o `fig6_cd_diagram.pdf` de 19/10 foi conferido: ranks 1,00 / 2,62 / 2,62 / 3,75 / 5,00 e CD 2,16 corretos): `fig2_ablation.pdf`, `fig3_go_sparse.pdf`, `fig_scaling_laws.pdf` sem uso; `make_fig4_cross_domain.py` com valores hardcoded e legenda “Same architecture (global MLP+R) used throughout” embora a barra do WOS seja o modelo fine-tuned. | K `:1222`; `docs/hmc-paper/figures/` | K T | 21 |

### B. Escrita científica (KBS e tese)

| # | Sev | Achado | Evidência | Spec |
|---|---|---|---|---|
| B1 | P1 | Calibração afirmada sem medida (o próprio texto diz que o erro de calibração não é medido). | K `:2125` “well-calibrated pre-reconciliation probabilities”; T `07-discussao.tex:73` “well-behaved” | 24 |
| B2 | P1 | Limiar “C > 1.000” sem medida (removido do IJCNN no spec 16). | K `:2132`; T `07-discussao.tex:78` | 24 |
| B3 | P1 | “correlates strongly” para ρ = 0,49; “over 80% … in Gene Ontology and FunCat” (FunCat é 74–75%, GO 89%); explicações causais sem teste (prevalência-base, “morphological contour descriptors”, mecanismo do Platt). | K `:2019`, `:2033`, `:2035-2044`; T `06-cross-domain.tex:76-77`, `:173-175`, `:178-180` | 24 |
| B4 | P1 | GAE híbrido: “small, consistent gain on FunCat” (7/10, p = 0,049 sem correção) contra “contributes nothing measurable” mais adiante. | K `:1636-1639`; T `05-funcat-go.tex:211-215` | 24 |
| B5 | P1 | “primary driver of performance gains” para um efeito de mediana +0,0016. | K `:165-168` | 24 |
| B6 | P1 | Highlights e título da KBS anunciam o que o paper refuta: “SR-GAE, H-MLCL+H-KNN, and depth calibration resolve sibling collapse and lift F1”; título “… Sibling-Repulsive Ontology Graphs, and Contrastive Learning …”. | `docs/hmc-paper/highlights.md:5`; K `:42` | 24 |
| B7 | P2 | Frases truncadas ou contraditórias. | K `:1066-1067` “does not close the gap to the autoencoder branch”; `:1250-1255` “wins on F1 by leveraging the trade-off effectively” | 24 |
| B8 | P2 | Negrito e sublinhado em diferenças menores que o desvio entre seeds; “best among verifiable methods” inclui o blend, cujo ganho o texto atribui a ruído. | T `05-funcat-go.tex:62-63`, `:84`; K tabela FunCat | 24 |
| B9 | P2 | Dezenas de p-values sem correção para comparações múltiplas e sem tamanho de efeito padronizado; 3 seeds por célula. A não independência dos 27 datasets já é declarada. | K/T seções estatísticas | 24 |
| B10 | P1 | Carta de apresentação com TODO sobre o status do IJCNN (sobreposição). | `docs/hmc-paper/cover_letter.md` (“[TODO before submitting…”) | 06 (já aberto) |

### C. Tese

| # | Sev | Achado | Evidência | Spec |
|---|---|---|---|---|
| C1 | P1 | Título centrado em “Contrastive Representation Learning”; a parte contrastiva é exploratória e tem resultado negativo (o próprio texto: “its main result is negative”). | `main.tex:60`; `07-discussao.tex:42-43` | 25 |
| C2 | P1 | Revisão bibliográfica curta: capítulo 2 com 162 linhas e `refs.bib` com 24 entradas (todas citadas). Faltam as literaturas de reconciliação post-hoc em hierarquias (true-path rule ensembles, regressão isotônica, AWX), métricas hierárquicas (hF) e predição de função com redes profundas. | `chapters/02-fundamentacao.tex`; `refs.bib` | 25 |
| C3 | P1 | RQ4 pergunta só sobre Surj/embeddings/retrieval/offsets; a contribuição “RQ4a” e a resposta no cap. 8 incluem a comparação entre benchmarks e o HPT, que nenhuma RQ pede. | `01-introducao.tex:86-93` × `:127-137`; `08-conclusao.tex:45-53` | 25 |
| C4 | P1 | RQ2 (escala) é respondida só com memória. Não há medida de tempo esparso × denso; a única tabela de tempo é a legada (A9). | `08-conclusao.tex:29-36` | 21, 25 |
| C5 | P1 | Ameaças à validade omitem: A4, A5, A14, 3 seeds, baselines publicados copiados (só o C-HMCNN foi reproduzido), um único encoder de texto (SPECTER2), ArXiv limitado a 50.000, um único hardware. | `07-discussao.tex:91-125` | 25 |
| C6 | P1 | Experimentos existentes e não reportados, listados como trabalho futuro: ESM-2 em `seq_FUN`/`seq_GO` (`output/protein_lm/`, protocolo completo, 3 seeds: AUPRC 0,528 contra 0,290 e 0,601 contra 0,446 com as features originais) e embeddings de rótulo com texto do `go.obo` (`output/label_embeddings/`, só `cellcycle_FUN`). | `07-discussao.tex:105-107`; `08-conclusao.tex:76-82`; `experiments/run_protein_lm.py` | 25 |
| C7 | P2 | “batch size 4 halves the epochs required for convergence” sem dado na tabela. | `05-funcat-go.tex:54-55` | 25 |
| C8 | P2 | 4 overfull (um `\vbox` de 90 pt e um `\hbox` de 75 pt) e 11 underfull. | `main.log` | 25 |
| C9 | P0 | A lacuna que motiva a RQ1 está descrita de forma incorreta. O artigo do C-HMCNN (arXiv 2010.10151, Seção 5.5, Tabela 3) já separa os dois mecanismos: `h+` (BCE + pós-processamento), `h+MCM` (BCE + módulo de máximo) e C-HMCNN (MCM + MCLoss), na validação dos 8 FunCat. C-HMCNN vence nos 8 (média 0,241 contra 0,237 do `h+MCM`; ranking 1,00 / 2,06 / 2,94) e o artigo conclui que MCM e MCLoss ajudam. Nenhum dos três manuscritos cita essa ablação. | T `01-introducao.tex:48-51` “as an indivisible package”, `02-fundamentacao.tex:135-136` “are always applied together”, `frontmatter/abstract.tex:5`; K `main.tex:63`, `:123`; I `main.tex:72-77` “typically evaluate the combination of both” | 25 |

### D. IJCNN (só o que restou depois dos specs 13–17)

| # | Sev | Achado | Evidência | Spec |
|---|---|---|---|---|
| D1 | P0 | Autores e afiliação no PDF; submissão é duplo-cego. | `main.tex:20-22` | 19 |
| D2 | P0 | URL do repositório com o nome do autor no abstract e na conclusão. | `main.tex:50`, `:617` | 19 |
| D3 | P1 | Sem declaração de uso de IA nos agradecimentos (a KBS tem, `:2273-2277`). | `main.tex` (sem seção) | 19 |
| D4 | P0 | A2 (BCE ponderado). | `main.tex:176` | 20 |
| D5 | P0 | A18: “An open-source implementation with per-seed artifacts” — os artefatos não estão publicados. | `main.tex:108` | 22 |
| D7 | P1 | C9: a introdução não cita a ablação da Seção 5.5 do C-HMCNN. | `:72-77` | 25 |
| D6 | P1/P2 | A3 (pares do hinge), A4/A5 (seleção de época e teto do EUR-Lex não declarados), A13 (256 tokens no E2E). | `:187`, `:281`, protocolo | 20 |

## Contagem

| | P0 | P1 | P2 |
|---|---|---|---|
| IJCNN | 4 (D1, D2, A2, A18) | 5 (D3, A3, A4, A5, C9) | 1 (A13) |
| KBS | 8 (A1, A2, A7, A9, A10, A17, A18, C9) | 15 (A3, A4, A5, A8, A11, A12, A14, A15, B1–B6, B10) | 7 (A6, A13, A19, B7, B8, B9, E1) |
| Tese | 7 (A1, A2, A7, A9, A10, A17, C9) | 18 (A3, A4, A5, A8, A11, A12, A14, A15, B1–B4, C1–C6) | 9 (A6, A13, A19, B8, B9, C7, C8, E1, E2) |
| Repositório | 1 próprio (A16); A17 e A18 são compartilhados com os manuscritos | — | 1 (E3) |

São 42 achados distintos (11 P0, 20 P1, 11 P2); o mesmo achado conta em cada manuscrito em que
aparece.

## Mapa achado → spec

| Spec | Prioridade | Achados | Compute |
|---|---|---|---|
| [19 — Anonimização do IJCNN](19-ijcnn-anonimizacao.md) | P0 | D1–D3 | não |
| [20 — Descrição do protocolo](20-descricao-do-protocolo.md) | P0 | A1–A6, A10–A13, A15, D4, D6 | não |
| [21 — Números fora do protocolo](21-numeros-fora-do-protocolo.md) | P0 | A7–A9, A19, C4, E1 | CPU (minutos) |
| [22 — Proveniência e artefatos](22-proveniencia-artefatos.md) | P0 | A16–A18, E3 | não |
| [23 — Controles experimentais](23-controles-experimentais.md) | P1 | A4, A5, A7 (variante), A14 | GPU (~6–8 h) |
| [24 — Afirmações da KBS e da tese](24-afirmacoes-kbs-tese.md) | P1 | A14, B1–B9, E2 | não |
| [25 — Tese: estrutura e arguição](25-tese-estrutura.md) | P0/P1 | C1–C9, D7 | não |

Já cobertos: B10 e metadados → spec 06.

### E. Achados da verificação complementar

| # | Sev | Achado | Evidência | Spec |
|---|---|---|---|---|
| E1 | P2 | Arquivos consolidados defasados. `consolidated_results.json`: `depthcal.full_pipeline` ainda diz 27/27 com fonte `output/sr_gae_sota` (a avaliação pareada dá 25/27); `gpu_speedup` é da tabela legada (A9); `wos_frozen.std` 0,0001 e `arxiv.std` 0,0004 contra 0,0002 e 0,0005 no texto; a fonte do EUR-Lex aponta para `output/protocol_eurlex_fixedtext/` (idêntico a `output/protocol/eurlex.json`). `protocol_summary.json`: as 324 médias conferem com os JSONs, mas os desvios são populacionais (ddof = 0) e o texto usa amostrais (ddof = 1). | `docs/hmc-paper/data/consolidated_results.json`; `docs/tese/data/protocol_summary.json` | 21 |
| E2 | P2 | GAE híbrido: a tese diz “+0.0016 … across the 10 FunCat datasets, Wilcoxon p=0.049”. Nos 10 datasets a média é +0,0014 (7/10, p = 0,049); +0,0016 é a dos 8 comuns. | T `05-funcat-go.tex:212-213`; `output/autoencoder/final/gae_hybrid@64/` | 24 |
| E3 | P2 | PyPI: o pacote existe, versão 0.1.1 (11/11/2026); o `pyproject.toml` em `dev` e `main` diz 0.0.11. | `pyproject.toml:7` | 22 |

## Verificado na segunda passada (13/11)

- **PyPI:** `hmc-torch` existe (0.1.1, enviado em 11/11/2026). Divergência de versão em E3.
- **Baselines publicados:** C-HMCNN e HMCN-F — os 20 pares de valores das tabelas de FunCat,
  GO e cross-domain conferem com a Tabela 2 do artigo do C-HMCNN, assim como as médias (0,264;
  0,420; 0,422; 0,849; 0,781). WOS — HiAGM 86,04, HTC-infoMAX 86,30, HiMatch 86,70, HGCLR
  87,11 e HILL 87,28 conferem com a Tabela 2 do HILL (arXiv 2403.17307); HPT 87,16 confere com
  arXiv 2402.18825; HTC-CLIP 87,86 confere com arXiv 2506.04381.
- **Protocolo do C-HMCNN:** Adam, weight decay 1e-5, paciência 20, dropout 0,7, batch 4, refit
  em treino+validação, 10 execuções, dimensão oculta e learning rate ajustados por dataset na
  validação. O nosso protocolo usa 3 seeds e learning rate fixo — diferença a declarar.
- **C9:** confirmado e promovido a P0 (ver tabela C).
- **`run_autoencoder.py`:** a triagem não pontua o test; os deltas de validação (+0,0007 e
  7/10 para `gae_hybrid@64`; +0,0002; −0,0003; −0,0047; −0,0105; −0,0155) conferem com
  `output/autoencoder/screen/`. Final: ver E2.
- **`run_label_embeddings.py`:** segue o `train_mlp` do protocolo; nenhum uso do test para
  seleção. Só há resultado para `cellcycle_FUN`.
- **`recover_seq_ids.py` (ESM-2):** o mapeamento usa só features (comprimento, composição de
  aminoácidos, cromossomo), nunca rótulos. 3.693 de 3.900 linhas (`seq_GO`) e 3.700 de 3.919
  (`seq_FUN`) têm ORF único; 6 ORFs aparecem em mais de um split, 3 deles entre test e
  treino/validação. Não há vazamento de rótulo; o resultado do ESM-2 (C6) pode ser reportado.
- **Arquivos consolidados:** conferidos (E1).
- **Figuras 5 e 6:** renderizadas e lidas (A8 rebaixado a P1; fig. 6 correta).
- **PPGCC/UFSCar** (Normas Complementares 2024 e Regimento Interno 2022):
  - Art. 4: a tese pode ser redigida e defendida em inglês.
  - Art. 63: para a defesa são exigidos (I) publicação ou aceite em periódico ou conferência do
    **Índice Restrito** do Qualis e (II) submissão de artigo a periódico do Índice Restrito; o
    aceite do periódico cumpre também o inciso I.
  - Art. 64: aluno como primeiro autor e orientador entre os autores.
  - Art. 66: comprovação no pedido de aprovação da banca.
  - Banca: número ímpar de membros; membros não devem ser coautores de artigo com aluno e
    orientador (exceto o orientador).
  - Regimento: versão original da tese em até 1.460 dias e defesa em até 1.490 dias da
    matrícula.
  - O plano de defesa fala em “B1+”; a norma fala em Índice Restrito. Confirmar com a
    secretaria que IJCNN e KBS estão no Índice Restrito vigente, e conferir os prazos contra a
    data de matrícula.

## Não verificado

- MLCL-KNN no WOS (0,8737): artigo da Neurocomputing sem acesso aberto.
- Números reportados pelo Surj (0,312 / 0,468 e por dataset): copiados de
  `surj_repro/results.md`, não conferidos no artigo original.
- Tamanho do efeito de A4 (época escolhida pelo score reconciliado): só o spec 23 mede.
- Classificação Qualis atual do IJCNN e da KBS e a data de matrícula (ver PPGCC acima).
