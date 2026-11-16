# 13 — Revisão do paper IJCNN 2027 (`docs/hmc-paper-ijcnn`, 2026-11-11)

Documento de apoio (não é uma tarefa sequencial). Revisão no papel de revisor do IJCNN.
Fontes lidas: `main.tex` (614 linhas), `refs.bib`, `README.md`, `Makefile`, spec 09, spec 11.
Build: `make check` OK — 6 páginas (limite 6), 0 referências indefinidas, 0 over/underfull.
O manuscrito **não foi alterado** por esta revisão.

## Veredito

O núcleo empírico é sólido e rastreável: as Tabelas 2, 3 e 5 e todas as estatísticas agregadas
foram recomputadas a partir de `output/protocol/*.json` e batem. O risco está em **duas
afirmações factualmente erradas** (proveniência do HPT; “within 0.003” atribuído ao BCE) e em uma
seção de escala que promete mais do que o dado mostra. As três são correções de redação, sem
experimento novo, e cabem nas 6 páginas se forem neutras em tamanho.

## Pontos fortes

- **Rastreabilidade.** Recomputados de `output/protocol/*.json` e idênticos ao paper: Tabela 2
  (`main.tex:331-340`), Tabela 3 linhas “Ours” (`main.tex:412-414`), Tabela 5
  (`main.tex:516-521`), linhas congeladas da Tabela 4 (`main.tex:495-498`), 26/27 e 25/27,
  medianas +0,0016 / +0,0003, Wilcoxon $p = 2{,}1\times10^{-7}$ / $1{,}3\times10^{-6}$
  (`main.tex:376-379`), Spearman $-0{,}07$ / $0{,}49$ (`main.tex:389-393`), Friedman
  $\chi^2 = 28{,}30$, CD 2,16 (`main.tex:427-428`), ganhos por dataset no GO (`main.tex:461-462`).
- **Protocolo declarado e cumprido:** época e limiar na validação, 3 seeds, desvio amostral;
  exceções (GBDT, ablação GO seed 0) estão rotuladas (`main.tex:289-293`).
- **Paridade com o C-HMCNN verificada** (`output/chmcnn_parity/summary.md`: diferença máxima de
  0,002 no FunCat), o que sustenta `main.tex:555`.
- **Limitações honestas:** a não independência dos 27 datasets e o tamanho pequeno do efeito
  estão no texto (`main.tex:586-589`).
- **Contagens consistentes:** 4 + 10 + 9 + 4 = 27; faixas da Tabela 1 batem com os JSONs.

## Achados

| # | Sev | Passe | Achado | Evidência | Spec |
|---|---|---|---|---|---|
| 1 | P0 | C/H | “official code” do HPT é a reimplementação do repo | `main.tex:530-531` | 14 |
| 2 | P0 | A/B | “within 0.003” atribuído a BCE + rec. e ao HMCN-F | `main.tex:606-607`, `:575` | 15 |
| 3 | P0 | A/C | Nemenyi separa C-HMCNN de BCE + rec.; omitido | `main.tex:426-431` | 15 |
| 4 | P1 | C | Tabela WOS: HPT reconciliado omitido; colunas enganosas | `main.tex:484-492` | 14 |
| 5 | P0 | A/C/H | Teste de escala GO: script legado fora do protocolo; OOM não ocorre como descrito | `main.tex:462-468`, `:77-79` | 16 |
| 6 | P1 | A | Afirmação por nível sem evidência e contradita | `main.tex:440-442` | 15 |
| 7 | P1 | A | GAE “beating C-HMCNN” dentro do ruído | `main.tex:432-435` | 15 |
| 8 | P1 | F | Anonimato (autores e URL do repositório) | `main.tex:20-22`, `:49`, `:609` | 09 |
| 9 | P2 | A | Exemplo de “melhora” que mostra queda | `main.tex:381-383` | 17 |
| 10 | P2 | E | Referência à Tabela 2 para valores que ela não tem | `main.tex:545-546` | 17 |
| 11 | P2 | E | Notação `\mathbf{p}` duplicada; MCLoss truncado | `main.tex:167`, `:216`, `:188-190` | 17 |
| 12 | P2 | H | Surj: “inflate both models by 1.09×–1.45×” | `main.tex:562-564` | 17 |
| 13 | P2 | C | Subamostra de 50 mil do ArXiv não declarada | `main.tex:264` | 17 |
| 14 | P2 | B | Negrito em empate visível na Tabela 2 | `main.tex:331` | 17 |

### 1. “Official code” do HPT (P0)

**Evidência:** `main.tex:530-531` — “Re-executing HPT's official code and selection rule on the
exact same split across seeds 0, 42, and 123 yields $0.8691 \pm 0.0023$”.
**Fonte:** `output/hpt/final/wos/bert_none_down_hpt_raw_seed{0,42,123}.json` → raw/thr0 F1 =
0,8672 / 0,8685 / 0,8716, exatamente os scores por seed citados em `hmc-paper/main.tex:899`.
Esses JSONs são produzidos por `experiments/run_hpt.py` sobre `src/hmc/models/global_classifier/hpt/`,
que o `CLAUDE.md` descreve como “reimplementação sobre transformers 5 / PyG 2.7; `HPT/` é só a
referência”. O spec 11 (linha 64) lista os desvios: inicialização dos rótulos sem `[CLS]`/`[SEP]`,
MLM só no vocabulário original, batch 8 × acumulação 2, avaliação padrão em vez do loop do HPT.
**Por que importa:** é uma afirmação de proveniência verificável e falsa; o paper tem como
contribuição um “reproducibility check of published baselines” (`main.tex:104-106`) e critica o
Surj por não ser reprodutível. O mesmo texto está em `hmc-paper/main.tex:897` e
`tese/chapters/04-r-matrix-texto.tex:102-103`.
**Correção:** “our reimplementation of HPT, with its configuration and selection rule”, com os
desvios em uma frase → spec 14.

### 2. “Within 0.003” com o sujeito errado (P0)

**Evidência:** `main.tex:606-607` — “BCE with reconciliation matches MCLoss while avoiding
raw-score collapse, stays within $0.003$ AUPRC of C-HMCNN on FunCat and GO”.
`main.tex:575` — “the sparse R-matrix stays within $0.003$ AUPRC of C-HMCNN and HMCN-F”.
**Fonte:** Tabela 3 (`main.tex:409-414`) e `output/protocol/` (8 datasets comuns): BCE + rec. =
0,2593 (FunCat) e 0,4163 (GO) contra 0,264 e 0,420 → diferenças de 0,005 e 0,004. Só o
MCLoss + rec. (0,2608; 0,4177) fica em 0,003. Contra o HMCN-F no GO, 0,418 vs 0,422 = 0,004.
**Por que importa:** a conclusão contradiz a tabela do próprio paper, na frase que resume o
resultado. O abstract (`main.tex:43-44`) está correto.
**Correção:** trocar o sujeito ou o número → spec 15.

### 3. Nemenyi: BCE + rec. é significativamente pior que o C-HMCNN (P0)

**Evidência:** `main.tex:428-430` — “does not separate our model (average rank $2.62$) from
HMCN-F ($2.62$) or C-HMCNN ($1.00$)”.
**Fonte:** `docs/tese/data/protocol_summary.json` → `funcat_stats`: 5 métodos (C-HMCNN, HMCN-F,
GBDT, MLP+R, MCLoss); `MLP+R` (BCE + rec.) tem rank 3,75; `significant_pairs` inclui
`["C-HMCNN", "MLP+R", 2.75]`, acima do CD de 2,157.
**Por que importa:** o paper recomenda BCE + reconciliação (`main.tex:578-582`) e o seu próprio
teste diz que essa configuração é separada do C-HMCNN no FunCat. O texto reporta só o rank do
MCLoss e cita 4 dos 5 métodos; um revisor que refizer a conta com a Tabela 3 encontra a omissão.
**Correção:** reportar o rank e o par significativo, e ajustar a mensagem (paridade vale para
MCLoss + rec.; BCE + rec. fica 0,004–0,005 atrás) → spec 15.

### 4. Tabela do WOS (P1)

**Evidência:** `main.tex:491` — “HPT (re-executed, raw@0) & 0.8691 $\pm$ 0.0023 & --”; cabeçalho
`main.tex:486` “WOS end-to-end comparison (like-for-like)”; baselines publicados
(`main.tex:487-490`) na coluna “Reconciled F1”.
**Fonte:** `output/hpt/final/wos/summary.json`: `raw/val` = 0,8697 ± 0,0023; `rec/thr0` =
0,8702 ± 0,0027; `rec/val` = 0,8712 ± 0,0029. A nossa linha (`main.tex:492`) usa limiar da
validação: 0,8702 / 0,8704.
**Por que importa:** no nosso protocolo (reconciliado, limiar na validação) a reimplementação do
HPT fica nominalmente **acima** do nosso modelo; a tabela mostra só a célula em que fica abaixo.
Nada disso muda a conclusão “dentro do ruído”, mas a seleção de célula parece conveniente.
Os números publicados não são reconciliados e não deveriam estar sob esse cabeçalho.
**Correção:** preencher a célula do HPT e renomear colunas/cabeçalho → spec 14.

### 5. Teste de escala no GO (P1)

**Evidência:** `main.tex:463-468` — “the dense $B\times C\times C$ R-matrix tensor ($\approx
8.8$~GB in float64; 4.3~GB in float32) causes an out-of-memory failure on a 12~GB GPU […]
improves AUPRC on all 9 datasets (mean $+0.0055$, up to $+0.0193$ on \texttt{eisen\_GO})”;
`main.tex:77-79` — “materializing the dense $C\times C$ ancestor-closure matrix is prohibitive in
GPU memory”.
**Fonte:** `hmc-paper/main.tex:1309` diz “Dense R **would** OOM at batch 64” (estimativa);
`hmc-paper/main.tex:1306` mostra ΔF1 médio de **−0,0051** (F1 cai em 8 de 9); `:1308` — treino
com “identity R-matrix”; `:1328-1332` — “At C-HMCNN's batch size of 4 the dense step does fit”.
O float64 vem de `src/hmc/models/global_classifier/constraint/utils.py:14` (`x.double()`).
**Por que importa:** (i) 4,3 GB cabem em 12 GB, então a frase se refuta sozinha; (ii) o OOM é
dado como observado; (iii) só o ganho de AUPRC do teste é reportado; (iv) a matriz em si ocupa
65 MB (`main.tex:238`), não é “prohibitive” — o gargalo é o intermediário em lote, e o protocolo
principal usa batch 4, em que o denso cabe.
**Correção:** redação exata e reporte do F1 → spec 16.

### 6. Afirmação por nível (P1)

**Evidência:** `main.tex:440-442` — “Across all families, per-level F1 drops steeply with depth,
confirming that deep, sparse levels are where bottom-up reconciliation modifies the most
predictions.”
**Fonte:** `per_level_f1` em `output/protocol/*.json` (BCE, média de 3 seeds), raw → Δ rec.:
Diatoms L1 +0,021, L2 0,000, L3 0,000; EUR-Lex L1–L3 ≈ +0,010, L6 −0,007; WOS L1 +0,005, L2 0,000;
`cellcycle_FUN` L5 −0,011. O F1 também não é monotônico (EUR-Lex L7 = 0,761 > L6 = 0,659).
**Por que importa:** não há tabela nem figura por nível no paper, e o dado aponta o contrário nas
árvores (a reconciliação só altera nós internos, `main.tex:200-201`).
**Correção:** remover a frase ou trocá-la pelo que o dado mostra → spec 15.

### 7. GAE “beating C-HMCNN” (P1)

**Evidência:** `main.tex:432-435` — “raises the FunCat average to $0.262$ (beating C-HMCNN on
\texttt{derisi\_FUN} at $0.197$ vs.\ $0.195$ and \texttt{spo\_FUN} at $0.217$ vs.\ $0.215$)”.
**Fonte:** `output/autoencoder/final/gae_hybrid@64/`: média 0,2625 vs 0,2608 do MCLoss + rec.
(+0,0016); derisi 0,1965 ± 0,0007, spo 0,2168 ± 0,0009; pior que o MCLoss + rec. em expr, gasch2
e seq.
**Por que importa:** diferença de 0,002 contra um número publicado sem variância, e o README do
paper declara que autoencoders ficam para a KBS.
**Correção:** suavizar para “within seed noise” ou cortar e recuperar espaço → spec 15.

### 8. Anonimato (P1, coberto)

**Evidência:** `main.tex:20-22` (bloco de autores ativo), `main.tex:49` e `:609`
(`github.com/Sette/hmc-torch`).
**Estado:** o spec 09 tem o item aberto “manter bloco anônimo se submissão double-blind”, mas não
menciona as duas URLs, que identificam o autor mesmo com o bloco anônimo. A regra do IJCNN 2027
não está escrita no projeto (ver “Não verificado”).

### 9–14. Polimento (P2)

- **9.** `main.tex:381-383` — “reconciliation brings BCE to $0.4332$” logo após BCE raw
  “($0.4333$)”: o exemplo escolhido para “consistently improves” mostra queda
  (`cellcycle_GO` é um dos 2 datasets em que o F1 não melhora).
- **10.** `main.tex:545-546` — “($0.537$ on Diatoms and $0.649$ on ImCLEF07a;
  Table~\ref{tab:decomp})”: a Tabela 2 só tem médias por família.
- **11.** `main.tex:167` define $\mathbf{p}$ como probabilidades e `main.tex:216` reutiliza
  $\mathbf{p}$ como vetor de pais; `main.tex:188-190` — “while masking descendants that are
  positive only when the target itself is negative” não é uma frase interpretável.
- **12.** `main.tex:562-564` — “they inflate both models by $1.09\times$--$1.45\times$”; na KBS
  (`hmc-paper/main.tex:1508-1512`) o nosso modelo chega a 1,09–1,12× do valor *reportado pelo
  Surj* e o C-HMCNN infla 1,40–1,45× o seu publicado.
- **13.** `main.tex:264` — “ArXiv (158 evaluated classes)”: `output/protocol/arxiv.json` tem
  32.000 / 8.000 / 10.000 (cap de 50 mil); o tamanho não é declarado.
- **14.** `main.tex:331` — Text AUPRC mostra 0,845 (BCE + rec.) e **0,845** (MCLoss + rec.);
  sem arredondar, 0,8450 vs 0,8454.

## Objeções prováveis e respostas

### 1. “Vocês rodaram mesmo o código do HPT?”
**Risco:** alto. **Resposta hoje:** não; é uma reimplementação que atinge 0,8716 na seed 123.
**Ação:** spec 14.

### 2. “Se BCE + reconciliação basta, por que o headline usa MCLoss?”
**Risco:** alto. **Resposta hoje:** no FunCat/GO o MCLoss + rec. é 0,002 melhor e é o único
dentro de 0,003 do C-HMCNN; BCE + rec. é separado pelo Nemenyi. **Ação:** spec 15.

### 3. “Os ganhos são de 0,002 — isso importa?”
**Risco:** alto. **Resposta hoje:** o paper já assume efeito pequeno e consistente
(`main.tex:586-589`) e vende a garantia de consistência. **Ação:** já mitigado (spec 08, item 2).

### 4. “A escalabilidade é real se o protocolo usa batch 4?”
**Risco:** médio. **Resposta hoje:** o operador remove o teto de batch; não é a única forma de
treinar. **Ação:** spec 16.

### 5. “Faltam os baselines 2023–2024 na tabela do WOS.”
**Risco:** médio. **Resposta hoje:** estão citados em prosa (`main.tex:533-536`) e ficam acima
do nosso número; a justificativa é que não garantem GHV = 0. **Ação:** já mapeado (spec 08,
item 3); os três valores não foram conferidos nos originais.

## Mapa achado → spec

| Achado | Spec | Ação principal |
|---|---|---|
| 1, 4 | 14 | redação “reimplementação” + tabela WOS, nos três manuscritos |
| 2, 3, 6, 7 | 15 | corrigir “within 0.003”, reportar Nemenyi do BCE, cortar frases sem suporte |
| 5 | 16 | OOM estimado, float64 explicado, F1 do teste, “prohibitive” |
| 9–14 | 17 | edição de texto |
| 8 | 09 | decidir anonimato; incluir as URLs no item |

## Já coberto

- Achado 8 → spec 09 (item aberto). O spec 09 está desatualizado em dois pontos: diz “5 páginas
  no estado atual” e mantém “preencher orientador” aberto, mas `main.tex:20` já traz o coautor.
- Objeções 3 e 5 → spec 08 (mitigações aplicadas na KBS).

## Não verificado

- Regras do IJCNN 2027 (double-blind, limite de palavras do abstract — o atual tem ~250): não
  estão escritas no projeto; conferir na chamada de trabalhos.
- Números publicados dos baselines (C-HMCNN, HMCN-F, HiAGM, HTCInfoMax, HGCLR, HPT, HILL,
  MLCL-KNN, HTC-CLIP) não foram conferidos nos papers originais.
- ~~Teste batch-64 do GO: JSON de origem não localizado.~~ **Resolvido em 11/11/2026:** a origem
  é o script legado `experiments/run_go_experiments.py` (train+valid, limiar no test, seed 42,
  `results.json` ausente) e o OOM foi medido — o denso float64 **cabe** em batch 64 na
  inferência (8,16 GiB) e só estoura no treino. O achado 5 passa a **P0**; detalhes no spec 16.
- “trains in seconds per dataset” (`main.tex:467`): sem fonte identificada.
- Surj: 25 configurações, cosseno ≥ 0,997 e +0,076 conferidos só contra o texto da KBS, não
  contra `output/surj/`.
- Linha GBDT da Tabela 3 (0,226) e baselines da Tabela 5: conferidos só contra
  `consolidated_results.json`.
