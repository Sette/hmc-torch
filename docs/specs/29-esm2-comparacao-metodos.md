# 29 — Comparação de métodos de FunCat/GO com ESM-2 (P1)

## Objetivo

Comparar, **com a mesma entrada (ESM-2 congelado)**, métodos da literatura que usam FunCat e
Gene Ontology contra o nosso (`global` + BCE + reconciliação, o melhor com ESM-2). Hoje só existe
a linha do nosso método; sem os baselines com as mesmas features, o ganho de 0,290 → 0,528
(`seq_FUN`) é do **encoder**, não do método (regra [[wos-no-sota]]: nada de "SOTA" sem
comparação de mesma entrada). Este spec fecha o passo condicional do [spec 27](27-kbs-submissao.md)
(passo 4) e o passo 6 do [spec 25](25-tese-estrutura.md).

## Escopo (decisão de projeto)

- **Em duas fases, com porta de decisão.** A Fase 1 é mínima (nosso método × C-HMCNN × MLP
  plano). Só se nosso método for melhor (ver "Porta de decisão") vale rodar a Fase 2.

- **Só `seq_FUN` e `seq_GO`.** ESM-2 embute sequências de proteína; os outros 7 conjuntos por
  família (cellcycle, derisi, eisen, expr, gasch1, gasch2, spo) são expressão gênica, sem
  sequência. Não há ESM-2 para eles.
- **Mesma entrada** para todos: embeddings ESM-2 (650M, `esm2_t33_650M_UR50D`, mean-pool)
  já em `src/hmc/features/protein.py`; mesmo mapeamento linha → ORF
  (`experiments/recover_seq_ids.py`, auditado) e mesmo tratamento das linhas sem ORF
  (média do treino + coluna indicadora).
- **Variante principal:** `esm` (só ESM-2). `esm+orig` fica como controle (já rodado: pior, 0,507).

## Fatos a usar

- Nosso método com ESM-2, BCE + rec., 3 seeds, AUPRC micro no test
  (`output/protein_lm/esm2_t33_650M_UR50D/{seq_FUN,seq_GO}.json`):
  `seq_FUN` 0,528 ± 0,004 (features do benchmark: 0,290; C-HMCNN publicado: 0,292);
  `seq_GO` 0,601 ± 0,004 (features do benchmark: 0,446). MCLoss + rec. com ESM-2: 0,524 em
  `seq_FUN`, isto é, BCE + rec. é o melhor.
- 4,8–6,5% das linhas sem ORF; 3 ORFs repetidos entre test e treino/validação.
- **Surj fora.** Não foi possível reproduzir os números relatados (`surj_repro` 0,232 contra
  0,312) e o código nunca foi liberado. Os 0,312 são nas 8 bases FunCat/GO da Tabela 2 do Surj,
  com as features do benchmark; não há evidência de que ele tenha usado ESM-2 (não conferi o
  texto original). Não existe número do Surj com ESM-2 para citar, e rodá-lo seria outra
  reprodução não verificável.
- Paridade do C-HMCNN com o loader original: `output/chmcnn_parity/` (seq_FUN 0,291 vs 0,292
  publicado). Para rodar o C-HMCNN, clone `EGiunchiglia/C-HMCNN` em `./C-HMCNN/` (gitignored;
  **não existe agora**) e importe de lá, como `experiments/compare_chmcnn_loader.py`.
  **Nunca copiar código GPL-3.0** (ver `CLAUDE.md`).
- `experiments/run_chmcnn_gap.py` já roda o modelo/R/MCLoss do C-HMCNN com a nossa receita
  (A–F); serve de base para o braço C-HMCNN.

## Métodos a comparar

**Fase 1 (escopo inicial)**

| # | Método | Referência | Origem do código | Como entra |
|---|---|---|---|---|
| 1 | **Nosso:** MLP global + BCE + rec. | este trabalho | `hmc-torch` | já rodado |
| 2 | Nosso: MLP global + MCLoss + rec. | idem | `hmc-torch` | já rodado |
| 3 | MLP plano (BCE, sem constraint) | baseline | `hmc-torch` | já rodado (`raw`) |
| 4 | **C-HMCNN(h)** (MCM + MCLoss) | Giunchiglia & Lukasiewicz, NeurIPS'20 | clone local (GPL) | rodar com ESM-2 |

**Fase 2 (só se a porta de decisão passar)**

| # | Método | Referência | Origem do código | Como entra |
|---|---|---|---|---|
| 5 | HMCN-F | Wehrmann et al., ICML'18 | reimplementar a partir do paper | rodar |
| 6 | GBDT/Platt | baseline do FunCat | `tabular_gbdt` | rodar |
| 7 | Outros que a revisão da literatura confirmar (AWX, HiLAP, ...) | — | só se houver receita fiel | avaliar |
| 8 | Clus-HMC / Clus-HMC-Ens | Vens et al., 2008 | só números publicados | **fora do rodar**; citar números originais |

**Excluído:** Surj (ver "Fatos a usar").

## Resultado da Fase 1 (2026-11-24)

`experiments/run_esm2_baselines.py` → `output/protein_lm_baselines/{seq_FUN,seq_GO}.json` e
`summary.md`. C-HMCNN = `ConstrainedFFNNModel` + MCLoss do clone (`e887855`), mesmos embeddings,
ORFs, splits, seeds (0, 42, 123), hiperparâmetros fixos (sem grid) e protocolo. AUPRC micro no
test, média ± desvio (ddof = 1):

| dataset | MLP plano (BCE, cru) | nosso BCE + rec. | nosso MCLoss + rec. | C-HMCNN | nosso − C-HMCNN |
|---|---|---|---|---|---|
| seq_FUN | 0,5215 ± 0,0037 | **0,5279 ± 0,0036** | 0,5236 ± 0,0021 | 0,5240 ± 0,0010 | +0,0039 (desvio 0,0036) |
| seq_GO | 0,5961 ± 0,0034 | **0,6011 ± 0,0038** | 0,5963 ± 0,0030 | 0,5938 ± 0,0041 | +0,0073 (desvio 0,0041) |

**Porta: passou, mas por margem curta** (diferença de ~1,1 e ~1,8 desvios, 3 seeds, sem grid).
Leituras honestas:
- O ganho do encoder (0,290 → 0,528 em `seq_FUN`) é duas ordens de grandeza maior que a
  diferença entre métodos (~0,004–0,007). A frase correta é que o encoder explica o salto.
- Nosso MCLoss + rec. ≈ C-HMCNN (esperado: mesmo objetivo e mesma restrição); a diferença vem
  do **objetivo BCE**, não de um modelo diferente.
- O MLP plano (cru) já fica a ~0,006 do nosso BCE + rec.: a reconciliação ajuda pouco com ESM-2.
- Com 3 seeds não há teste de significância; o grid de HP não foi rodado para nenhum dos lados.
- Nota técnica: o C-HMCNN estourou memória no `seq_GO` (R densa `B×C×C` na validação); o
  wrapper `ChunkedEval` só fatia a inferência em lotes de 8, sem mudar as saídas.

## Resultado da Fase 2 (2026-11-26)

Porta da Fase 1 passou, então a Fase 2 foi executada (`--methods hmcnf,gbdt` em
`run_esm2_baselines.py`), com os mesmos embeddings/ORFs/splits/seeds e hiperparâmetros fixos.
Revisão da literatura: a busca não achou métodos novos com FunCat/GO além dos já citados no
paper (C-HMCNN, HMCN-F, Clus-HMC, GBDT); AWX e HiLAP **não** rodados (sem receita fiel
verificada). Clus-HMC só com números publicados. Arquivos: `output/protein_lm_baselines/`
(`<dataset>.json` = C-HMCNN, `<dataset>_hmcnf.json`, `<dataset>_gbdt.json`, `summary.md`).

AUPRC micro no test, média ± desvio (ddof = 1), 3 seeds (GBDT: 1 seed), hiperparâmetros
fixos do protocolo:

| método | `seq_FUN` cru | `seq_FUN` rec. | `seq_GO` cru | `seq_GO` rec. |
|---|---|---|---|---|
| **HMCN-F** (reimpl. a partir do paper) | 0,5378 ± 0,0011 | **0,5432 ± 0,0012** | 0,6086 ± 0,0006 | **0,6154 ± 0,0001** |
| nosso MLP BCE | 0,5215 ± 0,0037 | 0,5279 ± 0,0036 | 0,5961 ± 0,0034 | 0,6011 ± 0,0038 |
| nosso MLP MCLoss | — | 0,5236 ± 0,0021 | — | 0,5963 ± 0,0030 |
| C-HMCNN | — | 0,5240 ± 0,0010 | — | 0,5938 ± 0,0041 |
| GBDT (1 seed; Platt) | 0,3900 (0,3304) | 0,3812 | 0,4782 (0,4509) | 0,4706 |

**O HMCN-F supera o nosso MLP nos dois datasets** (+0,015 em `seq_FUN`, +0,014 em `seq_GO`, ≥ 4
desvios), cru e reconciliado. Isso inverte a leitura da Fase 1: a vantagem de +0,004/+0,007
sobre o C-HMCNN não torna o nosso MLP global o melhor método com ESM-2. O GBDT fica ~0,14
abaixo nos dois.

Implicações para o texto (nada de "supera"/"SOTA"):
- A frase segura: o encoder explica o salto (0,290 → 0,52–0,54 em `seq_FUN`); entre os
  métodos, a ordem é HMCN-F > nosso BCE + rec. ≳ C-HMCNN ≫ GBDT, com diferenças pequenas
  entre os três primeiros.
- Reconciliar ajuda também o HMCN-F (+0,005 a +0,007): consistente com o argumento do paper de
  que a reconciliação é um pós-processamento barato e geral.
- **Ressalvas:** o HMCN-F é reimplementação nossa (hidden 384 por nível, β = 0,5, dropout e
  batch do protocolo), não o código dos autores; o nosso MLP usa hidden 2000 (`seq_FUN`) /
  9000 (`seq_GO`) e o HMCN-F 384 por nível, então a capacidade difere e o resultado pode
  refletir a arquitetura multi-nível, não o objetivo; só 2 datasets.
- O script foi retomado após uma queda de energia (JSONs de `seq_FUN` preservados); a seed
  123 do HMCN-F em `seq_GO` marcou 28.411 s por causa da suspensão da máquina, não do treino
  (~14 min por seed).

## Grid de hiperparâmetros (`experiments/run_esm2_grid.py`) — `seq_FUN` e `seq_GO` prontos

Mesmo grid para os três métodos neurais (hidden ×{0,5; 1; 2}, lr {1e-4; 3e-4}, dropout
{0,5; 0,7}; em `seq_GO` só ×{0,5; 1}); seed 0, só treino, validação AUPRC reconciliado; a melhor
configuração roda 3 seeds com refit e vai ao test uma vez. Saída: `output/protein_lm_baselines/grid/`.

`seq_FUN`, test, 3 seeds:

| método | melhor (hidden, lr, dropout) | val AUPRC | test cru | test rec. | antes do grid (rec.) |
|---|---|---|---|---|---|
| nosso MLP BCE | ×0,5, 1e-4, 0,7 | 0,4899 | 0,5219 | 0,5275 ± 0,0036 | 0,5279 |
| C-HMCNN | ×0,5, 1e-4, 0,7 | 0,4899 | 0,5241 | 0,5241 ± 0,0005 | 0,5240 |
| HMCN-F | ×1, 1e-4, 0,7 | 0,5036 | 0,5378 | 0,5432 ± 0,0012 | 0,5432 |

`seq_GO`, test, 3 seeds:

| método | melhor (hidden, lr, dropout) | val AUPRC | test cru | test rec. | antes do grid (rec.) |
|---|---|---|---|---|---|
| nosso MLP BCE | ×1, 1e-4, 0,7 | 0,5601 | 0,5927 | 0,5971 ± 0,0016 | 0,6011 |
| C-HMCNN | ×1, 1e-4, 0,7 | 0,5499 | 0,5938 | 0,5938 ± 0,0041 | 0,5938 |
| HMCN-F | ×0,5, 1e-4, 0,7 | 0,5748 | 0,6069 | 0,6148 ± 0,0003 | 0,6154 |

**O ajuste não muda nada:** em `seq_FUN` os três métodos ficam no valor do protocolo fixo (o
HMCN-F escolheu a própria configuração original e reproduz o número). Em `seq_GO` o nosso MLP
fica 0,004 abaixo (0,6011 → 0,5971, ~2 desvios; causa não investigada) e os
outros dois ficam iguais. A ordem da Fase 2 (HMCN-F > nosso MLP ≳ C-HMCNN ≫ GBDT) se sustenta
nos dois datasets com o mesmo orçamento de busca para todos. Em `seq_GO`, nosso MLP vs.
C-HMCNN: Δ = +0,0033, menor que 1 desvio do C-HMCNN (0,0041) — empate técnico, não vitória.

## Regra de redação: deixar claro que é ESM-2 (KBS, tese)

Os números desta seção **não são comparáveis** aos publicados pelos autores dos métodos
(features do benchmark, outro protocolo). Por isso, no manuscrito:

1. **Título e abertura da seção** dizem que é um experimento com **features ESM-2 congeladas**
   (`facebook/esm2_t33_650M_UR50D`, média sobre resíduos, 1281 atributos) em `seq_FUN`/`seq_GO`.
2. **Legenda da tabela** repete: "todos os métodos usam as mesmas features ESM-2, o mesmo
   protocolo e 3 seeds; os valores diferem dos publicados, que usam as features do benchmark".
3. **Rótulos**: "HMCN-F (reimplementação nossa, ESM-2)"; C-HMCNN e GBDT também "ESM-2, nosso
   protocolo". Surj não aparece (não reproduzido).
4. **Números publicados** só em tabela separada, de features do benchmark, sem comparar com
   as colunas ESM-2.
5. **Alegações** valem só entre métodos rodados por nós neste ambiente; nenhuma de SOTA.
6. Dizer que o salto 0,29 → 0,53 vem do **encoder**, e que a ordem entre métodos (HMCN-F >
   nosso MLP ≳ C-HMCNN ≫ GBDT) é a deste regime e de 2 datasets.

## Porta de decisão (após a Fase 1)

A Fase 2 só roda se, em **`seq_FUN` e `seq_GO`**, nosso BCE + rec. superar o C-HMCNN com ESM-2
por mais de 1 desvio-padrão (ddof = 1, 3 seeds), no mesmo orçamento de hiperparâmetros. Se não
passar, **parar**: reportar que o ganho 0,290 → 0,528 é do encoder e que o método não se separa
do C-HMCNN; a Fase 2 não é executada e a seção da KBS/tese fica com 4 linhas.

**Verificar antes de usar:** a lista acima é do meu conhecimento da literatura; confirmar no
paper do C-HMCNN (Tabela de FunCat/GO) e em `related work` do `main.tex` quais baselines os
trabalhos citados reportam, e se há trabalho mais recente (AWX, HiLAP, outros) que reporte
FunCat/GO. Cada método ganha uma linha na tabela só se houver código ou receita fiel.

## Protocolo (idêntico ao `run_protocol.py`)

- Treino só no split de treino; época e threshold na **validação**; test uma vez.
- 3 seeds (0, 42, 123); AUPRC micro no test sobre `to_eval`; F1 no threshold da validação.
- Cada método tem seu **orçamento de hiperparâmetros declarado**: o mesmo grid pequeno na
  validação para todos (hidden, lr, dropout), senão a comparação favorece quem teve mais ajuste
  (o C-HMCNN original ajusta hidden e lr por dataset; o nosso usa lr fixo 1e-4).
- Padronização das features (z-score com estatísticas do split de treino), igual para todos.
- Cru e reconciliado reportados lado a lado para todo método que admita reconciliação
  (`get_constr_out`/`BlockDiagonalR`; no GO, usar `BlockDiagonalR` com `nodes_idx` do dataset).
- Testes de significância: Wilcoxon pareado por seed × dataset é fraco com n=2 datasets;
  reportar média ± desvio (ddof = 1) e a diferença em desvios, sem p-value.

## Passos

**Fase 1**

1. **Revisão da literatura** (1 h): listar quais métodos reportam FunCat/GO e com que
   métrica/protocolo; atualizar a tabela acima e descartar o que não for comparável.
2. **Clonar o C-HMCNN** em `./C-HMCNN/`; confirmar `git log -1` e licença; nada vai para o `src/`.
3. **`experiments/run_esm2_baselines.py`** (novo): lê os embeddings ESM-2 já em cache de
   `run_protein_lm.py` (não recalcular), roda cada método da fase, grava
   `output/protein_lm_baselines/<dataset>.json` (`default=float` no `json.dump`).
   Reaproveitar `load_dataset`, `train_mlp` e `OBJECTIVES` de `run_protocol.py`.
4. **Braço C-HMCNN:** importar modelo/R/MCLoss do clone via o padrão de
   `compare_chmcnn_loader.py` e treinar com a entrada ESM-2 (~2 h de GPU); mesmo grid de HP
   do nosso método.
5. **Porta de decisão:** aplicar o critério acima e registrar o resultado neste spec.

**Fase 2 (condicional à porta)**

6. Rodar HMCN-F, GBDT/Platt e os demais confirmados no passo 1 com o mesmo script.
7. **Tabela e texto:** `tables/esm2_methods.tex` via `make_paper_artifacts.py` (não à mão) com
   colunas método × {seq_FUN, seq_GO} × {cru, rec.}; seção curta em KBS/tese; **IJCNN** só se
   couber no limite de 6 páginas (provavelmente não — é material da KBS de março).
8. **Redação** (também na Fase 1, com a tabela menor):
   - nosso ≈ C-HMCNN com ESM-2 → "o ganho é do encoder; a reconciliação é barata e não piora";
   - nosso > C-HMCNN → afirmar só com diferença maior que 1 desvio e declarar o orçamento de HP;
   - nosso < C-HMCNN → reportar assim.
   Em qualquer caso, tirar de Limitações "rather than end-to-end protein language models"
   (`main.tex:2211`) e declarar linhas sem ORF e os 3 ORFs repetidos.

**Passo novo (decorrente da Fase 2; em execução, ver seção do grid):** grid pequeno de HP na validação (hidden × {0,5; 1; 2},
lr {1e-4; 3e-4}, dropout {0,5; 0,7}; seed 0) para nosso BCE + rec., C-HMCNN e HMCN-F,
selecionando na validação antes de fixar o resultado de teste. Só então escolher a redação.

## Saídas esperadas

- `output/protein_lm_baselines/{seq_FUN,seq_GO}.json` com runs por seed.
- `tables/esm2_methods.tex` e uma seção de ~1 página no KBS e na tese.
- Decisão registrada sobre o que entra no IJCNN (provável: nada, só uma frase).

## Critério de aceite

- [x] Fase 1 concluída e porta de decisão registrada (passou por margem curta); Fase 2 só executada se a porta passou.
- [ ] Todos os métodos da tabela usam os mesmos embeddings, ORFs, splits e seeds.
- [ ] Orçamento de hiperparâmetros por método declarado no texto.
- [ ] Nenhum código do C-HMCNN em `src/` nem `experiments/` (só import do clone).
- [ ] Números da tabela vêm do JSON via script, não digitados.
- [ ] Nenhuma frase "SOTA" nem "supera" sem diferença > 1 desvio-padrão.
- [ ] Linhas sem ORF e ORFs repetidos declarados.

## Riscos

- **Só 2 datasets** (`seq_FUN`, `seq_GO`): a conclusão é de caso, não de família. Dizer.
- **Vazamento pelo mapeamento ORF:** já auditado (só features); manter a nota.
- **ESM-2 650M** pode ter visto as proteínas da levedura no pré-treino — não há rótulo GO
  nele, mas declare como ameaça à validade.
- **Comparação injusta com o C-HMCNN** se o grid de HP diferir; mitigado pelo passo de HP igual.

## Tempo/custo

Fase 1: ~½ dia de código + ~2 h de GPU (C-HMCNN). Fase 2 (se passar a porta): +~½ dia e ~2–4 h de GPU. Não bloqueia o
IJCNN; bloqueia a seção ESM-2 da KBS (março).
