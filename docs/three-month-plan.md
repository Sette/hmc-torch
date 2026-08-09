# Plano de 3 Meses para Fechar a Tese

## Ponto de partida (Agosto 2026)

**Pronto:**
- Paper ACL/EMNLP escrito (1086 linhas, 4 figuras, 10 tabelas)
- 25 datasets, 5 domínios, 2 SOTAs (ArXiv, WOS)
- Sparse R-Matrix validada em 9 datasets GO (0 violações, sem OOM)
- 91 testes, GPU 14x, artefatos reproduzíveis

**Lacunas:**
1. GO: sparse R não bate baseline (AUPRC 0.282 vs HMCN-F 0.422 vs WWW'22 0.468)
2. Sem multi-seed com desvio padrão
3. Sem análise por nível hierárquico
4. Sem teste estatístico

---

## Mês 1: Profundidade experimental (Setembro)

### Semana 1: 3 seeds em datasets principais

```
[ ] ArXiv global: seed 0, 42, 123 → F1=0.7295±0.0003 ✓ (já temos)
[ ] WOS globalE2E: seed 0, 42, 123 → F1=0.8743±? (1.5h/seed, 4.5h total)
[ ] cellcycle_FUN global: seed 0, 42, 123
[ ] seq_FUN global: seed 0, 42, 123
[ ] cellcycle_GO sparse R: seed 0, 42, 123
[ ] expr_GO sparse R: seed 0, 42, 123
```

**Output:** Tabela com mean±std para 6 datasets representativos. Números
para o paper: "our method is stable across seeds (std < 0.001 on ArXiv/WOS)".

### Semana 2-3: Análise por nível hierárquico

```
[ ] Para cada dataset (ArXiv, WOS, cellcycle_FUN, seq_FUN, cellcycle_GO):
    [ ] Computar F1 por nível de profundidade (raiz, meio, folhas)
    [ ] Computar precision/recall por nível
    [ ] Identificar onde o modelo erra mais
    [ ] Gerar gráfico: F1 vs profundidade na hierarquia
```

**Output:** Figura mostrando "F1 by hierarchy depth" para 5 datasets.
Insight esperado: F1 cai nas folhas, R-matrix ajuda mais nos níveis intermediários.

### Semana 4: Teste estatístico

```
[ ] Friedman test + Nemenyi post-hoc nos 8 datasets FUN
[ ] Critical difference diagram comparando global, GBDT, MLP, HMCN-F, C-HMCNN
[ ] Wilcoxon signed-rank: global vs GBDT (pareado por dataset)
```

**Output:** CD diagram (figura) + p-values. Paper: "global significantly
outperforms MLP (p<0.01) and is statistically tied with GBDT (p=0.12)".

---

## Mês 2: Fechar a lacuna do GO (Outubro)

### Semana 5-6: Sparse R + Ontology Embeddings

**Hipótese:** Combinar restrição binária (R-matrix) com embeddings
aprendidos da ontologia (GCN sobre o grafo GO) fecha o gap para HMCN-F.

```
[ ] Implementar OntologyLabelEncoder:
    - GCN de 2 camadas sobre o grafo da hierarquia GO
    - Produz embeddings de labels de dimensão d_label
    - Dot product com document embedding para scoring
[ ] Combinar com sparse R:
    scores = 0.5 * MLP_scores + 0.5 * ontology_scores
    reconciliar via sparse R
[ ] Rodar em 3 datasets GO: cellcycle, eisen, expr
[ ] Comparar: raw MLP vs +ontology vs +sparse R vs +both
```

**Critério de sucesso:** +ontology bate HMCN-F (AUPRC > 0.42) em pelo
menos 1 dataset GO. Se funcionar: contribuição original forte.

**Se não funcionar (Plano B):** Ablação ultra-profunda em 4 datasets.
Matriz de 3 seeds × 2 R on/off × 3 hidden dims × 3 losses = 54 experimentos
por dataset. Paper de análise: "When Does Hierarchical Consistency Help?"

### Semana 7-8: Experimentos finais + atualização do paper

```
[ ] Rodar configuração vencedora nos 9 datasets GO
[ ] Atualizar tabelas do paper com novos resultados
[ ] Escrever seção "Combining Sparse R with Ontology Embeddings"
[ ] Se SOTA no GO: atualizar abstract com "new SOTA on GO"
[ ] Se não: atualizar com análise de por que não funcionou
```

---

## Mês 3: Escrita da tese (Novembro)

### Semana 9-10: Estrutura e rascunho

```
Estrutura da tese (6 capítulos, ~150 páginas):

Cap 1: Introdução (10 pág)
  - Problema HMC, motivação, contribuições
  - [ ] Escrever

Cap 2: Fundamentação (25 pág)
  - HMC methods survey, R-matrix, evaluation metrics
  - [ ] Expandir related work do paper

Cap 3: Plataforma HMC-Torch (30 pág)
  - Design, contratos, encoders, heads
  - [ ] Expandir Cap 3 do paper com diagramas e exemplos de código

Cap 4: R-Matrix em texto (25 pág)
  - ArXiv/WOS SOTA, ablação, análise por nível
  - [ ] Paper 1 + experimentos multi-seed

Cap 5: Sparse R-Matrix para GO (25 pág)
  - Algoritmo, complexidade, experimentos GO
  - [ ] Paper 2 (se funcionou) ou análise sistemática

Cap 6: Conclusão (10 pág)
  - Contribuições, limitações, trabalhos futuros
  - [ ] Escrever
```

### Semana 11-12: Revisão e defesa

```
[ ] Revisão completa (2 passes)
[ ] Formatação ABNT (ou padrão da UFMG)
[ ] Figuras em alta resolução
[ ] Slides da defesa
[ ] Ensaio da apresentação (20min + 40min perguntas)
```

---

## Métricas de conclusão

| Marco | Data | Critério |
|-------|------|----------|
| Paper submetido | Jan 2027 | ACL/EMNLP, arXiv preprint |
| GO com ontology+R | Out 2026 | AUPRC > 0.40 em 1+ dataset |
| Todos os experimentos com 3 seeds | Set 2026 | mean±std nas tabelas |
| Tese rascunho completo | Nov 2026 | 6 capítulos, ~150 pág |
| Defesa | Dez 2026 | Aprovado |

## O que NÃO fazer

1. ❌ Novos datasets — já temos 25, é suficiente
2. ❌ Novos métodos (TabFM, ESM-2, visão) — não há tempo
3. ❌ Perseguir SOTA no FunCat (WWW'22 ontology) — não é nossa contribuição
4. ❌ Separar em 2 papers — 1 paper forte + tese é o caminho
5. ❌ Deixar escrita para depois — 1 página por dia = 90 páginas em 3 meses
