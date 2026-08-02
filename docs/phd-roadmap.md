# PhD Roadmap: HMC-Torch — Plano Comprimido (< 1 ano)

## Diagnóstico honesto

**Tempo restante:** < 12 meses. Não dá para 3 papers + proteína + GO + teoria.
É preciso escolher **1 contribuição original forte** e executá-la com profundidade.

**O que já está pronto:**
- Plataforma funcional (91 testes, GPU, 25 datasets)
- SOTA em ArXiv (+13.5 F1) e WOS (+0.2 F1)
- FunCat competitivo com baselines neurais
- Paper em LaTeX escrito

**O que NÃO dá tempo:**
- 2 papers separados → Junte em 1 paper forte
- Proteína (ESM-2) → Depende de dados brutos que podem não estar disponíveis
- 5 seeds em tudo → 3 seeds bem escolhidos nos datasets principais
- Todos os datasets GO → Foco nos 3-4 mais citados

---

## Plano comprimido: 3 fases em 10 meses

### Fase 1: Paper Único (Mês 1-4) — Submissão ACL/EMNLP Jan 2027

**Paper:** *"The R-Matrix Constraint as a First-Class Hierarchical Classification Component"*

Este paper substitui os Paper 1 e Paper 2 do plano anterior. É um paper de método
que introduz a plataforma E demonstra resultados SOTA. Não é demo track — é main track.

**Conteúdo:**
1. Introdução: o problema HMC, R-matrix como ideia herdada, plataforma como contribuição
2. Método: DatasetBundle, Hierarchy, R-matrix layer, losses, reconciliação
3. Experimentos:
   - ArXiv/WOS: 3 seeds, ablação R on/off, curvas por nível **← JÁ TEMOS**
   - FunCat: 3 seeds, 8 datasets, comparação honesta com SOTA **← JÁ TEMOS**
   - GPU speedup: tabela CPU vs GPU **← JÁ TEMOS**
4. Análise: onde R-matrix funciona e onde não, lições aprendidas

**O que falta para este paper:**
- [ ] 3 seeds (não 5) em ArXiv e WOS (±1 semana)
- [ ] Ablação R-matrix on/off em 3 datasets representativos (±2 semanas)
- [ ] Análise de erro por nível hierárquico (±1 semana)
- [ ] Refinar o .tex atual (±2 semanas)

**Prazo realista:** 6-8 semanas. Submissão para ACL 2027 (deadline ~Jan 2027).

---

### Fase 2: Contribuição Original (Mês 4-8)

**ESCOLHA ÚNICA** — não dá para fazer as duas. Escolha baseada no que é mais
viável e impactante:

#### Opção A: Sparse R-Matrix para GO (recomendada)

**Por que esta?**
- Não depende de dados externos (GO já está em `data/HMC_data_arff/datasets_GO/`)
- A ideia é simples de implementar e testar
- Se funcionar, é contribuição algorítmica clara + abre GO para métodos de restrição
- 9 datasets GO esperando para rodar

**Plano de execução (8 semanas):**
1. [ ] Semana 1-2: Implementar `SparseRMatrix` — 2 variantes:
   - Block-diagonal: R-matriz por nível, sem cross-level (O(N) memória)
   - Top-K ancestors: só os K ancestrais mais próximos (O(N*K) memória)
2. [ ] Semana 3-4: Validar em datasets FUN (500 classes) — comparar com R densa
3. [ ] Semana 5-6: Rodar em datasets GO (4000+ classes) — isso é inédito
4. [ ] Semana 7-8: Escrever seção de resultados + análise

**Métrica de sucesso:** R esparsa com >90% do F1 da R densa (onde esta couber em GPU)
e capacidade de rodar em datasets que a R densa não consegue (OOM).

**Risco:** Se a R esparsa for muito pior que a densa, a contribuição some.
Mitigação: testar rapidamente nas semanas 1-3 para decidir se continua ou pivota.

#### Opção B: Ablação Profunda + Survey (menos risco, menos impacto)

Se a R esparsa não funcionar:

1. [ ] Ablação completa da R-matrix: 3 seeds, 4 datasets, R on/off, 3 hidden dims,
   3 num_layers, 3 loss functions → matriz de 324 experimentos
2. [ ] Análise estatística rigorosa (Friedman + Nemenyi, critical difference diagrams)
3. [ ] Paper: "When Does Hierarchical Consistency Help? A Systematic Study of
   R-Matrix Constraints in HMC"

**Métrica de sucesso:** Insights fundamentais sobre *quando* e *por que* a restrição
funciona. Paper forte de análise empírica (não de método novo).

---

### Fase 3: Tese (Mês 8-10)

**Estrutura da tese (3 contribuições):**

1. **Plataforma HMC-Torch** (Cap. 3-4)
   - Design, contratos, modularidade
   - 25 datasets, GPU, artefatos reproduzíveis

2. **R-Matrix em texto** (Cap. 5, Paper 1)
   - SOTA em ArXiv (+13.5) e WOS (+0.2)
   - Ablação, análise por nível

3. **R-Matrix Esparsa em GO** (Cap. 6, Fase 2) OU **Análise sistemática** (Opção B)
   - Depende do resultado da Fase 2

**O que NÃO vai para a tese:**
- Proteína (ESM-2): sem dados brutos confirmados
- TabFM: depende de licença, não é contribuição nossa
- Visão: sem dados brutos
- Ontology learning: complexo demais para o tempo restante

---

## Cronograma comprimido (10 meses)

```
Ago 2026:  Paper 1 — experimentos finais (3 seeds, ablação, análise por nível)
Set 2026:  Paper 1 — escrita final, submissão arXiv (preprint)
Out 2026:  Fase 2 — implementação R esparsa + validação FUN + experimentos GO
Nov 2026:  Decisão: R esparsa funcionou? Sim → paper. Não → pivota para ablação.
Dez 2027:  Deadline ACL/EMNLP (Paper 1)
Jan 2027:  Fase 2 — escrita do paper/resultados
Fev 2027:  Início da escrita da tese
Mar 2027:  Tese — rascunho completo
Abr 2027:  Tese — revisão, formatação, defesa
Mai 2027:  Deadline final (assumindo prazo de ~Jun 2027)
```

## Métricas de sucesso realistas

| Marco | Prazo | Critério mínimo |
|-------|-------|-----------------|
| Paper submetido | Dez 2026 | ACL/EMNLP/EACL main track |
| R esparsa funcional | Dez 2026 | Roda em 1+ dataset GO sem OOM |
| Tese escrita | Mar 2027 | 3 contribuições claras, 1 paper submetido |
| Defesa | Mai 2027 | Banca marcada |

## O que NÃO fazer (armadilhas de tempo)

1. ❌ Tentar implementar proteína + visão + texto + TabFM
2. ❌ Esperar dados externos (proteína, imagens) — use o que já tem
3. ❌ Perseguir SOTA no FunCat (WWW'22) — não é sua contribuição
4. ❌ Separar em 2 papers — o tempo não permite
5. ❌ Deixar a tese para o último mês — comece a escrever AGORA
