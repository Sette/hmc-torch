# HMC-Torch: Implementação — Status e Próximos Passos

## Linha do tempo: < 1 ano até a defesa

---

## ✅ Completado (Marcos 0-7)

| Marco | O quê | Status |
|-------|-------|--------|
| 0 | Restaurar datasets, corrigir parser ARFF, pytest | ✅ |
| 1 | Contratos: DatasetBundle, Hierarchy, TreeHierarchy, DagHierarchy | ✅ |
| 2 | Heads: GlobalSigmoid, LocalLevel, TreePath + losses + reconciliação + calibração | ✅ |
| 3 | Baselines tabulares: GBDT One-vs-Rest, Residual MLP | ✅ |
| 4 | TabFM local-condicional (mock tests, implementação pronta) | ✅ |
| 5 | FeatureEncoders: tabular, texto, vision/protein/expression (placeholders) | ✅ |
| 6 | YAML configs, manifestos, optional-dependencies | ✅ |
| 7 | CI (GitHub Actions), 91 testes, mocks | ✅ |

**Resultados experimentais prontos:**
- ArXiv: Micro-F1 0.7295 (SOTA, +13.5)
- WOS: Micro-F1 0.8743 (SOTA, +0.2)
- FunCat: 8 datasets, competitivo com HMCN-F/C-HMCNN, abaixo do WWW'22
- GPU: 14× speedup (RTX 3060)

---

## 🔴 Prioridade 1: Paper (Ago—Dez 2026, ~8 semanas de trabalho)

### Experimentos que faltam para o paper

```
Semana 1-2: Seeds e ablação
  [ ] Rodar ArXiv global com 3 seeds (seed=0, 42, 123)
  [ ] Rodar WOS globalE2E com 3 seeds
  [ ] Ablação R-matrix on/off: ArXiv, WOS, cellcycle_FUN, seq_FUN
  [ ] Salvar desvio padrão em todos os resultados

Semana 3-4: Curvas e análise
  [ ] Curvas de aprendizado (F1 por época) para global e MLP
  [ ] Análise de erro por nível hierárquico (raiz, meio, folhas)
  [ ] Matriz de confusão hierárquica

Semana 5-6: Refinamento do paper
  [ ] Atualizar .tex com resultados multi-seed
  [ ] Adicionar figuras (curvas, ablação, erro por nível)
  [ ] Revisão de related work
  [ ] Submeter para arXiv como preprint

Semana 7-8: Submissão
  [ ] Formatar para ACL/EMNLP
  [ ] Revisão final
  [ ] Submeter (deadline ~Jan 2027)
```

### Paper: estrutura final

**Título:** *"The R-Matrix Constraint as a First-Class Hierarchical Classification Component"*

**Contribuição:** Plataforma modular que trata a R-matrix (Giunchiglia 2018) como
componente reutilizável + demonstração empírica de que restrições hierárquicas
simples superam GCNs em taxonomias limpas.

**Tabelas necessárias:**
1. ArXiv/WOS SOTA comparison (já temos)
2. FunCat AUPRC comparison com SOTA (já temos)
3. Ablação R-matrix on/off (falta)
4. GPU speedup (já temos)
5. Erro por nível hierárquico (falta)

---

## 🟡 Prioridade 2: R Esparsa para GO (Out—Dez 2026)

**Pergunta:** Dá para estender a R-matrix para datasets GO (4000+ classes) usando
aproximação esparsa?

**Por que isso importa:** A R-matrix densa é O(N²) — impossível para GO.
Se funcionar, é contribuição algorítmica original + abre 9 datasets GO.

**Implementação (4-6 semanas):**

```
Semana 1-2: SparseRMatrix
  [ ] BlockDiagonalRMatrix: uma R por nível, sem cross-level
  [ ] TopKRMatrix: só K ancestrais mais próximos
  [ ] Testes unitários para ambas

Semana 3-4: Validação em FUN
  [ ] Comparar R esparsa vs R densa nos 8 FUN
  [ ] Medir: F1, AUPRC, memória, tempo
  [ ] Critério: R esparsa ≥ 90% do F1 da R densa?

Semana 5-6: Experimentos GO
  [ ] Rodar em 3-4 datasets GO (cellcycle_GO, eisen_GO, expr_GO, seq_GO)
  [ ] Comparar com baselines sem R (global sem constraint)
  [ ] Documentar limites de memória
```

**Se funcionar:** +1 capítulo da tese, +1 contribuição original clara.

**Se não funcionar:** Pivota para ablação profunda (Opção B do roadmap).
Ablação de 324 experimentos com análise estatística rigorosa também é tese.

---

## 🟢 Prioridade 3: Tese (Mar—Mai 2027)

### Estrutura (3 contribuições)

```
Cap 1: Introdução
Cap 2: Fundamentação (HMC, R-matrix, métodos)
Cap 3: Plataforma HMC-Torch (design, contratos, modularidade)
Cap 4: Experimentos em texto (ArXiv/WOS SOTA + ablação)
Cap 5: R-matrix esparsa para GO OU Análise sistemática
Cap 6: Conclusão
```

### Regra de ouro: comece a escrever AGORA

Cada experimento que rodar, escreva 1 parágrafo na tese. Não deixe acumular.
A tese são 6 capítulos. Se você escrever 1 página por semana, em 30 semanas
são 180 páginas — mais que suficiente.

---

## ❌ O que NÃO vai para o plano

| Item | Motivo |
|------|--------|
| Proteína (ESM-2) | Sem dados brutos confirmados. Risco alto de atraso. |
| Visão (DINOv2) | Sem dados brutos. |
| TabFM | Licença não-comercial, não é contribuição nossa. |
| Ontology learning (WWW'22) | Complexo, 6+ meses, não é nossa contribuição. |
| 2 papers separados | Tempo insuficiente. 1 paper forte + tese. |
| Todos os datasets GO | Foco em 3-4 representativos. |

---

## Linha do tempo visual

```
Ago 2026  ████████████░  Paper: experimentos (3 seeds, ablação, curvas)
Set 2026  ████████████░  Paper: escrita, arXiv preprint
Out 2026  ████████████░  R esparsa: implementação + validação FUN + experimentos GO
Nov 2026  ████████████░  Decisão GO + submissão ACL
Dez 2026  ██░░░░░░░░░░░  Paper sob revisão. Início escrita tese.
Jan 2027  ████████████░  Tese: Cap 2-3 (fundamentação + plataforma)
Fev 2027  ████████████░  Tese: Cap 4-5 (resultados + contribuição)
Mar 2027  ████████████░  Tese: Cap 1,6 + revisão
Abr 2027  ████████████░  Formatação final, defesa
Mai 2027  ██░░░░░░░░░░░  Deadline
```
