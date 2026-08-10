# Elevando a contribuição científica do HMC-Torch

> Análise do estado atual, plano de ação, e estratégia de dois papers para ir
> além de "um bom pacote com uma contribuição real".

---

## Estratégia de dois papers

```
Paper 1 (atual)                    Paper 2 (este plano)
─────────────────────────────      ─────────────────────────────
Infraestrutura + método            Análise científica
Plataforma modular                 Entendimento dos mecanismos
Block-diagonal R-matrix            Quando/por que/quanto ajuda
SOTA em ArXiv, WOS, AAPD           Leis de escala + ablação
Honest failure analysis            LLM comparison (2026)
                                   Multi-seed robusto
```

O Paper 1 mostra **que** a R-matrix funciona. O Paper 2 investiga **quando, por
que e quanto** — decompondo o mecanismo, medindo leis de escala, e
contextualizando contra LLMs.

### Paper 2 — Sugestões de título

- *"Understanding Hierarchical Constraints: When and Why the R-Matrix Helps in HMC"*
- *"Scaling Laws for Hierarchical Multi-Label Classification"*
- *"Decomposing the R-Matrix: Training Consistency vs Inference Reconciliation"*

### Venues para o Paper 2

| Venue | Fit | Deadline típico |
|---|---|---|
| **EMNLP 2027** | HMC em NLP + scaling laws | ~Junho 2027 |
| **ACL 2027** | Análise + LLM comparison é quente | ~Dezembro 2026 |
| **JMLR** | Journal, permite profundidade | Rolling |
| **TMLR** | Journal aberto, rápido | Rolling |
| **NeurIPS 2027** | Scaling laws + ablations são core ML | ~Maio 2027 |

---

## Diagnóstico atual

O trabalho tem três bases sólidas:

1. **Block-diagonal R-matrix** para DAGs grandes — extensão esparsa O(N) que
   viabiliza R-matrix em Gene Ontology (4000+ classes), com zero violações
   hierárquicas e >1400× de redução de memória.
2. **SOTA em benchmarks estabelecidos** — ArXiv (+13.5 pts sobre HiAGM), WOS e
   AAPD, usando embeddings frozen (SPECTER2) + MLP + R-matrix.
3. **Honestidade intelectual** — o paper credita Giunchiglia & Lukasiewicz
   (2018) pela R-matrix original, documenta onde o método falha (FunCat/GO:
   ontology learning ganha por +39–71% AUPRC), e não faz claims exagerados.

As limitações atuais:

- Nenhuma contribuição teórica (inteiramente empírico)
- Comparação contra baselines de 2018–2022 (nada de 2024–2026)
- Ablação mencionada mas não mostrada sistematicamente
- Multi-seed limitado (1 seed em WOS E2E)
- R-matrix é binária e fixa — não aprende

---

## Quick Wins — Status da implementação

Os 4 quick wins foram implementados como scripts autônomos em `experiments/`.
**Código pronto, experimentos pendentes de execução.**

### Quick Win 1: Ablação sistemática ✅ código pronto

**Script:** `experiments/run_ablation.py`
**Visualização:** `experiments/plot_ablation.py`

4 configurações × 3 tipos de hierarquia:

| Modo | Treino (MC-loss) | Eval (get_constr_out) |
|---|---|---|
| `bce_only` | Identidade | Identidade (`baseline_model=True`) |
| `consistency_loss` | R densa | Identidade (`baseline_model=True`) |
| `reconciliation` | Identidade | R densa |
| `both` | R densa | R densa |

Datasets: ArXiv (árvore rasa), cellcycle_FUN (árvore profunda), cellcycle_GO
(DAG, BlockDiagonalR).

Hipótese: `reconciliation` sozinha provê a maior parte do ganho; `consistency_loss`
adiciona pouco — confirmando que $\lambda=0$ é a escolha certa.

**Output:** `output/ablation/results.json` →
`docs/hmc-paper/figures/fig_ablation.pdf`

### Quick Win 2: Leis de escala ✅ código pronto

**Script:** `experiments/run_scaling_laws.py`
**Visualização:** `experiments/plot_scaling_laws.py`

Roda `global` (MLP + R-matrix) em 25 datasets, coletando:
- Métricas: `f1`, `auprc`, `time_s`
- Propriedades da hierarquia: `n_nodes`, `max_depth`, `avg_branching`,
  `n_roots`, `n_leaves`
- Propriedades dos dados: `n_train`, `n_test`, `n_features`, `ratio`
  (n_train / n_classes)

4 gráficos gerados:
1. F1 vs n_classes (log scale), colorido por modalidade
2. F1 vs max_depth
3. Tempo de treino vs n_classes × n_train
4. Histograma da razão dados/classes por modalidade

**Output:** `output/scaling_laws/results.json` →
`docs/hmc-paper/figures/fig_scaling_laws.pdf`

### Quick Win 3: Multi-seed (5 seeds) ✅ código pronto

**Script:** `experiments/run_multi_seed_full.py`

6 datasets × 5 seeds [42, 123, 456, 789, 1024]:
- Texto: ArXiv, WOS, AAPD (frozen SPECTER2 + R)
- Tabular: cellcycle_FUN, seq_FUN (MLP + R)
- DAG: cellcycle_GO (MLP + BlockDiagonalR)

Métricas com mean ± std (ddof=1) para F1 e AUPRC.

**Output:** `output/multi_seed_full/results.json`

### Quick Win 4: Comparação com LLMs ✅ código pronto

**Script:** `experiments/run_llm_comparison.py`

Zero-shot hierarchical classification em 200 amostras de teste do ArXiv e WOS.
Detecta automaticamente APIs disponíveis:
- OpenAI (GPT-4o) — `OPENAI_API_KEY`
- Anthropic (Claude) — `ANTHROPIC_API_KEY`
- Ollama local (fallback, sem API key)

Prompt pede caminhos completos (ex: `cs.AI stat.ML`), ativa ancestrais
automaticamente, calcula micro-F1 e AUPRC.

**Output:** `output/llm_comparison/results.json`

### Como executar

```bash
export PYTHONPATH=src
. .venv/bin/activate

python experiments/run_ablation.py        # ~30 min GPU
python experiments/plot_ablation.py

python experiments/run_scaling_laws.py    # ~2-4 horas GPU
python experiments/plot_scaling_laws.py

python experiments/run_multi_seed_full.py # ~3-6 horas GPU

OPENAI_API_KEY=sk-... python experiments/run_llm_comparison.py
```

---

## 1. Profundidade teórica

### 1.1 Caracterização formal de quando a R-matrix ajuda

**Problema:** O paper observa que a R-matrix funciona bem em "hierarquias
limpas e com poucos dados", mas isso é observação, não teoria.

**Direção:** Provar bounds formais. Dada uma hierarquia com profundidade $D$,
fator de branching $B$, e $N$ amostras de treino, qual o ganho esperado da
restrição? Sob quais condições a R-matrix é garantia de melhoria?

**Resultado esperado:** Um teorema do tipo: "Para uma hierarquia tree-structured
com depth $D$ e $N$ amostras, a R-matrix constraint reduz o erro de
generalização em $\Omega(D / \sqrt{N})$ quando as classes folha têm poucos
exemplos positivos."

### 1.2 Gap de aproximação do block-diagonal

**Problema:** O algoritmo é correto (produz mesmo resultado do denso), mas não
há análise de complexidade fina.

**Direção:** Quantas iterações são necessárias em função da topologia do DAG?
Qual o worst-case para DAGs com múltiplos pais por nó? Provar que o número de
iterações é limitado por $\min(D, P_{max})$ onde $P_{max}$ é o número máximo
de pais por nó.

**Resultado esperado:** Um resultado técnico publicável: "Block-diagonal
reconciliation converges in at most $\min(D, P_{max})$ iterations for any DAG."

### 1.3 Conexão com graph signal processing

**Problema:** A reconciliação bottom-up é essencialmente um filtro de grafo,
mas isso não é explorado teoricamente.

**Direção:** Conectar com spectral graph theory. A R-matrix é um caso
particular de operador de smoothing em grafos hierárquicos. Mostrar que a
reconciliação max-propagation é equivalente a um filtro passa-baixa no grafo
da hierarquia.

**Resultado esperado:** Abre diálogo com literatura de GNNs e graph signal
processing, aumentando o público potencial do paper.

---

## 2. Profundidade empírica

### 2.1 Ablação sistemática por componente

**Status:** Script implementado (`experiments/run_ablation.py`).

**O que o script faz:** 4 configurações (bce_only, consistency_loss,
reconciliation, both) × 3 tipos de hierarquia. Separa o efeito do MC-loss
(treino) da reconciliação bottom-up (inferência).

**Resultado esperado:** Mostrar que a reconciliação na inferência domina o
ganho, e a consistency loss no treino é redundante quando o modelo aprende
naturalmente. Isso vira um insight científico, não só uma escolha de engenharia.

### 2.2 Leis de escala para HMC

**Status:** Script implementado (`experiments/run_scaling_laws.py`).

**Problema:** Ninguém na literatura de HMC estudou como o desempenho escala
com propriedades da hierarquia.

**O que o script coleta:** 25 datasets com propriedades da hierarquia
(n_nodes, max_depth, avg_branching) + métricas (f1, auprc, time_s).

**Resultado esperado:** "Scaling Laws for Hierarchical Multi-Label
Classification" — um paper só com isso seria citado como referência.

### 2.3 Multi-seed completo

**Status:** Script implementado (`experiments/run_multi_seed_full.py`).

**O que existia:** ArXiv com 3 seeds, WOS E2E com 1 seed ("devido a custo
computacional").

**O que o script faz:** 5 seeds com desvio padrão para 6 datasets principais
(ArXiv, WOS, AAPD, cellcycle_FUN, seq_FUN, cellcycle_GO).

### 2.4 Comparação com métodos 2024–2026

**Status:** Script implementado (`experiments/run_llm_comparison.py`).

**O que existia:** Baselines de 2018–2022 (HiAGM, HTC-infoMAX, HiMatch,
HMCN-F, C-HMCNN).

**O que o script faz:** Zero-shot com GPT-4o / Claude / Ollama vs HMC-Torch.

**A fazer (não implementado):**
- Graph transformers aplicados a HMC (se existirem)
- Métodos com contrastive learning em hierarquias
- Qualquer SOTA publicado em 2024–2026 para ArXiv/WOS

---

## 3. Métodos novos (além da R-matrix binária)

### 3.1 R-matrix aprendível

**Ideia:** Hoje $R_{ij} \in \{0, 1\}$ é fixa e binária. Uma versão com pesos
aprendíveis $R_{ij} \in [0, 1]$ permite que o modelo aprenda a importância
relativa de cada relação ancestral.

```python
# Versão atual (fixa)
loss = max(0, p_child - p_parent + margin)

# Versão aprendível
w = sigmoid(learnable_param[parent, child])  # peso em [0, 1]
loss = w * max(0, p_child - p_parent + margin)
```

**Hipótese:** Em DAGs (GO), algumas arestas são semanticamente mais
importantes que outras. A versão aprendível capturaria isso.

**Esforço:** Médio — requer modificação nos modelos e experimentos comparativos.

### 3.2 R-matrix + ontology embeddings

**Problema:** O paper admite que ontology learning (WWW'22) ganha por
+39–71% AUPRC em FunCat. A R-matrix binária não captura semântica das classes.

**Ideia:** Integrar embeddings de ontologia (aprendidos via grafo GO) como
pesos ou features da R-matrix:

1. Pré-treinar embeddings de classes usando a estrutura da ontologia
2. Usar similaridade entre embeddings para ponderar a R-matrix:
   $R_{ij}^{weighted} = R_{ij} \cdot \text{sim}(e_i, e_j)$
3. Ou concatenar ontology embeddings como features adicionais no classificador

**Resultado esperado:** Fechar o gap com ontology learning mantendo a
simplicidade da R-matrix.

**Esforço:** Alto — requer integração de embeddings de ontologia no pipeline.

### 3.3 R-matrix com atenção hierárquica

**Ideia:** Substituir a propagação max-bottom-up por um mecanismo de atenção
que aprende quais filhos são mais relevantes para cada pai:

```python
# Versão atual (hard max)
p_parent = max(p_parent, max(p_children))

# Versão com atenção (soft)
attention_weights = softmax(score(parent_embed, child_embeds))
p_parent = max(p_parent, sum(w * p_children))
```

**Hipótese:** Em hierarquias onde filhos contribuem desigualmente para o pai
(ex: "cs.AI" é mais central para "cs" do que "cs.DC"), a atenção melhora a
reconciliação.

**Esforço:** Médio — implementar e comparar com versão max.

### 3.4 Block-diagonal para além de GO

**O que existe:** Block-diagonal validado só em Gene Ontology.

**O que fazer:** Aplicar a outros DAGs grandes:
- WordNet (117k synsets, DAG)
- MeSH (Medical Subject Headings, 29k termos)
- ICD-10 (14k códigos, hierarquia com múltiplos pais)
- SNOMED CT (350k conceitos)
- USPTO patent classification (IPC/CPC, 70k+ códigos)

**Resultado esperado:** Mostrar que o método é geral para qualquer DAG, não só GO.

**Esforço:** Médio-Alto — requer novos datasets e managers.

---

## 4. Impacto mais amplo

### 4.1 Case study em problema real

**Ideia:** Aplicar HMC-Torch a um problema não-benchmark:

- **Classificação de patentes** (IPC/CPC com 70k+ códigos)
- **ICD-10 para codificação clínica** automática a partir de textos médicos
- **Classificação taxonômica** de espécies a partir de DNA barcoding
- **Moderação de conteúdo** hierárquica (categorias de violação aninhadas)

### 4.2 Comparação com LLMs

**Status:** Script implementado (`experiments/run_llm_comparison.py`) —
zero-shot com GPT-4o, Claude, Ollama.

**A expandir:**
1. Few-shot: fornecer 5–10 exemplos por classe no prompt
2. Fine-tuned: fine-tune LLM vs. HMC-Torch
3. Análise de erros: onde o LLM erra que o HMC-Torch acerta?

**Hipótese:** Para datasets pequenos/médios, HMC-Torch com R-matrix supera
LLMs zero-shot e compete com few-shot, a fração do custo.

### 4.3 User study com pesquisadores

**Claim central:** "Plataforma modular acelera experimentação."

**Como provar:** Estudo com 5–10 pesquisadores de HMC:
1. Tarefa: adicionar um dataset novo e rodar baseline
2. Medir: tempo até primeiro resultado, número de bugs, satisfação
3. Comparar: HMC-Torch vs. código monolítico típico da literatura

**Esforço:** Alto — requer recrutamento e coordenação.

---

## Plano de ação: ordem de prioridade

### Fase 1: Quick Wins — IMPLEMENTADO ✅

| # | Ação | Script | Status |
|---|---|---|---|
| 1 | Ablação sistemática | `experiments/run_ablation.py` | ✅ Código pronto |
| 1b | Visualização ablação | `experiments/plot_ablation.py` | ✅ Código pronto |
| 2 | Leis de escala | `experiments/run_scaling_laws.py` | ✅ Código pronto |
| 2b | Visualização scaling | `experiments/plot_scaling_laws.py` | ✅ Código pronto |
| 3 | Multi-seed (5 seeds) | `experiments/run_multi_seed_full.py` | ✅ Código pronto |
| 4 | LLM comparison | `experiments/run_llm_comparison.py` | ✅ Código pronto |

**Próximo passo:** Executar os experimentos e coletar resultados.

### Fase 2: Métodos novos (Paper 2 — diferencial competitivo)

| # | Ação | Categoria | Impacto | Esforço | Prazo |
|---|---|---|---|---|---|
| **5** | R-matrix aprendível | Método novo | Médio-Alto | Médio | 3–4 semanas |
| **6** | Block-diagonal para WordNet + MeSH | Método novo | Médio | Médio-Alto | 4–6 semanas |
| **7** | R-matrix com atenção hierárquica | Método novo | Médio | Médio | 4–6 semanas |
| **8** | Expandir LLM comparison (few-shot, fine-tuned) | Impacto | Alto | Médio | 3–4 semanas |

### Fase 3: Teoria + impacto (Papers futuros)

| # | Ação | Categoria | Impacto | Esforço | Prazo |
|---|---|---|---|---|---|
| **9** | Prova teórica do gap do block-diagonal | Teoria | Alto | Alto | 6–8 semanas |
| **10** | Caracterização formal de quando R-matrix ajuda | Teoria | Alto | Alto | 8–12 semanas |
| **11** | R-matrix + ontology embeddings | Método | Alto | Alto | 8–12 semanas |
| **12** | Case study (patentes, ICD-10) | Impacto | Alto | Alto | 12+ semanas |
| **13** | User study com pesquisadores | Impacto | Médio | Alto | 12+ semanas |

---

## Estratégia de publicação

### Paper 1: Plataforma + Método (atual)

**Status:** Em desenvolvimento. Alvo: JMLR Software Track ou NeurIPS Datasets & Benchmarks.

- Block-diagonal R-matrix para DAGs grandes
- SOTA em ArXiv (+13.5 pts), WOS, AAPD
- Plataforma modular com 28 datasets

### Paper 2: Análise Científica (quick wins + métodos novos)

**Status:** Scripts implementados, experimentos pendentes.

**Opção A — "Understanding the R-Matrix"** (alvo: EMNLP/ACL):
- Ablação 4-way (Quick Win 1)
- Scaling laws (Quick Win 2)
- Multi-seed robusto (Quick Win 3)
- LLM comparison (Quick Win 4)
- R-matrix aprendível (Fase 2, #5)

**Opção B — "Scaling Laws for HMC"** (alvo: NeurIPS/JMLR):
- Scaling laws como contribuição central (Quick Win 2)
- Ablação como validação do mecanismo (Quick Win 1)
- Prova teórica do block-diagonal (Fase 3, #9)
- Multi-seed como standard de reproducibilidade (Quick Win 3)

**Opção C — "HMC in the LLM Era"** (alvo: ACL/EMNLP):
- LLM comparison como contribuição central (Quick Win 4, expandido)
- Few-shot e fine-tuned LLMs vs HMC-Torch (Fase 2, #8)
- Análise de erros: onde cada abordagem falha
- Custo-benefício: HMC-Torch vs APIs de LLM

### Paper 3: Teoria + Ontologia (longo prazo)

- Provas teóricas (Fase 3, #9, #10)
- R-matrix + ontology embeddings (Fase 3, #11)
- Case study real (Fase 3, #12)

---

## Artefatos implementados

```
experiments/
├── run_ablation.py              # Quick Win 1a: ablação 4-way
├── plot_ablation.py             # Quick Win 1b: visualização
├── run_scaling_laws.py          # Quick Win 2a: coleta de métricas
├── plot_scaling_laws.py         # Quick Win 2b: 4 gráficos
├── run_multi_seed_full.py       # Quick Win 3: 5 seeds × 6 datasets
└── run_llm_comparison.py        # Quick Win 4: zero-shot LLM vs HMC
```

## Notas

- Os quick wins (Fase 1) estão com código pronto. Rodar os experimentos é o
  próximo passo imediato.
- A Fase 2 (métodos novos) é o que diferencia o Paper 2 de "apenas mais
  experimentos" para "novos algoritmos".
- A Fase 3 (teoria) é o caminho mais difícil mas de maior prestígio.
- Os cenários de publicação não são mutuamente exclusivos; o ideal é combinar
  elementos de cada um.
- O Paper 1 e o Paper 2 podem ser submetidos em paralelo a venues diferentes
  ou sequencialmente (Paper 2 cita Paper 1 como infraestrutura).
