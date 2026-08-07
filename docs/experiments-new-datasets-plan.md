# Plano: Experimentos em Novos Datasets para o HMC-Torch

**Autor:** Bruno Sette  
**Data:** Agosto 2026  
**Status:** Proposta (revisar antes de implementar)

---

## 1. Contexto e Motivação

### Estado atual (Agosto 2026)

O HMC-Torch suporta **25 datasets** em 5 domínios:

| Família | Tipo | # Datasets | Features | Hierarquia |
|---------|------|-----------|----------|------------|
| ArXiv + WOS | **Texto** → Transformer | 2 | SPECTER2 (768d) | Tree, 2 níveis, ~150 labels |
| FunCat (FUN) | Tabular (ARFF) | 10 | Pré-computadas (77-561d) | Tree, 3-4 níveis, ~499 labels |
| Gene Ontology (GO) | Tabular (ARFF) | 9 | Pré-computadas (77-561d) | DAG, 10-16 níveis, ~4000 labels |
| Others | Tabular (ARFF) | 4 | Pré-computadas | Tree/DAG variados |

### Por que novos datasets?

O paper atual (ACL/EMNLP-style) reporta SOTA no ArXiv e WOS. **Adicionar datasets de texto** fortaleceria o claim de generalização:

> "O R-matrix funciona em qualquer domínio textual com hierarquia, não só em papers científicos."

Além disso, datasets textuais externos permitem comparação direta com métodos publicados (C-HMCNN, HMCN-F, etc.) que já reportam resultados neles.

### Conflito com o plano de 3 meses

O `three-month-plan.md` diz "❌ Novos datasets — já temos 25, é suficiente". Este plano **substitui** essa restrição para datasets de texto, mantendo-a para tabulares. Justificativa:

- Datasets de texto ≠ tabulares: testam o pipeline completo (transformer → R-matrix)
- 2 datasets de texto é pouco para claims de generalização
- Datasets padrão da literatura permitem comparação externa
- O esforço é concentrado: 1 manager por dataset ≈ 200 linhas

---

## 2. Datasets Propostos

### Tier 1: Essenciais (alto impacto, esforço moderado)

#### 2.1 RCV1-V2 (Reuters Corpus Volume 1, v2)

| Propriedade | Valor |
|-------------|-------|
| **Documentos** | 804,414 notícias |
| **Labels** | 103 tópicos (4 níveis: Root → CCAT/ECAT/GCAT/MCAT → subtópicos) |
| **Hierarquia** | Tree, profundidade 4 |
| **Features** | Texto (título + corpo) → SPECTER2 embeddings |
| **Split padrão** | 23,149 train / 781,265 test (LYRL2004 split) |
| **Disponibilidade** | Pública (NIST license, gratuita para pesquisa) |
| **Uso na literatura** | HMCN-F (2018), C-HMCNN (2021), HiAGM (2019), HTCInfoMax (2020) |

**Por que é o mais importante:**
- É **O** benchmark padrão de HMC textual
- Todos os SOTAs reportam resultados nele
- 103 labels → hierarquia pequena/média (R-matrix denso cabe em GPU)
- Permite comparação direta com C-HMCNN, HMCN-F, etc.
- Texto jornalístico (domínio diferente de ArXiv/WOS)

**Métodos publicados (Micro-F1):**
- C-HMCNN: 0.836 (2021)
- HMCN-F: 0.838 (2018)  
- HiAGM: 0.834 (2019)
- HTCInfoMax: 0.847 (2020)

**Plano de amostragem:** Usar 50K documentos para manter viabilidade computacional (≈ mesmo tamanho que ArXiv).

#### 2.2 DBLP (Computer Science Bibliography)

| Propriedade | Valor |
|-------------|-------|
| **Documentos** | ~100K+ papers de CS |
| **Labels** | ACM CCS 2012 hierarchy (taxonomia de tópicos de computação) |
| **Hierarquia** | Tree, 3 níveis, ~150-200 labels |
| **Features** | Texto (título) → SPECTER2 embeddings |
| **Disponibilidade** | Pública (dblp.org XML dump) |
| **Uso na literatura** | HiAGM (2019), HTCInfoMax (2020), vários papers de text HMC |

**Por que:**
- Mesmo domínio que ArXiv (papers de CS) → comparação interessante
- Hierarquia diferente: ACM CCS vs ArXiv taxonomy
- Títulos curtos (só título, sem abstract) → teste de robustez com pouco texto
- Download simples, sem licença

### Tier 2: Alta escala (testa sparse R)

#### 2.3 EUR-Lex 4.3K

| Propriedade | Valor |
|-------------|-------|
| **Documentos** | ~65K documentos legais da UE |
| **Labels** | ~4,000 EUROVOC concepts em hierarquia profunda |
| **Hierarquia** | Tree, até 8 níveis |
| **Features** | Texto (título + corpo) |
| **Disponibilidade** | Pública (EU Open Data) |
| **Uso na literatura** | Extreme classification, HMC com >1000 labels |

**Por que é interessante:**
- ~4K labels → testa sparse R-matrix em **texto** (não só GO tabular)
- Hierarquia profunda (8 níveis) vs ArXiv (2 níveis)
- Domínio legal (diferente de científico e jornalismo)
- Tamanho similar ao GO mas com **features textuais**

### Tier 3: Stretch goals (se houver tempo)

#### 2.4 PubMed Sample com MeSH

- **Por que:** Biomedical, hierarquia MeSH (~28K labels, DAG)
- **Desafio:** Dataset completo é enorme (>30M docs); faríamos sample de 100K
- **Diferencial:** Hierarquia DAG + texto, testa o pipeline completo em cenário real

#### 2.5 Amazon Product Reviews (amostra)

- **Por que:** Domínio de e-commerce, hierarquia de produtos
- **Desafio:** Hierarquia muito desbalanceada
- **Disponibilidade:** Pública (Amazon Review Data)

---

## 3. Plano de Implementação

### Fase 1: RCV1-V2 (Semana 1-2)

```
[ ] Download RCV1-V2 (LYRL2004 split)
[ ] Implementar RCV1Manager (inspirado no ArXivManager):
    - _load_records(): parser XML do Reuters
    - _build_hierarchy(): 103 tópicos, 4 níveis
    - _compute_features(): SPECTER2 embeddings com cache MD5
    - _create_splits(): usar split LYRL2004 padrão
[ ] Adicionar dispatch em dataset_manager.py
[ ] Adicionar defaults em registry.py
[ ] Criar download_rcv1.py (via NIST ou mirror)
[ ] Atualizar Makefile (make download-rcv1)
```

**Interface esperada do RCV1Manager:**
```python
mgr = RCV1Manager(
    data_dir="./data/rcv1",
    model_name="allenai/specter2_base",
    max_records=50_000,
    model_cache_dir="./models",
)
train, valid, test = mgr.get_datasets()
# Cada split: .x (embeddings), .y (global labels), .y_local (per-level)
```

### Fase 2: DBLP (Semana 2-3)

```
[ ] Download DBLP XML dump + ACM CCS mapping
[ ] Implementar DBLPManager:
    - Parser XML do DBLP
    - Mapeamento ACM CCS topics → hierarquia
    - SPECTER2 embeddings (só título)
[ ] Adicionar dispatch e registry
[ ] Download script
```

### Fase 3: EUR-Lex (Semana 3-4)

```
[ ] Download EUR-Lex dataset
[ ] Implementar EURLexManager:
    - Parser XML dos documentos EU
    - EUROVOC hierarchy (~4K labels)
    - SPECTER2 embeddings
    - Sparse R-matrix (BlockDiagonalR) para >1000 labels
[ ] Adicionar dispatch e registry
[ ] Download script
```

---

## 4. Grid de Experimentos

### 4.1 Experimentos principais (por dataset)

| # | Experimento | Métodos | Seeds | Output |
|---|------------|---------|-------|--------|
| 1 | **Benchmark completo** | global, globalE2E, globalSOTA, local | 3 | Tabela 1 do paper |
| 2 | **Ablação R-matrix** | global (com/sem R) | 3 | Delta F1 por dataset |
| 3 | **Análise por nível** | best method | 1 | Figura F1 vs profundidade |
| 4 | **Comparação SOTA** | best method vs published | 3 | Tabela comparativa |

### 4.2 Estimativa de tempo (GPU)

| Dataset | Método | Tempo/seed | Seeds | Total |
|---------|--------|-----------|-------|-------|
| RCV1-V2 (50K) | global | ~2h | 3 | 6h |
| RCV1-V2 | globalE2E | ~4h | 3 | 12h |
| RCV1-V2 | globalSOTA | ~5h | 3 | 15h |
| RCV1-V2 | local | ~1.5h | 3 | 4.5h |
| **RCV1-V2 subtotal** | | | | **~37.5h** |
| DBLP (50K) | todos | similar | 3 | ~35h |
| EUR-Lex (65K) | todos | 1.5-2x (mais labels) | 3 | ~55h |
| **Total estimado** | | | | **~128h GPU** |

### 4.3 Script automatizado

Criar `run_new_datasets.py` que:
1. Itera sobre datasets (rcv1, dblp, eurlex)
2. Para cada um, roda benchmark com 3 seeds
3. Salva resultados em `output/new_datasets/{dataset}/`
4. Gera tabelas LaTeX automáticas

---

## 5. Impacto no Paper

### O que muda no paper atual

| Seção | Antes | Depois |
|-------|-------|--------|
| Abstract | "25 datasets spanning 5 domains" | "28 datasets spanning 7 domains" |
| Datasets (Table 1) | ArXiv, WOS, FUN, GO, Others | **+ RCV1-V2, DBLP, EUR-Lex** |
| Results (Table 2-3) | SOTA em ArXiv/WOS | SOTA também em RCV1-V2/DBLP |
| Ablation (Sec 5.5) | R-matrix on/off em ArXiv/WOS | R-matrix on/off em **4 datasets texto** |
| Per-level (Fig 5) | 4 datasets | **7 datasets** (mais robusto) |
| Generalization | Claim fraco ("2 text datasets") | Claim forte ("5 text datasets, 3 domains") |

### Nova contribuição central

> "R-matrix constraints consistently improve hierarchical F1 across text domains (scientific, news, legal, CS bibliography), with gains of 0.02-0.08 Micro-F1 compared to unconstrained baselines."

### Claim de SOTA

Se o R-matrix bater os SOTAs publicados no RCV1-V2:
- HiAGM: 0.834, HTCInfoMax: 0.847, C-HMCNN: 0.836
- Isso seria um **resultado de alto impacto** — o R-matrix é mais simples que GCNs/Tree-LSTMs e potencialmente mais eficaz

---

## 6. Riscos e Mitigação

| Risco | Prob. | Mitigação |
|-------|-------|-----------|
| RCV1-V2 precisa licença NIST | Média | Solicitar com antecedência; usar mirror público como fallback |
| Performance abaixo do SOTA | Média | Rodar ablation para entender por que; ajustar hiperparâmetros |
| Tempo de GPU insuficiente | Alta | Reduzir para 1 seed inicialmente; priorizar RCV1-V2 |
| EUR-Lex muito grande | Alta | Sample de 30K docs; usar batch_size menor |
| DBLP hierarquia mal definida | Baixa | ACM CCS é bem documentada |

---

## 7. Decisão: Go / No-Go

### Recomendação: **GO nos 3 datasets, mas faseado**

1. **Julho-Agosto 2026:** RCV1-V2 + DBLP (os 2 mais importantes)
2. **Setembro 2026:** Se resultados promissores, adicionar EUR-Lex
3. **Outubro 2026:** Se necessário, PubMed sample (só se tempo permitir)

### Critério de sucesso para continuar:
- RCV1-V2 Micro-F1 ≥ 0.80 (competitivo com SOTA) → continuar para DBLP
- DBLP rodando sem erros → continuar para EUR-Lex

### Não fazer:
- ❌ Datasets tabulares novos (FUN/GO já cobrem bem)
- ❌ Datasets que exigem licença paga (NYT via LDC)
- ❌ Datasets com hierarquia plana (sem estrutura para R-matrix)
- ❌ Mais de 5 datasets novos (foco!)

---

## 8. Próximos Passos Imediatos

1. [ ] **Hoje:** Solicitar licença RCV1-V2 no site da NIST (se necessário)
2. [ ] **Hoje:** Baixar DBLP XML dump (não precisa de licença)
3. [ ] **Amanhã:** Implementar `RCV1Manager` (seguir padrão `ArXivManager`)
4. [ ] **Esta semana:** Rodar benchmark inicial no RCV1-V2 (1 seed)
5. [ ] **Semana que vem:** Se F1 > 0.80, expandir para 3 seeds + DBLP

---

## Referência: Estrutura de diretórios esperada

```
data/
├── arxiv/          # existente
├── wos/            # existente
├── gofun/          # existente (ARFF files)
├── rcv1/           # NOVO
│   ├── rcv1v2-ids.dat
│   ├── lyrl2004_tokens_train.dat
│   └── lyrl2004_tokens_test_pt0.dat
├── dblp/           # NOVO
│   ├── dblp.xml
│   └── ACMCCS.xml
└── eurlex/         # NOVO
    └── eurlex_4.3k/
```
