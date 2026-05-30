# HMC no ArXiv: Estado da Arte e Melhorias

## Contexto do benchmark

O ArXiv como dataset de HMC não tem uma versão canônica única — papers usam subconjuntos diferentes (só CS, CS+Math, full ArXiv), splits distintos e avaliações em níveis diferentes, o que dificulta comparação direta. Os benchmarks mais citados para HTC (Hierarchical Text Classification) em geral são **WOS**, **RCV1** e **NYT**, mas o ArXiv aparece em trabalhos mais recentes.

## Trabalhos relevantes e resultados típicos

| Método | Feature | Arquitetura | Micro-F1 (aprox.) | Referência |
|---|---|---|---|---|
| TF-IDF + MLP | BoW | MLP flat | 65–72% | baseline |
| **C-HMCNN** | TF-IDF | MLP + R-matrix | 70–76% | Giunchiglia & Lukasiewicz, NeurIPS 2020 |
| **HiAGM** | BERT | TreeLSTM + label graph | 82–87% | Zhou et al., ACL 2020 |
| **HGCLR** | BERT | Graph + contrastive learning | 85–89% | Wang et al., ACL 2022 |
| **HPT** | BERT/RoBERTa | Hierarchy-aware fine-tuning | 87–91% | Wang et al., ACL 2022 |
| **SPECTER + MLP** | SciBERT domain | Transformer embeddings | 83–88% | Cohan et al., ACL 2020 |

O gap entre a implementação atual (TF-IDF + MLP + R-matrix) e o estado da arte é **~15–20 pontos de Micro-F1**, quase todo explicado por três fatores descritos abaixo.

---

## Análise da implementação atual

### O que está bom

- R-matrix / `get_constr_out()` segue a abordagem do C-HMCNN original
- Hierarchical consistency loss penaliza `P(filho) > P(pai)` durante treino
- Feature cache com hash impede recomputação desnecessária

### Lacunas principais

**1. Features fracas** — o gargalo mais crítico

O TF-IDF + SVD(256) descarta semântica distribucional completamente. Para texto científico o delta é enorme:

```
TF-IDF → 256 dims    →  MLP  →  ~70% Micro-F1
SciBERT → 768 dims   →  MLP  →  ~85% Micro-F1  (+15pp com a mesma arquitetura)
SPECTER → 768 dims   →  MLP  →  ~87% Micro-F1  (pré-treinado em citações ArXiv)
```

O modo `embedding` já está implementado no `ArXivManager`, mas usa `all-MiniLM-L6-v2` (modelo genérico).
Trocar para **SPECTER** ou **SciBERT** daria o maior ganho isolado.

**2. R-matrix não aplicada na inferência ArXiv**

O código aplica `get_constr_out()` na inferência para datasets ARFF, mas no pipeline ArXiv o constraint
hierárquico só aparece como perda suave durante treino. Aplicar a inferência constrained também ao ArXiv
é um fix de poucas linhas com ganho direto em consistência (e geralmente em F1 também).

**3. Arquitetura flat ignora estrutura do label graph**

O MLP global trata todos os ~500 nós do grafo de categorias ArXiv como saídas independentes.
Modelos que propagam informação pelo grafo de labels consistentemente superam MLPs flat:

- **GCN/GAT sobre o label graph** — já existe `BuildClassification` com GCN/GAT em `models/local_classifier/networks.py`, mas não integrado ao global classifier
- **TreeLSTM no grafo de hierarquia** — estilo HiAGM

---

## Plano de melhorias

Ordenado por impacto / esforço:

### Passo 1 — Trocar a feature (maior ROI)

```yaml
# config.yaml
arxiv_feature_type: embedding
arxiv_model_name: allenai/specter2  # ou allenai/scibert_scivocab_uncased
```

SPECTER2 é pré-treinado com objetivo de recuperação de papers científicos (título + abstract → embedding),
exatamente o dado disponível no ArXiv JSONL. Não requer mudança de arquitetura.

**Alternativas de modelo:**

| Modelo | Dims | Foco |
|---|---|---|
| `allenai/specter2` | 768 | Recuperação de papers científicos |
| `allenai/scibert_scivocab_uncased` | 768 | Linguagem científica geral |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Semântica geral, forte baseline |
| `allenai/specter2_base` | 768 | Versão leve do SPECTER2 |

### Passo 2 — Habilitar R-matrix constraint na inferência ArXiv

Em `pipeline/global_classifier/core/train.py`, a chamada a `get_constr_out()` precisa receber a `r_matrix`
do `ArXivManager`. A estrutura já computa `self.a` (adjacency) e `self.nodes_idx` — falta construir e
registrar a R-matrix no manager e expô-la para o pipeline global.

```python
# Em ArXivManager._build_hierarchy():
# Adicionar: self.r_matrix = compute_r_matrix(adjacency_matrix)

# Em pipeline/global_classifier/core/train.py na inferência:
# Usar get_constr_out(logits, args.hmc_dataset.r_matrix) em vez de logits raw
```

### Passo 3 — Label graph GNN

O grafo de hierarquia ArXiv tem ~500 nós e ~450 arestas — pequeno o suficiente para GCN.

```
[Input features] → MLP encoder → node_embedding
[Label hierarchy] → GCN        → label_embedding
cosine(node_embedding, label_embedding) → prediction por nó
```

O `BuildClassification` em `models/local_classifier/networks.py` já suporta GCN/GAT, mas está
desconectado do global classifier. Integrar ao `ConstrainedModel` é a mudança arquitetural com maior
potencial de ganho.

**Referência:** HiAGM (Zhou et al., ACL 2020) — usa TreeLSTM exatamente com essa lógica.

### Passo 4 — Hierarquia contrastiva

HGCLR (Wang et al., ACL 2022) mostra que **contrastive learning no espaço de labels** dá ganhos
significativos:

- amostras do mesmo sub-ramo = pares positivos
- amostras de sub-ramos distantes = negativos difíceis

```python
def hierarchical_contrastive_loss(embeddings, labels, hierarchy, temperature=0.07):
    # amostras que compartilham ancestral próximo → alta similaridade
    # amostras em sub-árvores distintas → baixa similaridade
    ...
```

Combinar com a BCE/focal loss existente via ponderação `lambda_contrast`.

### Passo 5 — Fine-tuning end-to-end do transformer

Ao invés de extrair embeddings frozen, fine-tunar o SciBERT/SPECTER junto com o classificador.
O cache de features atual não suporta isso (features mudam a cada época), mas o ganho pode ser
+3–5 pp sobre embeddings frozen.

Requer: desabilitar o cache ou usar cache por época + muito mais VRAM.

---

## Prioridade resumida

```
Impacto  │  ████████████  SPECTER/SciBERT embedding    (config.yaml, ~2h)
         │  ████████      R-matrix na inferência ArXiv  (código, ~1 dia)
         │  ██████        GCN no label graph             (~2 dias)
         │  ████          Contrastive loss                (~3 dias)
         │  ██            Fine-tuning end-to-end          (~1 semana)
Esforço  └──────────────────────────────────────────────────────────────
                  baixo                                            alto
```

O passo 1 (SPECTER + embedding mode) provavelmente leva de ~70% para ~85% Micro-F1 sem mudar
uma linha de código de modelo. A R-matrix na inferência ArXiv é o próximo ganho mais barato.

---

## Referências

- Giunchiglia & Lukasiewicz (2020). *Coherent Hierarchical Multi-label Classification Networks*. NeurIPS.
- Zhou et al. (2020). *Hierarchy-Aware Global Model for Hierarchical Text Classification*. ACL.
- Wang et al. (2022). *Incorporating Hierarchy into Text Encoder: a Contrastive Learning Approach for Hierarchical Text Classification*. ACL. (**HGCLR**)
- Wang et al. (2022). *HPT: Hierarchy-aware Prompt Tuning for Hierarchical Text Classification*. EMNLP.
- Cohan et al. (2020). *SPECTER: Document-level Representation Learning using Citation-informed Transformers*. ACL.
- Beltagy et al. (2019). *SciBERT: A Pretrained Language Model for Scientific Text*. EMNLP.
