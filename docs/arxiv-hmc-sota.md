# HMC no ArXiv: Estado da Arte e Melhorias

## Contexto do benchmark

O ArXiv como dataset de HMC não tem uma versão canônica única. Trabalhos usam subconjuntos diferentes, como apenas CS, CS+Math ou o ArXiv completo, além de splits e profundidades de label distintas. Por isso, os números externos devem ser lidos como faixas aproximadas, não como comparação direta de leaderboard.

## Trabalhos relevantes e resultados típicos

| Método | Encoder | Arquitetura | Micro-F1 aprox. | Referência |
|---|---|---|---|---|
| **hmc-torch `global`** | SPECTER2 frozen | MLP + R-matrix | 72,6% | este repositório |
| **HiAGM** | BERT | Text encoder + label graph | 82-87% | Zhou et al., ACL 2020 |
| **HGCLR** | BERT | Graph + contrastive learning | 85-89% | Wang et al., ACL 2022 |
| **HPT** | BERT/RoBERTa | Hierarchy-aware prompt tuning | 87-91% | Wang et al., EMNLP 2022 |
| **SPECTER + MLP** | SPECTER/SciBERT domain | Transformer embeddings | 83-88% | Cohan et al., ACL 2020 |

O resultado atual com SPECTER2 frozen fica aproximadamente 15 pontos de Micro-F1 abaixo dos melhores métodos baseados em transformers ajustados à hierarquia.

---

## Análise da implementação atual

### O que está bom

- SPECTER2 é o encoder padrão para ArXiv e usa título + abstract.
- Modelos SPECTER usam CLS pooling, que é mais adequado ao objetivo de pré-treino de recuperação de papers.
- R-matrix / `get_constr_out()` segue a abordagem do C-HMCNN original.
- Hierarchical consistency loss penaliza `P(filho) > P(pai)` durante treino.
- Feature cache com hash impede recomputação desnecessária dos embeddings.

### Lacunas principais

**1. Encoder frozen**

O `global` usa embeddings SPECTER2 congelados. Isso é eficiente e reprodutível, mas não adapta o encoder à fronteira de decisão da tarefa hierárquica.

**2. Arquitetura flat no caminho principal**

O MLP global trata os nós de categoria como saídas preditivas. Métodos SOTA propagam informação pelo grafo de labels ou incorporam a hierarquia diretamente no encoder.

**3. Sinal contrastivo hierárquico**

HGCLR mostra ganhos relevantes ao aproximar amostras do mesmo sub-ramo e afastar sub-ramos distantes. A flag `--use_contrastive_loss` já existe no pipeline.

---

## Plano de melhorias

Ordenado por impacto / esforço:

### Passo 1 — Fine-tuning end-to-end

Usar `globalE2E` para ajustar o transformer junto com o classificador:

```bash
python -m hmc.main \
  --dataset_name arxiv --method globalE2E --device cuda \
  --arxiv_model_name allenai/specter2_base \
  --dataset_path ./data --output_path ./output
```

O ganho esperado vem da adaptação do encoder científico à tarefa de classificação hierárquica.

### Passo 2 — Label graph GNN

Usar `globalSOTA`, que combina o text encoder com um GCN sobre a hierarquia de labels:

```bash
python -m hmc.main \
  --dataset_name arxiv --method globalSOTA --device cuda \
  --arxiv_model_name allenai/specter2_base \
  --dataset_path ./data --output_path ./output
```

O grafo de hierarquia ArXiv é pequeno o suficiente para GCN/GAT, e a arquitetura segue a família de métodos do HiAGM.

### Passo 3 — Hierarquia contrastiva

Combinar a BCE/focal loss existente com `--use_contrastive_loss true` e calibrar `--lambda_contrastive`. A ideia é usar a distância na hierarquia para construir pares positivos e negativos mais informativos.

### Passo 4 — Modelos científicos alternativos

Comparar `allenai/specter2_base` com:

| Modelo | Dims | Foco |
|---|---:|---|
| `allenai/scibert_scivocab_uncased` | 768 | Linguagem científica geral |
| `sentence-transformers/all-mpnet-base-v2` | 768 | Baseline semântico geral forte |
| `allenai/specter2_base` | 768 | Recuperação de papers científicos |

---

## Prioridade resumida

```text
Impacto  |  fine-tuning end-to-end        alto
         |  GCN no label graph            alto
         |  contrastive loss              medio
         |  troca de encoder cientifico   medio
Esforco  |  baixo                     alto
```

O caminho recomendado é partir do resultado `global` com SPECTER2 frozen, executar `globalE2E` e então `globalSOTA` com o mesmo split e semente para medir o ganho incremental.

---

## Referências

- Zhou et al. (2020). *Hierarchy-Aware Global Model for Hierarchical Text Classification*. ACL.
- Wang et al. (2022). *Incorporating Hierarchy into Text Encoder: a Contrastive Learning Approach for Hierarchical Text Classification*. ACL. (HGCLR)
- Wang et al. (2022). *HPT: Hierarchy-aware Prompt Tuning for Hierarchical Text Classification*. EMNLP.
- Cohan et al. (2020). *SPECTER: Document-level Representation Learning using Citation-informed Transformers*. ACL.
- Beltagy et al. (2019). *SciBERT: A Pretrained Language Model for Scientific Text*. EMNLP.
