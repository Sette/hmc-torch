# Próximos passos: implementação multimodal de HMC

## Resultado pretendido

Evoluir o pacote de um conjunto de pipelines específicos para uma plataforma de
classificação hierárquica multimodal, mantendo compatibilidade com os comandos
atuais (`local`, `global`, `globalSOTA`) e com os splits originais.

O fluxo-alvo será:

```text
DatasetAdapter
    -> FeatureEncoder (tabular, texto, visão, proteína)
    -> HierarchicalHead (árvore, DAG ou local condicional)
    -> calibrador e reconciliador
    -> métricas e artefatos reproduzíveis
```

Não implementar todos os encoders de uma vez. A ordem de entrega começa pelo
loader GoFun e pelo baseline tabular, depois TabFM em `seq_FUN`; texto, visão e
embeddings de proteína dependem da disponibilidade dos dados brutos.

## Princípios de implementação

- Não alterar nem reagrupar os splits oficiais ARFF/JSONL.
- Ajustar normalização, seleção de atributos, calibração e thresholds somente
  no treino/validação.
- Separar `features`, `encoder`, `head`, `postprocess` e `evaluation`; nenhum
  pipeline deve precisar conhecer detalhes de uma modalidade.
- Manter predições por nó antes e depois da reconciliação para auditoria.
- Tratar árvores e DAGs explicitamente: não reutilizar lógica de árvore para
  GO sem teste específico de múltiplos pais.
- Tornar dependências de modelos grandes opcionais; o pacote básico deve
  continuar instalável e testável sem TabFM, ESM ou modelos visuais.

## Marco 0 — tornar o estado atual executável

O commit atual removeu `src/hmc/datasets`, mas `arguments.py` e os pipelines
ainda importam `hmc.datasets.*`. Isso precisa ser resolvido antes de adicionar
uma nova arquitetura.

### Ações

1. Restaurar os módulos removidos a partir do commit anterior ou movê-los para
   `src/hmc/data/`, atualizando todos os imports de forma atômica.
2. Restaurar, no mínimo:
   - `dataset_manager.py`;
   - `registry.py`;
   - parser ARFF GoFun (`gofun/dataset_arff.py`);
   - manager GoFun (`gofun/manager.py`);
   - adaptadores ArXiv já referenciados pelo README.
3. Corrigir a inconsistência de caminhos em `eisen_GO` antes de usar o loader.
4. Fazer o teste de integração de `seq_FUN` passar em CPU, com um fixture ARFF
   pequeno e sem depender de dados externos.

### Critério de aceite

`python -m hmc.main ... --dataset_name seq_FUN --method local` carrega dados e
completa uma época de treino; `pytest` executa sem imports ausentes.

## Marco 1 — contratos comuns de dados e hierarquia

### Novos módulos

```text
src/hmc/data/
  base.py                 # DatasetBundle, Split, FeatureMetadata
  hierarchy.py            # Hierarchy, TreeHierarchy, DagHierarchy
  gofun/                  # parser/manager ARFF migrados
  arxiv/                  # manager JSONL migrado

src/hmc/features/
  base.py                 # FeatureEncoder abstrato
  tabular.py              # imputação, scaling, seleção e cache
  text.py                 # placeholder para texto bruto
  vision.py               # placeholder para imagens brutas
  protein.py              # placeholder para sequências brutas
```

### Contratos

```python
@dataclass
class DatasetBundle:
    train: Split
    valid: Split | None
    test: Split
    hierarchy: Hierarchy
    metadata: FeatureMetadata

class FeatureEncoder(Protocol):
    def fit(self, split: Split) -> "FeatureEncoder": ...
    def transform(self, split: Split) -> EncodedSplit: ...
```

`FeatureMetadata` deve informar modalidade, quantidade de atributos, presença
de valores ausentes, esparsidade, IDs de amostra e se os dados brutos estão
disponíveis. `Hierarchy` deve expor nós, pais, filhos, níveis, matriz de
ancestrais e operações de fechamento.

### Ações

1. Migrar o parser ARFF para retornar `DatasetBundle`, preservando os arrays e
   nomes de nós atuais.
2. Implementar `TreeHierarchy` para FunCat e `DagHierarchy` para GO.
3. Implementar validações: todo rótulo positivo deve conter seus ancestrais;
   todos os IDs devem ser exclusivos entre os splits quando disponíveis.
4. Criar comando de auditoria, por exemplo `python -m hmc.audit --dataset_name
   seq_FUN`, que escreve `dataset-card.json` com estatísticas por nível e
   prevalência por termo.

### Critério de aceite

Testes unitários para árvore, DAG com múltiplos pais, fechamento de rótulos e
reprodução do número de atributos/nós conhecido para `seq_FUN`.

## Marco 2 — cabeça HMC e pós-processamento reutilizáveis

### Novos módulos

```text
src/hmc/models/hierarchical/
  heads.py                # GlobalSigmoidHead, LocalLevelHead, TreePathHead
  label_graph.py          # GCN/GAT de embeddings de termos
  losses.py               # BCE ponderada, focal, consistência, contrastiva
  postprocess.py          # calibração, closure_tree, closure_dag
```

### Ações

1. Extrair a lógica de constraint/R-matrix para `postprocess.py`, aceitando
   scores de qualquer encoder.
2. Implementar `GlobalSigmoidHead` para multi-label e `TreePathHead` para
   tarefas de caminho único.
3. Manter `LocalLevelHead` somente para métodos locais existentes, mas usar a
   mesma API de saída: `node_scores: [batch, n_nodes]`.
4. Implementar `DagHierarchy.reconcile(scores, strategy)` com estratégias
   `ancestor_max` e `max_path`; validar que o score de todo pai seja maior ou
   igual ao do filho.
5. Implementar calibradores Platt e isotônico, ajustados exclusivamente no
   validation split.

### Critério de aceite

Os modelos existentes podem usar a nova camada de reconciliação sem mudar suas
métricas quando a reconciliação está desativada; com ela ativada, nenhuma
violação hierárquica permanece.

## Marco 3 — baseline tabular forte

### Novos módulos

```text
src/hmc/models/tabular/
  gbdt_ovr.py             # One-vs-Rest com backend opcional
  mlp.py                  # MLP residual compartilhado
  preprocessing.py        # seleção de atributos ajustada no treino

src/hmc/pipeline/tabular/
  main.py
  train.py
  predict.py
```

### Ações

1. Implementar `tabular_gbdt` com backend inicial `HistGradientBoosting` do
   scikit-learn; oferecer LightGBM/CatBoost apenas como extras opcionais.
2. Implementar `tabular_mlp` como encoder residual + `GlobalSigmoidHead`.
3. Usar pesos por prevalência ou focal loss para termos raros; registrar a
   política adotada no arquivo de resultados.
4. Expor `--method tabular_gbdt` e `--method tabular_mlp` sem remover métodos
   atuais.
5. Criar uma matriz de resultados para oito FunCat e oito GO, começando por
   `seq_FUN`.

### Critério de aceite

Cada baseline produz `scores_before_postprocess.npz`,
`scores_final.npz`, `metrics.json` e `run-config.json` em diretório único de
execução.

## Marco 4 — TabFM local condicional

### Dependência opcional

Adicionar um extra, sem inserir TabFM nas dependências obrigatórias:

```toml
[project.optional-dependencies]
tabfm = ["tabfm[pytorch]"]
```

Registrar no README e no artefato de execução que os pesos TabFM atuais possuem
licença não comercial.

### Novos módulos

```text
src/hmc/models/tabular/tabfm/
  adapter.py              # carregamento lazy e checagem de dependência
  context.py              # amostragem estratificada, sem leakage
  conditional.py          # uma tarefa binária por nó
  cache.py                # cache de contextos e probabilidades

src/hmc/pipeline/tabfm/
  main.py
  evaluate.py
```

### Ações

1. Criar `TabFMAdapter` com import lazy: a ausência do pacote deve retornar
   mensagem de instalação clara, não erro na importação de `hmc`.
2. Implementar `ConditionalNodeDataset`: para cada nó, usar somente linhas de
   treino nas quais o pai é positivo; para a raiz, usar todas as linhas.
3. Criar `StratifiedContextSampler` com 50 positivos/50 negativos como padrão,
   seed explícita e estratégias para baixa prevalência.
4. Aplicar seleção de atributos apenas no treino. Para mais de 500 atributos,
   comparar `variance`, `mutual_info` e `svd`; salvar a transformação.
5. Compor scores em `TreeHierarchy`; implementar `max_path` para `DagHierarchy`.
6. Calibrar no validation split e comparar um, oito e 32 contextos por nó.
7. Criar método `tabfm_local` primeiro para `seq_FUN`, depois expandir apenas
   se os critérios de qualidade forem atendidos.

### Critério de aceite

Teste com uma árvore sintética verifica: contexto sem IDs de validação/teste,
limite de duas classes por chamada, composição correta e predições consistentes.
Uma execução `seq_FUN` deve produzir tabela comparável com `tabular_gbdt` e
`tabular_mlp`.

## Marco 5 — adaptadores por modalidade

Este marco só inicia depois de inventariar os arquivos brutos; os ARFF atuais
podem conter somente features já extraídas.

### 5A. Expressão gênica

```text
src/hmc/features/expression.py
src/hmc/models/expression/autoencoder.py
```

1. Implementar normalização e denoising/masked autoencoder.
2. Fine-tunar o encoder com `GlobalSigmoidHead` ou `LabelGraphHead`.
3. Investigar transferência multi-experimento por IDs de genes; bloquear splits
   cruzados quando o mesmo gene apareça em treino e teste.

### 5B. Sequência de proteína

```text
src/hmc/features/protein.py
src/hmc/data/protein_sources.py
```

1. Definir fonte, licença e mapeamento de IDs para sequências brutas.
2. Gerar embeddings ESM-2/ProtT5 congelados em cache por sequência e versão do
   modelo.
3. Treinar somente a cabeça HMC inicialmente; fine-tuning só após uma baseline
   congelada superar as features `seq` originais.

### 5C. Visão

```text
src/hmc/features/vision.py
src/hmc/models/vision/hierarchical_classifier.py
```

1. Confirmar se imagens Diatoms/ImCLEF originais estão disponíveis.
2. Se sim, usar encoder pré-treinado (DINOv2/ConvNeXt/ViT) e cabeças por nível.
3. Distinguir caminho único de multi-label pela auditoria dos rótulos; usar
   softmax mascarado por descendentes no caso de caminho único.
4. Se só existirem vetores ARFF, tratar como tabular e não adicionar dependência
   de visão.

### 5D. Texto

```text
src/hmc/features/text.py
src/hmc/models/text/hierarchical_transformer.py
```

1. Enron: confirmar se emails brutos existem. Sem eles, priorizar classificador
   linear esparso; com eles, definir limpeza de headers/threads e encoder.
2. ArXiv: migrar o caminho existente de SPECTER2/SciBERT para a interface
   `FeatureEncoder`, preservando seu cache.
3. Adicionar embeddings de labels + GCN somente depois de reproduzir o baseline
   SPECTER2 com a mesma métrica e split.

## Marco 6 — CLI, configuração e resultados

### Ações

1. Substituir flags específicas por grupos compatíveis, sem quebrar as atuais:
   - `--encoder {tabular_mlp,gbdt,tabfm,expression,protein,text,vision}`;
   - `--head {global_sigmoid,local_level,tree_path,label_graph}`;
   - `--reconcile {none,ancestor_max,max_path,noisy_or}`.
2. Manter `--method` como atalho que define combinações compatíveis.
3. Criar configurações YAML versionadas em `configs/` para cada experimento.
4. Persistir manifesto com git SHA, seeds, versões das dependências, licença do
   modelo, dimensões, transformações e custo computacional.
5. Criar relatório Markdown/CSV agregado por dataset e nível.

### Critério de aceite

Uma configuração YAML reproduz uma execução completa sem argumentos manuais e
o relatório permite comparar métodos sem reprocessar predições.

## Marco 7 — testes e integração contínua

| Camada | Testes necessários |
| --- | --- |
| Dados | parser ARFF, splits, IDs, atributos ausentes, árvore/DAG |
| Hierarquia | fechamento, R-matrix, múltiplos pais, consistência de scores |
| Encoders | `fit` só em treino, dimensões estáveis, cache válido |
| TabFM | amostragem sem leakage, classes binárias, fallback raro |
| Pipeline | treino mínimo em CPU com fixture sintético |
| Reprodutibilidade | mesma seed gera mesmos contextos e artefatos |

Não adicionar downloads de modelos a testes automáticos. Usar mocks para TabFM,
encoders de proteína, visão e Transformers; rodar integrações reais em workflow
manual/noturno separado.

## Ordem recomendada e pontos de decisão

1. **Marco 0:** recuperar dataset loader e testes.
2. **Marcos 1–2:** contratos e camada hierárquica comum.
3. **Marco 3:** GBDT/MLP tabulares para estabelecer baseline real.
4. **Marco 4:** TabFM apenas em `seq_FUN`.
5. **Decisão:** expandir TabFM somente se superar o baseline em Pooled AUPRC e
   Micro-F1 de forma estável em cinco seeds.
6. **Marco 5:** escolher um único adaptador de dados brutos com maior retorno:
   proteína, se as sequências forem obtidas; caso contrário, texto ArXiv ou
   imagens, conforme disponibilidade.
7. **Marcos 6–7:** consolidar configuração, artefatos e CI antes de ampliar a
   matriz de experimentos.

## Fora de escopo inicial

- fine-tuning de ESM/ProtT5, DINOv2 ou LLMs grandes;
- uso comercial dos pesos TabFM v1.0;
- misturar exemplos entre datasets sem auditoria de IDs;
- alegações de SOTA antes de reproduzir split, métrica e baseline publicados.
