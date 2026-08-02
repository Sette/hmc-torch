# Plano: TabFM para classificação hierárquica no GoFun

## Objetivo

Avaliar se o **TabFM** pode melhorar os resultados dos datasets GoFun/FunCat e
Gene Ontology (GO), sem comprometer o protocolo experimental necessário para
uma alegação de estado da arte (SOTA).

O primeiro alvo será `seq_FUN`, pois ele já é o dataset padrão do projeto. A
expansão para os demais datasets FunCat só ocorrerá depois de validar o método
e o protocolo nesse dataset.

## Contexto e hipótese

TabFM é um foundation model zero-shot para dados tabulares. Ele usa atenção por
linha e coluna, comprime cada linha e faz in-context learning usando as linhas
de treino como contexto. O modelo foi pré-treinado em centenas de milhões de
tabelas sintéticas [Google Research](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/).

A hipótese é que o modelo consiga capturar relações não lineares entre os
atributos dos genes que os classificadores atuais não capturam. A hipótese deve
ser testada com cuidado: o modelo não foi projetado para classificação
hierárquica multi-label, nem especificamente para predição de função proteica.

## Restrições que definem a arquitetura

O TabFM v1.0 atual:

- suporta classificação binária e multiclasse, mas não multi-label nativamente;
- suporta no máximo 10 classes por chamada;
- foi otimizado para tabelas de até 500 atributos;
- usa contexto limitado: por padrão, até 100 linhas de treino por contexto;
- tem pesos sob licença não comercial e não apropriados para produção/comercial.

Essas restrições são documentadas na [model card do TabFM](https://huggingface.co/google/tabfm-1.0.0-pytorch).

Portanto, **não** será usado um único TabFM para prever todos os termos do
GoFun. Isso violaria tanto a semântica multi-label quanto o limite de classes.

## Arquitetura proposta

### Classificadores binários condicionais por nó

Para cada nó `v` da taxonomia, treinar/inferir uma tarefa binária:

```text
P(v ativo | pai(v) ativo, atributos do gene)
```

O contexto de cada tarefa conterá apenas exemplos em que o pai é ativo. A
probabilidade global será composta ao longo da hierarquia:

```text
P(v ativo) = P(pai(v) ativo) × P(v ativo | pai(v) ativo)
```

Cada chamada ao TabFM terá duas classes (`ativo` e `inativo`), respeitando o
limite de 10 classes.

```text
atributos do gene
       |
       +-- TabFM binário: P(nó raiz)
       +-- TabFM binário: P(filho | pai), para cada nó
       |
       +-- composição dos caminhos
       +-- calibração no validation split
       +-- fechamento ancestral / predições consistentes
```

### Diferenças entre FunCat e GO

- **FunCat:** a estrutura é uma árvore; a composição acima é direta.
- **GO:** a estrutura é um DAG e um termo pode ter múltiplos pais. A primeira
  implementação deve calcular o score do termo pelo maior score entre caminhos
  válidos (`max-path`), preservando a consistência. Uma segunda variante deve
  comparar uma composição `noisy-OR`, tomando cuidado para não contar a mesma
  evidência duas vezes.

### Sem poda precoce no primeiro experimento

Na primeira versão, todos os nós terão score calculado. Só depois os scores
serão reconciliados pela hierarquia. Poda top-down durante a inferência reduz
custo, mas pode eliminar um ramo correto por erro de um ancestral e reduzir
recall; será uma otimização posterior.

### Contexto e classes raras

Para cada classificador binário:

- amostrar contexto estratificado, com no máximo 50 positivos e 50 negativos;
- usar múltiplos contextos/estimadores para diminuir a variância;
- nunca colocar exemplos de validação ou teste no contexto;
- para termos muito raros, usar um fallback explícito: prior calibrado ou
  classificador supervisionado convencional.

O limiar de raridade será escolhido depois da auditoria de prevalência; como
ponto inicial, avaliar termos com menos de 10 positivos condicionais.

## Protocolo experimental

Uma melhoria só pode ser chamada de SOTA se comparar exatamente o mesmo split,
as mesmas métricas e o mesmo protocolo dos baselines. Os benchmarks históricos
de função gênica em levedura usam particularmente **Pooled AUPRC**, além de
outras medidas hierárquicas [benchmark atualizado](https://pmc.ncbi.nlm.nih.gov/articles/PMC6755698/).

Para toda execução, registrar:

- split ARFF original, seed e versão do código;
- Pooled AUPRC, Micro-F1, Macro-F1 e F1 hierárquico;
- consistência hierárquica antes e depois da reconciliação;
- métricas por nível e por termo, especialmente termos raros;
- média e desvio padrão de pelo menos cinco execuções quando houver
  aleatoriedade;
- custo de inferência, memória e número de chamadas TabFM.

Toda transformação de atributos, seleção de atributos, calibração e escolha de
limiar deve ser ajustada exclusivamente no treino/validação. O teste é usado
uma única vez para o resultado final.

## Fases de execução

### Fase 0 — recuperar e estabilizar o benchmark

1. Restaurar ou migrar os módulos `hmc.datasets` removidos recentemente: os
   pipelines ainda os importam, portanto o fluxo GoFun atual não é executável
   no estado do repositório.
2. Confirmar que os três splits ARFF originais (`train`, `valid`, `test`) são
   preservados.
3. Criar relatório automático por dataset com:
   - número de exemplos e atributos;
   - níveis, nós por nível e total de nós;
   - número máximo de filhos por pai;
   - prevalência de cada termo e de cada tarefa condicional;
   - percentual de atributos ausentes.
4. Implementar/validar as métricas do protocolo.

**Entrega:** comando reproduzível que gera métricas e predições do baseline
atual em `seq_FUN`.

### Fase 1 — baselines fortes e comparáveis

1. Reproduzir `local` e o baseline global atuais.
2. Implementar One-vs-Rest supervisionado com um modelo tabular forte
   (CatBoost, LightGBM ou XGBoost), seguido de fechamento hierárquico.
3. Implementar TabFM binário *flat* por termo, sem a composição hierárquica.

O baseline flat responde uma pergunta essencial: o eventual ganho vem do
TabFM, ou somente do mecanismo hierárquico?

**Entrega:** tabela de validação com todos os baselines e análise por nível.

### Fase 2 — prova de conceito `tabfm_local` em `seq_FUN`

1. Adicionar um método separado, por exemplo `tabfm_local`, sem alterar o
   método `local` existente.
2. Construir tarefas binárias condicionais e contextos estratificados por nó.
3. Compor probabilidades na árvore FunCat e gerar predições globalmente
   consistentes.
4. Aplicar calibração Platt no validation split como primeira variante.
5. Comparar um, oito e 32 contextos por tarefa; medir qualidade versus custo.

**Entrega:** resultados cegos no test split de `seq_FUN`, scripts reproduzíveis
e arquivo de predições por termo.

### Fase 3 — robustez e ganho de qualidade

1. Comparar seleção por variância/MI e PCA/SVD, ajustadas somente no treino.
   Isso é especialmente necessário em `seq_FUN` (529 atributos) e `expr_FUN`
   (561), acima do limite recomendado de 500 atributos.
2. Implementar ensemble de contextos estratificados e subconjuntos de
   atributos.
3. Criar híbrido por nó: blend calibrado de TabFM com o melhor baseline
   supervisionado.
4. Adicionar refinador leve no grafo de labels, recebendo os scores dos nós e
   impondo coerência adicional.
5. Expandir aos outros sete datasets FunCat apenas se o resultado em `seq_FUN`
   justificar a continuação.

### Fase 4 — Gene Ontology e DAG

1. Portar a composição para múltiplos pais.
2. Comparar `max-path` e `noisy-OR`.
3. Usar métricas apropriadas ao DAG e analisar inconsistências por termo.

## Critérios de decisão

Continuar a linha TabFM somente se ela melhorar, de forma estável, Pooled AUPRC
e Micro-F1 em relação ao baseline supervisionado forte, sem degradação material
de consistência e com custo de inferência aceitável.

Se o TabFM puro não superar os baselines, manter a variante híbrida somente se
ela vencer de modo estatisticamente consistente. Caso contrário, encerrar a
linha TabFM e investir no caminho de maior potencial abaixo.

## Caminho alternativo de maior potencial

Para maximizar desempenho em função proteica, o avanço mais promissor não é um
LLM textual, mas embeddings de modelos fundacionais de proteína (por exemplo,
ESM ou ProtT5) a partir das sequências proteicas brutas, combinados com uma
cabeça multi-label hierárquica/GCN de termos GO. Isso exige recuperar as
sequências originais, que não parecem estar disponíveis nas features tabulares
atuais. O TabFM deve ser tratado como experimento tabular complementar, não
como substituto garantido para esse caminho.

## Extensão: propostas para os demais grupos de datasets

O repositório não contém apenas GoFun. Há três modalidades principais e uma
arquitetura única não deve tentar tratá-las como se fossem a mesma coisa.

```text
dados da modalidade --> adaptador/encoder específico --> embedding comum
                                                       |
                                                       +--> cabeça HMC
                                                             - árvore: caminhos condicionais
                                                             - DAG: scores por nó + reconciliação
```

A cabeça HMC deve ser compartilhada entre modalidades: loss multi-label com
ponderação para cauda longa, calibração, fechamento hierárquico, métricas e
relatórios. Apenas o encoder de entrada varia.

| Grupo | Dados disponíveis no benchmark atual | Proposta prioritária | Papel do TabFM |
| --- | --- | --- | --- |
| `*_FUN`, `*_GO` | atributos numéricos de genômica; `seq` são estatísticas de sequência e os demais são sobretudo microarrays de expressão | gradient boosting/MLP com cabeça HMC; encoder de proteína se as sequências brutas forem recuperadas | experimento binário condicional e ensemble para atributos tabulares |
| `diatoms_others`, `imclef07a_others`, `imclef07d_others` | vetores numéricos de descritores visuais no ARFF; imagens brutas não estão confirmadas no repositório | se houver imagens: encoder visual pré-treinado + cabeças por nível; se houver apenas vetores: GBDT/MLP | candidato razoável sobre os descritores, mas não substitui encoder visual sobre pixels |
| `enron_others` | representação numérica/sparsa de emails no ARFF; texto bruto não está confirmado | linear esparso forte; se houver emails brutos, transformer de texto com cabeça HMC | não recomendado para vetores esparsos de texto com ~1000 atributos |
| `arxiv` | título/resumo em JSONL, já com suporte a Transformer | SPECTER2/SciBERT fine-tuned + embeddings de labels e GNN | não aplicável: é texto, não tabela |

As coleções FunCat e GO preservam as mesmas fontes de atributos, mas mudam a
ontologia: FunCat é árvore relativamente pequena; GO é um DAG muito maior e
mais profundo. Os dados de `seq` são estatísticas de sequência, enquanto
Cellcycle, Derisi, Eisen, Expr, Gasch1, Gasch2 e Spo são dados de microarray
[descrição dos datasets](https://www.sciencedirect.com/science/article/pii/S1532046414002767).

### Genômica tabular: propostas além de TabFM

1. **Encoder de expressão auto-supervisionado.** Pré-treinar um denoising/masked
   autoencoder com os atributos de expressão, depois fine-tunar a cabeça HMC.
   É barato e compatível com as poucas milhares de amostras desses benchmarks.
2. **Transferência multimodal entre datasets.** Os datasets descrevem genes de
   levedura sob condições experimentais diferentes. Verificar interseção de IDs
   de genes entre os splits e, caso não exista vazamento, treinar encoder
   compartilhado com adaptadores por experimento e uma cabeça de ontologia
   compartilhada. Isso permite que expressão e estatísticas de sequência se
   complementem.
3. **Sequência bruta para `seq`.** Recuperar sequências pelo ID de proteína e
   extrair embeddings ESM-2/ProtT5 congelados; treinar somente a cabeça HMC e
   comparar com as estatísticas tabulares. A separação de IDs precisa ser feita
   antes de buscar/combinar os dados para impedir leakage entre treino e teste.
4. **Grafo de termos.** Para GO, usar embeddings aprendidos dos termos e
   message passing no DAG, com scores iniciais provenientes do encoder. Esse
   componente é mais promissor para GO do que milhares de chamadas TabFM.

### Visão: Diatoms e ImCLEF

Os datasets Diatoms e ImCLEF são de imagem, porém o ARFF pode conter somente
descritores visuais previamente extraídos. Além disso, Diatoms e ImCLEF podem
ser problemas de caminho único na hierarquia, e não multi-label pleno; isso
deve ser auditado antes de reutilizar BCE multi-label. Essa observação também é
reportada em análise recente dos benchmarks [Ontologue supplementary material](https://proceedings.neurips.cc/paper_files/paper/2022/file/8cf04c64d1734e5f7e63418a2a4d49de-Supplemental-Datasets_and_Benchmarks.pdf).

- **Com imagens originais:** comparar DINOv2, ConvNeXt ou ViT pré-treinado,
  congelado e depois fine-tunado, com cabeças multiclasses por nível e máscara
  de descendentes válidos. Isso é superior conceitualmente a aprender sobre
  descritores manuais.
- **Somente com ARFF:** comparar CatBoost/LightGBM, MLP residual e TabFM;
  manter classificação hierárquica por caminho se cada amostra tiver um único
  ramo verdadeiro.
- **Métrica adicional:** distância taxonômica do erro. Ela distingue errar uma
  espécie pelo gênero vizinho de errar por uma família distante.

### Texto: Enron e ArXiv

- **Enron com vetores ARFF:** usar Logistic Regression/Linear SVM calibrada e
  uma cabeça hierárquica antes de modelos profundos. Em matriz de palavras
  esparsa e pequena, esse baseline é tipicamente muito competitivo; TabFM perde
  tanto em adequação de modalidade quanto no limite de atributos/contexto.
- **Enron com emails brutos:** usar encoder de texto moderno, com atenção ou
  decoder hierárquico. É necessário definir remoção de cabeçalhos, assinaturas,
  destinatários e threads antes de medir qualquer resultado.
- **ArXiv:** continuar com SPECTER2 ou SciBERT fine-tuned, embeddings de labels
  e GNN da hierarquia. O projeto já tem um caminho `globalSOTA`; a prioridade é
  validar o R-matrix na inferência, o split e os resultados antes de trocar a
  arquitetura.

### Roteiro transversal de implementação

1. Criar uma interface `FeatureEncoder` que devolve `embedding` e metadados da
   modalidade; preservar o atual parser ARFF como `TabularEncoder`.
2. Criar uma única interface `HierarchicalHead` com variantes `tree_path`,
   `dag_global` e `local_conditional`.
3. Criar adaptadores: `tabular_tabfm`, `tabular_gbdt`, `expression_autoencoder`,
   `protein_embedding`, `vision_encoder` e `text_encoder`.
4. Definir um manifesto de cada dataset: modalidade, fonte, IDs, tipo real de
   rótulo (multi-label ou caminho único), estrutura (árvore/DAG) e métricas.
5. Só introduzir dependências pesadas (ESM, DINOv2, modelos de texto) quando os
   dados brutos correspondentes estiverem efetivamente disponíveis e a licença
   permitir o uso desejado.

## Riscos e mitigação

| Risco | Mitigação |
| --- | --- |
| TabFM não generaliza para dados biológicos | Baselines supervisionados fortes e blend híbrido. |
| Muitos classificadores tornam a inferência cara | Cálculo batelado, cache de contextos e poda apenas após medir recall. |
| Termos raros não têm contexto positivo suficiente | Prior/fallback supervisionado e análise estratificada por prevalência. |
| Vazamento entre contexto e avaliação | API de splits imutável e testes que rejeitam IDs de validação/teste no contexto. |
| Comparação inválida com literatura | Preservar split, métrica, seeds e artefatos de predição. |
| Uso incompatível com a licença | Restringir experimentos a pesquisa não comercial e registrar a licença com o artefato. |
