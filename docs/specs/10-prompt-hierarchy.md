# 10 — Prompt-based hierarchy encoder (P1, KBS mar/2027)

## Objetivo

Especificar, implementar e avaliar o "prompt-based hierarchy encoder" citado como next step em
`docs/hmc-paper-ijcnn/main.tex:377-384`, para a versão estendida da KBS (março/2027). A pergunta:
injetar a hierarquia no encoder de texto via prompts de rótulo melhora o HPT e o nosso
`globalE2E`, sobretudo em hierarquias profundas e rótulos raros?

## Lacuna e hipótese

O HPT atual (`globalHPT`, `models/global_classifier/hpt/`) insere no prompt apenas **um token de
nível por nível** e **um `[PRED]` por nível**. Os rótulos entram só como linhas do verbalizador
(produto interno com a saída dos `[PRED]`), nunca como tokens do prompt. Um encoder de hierarquia
via prompt transforma o texto do rótulo e o caminho na taxonomia em tokens de prompt,
contextualizados pelo grafo de rótulos (GAT) e atendidos pelos `[PRED]` de cada nível.

Hipótese falsificável: o ganho concentra-se em rótulos com poucos positivos e em níveis
profundos; se for uniforme por frequência, ou desaparecer contra um controle de prompts
aleatórios com o mesmo número de parâmetros, o mecanismo não é semântica de rótulo.

## Métodos candidatos

| ID | Método | Descrição | Papel |
|---|---|---|---|
| M0 | HPT reexecutado (spec 11) | baseline congelado | controle |
| M1 | Verbalizador com texto do rótulo | trocar a média de word embeddings por um encoder de frase congelado sobre nome + caminho (+ definição quando houver); manter GAT e produto interno | primeiro experimento e fallback |
| M2 | Hierarchy prompt encoder | por nível, K tokens de prompt produzidos por um pequeno Transformer/GAT sobre textos de rótulo e grafo, inseridos antes de cada `[PRED]`; scoring por verbalizador ou cross-attention | método principal |
| M3 | Prefix tuning | key/values de prefixo por nível injetados nas camadas do BERT (P-tuning v2) | fallback/ablation |

Não-objetivo: substituir a R-matrix. O paper testa complementaridade, não substituição.

## Pontos de plugue

- `src/hmc/models/global_classifier/hpt/model.py`: `GraphLayer` (40–52), `build_edge_index`
  (55–71), `graph_nodes` (201–206), construção do prompt/scoring (236–255), `_label_init`
  (267–309). Criar uma classe nova em vez de alterar `HPTPromptModel`, para congelar o baseline
  do spec 11.
- `src/hmc/pipeline/hpt/main.py`: `HPTConfig` (47–69), `label_space` (71–102; adicionar
  `label_paths`), `fit_hpt` (125–252), `evaluate_hpt` (255–284).
- `experiments/run_hpt.py`: `run_name` (61–63) e CLI (150–190); incluir toda opção nova no nome
  do JSON para não pular runs silenciosamente.
- `src/hmc/models/hierarchical/sparse_r.py`: inalterado; a decomposição raw/reconciliado sai do
  mesmo JSON.
- Textos de rótulo: termos do WOS; ArXiv/AAPD via `arxiv_label_names`; títulos de conceito do
  EUR-Lex via `_load_concept_titles`; GO via `go_term_texts` (fora de escopo, sem texto de
  documento).

Flags propostas: `--label-encoder {mean,sentence,crossattn,prefix}`, `--label-text
{name,path,definition}`, `--label-prompt-tokens {1,4,8}`, `--score-head
{verbalizer,crossattn}`; sufixo no `run_name`: `_le-{encoder}_{text}_k{K}`.

## Dados, baselines e controles

- Dados: WOS (primário), AAPD (sanidade/sem regressão), EUR-Lex 57K (hierarquia profunda, 4.271
  conceitos), ArXiv (split próprio, referência). RCV1 só se o spec 12 passar; HPT/prompt tuning
  em 781k documentos de teste é caro demais para P0/P1.
- Baselines: HPT reexecutado (spec 11); nosso `globalE2E` 0,8704 ± 0,0033; HiAGM/HTCInfoMax/
  HiMatch/HGCLR publicados, com a ressalva de que os três primeiros são reexecuções BERT do HPT.
- Controles obrigatórios: banco de prompts aleatórios com o mesmo número de parâmetros (isola
  semântica); M1 vs M2 (isola o encoder); HPT com tokens extras sem texto de rótulo (isola
  capacidade); raw vs reconciliado para todo modelo.

## Ablations (WOS, seed 0, 10 épocas salvo indicação)

1. Texto do rótulo: nome vs caminho vs nome+caminho.
2. Encoder do rótulo congelado vs fine-tuned.
3. K = 1, 4, 8 tokens por nível.
4. Camadas do grafo 0, 1, 2; direção `down` vs `both`.
5. Scorer: verbalizador vs cross-attention.
6. Controle de prompts aleatórios e controle de parâmetros.
7. M1 vs M2 vs M0.
8. Método principal no EUR-Lex: 3 seeds; F1 por profundidade e em rótulos raros.

## Protocolo e métricas

- Validation-only: época e threshold na validação, test uma vez; 3 seeds 0/42/123.
- Primária: micro-F1 reconciliado no WOS. Secundárias: micro-F1 no EUR-Lex, F1 em rótulos raros
  (<10 positivos no treino), F1 por profundidade, macro-F1.
- Reportar média ± desvio e IC bootstrap de 95% sobre os documentos de teste; não alegar
  significância só com 3 seeds.
- Reusar o schema JSON de `fit_hpt`/`evaluate_hpt` para manter raw/reconciliado e thresholds
  idênticos aos do spec 11.

## Custo e cronograma

- P0a (M1, WOS, 3 seeds): ~15 h.
- P0b (M2, WOS, 3 seeds): ~20 h.
- Ablations P0 (6 configs, seed 0): ~21 h.
- P1 (AAPD + EUR-Lex, método principal, 3 seeds): ~36 h.
- P2 (ArXiv/RCV1, opcional): ~20 h.
- Total: ~90–115 GPU-h, 4–6 semanas de calendário na RTX 3060 se exclusiva.

Datas: implementar até 01/12/2026; go/no-go em 15/01/2027. Se não houver resultado positivo no
WOS até lá, a KBS de março mantém o prompt encoder como parágrafo de future work e o spec segue
como trabalho futuro.

## Critério de sucesso

- M2 supera o HPT reexecutado no WOS em ≥0,004 micro-F1 com IC95% excluindo zero, e não regride
  mais de 0,002 no AAPD.
- M2 supera o controle de prompts aleatórios em ≥0,003 e o M1 em ≥0,002 no WOS.
- F1 em rótulos raros melhora ≥0,02; micro-F1 no EUR-Lex melhora ≥0,005.
- Se não bater: publicar a análise (texto do rótulo fraco, hierarquia já capturada pela
  reconciliação) sem oversell.

## Riscos

- WOS saturado e ruído entre seeds; mitigar com EUR-Lex e endpoints de rótulos raros.
- Textos de rótulo fracos (WOS não tem definições); testar caminhos e títulos do EUR-Lex.
- Confundimento de capacidade extra; controles aleatório e de parâmetros são obrigatórios.
- Compute e prazo; gates por etapa, WOS primeiro.
- Incerteza do baseline; congelar os números do HPT do spec 11 antes de começar o spec 10.
- Escopo da tese; ver a nota abaixo.

## Atualizações no paper/tese

- KBS: nova subseção de método, `tab:prompt` (WOS/AAPD/EUR-Lex), tabela de ablations,
  abstract/contribuições/limitações. Ablations detalhadas vão para material suplementar.
- IJCNN: fora de escopo; não tocar na versão de 6 páginas.
- `docs/tese/README.md:68`: hoje diz "HPT, RCV1 e ESM-2 estão fora do escopo". Substituir por:
  a reexecução do HPT (spec 11) e o RCV1 (spec 12) alimentam a KBS/tese; ESM-2 e o prompt
  encoder (spec 10) continuam fora do escopo da tese.

## Checklist

- [ ] Spec revisado e aprovado.
- [ ] Implementação do M1 (verbalizador com texto) + teste.
- [ ] Implementação do M2 (prompt encoder) + teste.
- [ ] Controles aleatório e de parâmetros.
- [ ] Screen no WOS (seed 0, validação) e gate.
- [ ] Final 3 seeds no WOS.
- [ ] EUR-Lex 3 seeds (se o WOS passar).
- [ ] Ablations e tabelas.
- [ ] KBS atualizada; IJCNN intocada.
- [ ] Spec marcado no README.
