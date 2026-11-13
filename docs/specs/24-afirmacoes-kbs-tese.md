# 24 — Afirmações mais fortes que o dado na KBS e na tese (P1)

## Objetivo

Ajustar título, highlights e passagens da KBS e da tese em que a afirmação excede a evidência
ou contradiz o resultado negativo que o próprio texto reporta. Fecha os achados A14 e B1–B9 do
[relatório 18](18-audit-geral.md). Só edição de texto.

## Pré-requisitos

- Specs 20 e 21 aplicados (para não editar duas vezes os mesmos parágrafos).
- `output/protocol/structure.json` para as porcentagens de nós internos.

## Fatos a usar

- Reconciliação (BCE): mediana +0,0016 AUPRC, IQR [+0,0008, +0,0044], 26/27.
- Spearman do ganho com o aumento médio de score: ρ = 0,49 (p = 0,009) — moderado.
- Fração de anotações positivas em nós internos: FunCat 74–75%; GO ~89%.
- GAE híbrido no FunCat: +0,0016 em média, maior em 7/10, Wilcoxon p = 0,049 sem correção;
  abaixo do MCLoss + rec. em 3 dos 8 datasets comuns.
- Distância ao C-HMCNN (8 comuns): FunCat BCE + rec. −0,005, MCLoss + rec. −0,003; GO −0,004 e
  −0,002. Cross-domain: BCE + rec. 0,857 contra 0,849.
- Avaliação pareada: γ = 0 em 78/81; protótipos aleatórios +0,0025 contra +0,0022.
- Calibração não é medida em nenhum experimento.

## Passos

1. **Título da KBS** (`main.tex:42`) — hoje “Scalable R-Matrix Reconciliation, Sibling-Repulsive
   Ontology Graphs, and Contrastive Learning for …”. Propor um título centrado no achado, por
   exemplo “Where Does the Hierarchical Constraint Help? Inference-Time Reconciliation, a
   Sparse R-Matrix, and a Negative Result on Ontology Embeddings”. Atualizar
   `cover_letter.md` e o pacote de submissão. Decisão do orientador.
2. **Highlights** (`highlights.md:5`) — trocar “SR-GAE, H-MLCL+H-KNN, and depth calibration
   resolve sibling collapse and lift F1” por algo que o paper sustenta (≤ 85 caracteres), por
   exemplo “Repaired ontology prototypes add nothing; k-NN retrieval gives +0.002 AUPRC”.
3. **Objetivo por família** (A14) — KBS `:76-78` (abstract), tese `frontmatter/abstract.tex`,
   `frontmatter/resumo.tex`, `01-introducao.tex:129-132`, `08-conclusao.tex:47-50`: nomear o
   braço de cada número (“with MCLoss … with BCE …”) e dar também a distância do BCE + rec. no
   FunCat/GO, como já está no IJCNN.
4. **“primary driver”** — KBS `:165-168`: trocar por formulação com o tamanho do efeito
   (“accounts for the measured gain, which is small: median +0.0016 AUPRC”).
5. **Calibração** — KBS `:2125` (“well-calibrated pre-reconciliation probabilities”), tese
   `07-discussao.tex:73` (“well-behaved”): trocar por “without the drop in raw internal-node
   AUPRC observed under MCLoss”.
6. **“C > 1,000”** — KBS `:2132`, tese `07-discussao.tex:78`: substituir pelo que foi medido
   (OOM no treino denso em float64 com B = 64 e C = 4.134 numa GPU de 12 GB), como no IJCNN.
7. **Correlação e estrutura** — KBS `:2019`: “correlates strongly” → “correlates moderately”;
   `:2033`: “over 80% in Gene Ontology and FunCat” → os dois valores; KBS `:2035-2044` e tese
   `06-cross-domain.tex:173-175`, `:178-180`: marcar as explicações como hipóteses (“a possible
   explanation is …”) ou cortar. Com N = 27 e várias propriedades testadas, dizer quantas
   foram testadas.
8. **GAE híbrido** — KBS `:1636-1639`: “small, consistent gain” → “a gain of 0.0016 that is
   within seed noise (higher on 7 of 10 datasets; uncorrected p = 0.049)”; tese
   `05-funcat-go.tex:211-215`: retirar “a modest complementary benefit”.
9. **Frases quebradas** — KBS `:1066-1067` (reescrever ou cortar a oração sobre “the gap to the
   autoencoder branch”); `:1250-1255` (retirar “wins on F1 by leveraging the trade-off
   effectively” e deixar os três números).
10. **Negrito nas tabelas** — tese `05-funcat-go.tex:62-63`, `:84`, `:143` e tabelas
    equivalentes da KBS: negrito só quando a diferença excede um desvio-padrão entre seeds, ou
    retirar o negrito da coluna do blend e dizer na legenda que as diferenças entre as colunas
    “Ours” estão dentro do ruído.
11. **Estatística** — acrescentar às limitações (KBS “Limitations”; tese
    `07-discussao.tex:114-125`): os p-values não são corrigidos para comparações múltiplas; 3
    seeds por célula; reportar o tamanho do efeito ao lado de cada p. Se quiser corrigir,
    aplicar Holm aos testes principais (raw × rec em AUPRC e F1; blend × baseline; aleatório ×
    SR-GAE; DepthCal) e reportar quais permanecem.
12. `make check` na KBS e na tese; `make submission` na KBS.

## Saídas esperadas

- Título, highlights, resumo e conclusões coerentes com o resultado negativo.
- Nenhuma afirmação causal ou de calibração sem medida.

## Critério de aceite

- [ ] `highlights.md`: 5 itens, ≤ 85 caracteres cada, nenhum afirma ganho de SR-GAE ou DepthCal.
- [ ] `grep -n -i "well-calibrated\|correlates strongly\|primary driver\|1{,}000\$" ` na KBS e na
      tese não retorna nada.
- [ ] Todo “0.003”/“0.002” de distância ao C-HMCNN nomeia o objetivo.
- [ ] As limitações mencionam comparações múltiplas.
- [ ] KBS e tese compilam sem referências indefinidas.

## Tempo/custo

~3 h; sem compute.

## Falhas e rollback

- **Orientador mantém o título:** registrar a decisão aqui e ao menos alinhar os highlights.
- **Holm derruba algum teste secundário** (por exemplo p = 0,028): reportar como não
  significativo após correção; os testes centrais (p ≤ 2e-5) sobrevivem.
- Rollback: `git checkout` na KBS; backup da tese.

## Atualizações no paper/tese

- KBS: `main.tex:42`, `:76-78`, `:165-168`, `:1066-1067`, `:1250-1255`, `:1636-1639`, `:2019`,
  `:2033-2044`, `:2125`, `:2132`; `highlights.md`; `cover_letter.md`.
- Tese: `frontmatter/abstract.tex`, `frontmatter/resumo.tex`, `chapters/01-introducao.tex`,
  `05-funcat-go.tex`, `06-cross-domain.tex`, `07-discussao.tex`, `08-conclusao.tex`.
- IJCNN: nada (já tratado nos specs 15–17).

## Checklist

- [ ] Título decidido com o orientador.
- [ ] Highlights reescritos.
- [ ] Objetivo nomeado em resumo, introdução e conclusão.
- [ ] Calibração, “C > 1,000”, “strongly”, “over 80%”, hipóteses causais.
- [ ] GAE híbrido.
- [ ] Frases quebradas.
- [ ] Negrito nas tabelas.
- [ ] Limitações estatísticas.
- [ ] `make check` e `make submission`.
- [ ] Spec marcado como concluído no `README.md`.
