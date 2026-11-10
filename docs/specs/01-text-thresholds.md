# 01 — Text benchmarks com limiar validation-only (P0)

## Objetivo

Substituir os números de texto congelado do paper (ArXiv, WOS frozen, AAPD, EUR-Lex) por runs do
protocolo `run_protocol.py`, que escolhe época e limiar **só na validação** e usa o test uma vez.
Os números antigos do paper vieram de runs legados com limiar no test; o caveat está em
`docs/hmc-paper/main.tex:713-724`.

## Pré-requisitos

- `data/aapd`, `data/wos`, `data/eurlex` presentes.
- `data/arxiv` **ausente**: o manager espera `data/arxiv/arxiv-metadata-oai-snapshot.json`.
  O alvo `make download-arxiv` tem um bug de caminho (grava na raiz de `data/`); o comando certo é
  `PYTHONPATH=src uv run python -m hmc.datasets.arxiv.download_arxiv --output_dir ./data/arxiv`
  (kagglehub; ~1 GB). Se o download falhar, seguir sem ArXiv.
- GPU livre; `uv run` para ter matplotlib/torch.
- `output/protocol/` não tem JSONs de texto hoje (só FunCat e `*_others`).

## Comandos

```bash
# opcional: ArXiv
PYTHONPATH=src uv run python -m hmc.datasets.arxiv.download_arxiv --output_dir ./data/arxiv

# texto congelado, 3 objetivos, 3 seeds (validation-only por construção)
PYTHONPATH=src uv run python experiments/run_protocol.py \
  --datasets aapd,wos,eurlex --objectives bce,hinge,mc --seeds 0,42,123
# se ArXiv estiver disponível: incluir arxiv na lista
```

## Decisões desta execução

- **EUR-Lex (texto):** usar o release **original da AUEB** (`dataset.zip` do archive.org), com
  `text = header + "\n" + recitals + "\n" + main_body`. O mirror HF
  `jonathanli/eurlex` usa apenas `header + recitals`; a verificação mostrou IDs/rótulos/splits
  idênticos, mas texto incompleto. O downloader (`download_eurlex.py`) agora baixa o original
  por padrão e mantém o HF como fallback. **O re-run confirmou os mesmos números**: o tokenizer
  trunca em 256 tokens e o `main_body` (anexado no fim) fica fora da janela; as features antigas
  e novas são byte-idênticas (mesmo SHA-256).
- **EUR-Lex (dados):** release completa (45k treino / 4.519 labels), que é o que o
  `run_protocol.py` carrega; atualizar a descrição do dataset e os números no paper (o valor
  antigo de 0,666 era da subamostra de 10k/2.901).
- **Épocas:** o script usa 100 épocas/patience 10; alinhar a seção de treino do paper a isso.
- **ArXiv:** baixado em 03/11/2026 (5,7 GB em `data/arxiv/arxiv-metadata-oai-snapshot.json`,
  via kagglehub); rodar o protocolo no ArXiv depois que os três datasets atuais terminarem.

## Correção (04/11/2026) — o EUR-Lex usava só o título

Ao revisar as variantes de texto, descobri que `EURLexManager._extract_text` montava
`title + header + recitals + main_body`, mas os registros gravados pelo downloader têm apenas
`title` e `text`. Como o título nunca é vazio, o fallback para `text` nunca era atingido:
**todos os números de EUR-Lex anteriores (0,7863 BCE / 0,7850 hinge / 0,7898 MCLoss) foram
calculados com o título apenas (≈407 caracteres)**, não com o texto do release. As primeiras
"variantes" testadas antes desta correção também foram um no-op: mexiam no campo `text`, que o
manager ignorava; por isso as três deram exatamente o mesmo resultado.

Correções:

- `_extract_text` agora usa `text` como corpo quando `header`/`recitals`/`main_body` estão
  ausentes e prefixa o título quando ele ainda não está no texto
  (`src/hmc/datasets/eurlex/manager.py`).
- Testes de regressão em `tests/test_eurlex_text.py`.
- Cache de features apagado (`data/eurlex/.feature_cache/*.npy`) para não reutilizar embeddings
  de título.

Re-runs (04/11/2026):

- `output/protocol_eurlex_fixedtext/eurlex.json` — texto completo (title+header+recitals+
  main_body), bce/hinge/mc, 3 seeds. Resultado: BCE **0,7888 ± 0,0005** (AUPRC 0,8566),
  hinge **0,7875 ± 0,0006** (0,8553), MCLoss **0,7927 ± 0,0009** (0,8587). Contra o título-only:
  +0,0025 / +0,0025 / +0,0029 F1.
- `output/protocol_eurlex_{title_body,body_header,title_body_only}_fixed/` — variantes de ordem,
  BCE, 3 seeds. Resultados: `title_body` **0,7830 ± 0,0007** (AUPRC 0,8508; −0,0058),
  `body_header` **0,7624 ± 0,0002** (0,8277; −0,0264) e `title_body_only` **0,7826 ± 0,0005**
  (0,8507; −0,0062), todas abaixo do texto completo (0,7888). Conclusão: manter `title + header +
  recitals + main_body`; colocar o `main_body` na frente da janela de 256 tokens piora, e remover
  o título piora muito.

Os números da tabela abaixo são os antigos (título-only) e serão substituídos quando os runs
terminarem.

## Resultados (03/11/2026)

Protocolo validation-only, 3 seeds, `rec` (micro-F1 após reconciliação):

| Dataset | BCE | hinge | MCLoss | AUPRC (BCE) | Observação |
|---|---|---|---|---|---|
| AAPD | 0,8232 ± 0,0003 | 0,8229 ± 0,0004 | 0,8229 ± 0,0007 | ~0,907 | próximo do 0,8228 antigo |
| WOS frozen | 0,7462 ± 0,0001 | 0,7446 ± 0,0007 | 0,7441 ± 0,0006 | ~0,813 | abaixo do 0,7519 antigo (limiar no test) |
| EUR-Lex | 0,7863 ± 0,0009 | 0,7850 ± 0,0002 | 0,7898 ± 0,0009 | ~0,852 | **título-only (bug corrigido em 04/11; ver acima)**; 4.271 conceitos, 4.520 nós, 5.237 arestas |
| ArXiv | 0,7234 ± 0,0004 | 0,7235 ± 0,0007 | 0,7238 ± 0,0007 | ~0,803 | 32k/8k/10k, 159 nós; abaixo do 0,7295 antigo (limiar no test) |

## Saídas esperadas

- `output/protocol/{aapd,wos,eurlex,arxiv}.json` (um por dataset).
- Cada JSON: `protocol` = “train on train; early stopping and threshold on valid; report on test”;
  `runs[]` com `objective`, `seed`, `best_epoch`, `eval.raw/rec` (f1, auprc, threshold, val_f1).

## Critério de aceite

- [ ] JSONs existem e o campo `protocol` confirma validation-only.
- [ ] Números novos extraídos e comparados com os antigos do paper.
- [ ] `tab:text`, `tab:new_text`, `tab:ablation` e o parágrafo de métricas atualizados.
- [ ] Caveat `% TODO(P0)` removido.

## Tempo/custo

~1,5–3 h na RTX 3060 (features SPECTER2 congeladas; MLP textual). EUR-Lex pode chegar a ~1–2 h
se usar a release completa.

## Falhas e rollback

- **ArXiv sem dados:** rodar sem ArXiv e marcar o número legado como “reference only,
  test-threshold” (ou remover a coluna); registrar a decisão neste spec.
- **EUR-Lex:** o script usa a release completa (45k treino / 4.519 labels), não a subamostra de
  10k/2.901 do paper. Decidir: reproduzir a subamostra (pequeno ajuste no manager/flag) ou adotar
  a release completa e atualizar a descrição do dataset no paper.
- **Épocas:** o script usa 100 épocas/patience 10; o paper cita 50 (AAPD) e 30 (EUR-Lex).
  Registrar os valores efetivos e alinhar a seção de treino do paper.
- **JSON existente:** o script pula; apagar o JSON ou usar `--out_dir` novo (guardar backup).

## Atualizações no paper/tese

- `main.tex:713-724` (parágrafo de métricas; remover exceção e TODO).
- `tab:text` (733–757, linha 751), prosa 768–785.
- `tab:new_text` (806–827, linhas 823–824), prosa 829–853.
- `tab:ablation` (1453–1476, linhas 1462–1465).
- Discussão 1544–1547; per-level 1440–1442.
- Figuras hardcoded: `figures/make_fig1_text.py:29`, `make_fig4_cross_domain.py:36`,
  `make_fig_ablation.py:35`; regerar PDFs.
- `data/consolidated_results.json` (`text.*`).

## Checklist

- [x] Pré-requisitos verificados (dados, GPU, JSONs).
- [x] Decisão sobre ArXiv registrada (baixado e rodado).
- [x] Decisão sobre EUR-Lex (release completa) registrada.
- [x] Comando executado; JSONs conferidos (4 datasets, 9 runs cada).
- [x] Números novos extraídos.
- [ ] Paper atualizado e recompilado (spec 05).
- [x] Spec marcado como concluído no `README.md` (runs; paper em 05).
