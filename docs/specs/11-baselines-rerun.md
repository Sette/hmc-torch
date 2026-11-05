# 11 — Reexecução de baselines no WOS (P0)

## Objetivo

Trocar a única comparação de texto ainda não verificada — HPT 0,8716 no WOS — por uma
reexecução nossa, com JSON de origem, e documentar no paper a proveniência de **cada** número de
baseline. O HPT é o único número de texto acima do nosso (0,8704 ± 0,0033) e o único que o paper
usa como teto; os demais já vêm de reexecuções de terceiros ou foram reproduzidos por nós.

## Resposta curta: não é preciso rodar tudo de novo

A tabela do HPT (EMNLP'22, Tabela 1, WOS) separa as linhas originais das linhas com BERT:

| Modelo | Micro-F1 | Macro-F1 | Proveniência |
|---|---:|---:|---|
| HiAGM | 85,82 | 80,28 | paper original |
| HTCInfoMax | 85,58 | 80,05 | paper original |
| HiMatch | 86,20 | 80,53 | paper original |
| BERT+HiAGM | **86,04** | 80,19 | reexecução do HPT com BERT |
| BERT+HTCInfoMax | **86,30** | 79,97 | reexecução do HPT com BERT |
| BERT+HiMatch | 86,70 | 81,06 | reexecução do HPT com BERT |
| HGCLR | 87,11 | 81,20 | paper do HGCLR (BERT) |
| HPT | **87,16** | 81,93 | auto-reportado, sem desvio |

Nosso paper usa **86,04** e **86,30** para HiAGM/HTCInfoMax: são as linhas BERT do HPT, ou seja,
já são reexecuções de terceiros no mesmo split (30.070/7.518/9.397), não números copiados de
setups diferentes. O HGCLR é BERT por construção e reportado no mesmo split. O HPT 0,8716 é o
único número auto-reportado acima de nós, sem desvio-padrão e sem seed/época declaradas.

Logo, o mínimo defensável é:

| Tier | Baseline | Ação | Custo |
|---|---|---|---|
| A | HPT no WOS, `bert/none/down`, seeds 0/42/123 | **Rodar agora** | ~14–25 h GPU |
| B1 | HPT `bert/none/both`, seed 0 | Opcional, só se o Tier A divergir | ~5–8 h |
| B2 | HPT `specter2/none/down` | Não rodar: não é baseline publicado e o MLM do SPECTER2 é aleatório | — |
| C | HiAGM, HTCInfoMax, HiMatch, HMCN-F | Não reexecutar: os três primeiros já são reexecuções BERT do HPT; HMCN-F não é baseline de texto | — |
| C2 | HGCLR | Não reexecutar agora; só se a proveniência for questionada | 0 ou 12–24 h |
| D | C-HMCNN (FunCat), Surj | Concluído: paridade ≤0,003 AUPRC; Surj não verificável | — |

## Pré-requisitos

- `data/wos` presente; split HPT 30.070/7.518/9.397.
- Modelos em cache: `models/bert-base-uncased__AutoModelForMaskedLM` (existe).
- GPU livre (a execução do EUR-Lex corrigido deve terminar antes).
- `PYTHONPATH=src`, `uv run`.
- `output/hpt/` está vazio hoje; não há resultados antigos para reaproveitar.

## Correção de fidelidade (obrigatória antes do Tier A)

O `fit_hpt` seleciona época pelo micro-F1 **reconciliado** na validação com threshold ajustado.
O HPT seleciona pelo **macro-F1 cru** na validação, a cada época, com threshold logit 0
(probabilidade 0,5), paciência 5 e **sem cap de épocas**. Rodar com a seleção padrão não
reproduz o HPT; por isso foi adicionado um modo de fidelidade:

- `HPTConfig.selection`: `rec_val_micro` (default, nosso protocolo) ou `hpt_raw` (HPT).
- `selection_score()` em `src/hmc/pipeline/hpt/main.py`: `hpt_raw` = macro-F1 cru na validação em
  0,5; `rec_val_micro` = micro-F1 reconciliado no threshold ajustado.
- `experiments/run_hpt.py`: flags `--selection {rec_val_micro,hpt_raw}` e `--patience N`
  (default 6; usar 5 no Tier A); o nome do JSON ganha o sufixo `_hpt_raw` para não colidir com
  runs do protocolo padrão.
- Testes: `tests/test_hpt.py` cobre os dois modos.

Desvios remanescentes do HPT de referência, a registrar no spec/paper: inicialização dos rótulos
sem `[CLS]`/`[SEP]`; MLM só no vocabulário original; batch 8 × acumulação 2 (16) em 12 GB;
avaliação micro/macro-F1 padrão em vez do loop customizado do HPT.

## Comandos

```bash
cd /home/brunos/git/hmc-torch
mkdir -p output/hpt/logs

# 0. Smoke test (<5 min): valida o modo hpt_raw e o caminho de saída
PYTHONPATH=src uv run python experiments/run_hpt.py \
  --dataset wos --stage final --backbone bert --loss none --graph down \
  --selection hpt_raw --patience 5 --seed 0 --max_epochs 1 --limit 512 \
  --out_dir output/hpt_smoke

# 1. Seed 0, WOS completo (4,5–8,5 h)
PYTHONPATH=src uv run python experiments/run_hpt.py \
  --dataset wos --stage final --backbone bert --loss none --graph down \
  --selection hpt_raw --patience 5 --seed 0 --max_epochs 30 \
  --out_dir output/hpt 2>&1 | tee output/hpt/logs/spec11_seed0.log

# 2. Se o seed 0 passar do gate, seeds 42 e 123 (sequenciais)
PYTHONPATH=src uv run python experiments/run_hpt.py \
  --dataset wos --stage final --backbone bert --loss none --graph down \
  --selection hpt_raw --patience 5 --seed 42 --max_epochs 30 \
  --out_dir output/hpt 2>&1 | tee output/hpt/logs/spec11_seed42.log

PYTHONPATH=src uv run python experiments/run_hpt.py \
  --dataset wos --stage final --backbone bert --loss none --graph down \
  --selection hpt_raw --patience 5 --seed 123 --max_epochs 30 \
  --out_dir output/hpt 2>&1 | tee output/hpt/logs/spec11_seed123.log

# 3. Opcional B1, só depois dos três seeds
PYTHONPATH=src uv run python experiments/run_hpt.py \
  --dataset wos --stage final --backbone bert --loss none --graph both \
  --selection hpt_raw --patience 5 --seed 0 --max_epochs 30 \
  --out_dir output/hpt 2>&1 | tee output/hpt/logs/spec11_both_seed0.log
```

**Não usar `--max_epochs 10`**: o orçamento de 10 épocas é do nosso `globalE2E`, não do HPT. Se
`best_epoch == 30` em algum seed, o run foi truncado; repetir com `--max_epochs 40` antes de
interpretar.

## Saídas esperadas

- `output/hpt/final/wos/bert_none_down_hpt_raw_seed0.json`
- `..._seed42.json`, `..._seed123.json`
- Opcional: `bert_none_both_hpt_raw_seed0.json`
- Logs em `output/hpt/logs/`
- Resumo agregado (script novo `experiments/summarize_hpt.py`):
  `output/hpt/final/wos/summary.json` com média/desvio por seed para `raw/thr0` e `rec/val`
  (F1, macro-F1, AUPRC, GHV), `seeds`, `best_epochs`, tamanhos dos splits e caminhos de origem.

Cada JSON tem `config` (com `selection` e `patience`), `best_epoch`, `history` (com
`selection_score`), `train_time_s`, `n_train/n_valid/n_test` e `raw`/`rec` × `thr0`/`val`.
O número comparável ao HPT é **`raw/thr0` micro-F1** (threshold logit 0).

## Critério de aceite

- [ ] Patch de fidelidade aplicado; `pytest tests/test_hpt.py -q` passa.
- [ ] Três JSONs completos com `n_train=30070`, `n_valid=7518`, `n_test=9397`;
      `config.selection=="hpt_raw"`, `config.patience==5`, `best_epoch < max_epochs`.
- [ ] `summary.json` com média/desvio e caminhos das fontes.
- [ ] IJCNN atualizado antes (abstract ~38, prosa ~298–300, `tab:wos` ~315–327, Limitations
      ~377–383, Threats ~388–390); `make check` ≤6 páginas.
- [ ] KBS atualizado (caption ~737–738, linha do HPT ~747, prosa ~783, contribuição ~148,
      conclusão ~1632) e uma tabela curta `tab:hpt-repro` na subseção de reprodutibilidade.
- [ ] `docs/hmc-paper/figures/make_fig1_text.py:35` atualizado se o valor plotado mudar.
- [ ] `docs/hmc-paper/data/consolidated_results.json` `text.wos` com `hpt_repro` e fontes.
- [ ] Spec marcado no `docs/specs/README.md`.

## Resultado parcial (04/11/2026)

Seed 0 (`bert/none/down`, `hpt_raw`, paciência 5, cap 30): melhor época **8**, early stop na 13,
3,08 h; `raw/thr0` micro-F1 **0,8672**, macro-F1 0,8085, AUPRC 0,9185; `rec/thr0` 0,8678.
Fica **0,0044 abaixo** do 0,8716 publicado, fora da banda de consistência de ±0,002. Pela regra
de decisão: reportar publicado e reexecutado, **não** alegar que batemos o HPT; rodar seeds 42/123
para a média e o diagnóstico `both` (B1) para ver se o grafo reverso explica a diferença.

## Tempo/custo

- Seed 0: 3,1 h (13 épocas, early stop); três seeds: ~9 h; B1: +3 h.
- Calendário: 1–2 dias com runs noturnos.
- OOM: o WOS cabe em 12 GB; liberar a GPU antes. Se precisar, reduzir batch e registrar.

## Falhas e rollback

- **Interrupção:** `run_hpt.py` pula JSON/.lock existente; remover o `.lock`/JSON parcial e
  rodar de novo. Não há resume.
- **`best_epoch == max_epochs`:** truncado; repetir com cap maior.
- **Divergência do 0,8716:**

| Média reexecutada (`raw/thr0` micro-F1) | Ação no paper |
|---|---|
| 0,8686–0,8746 | Reportar `X ± Y`; consistente com 0,8716; diferença para 0,8704 dentro do ruído |
| < 0,8686 | Reportar publicado e reexecutado; checar época < cap, init dos rótulos e rodar `both`; não alegar que batemos o HPT |
| > 0,8746 | Reportar ambos; o publicado pode subestimar o HPT; checar vazamento; não alegar margem |
| desvio entre seeds > 0,006 | Adicionar um quarto seed (3, default do HPT) antes de atualizar o paper |

- **Sem ganho/ruído:** manter o resultado medido; o paper não é SOTA e não deve forçar número.

## Atualizações no paper/tese

- `docs/hmc-paper/main.tex`: WOS ~780–791; caption da tabela de texto ~731–738; contribuições
  ~142–150; conclusão ~1631–1634; nova `tab:hpt-repro`.
- `docs/hmc-paper-ijcnn/main.tex`: WOS ~298–300; tabela ~310/320–321; Limitations/Threats
  ~377–390.
- `data/consolidated_results.json` e scripts de figura, se o número plotado mudar.
- `docs/specs/05-paper-numbers.md` e `docs/specs/07-verify.md`.
- Tese: capítulos 4/7 são esqueletos; atualizar só a nota de escopo.

## Checklist

- [x] Spec escrito e revisado.
- [x] Patch de fidelidade + testes.
- [x] Smoke test ok.
- [x] Seed 0 rodado; gate conferido (0,8672; rodar 42/123 + `both`).
- [ ] Seeds 42/123 rodados.
- [ ] Summary agregado.
- [ ] IJCNN e KBS atualizados e compilando.
- [ ] `consolidated_results.json`/figuras atualizados.
- [ ] Spec marcado no README.
