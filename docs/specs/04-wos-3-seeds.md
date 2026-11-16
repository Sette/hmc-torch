# 04 — WOS end-to-end com 3 seeds (P0)

## Objetivo

Substituir o resultado de seed única (0,8671) pela média ± desvio das seeds 0, 42 e 123 do
`globalE2E` no WOS, sob o protocolo do HPT (época e limiar na validação).

## Pré-requisitos

- `output/protocol/wos_ep10/wos_e2e_seed0.json` existe (seed 0, `--epochs 10`).
- Dados de WOS em `data/wos`.
- GPU RTX 3060 (~4,47 h/seed; ~26 min/época).
- `run_wos_e2e.py` **pula** se `wos_e2e_seed<N>.json` ou `.lock` existir; sem resume.

## Comandos

```bash
# uma seed por vez, em background; 10 épocas (o default do script é 4!)
PYTHONPATH=src uv run python experiments/run_wos_e2e.py \
  --seed 42 --epochs 10 --out_dir output/protocol/wos_ep10
PYTHONPATH=src uv run python experiments/run_wos_e2e.py \
  --seed 123 --epochs 10 --out_dir output/protocol/wos_ep10

# consolidar as 3 seeds no nível superior para o make_paper_artifacts.py
cp output/protocol/wos_ep10/wos_e2e_seed*.json output/protocol/
```

## Saídas esperadas

- `output/protocol/wos_ep10/wos_e2e_seed42.json`, `wos_e2e_seed123.json` (seed 0 já existe).
- Cada JSON: `seed`, `best_epoch`, `history`, `eval.raw/rec` (micro-F1, AUPRC).
- Média ± desvio das 3 seeds.

## Critério de aceite

- [ ] 3 JSONs de seed.
- [ ] Média/desvio calculados.
- [ ] `tab:text` (linha 752) e Δ vs HPT (754) atualizados.
- [ ] Prosa 782–792, 1076–1081, `tab:gpu` 1093–1094, contribuição 147–148 e conclusão
      1616–1617 atualizadas.
- [ ] Figuras `make_fig1_text.py:36` e `make_fig4_cross_domain.py:37` regeradas.
- [ ] `data/consolidated_results.json` (`text.wos.seeds`, `micro_f1`) atualizado.

## Tempo/custo

~4,47 h/seed; ~9 h para as duas seeds. Rodar em background e monitorar.

## Falhas e rollback

- **Interrupção:** não há resume; remover o `.lock`/JSON parcial e rodar de novo.
- **OOM:** o script usa batch 8 fixo; reduzir `--batch_size` se necessário (e registrar).
- **Números piores:** manter o resultado; o paper reporta o que foi medido, com a ressalva.

## Atualizações no paper/tese

- `main.tex:751-754` (tabela), 782–792 (prosa), 1076–1081 (`single seed`), 1093–1094 (GPU),
  147–148 (contribuição), 1616–1617 (conclusão).
- `figures/make_fig1_text.py`, `figures/make_fig4_cross_domain.py`.
- Tese cap. 4, linha 17.

## Checklist

- [ ] Seed 42 rodada e JSON conferido.
- [ ] Seed 123 rodada e JSON conferido.
- [ ] Média/desvio calculados.
- [ ] Paper/figuras atualizados.
- [ ] Spec marcado como concluído no `README.md`.
