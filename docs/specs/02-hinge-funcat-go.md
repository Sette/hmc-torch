# 02 — Hinge em FunCat/GO e consolidação do protocolo (P1)

## Objetivo

Adicionar o objetivo `hinge` aos JSONs de FunCat e GO, consolidar os JSONs de GO em
`output/protocol/` e destravar o `make_paper_artifacts.py` (hoje quebra com
`KeyError: hinge_raw`) e o `make sync` da tese.

## Pré-requisitos

- `output/protocol/*_FUN.json` existentes (10 FunCat, com BCE/MCLoss/GBDT).
- GO dividido em `output/protocol_go_recipe/{bce,mc}/<dataset>.json` (9 × 2).
- `run_protocol.py` **pula** dataset cujo `<out_dir>/<dataset>.json` existe; não faz merge
  incremental. Rodar em `--out_dir` novo.
- `--no_gbdt` no staging (o GBDT de FunCat custa ~0,8 h e já existe).

## Comandos

```bash
# 1. consolidar GO bce+mc em output/protocol/<ds>_GO.json
python3 docs/specs/merge_runs.py output/protocol_go_recipe/bce output/protocol
python3 docs/specs/merge_runs.py output/protocol_go_recipe/mc  output/protocol

# 2. rodar hinge em diretório novo (não pula por não existir lá)
PYTHONPATH=src uv run python experiments/run_protocol.py \
  --group funcat --objectives hinge --out_dir output/protocol_hinge --no_gbdt
PYTHONPATH=src uv run python experiments/run_protocol.py \
  --group go --objectives hinge --out_dir output/protocol_hinge --no_gbdt

# 3. merge dos runs hinge nos JSONs principais (dedup por objective+seed)
python3 docs/specs/merge_runs.py output/protocol_hinge output/protocol

# 4. regenerar artefatos e sincronizar a tese
PYTHONPATH=src uv run python experiments/make_paper_artifacts.py --paper_dir docs/tese
cd docs/tese && make sync
```

O `merge_runs.py` cria o JSON de destino se ele não existir (caso do GO) e preserva os runs
existentes; a deduplicação é por `(objective, seed)`.

## Saídas esperadas

- Cada `output/protocol/<ds>_FUN.json` e `<ds>_GO.json` com runs `bce`, `mc` e `hinge` × 3 seeds.
- `make_paper_artifacts.py` completa; `docs/tese/{data,figures,tables}` preenchidos.

## Critério de aceite

- [ ] 10 FunCat e 9 GO com hinge × 3 seeds.
- [ ] Números BCE/MCLoss existentes inalterados.
- [ ] `KeyError: hinge_raw` não ocorre mais.
- [ ] `make sync` da tese completa.

## Tempo/custo

- Hinge FunCat: ~1,1 h. Hinge GO: ~2,0 h. Total ~3,1 h (sem GBDT).

## Falhas e rollback

- **GO sem JSON principal:** consolidar `protocol_go_recipe/{bce,mc}` antes do merge; se faltar
  algum, rodar bce/mc também (custo adicional).
- **Merge duplicado:** deduplicar por `(objective, seed)`; manter o registro mais recente.
- **`make sync` ainda falha:** conferir se todos os datasets da família têm hinge e se
  `structure.json` (spec 03) existe.

## Atualizações no paper/tese

- `tab:funcat` (861–900): nova coluna “Ours hinge + rec.”; média 891; legenda/análise 863–869,
  902–914; Friedman/Nemenyi 934–941 se hinge entrar no ranking.
- `tab:go_sota` (1182–1216): nova coluna; média 1208; prosa 1218–1233.
- `tab:others` (981–1011): hinge **existe nos dados** (média ~0,855), mas não na tabela; adicionar
  coluna em 996–998, linhas 1000–1003, média 1005 e F1 no rodapé 1007–1008.
- `make_paper_artifacts.py`: `funcat.tex`/`go.tex` (405–427) e `others.tex` (429–440) não emitem
  hinge; ajustar se as tabelas geradas forem usadas.
- Tese: cap. 5 (linhas 17 e 25) e cap. 6 (linha 13) se hinge entrar no cross-domain.

## Checklist

- [x] GO consolidado em `output/protocol/` (9 JSONs, bce+mc × 3 seeds).
- [x] Hinge FunCat/GO rodado (19 JSONs, 3 seeds) e mesclado (57 runs).
- [x] `make_paper_artifacts.py` completa (sem `KeyError: hinge_raw`).
- [x] `make sync` da tese completa.
- [ ] Tabelas do paper atualizadas (spec 05).
- [x] Spec marcado como concluído no `README.md` (paper em 05).
