# 05 — Atualizar os números do paper (P0/P1)

## Objetivo

Propagar para o `docs/hmc-paper/main.tex` os resultados dos specs 01–04, remover o caveat de
limiar no test e garantir que todo número tenha origem em um JSON verificável.

## Pré-requisitos

- Specs 01–04 concluídos.
- `make_paper_artifacts.py` rodando (depende do spec 02 para hinge e do 03 para structure).

## O que atualizar (números hardcoded; não há macros)

### Texto (`tab:text`, `tab:new_text`, `tab:ablation`)
- `main.tex:713-724` — remover a exceção de limiar no test e o `% TODO(P0)`; afirmar o protocolo
  validation-only único.
- `tab:text` (733–757): linha 751 (ArXiv/WOS frozen), 752 (WOS E2E média ± desvio), 754 (Δ vs
  HPT); legenda 735–740.
- Prosa 768–785 (ArXiv/WOS frozen) e 782–792 (WOS E2E).
- `tab:new_text` (806–827): linhas 823–824; prosa 829–853.
- `tab:ablation` (1453–1476): linhas 1462–1465.
- Discussão 1544–1547; per-level 1440–1442.

### Hinge
- `tab:funcat` (861–900): nova coluna; média 891; legenda 863–869; análise 902–914;
  Friedman/Nemenyi 934–941 se hinge entrar no ranking.
- `tab:go_sota` (1182–1216): nova coluna; média 1208; prosa 1218–1233.
- `tab:others` (981–1011): hinge existe nos dados (~0,855), mas não na tabela; adicionar coluna.
- Abstract 69/75–76, contribuições 153–157, discussão 1566–1568, conclusão 1620–1633 (só se
  hinge virar número de destaque).

### Estrutura (spec 03)
- Seção 5.2 e `fig_structure.pdf`; `structure_spearman`.

### Figuras e resumo
- Regerar `fig1_sota_comparison.pdf`, `fig4_cross_domain.pdf`, `fig_ablation.pdf` a partir dos
  scripts em `docs/hmc-paper/figures/` (atualizar os valores hardcoded).
- `data/consolidated_results.json`: `text.*`, `total_datasets`, `seeds`.
- `cover_letter.md` e `submission/cover_letter.md` se os números de manchete mudarem.

## Critério de aceite

- [ ] Nenhum `% TODO(P0)` remanescente.
- [ ] Toda tabela/prosa atualizada e consistente com os JSONs.
- [ ] `make pdf` + `make check`: 0 undefined, 0 overfull.
- [ ] Figuras regeradas e coerentes.
- [ ] `data/consolidated_results.json` atualizado.

## Verificação

```bash
cd docs/hmc-paper
make check
grep -n "TODO(P0)\|test split" main.tex
grep -c "Overfull \\hbox" main.log
```

## Checklist

- [x] Texto atualizado (ArXiv/WOS frozen/AAPD/EUR-Lex; WOS E2E 3 seeds).
- [x] Hinge atualizado (colunas em `tab:funcat`, `tab:go_sota`, `tab:others`).
- [x] Caveat `% TODO(P0)` removido; protocolo validation-only único.
- [x] Figuras regeradas (fig1, fig4, fig_ablation).
- [x] `data/consolidated_results.json` atualizado.
- [x] Build limpo: 0 undefined, 0 overfull.
- [x] Structure: `structure.json`/`fig_structure.pdf` gerados; a análise fica na tese (o paper não tem subseção própria).
- [x] Spec marcado como concluído no `README.md`.
