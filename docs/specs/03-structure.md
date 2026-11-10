# 03 — `output/protocol/structure.json` (P1)

## Objetivo

Gerar `output/protocol/structure.json`, que alimenta a análise de estrutura da Seção 5.2 e a
`fig_structure.pdf`. Sem ele, o `make_paper_artifacts.py` pula a análise em silêncio.

## Pré-requisitos

- `experiments/hierarchy_structure.py` (treina BCE seed 0 nos 27 datasets).
- GPU livre; `uv run`.
- Os JSONs de protocolo não são necessários para este script, mas ele usa os mesmos módulos de
  dados (FunCat, GO, others, texto).

## Comando

```bash
PYTHONPATH=src uv run python experiments/hierarchy_structure.py
```

## Saídas esperadas

- `output/protocol/structure.json` com, por dataset:
  `internal_label_share`, `changed_share`, `mean_change`.
- `make_paper_artifacts.py` passa a gerar `fig_structure.pdf` e `structure_spearman`.

## Critério de aceite

- [ ] JSON cobre os 27 datasets.
- [ ] `fig_structure.pdf` é gerado.
- [ ] Seção 5.2 e cap. 5 da tese atualizados com os números.

## Tempo/custo

~4–6 h (BCE seed 0 em 27 datasets; é o segundo run mais longo depois do WOS).

## Falhas e rollback

- **Dataset sem dados:** o script deve pular; conferir a lista final.
- **Interrupção:** o script não tem resume; rodar de novo (guarda o JSON ao final).

## Atualizações no paper/tese

- `main.tex` Seção 5.2 (análise de estrutura) e `fig_structure.pdf`.
- `docs/tese/chapters/05-funcat-go.tex` (seção de estrutura).

## Checklist

- [x] Script executado.
- [x] JSON com os 27 datasets (campos `internal_label_share`, `changed_share`, `mean_change`).
- [x] `fig_structure.pdf` gerado.
- [ ] Paper/tese atualizados (spec 05).
- [x] Spec marcado como concluído no `README.md` (paper em 05).
