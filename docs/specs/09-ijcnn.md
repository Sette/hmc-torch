# 09 — IJCNN 2027 (rota principal)

## Objetivo

Submeter a versão focada de **6 páginas** ao IJCNN 2027 (Cape Town, 14–18/06/2027). Se aceita,
o aceite em **15/03/2027** satisfaz o requisito (a) do PPGCC com folga; a **KBS estendida** é
submetida em março para o requisito (b) (submissão a periódico).

## Fatos

- Submissão abre 01/12/2026; prazo **31/01/2027**; notificação **15/03/2027**;
  camera-ready **12/04/2027**.
- **6 páginas** IEEE duas colunas, incluindo referências; até 4 páginas extras pagas
  (US$100/página, contadas pela maior versão — mirar 6).
- Qualis A2/A1 conforme a fonte; atende B1+.
- Template: `IEEEtran` classe `conference`.

## Projeto

`docs/hmc-paper-ijcnn/` (separado de `docs/hmc-paper`, que fica congelado como KBS):

- `main.tex` (IEEEtran, 5 páginas no estado atual), `refs.bib`, `figures/`
  (`fig_ablation.pdf`, `fig3_go_sparse.pdf`), `Makefile`, `README.md`, `.gitignore`.
- `make check` falha se passar de 6 páginas ou houver citação indefinida.

## Recorte

**Mantém:** R-matrix, reconciliação, R esparsa O(C+E), FunCat/GO/cross-domain, WOS,
decomposição raw/reconciliado, Surj (resumo), limitações.
**Deixa para a KBS/tese:** detalhes de plataforma, EUR-Lex em profundidade, Surj por dataset,
autoencoders, análise por nível detalhada, calibração.

## Checklist

- [x] Projeto e template IEEEtran criados.
- [x] Versão de 5 páginas compilando (≤6).
- [x] Tabelas (datasets, FunCat/GO, WOS, texto/cross-domain), figuras (decomposição, GO
      esparsa), orientação prática e limitações.
- [ ] Preencher nome/e-mail/ORCID do orientador.
- [ ] Revisão final de texto/figuras e conferência das datas no site do IJCNN.
- [ ] Submissão até 31/01/2027.
- [ ] Após 15/03: se aceito, camera-ready 12/04 e KBS estendida em março; se rejeitado,
      KBS imediatamente e reavaliar a defesa.

## Riscos

- **Rejeição em 15/03:** a KBS submetida em março só decide em agosto; a defesa de
  maio/junho ficaria sem aceite. Mitigação: IEEE Access (A2, rápido) ou adiamento.
- **Corte de 50%:** não sacrificar a R esparsa nem o achado da reconciliação.
- **Páginas extras:** não passar de 6 na submissão inicial (a taxa conta a maior versão).
