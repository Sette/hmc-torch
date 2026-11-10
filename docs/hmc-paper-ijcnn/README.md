# IJCNN 2027 paper (6 pages, IEEE two-column)

Versão focada do paper para o **IJCNN 2027** (Cape Town, 14–18/06/2027; submissão até
31/01/2027; notificação 15/03/2027). A versão completa para a **KBS** permanece em
`docs/hmc-paper/` e será submetida depois da decisão do IJCNN.

## Build

```bash
make deps   # instala IEEEtran no ~/texmf, se necessário
make pdf    # gera main.pdf
make check  # falha se houver citação indefinida ou passar de 6 páginas
```

## Recorte

- Mantém: R-matrix, reconciliação, R esparsa O(C+E), FunCat/GO/cross-domain, WOS.
- Deixa para a KBS/tese: detalhes de plataforma, EUR-Lex, Surj, autoencoders, análise
  por nível detalhada, calibração.
- Números: os do paper da KBS atualizado (validation-only, 3 seeds).

## Estado

- [x] Esqueleto e template IEEEtran
- [x] Texto de 6 páginas
- [x] Tabelas/figura
- [x] Build ≤6 páginas
- [ ] Submissão 31/01/2027
