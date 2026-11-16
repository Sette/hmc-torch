# 07 — Verificação final

## Objetivo

Fechar a bateria: builds, pacote de submissão, sincronização da tese e revisão do diff, sem
commits não solicitados.

## Comandos

```bash
# Paper
cd docs/hmc-paper
make check
make submission

# Tese
cd docs/tese
make pdf
make check
make sync          # se spec 02 concluído
make sync-paper    # fallback

# Estado dos repos
cd /home/brunos/git/hmc-torch
git status --short
git -C docs/hmc-paper diff --stat
git -C docs/tese status --short
```

## Critério de aceite

- [ ] Paper: 0 undefined, 0 overfull; `submission/` com PDF + cover letter + highlights.
- [ ] Tese: `main.pdf` compila; `make check` OK; `make sync` completa (ou fallback justificado).
- [ ] Nenhum número do paper sem JSON de origem.
- [ ] `docs/specs/README.md` com todos os estados atualizados.
- [ ] Diff revisável; nenhum commit sem pedido.

## Checklist

- [x] Paper verificado (0 undefined, 0 overfull).
- [x] Tese verificada (`make check` OK).
- [x] `submission/` conferido (PDF + cover letter + highlights).
- [x] Status dos specs atualizado.
- [x] Pendências remanescentes listadas: metadados do orientador/reviewers/funding (spec 06).
