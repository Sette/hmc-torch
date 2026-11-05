# 06 — Metadados da submissão (orientador, reviewers, funding)

## Objetivo

Preencher os placeholders de autoria e submissão do paper/tese. Parte dos dados depende do
usuário; o que estiver disponível é aplicado, o resto fica listado como pendência explícita.

## Pré-requisitos

- Nome completo, e-mail institucional, ORCID e afiliação do orientador.
- 4–6 suggested reviewers (nome, e-mail, afiliação), sem conflito.
- Agência/grant de financiamento (CAPES/CNPq/FAPESP e número).
- Papéis CRediT do orientador.

## O que preencher

- `docs/hmc-paper/main.tex`:
  - `\author[ufscar]{[Orientador]}` e `\ead{[orientador@ufscar.br]}`.
  - CRediT: `[Orientador]: Supervision, ...`.
  - Acknowledgments: agência/grant.
  - Declaração de IA (se exigida pela KBS).
- `docs/hmc-paper/cover_letter.md`: reviewers sugeridos; nome do orientador.
- `docs/tese/main.tex`: `\orientador{[Orientador]}`.
- `docs/tese/frontmatter/folha-aprovacao.tex`: banca (quando definida).

## Critério de aceite

- [ ] Nenhum `[Orientador]`/`[orientador@ufscar.br]` no PDF final (ou pendência listada).
- [ ] Reviewers sugeridos na cover letter.
- [ ] Funding/agência preenchidos.
- [ ] Build do paper e da tese limpos.

## Falhas

- Sem os dados do orientador/reviewers: manter os placeholders e registrar a pendência no
  `README.md`; não inventar nomes, e-mails ou grants.

## Checklist

- [ ] Dados do orientador aplicados (ou pendência registrada).
- [ ] Reviewers aplicados (ou pendência).
- [ ] Funding aplicado (ou pendência).
- [ ] Spec marcado como concluído no `README.md`.
