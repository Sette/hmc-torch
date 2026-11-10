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

## Dados fornecidos (10/11/2026)

- Autor: Bruno Silva Sette, `bruno.sette@ufscar.br`.
- Orientador: Diego Furtado Silva (UFSCar); e-mail e ORCID pendentes.
- Financiamento: sem agência; declaração de autofinanciamento no KBS.
- Declaração de IA: padrão Elsevier adicionada ao KBS (Claude/Anthropic; revisão e
  responsabilidade dos autores).
- IJCNN: double-blind; bloco nomeado mantido por enquanto; anonimizar antes da submissão.
- Abertos: e-mail/ORCID do orientador, 4–6 revisores, banca e agradecimentos adicionais.

## Critério de aceite

- [x] Nenhum `[Orientador]`/`[orientador@ufscar.br]` no PDF final (e-mail/ORCID do
  orientador ainda pendentes, sem placeholder no PDF).
- [ ] Reviewers sugeridos na cover letter.
- [x] Funding/agência preenchidos (sem agência; autofinanciamento declarado).
- [x] Build do paper e da tese limpos.

## Falhas

- Sem os dados do orientador/reviewers: manter os placeholders e registrar a pendência no
  `README.md`; não inventar nomes, e-mails ou grants.

## Checklist

- [x] Dados do orientador aplicados (nome; e-mail/ORCID pendentes).
- [ ] Reviewers aplicados (ou pendência).
- [x] Funding aplicado (sem agência; autofinanciamento declarado).
- [x] Declaração de IA (Elsevier) aplicada.
- [ ] Spec marcado como concluído no `README.md` (permanece `[~]`).
