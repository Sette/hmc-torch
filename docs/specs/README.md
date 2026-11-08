# Specs — pendências do paper (KBS) e da tese

Execução **sequencial**: uma tarefa só começa depois que a anterior fecha o checklist.
Atualize a tabela ao concluir cada spec.

| # | Spec | Prioridade | Estado | Depende de |
|---|---|---|---|---|
| 01 | [Text thresholds](01-text-thresholds.md) | P0 | [x] runs concluídos | dados de texto |
| 02 | [Hinge FunCat/GO](02-hinge-funcat-go.md) | P1 | [x] rodado/mesclado | — |
| 03 | [Structure](03-structure.md) | P1 | [x] rodado | — |
| 04 | [WOS 3 seeds](04-wos-3-seeds.md) | P0 | [x] rodado | — |
| 05 | [Paper numbers](05-paper-numbers.md) | P0/P1 | [x] atualizado | 01–04 |
| 06 | [Metadata](06-metadata.md) | — | [~] aguardando dados do usuário | usuário |
| 07 | [Verify](07-verify.md) | — | [x] verificado | 01–06 |
| 08 | [Reviewer risks (KBS)](08-reviewer-risks.md) | P0/P1 | [x] mitigações aplicadas em `docs/hmc-paper/` | 01–07 |
| 09 | [IJCNN 2027](09-ijcnn.md) | P0 | [x] versão de 6 p. revisada e verificada; falta apenas orientador | — |
| 10 | [Prompt hierarchy](10-prompt-hierarchy.md) | P1 (KBS 2027) | [ ] especificado; execução após 11 | 11 |
| 11 | [Baselines re-run](11-baselines-rerun.md) | P0 | [x] HPT reexecutado (0,8691 ± 0,0023); papers atualizados | — |
| 12 | [RCV1 rota aberta](12-rcv1.md) | P2 | [ ] especificado; gate de decisão | — |

> **Rota atual:** IJCNN 2027 é a submissão principal (31/01/2027; notificação 15/03/2027).
> A versão completa da KBS fica **congelada** em `docs/hmc-paper/` e será submetida em março,
> após a decisão do IJCNN, para o requisito de submissão a periódico.
>
> **Next steps:** o spec 11 (reexecução do HPT no WOS) é o único em execução agora e precisa
> entrar no IJCNN antes de 31/01/2027. O spec 10 (prompt encoder) tem go/no-go em 15/01/2027
> para a KBS de março; o spec 12 (RCV1 aberto) é um gate de decisão, não uma execução imediata.

Documento de apoio (não sequencial): [08 — Riscos de revisão na KBS](08-reviewer-risks.md) —
objeções prováveis e mitigações.

Regras:

- Nenhum commit sem pedido.
- Os números do paper são **hardcoded** no `main.tex`; a atualização é manual.
- O `make_paper_artifacts.py` gera fragmentos de referência, mas o paper atual não os `\input`.
- Se um número mudar, atualize também `data/consolidated_results.json` e os scripts de figura.
