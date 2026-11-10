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
| 07 | [Verify](07-verify.md) | — | [~] revisão editorial concluída; RCV1 JSON concluído; falta spec 06 (metadados) | 01–06 |
| 08 | [Reviewer risks (KBS)](08-reviewer-risks.md) | P0/P1 | [x] mitigações aplicadas em `docs/hmc-paper/` | 01–07 |
| 09 | [IJCNN 2027](09-ijcnn.md) | P0 | [x] 6 p., 0 over/underfull após a revisão; falta apenas orientador | — |
| 10 | [Prompt hierarchy](10-prompt-hierarchy.md) | P1 (KBS 2027) | [x] M1 e M2 implementados e testados (`tests/test_hpt.py`) | 11 |
| 11 | [Baselines re-run](11-baselines-rerun.md) | P0 | [x] HPT reexecutado (0,8691 ± 0,0023); KBS 49 p., IJCNN 6 p. | — |
| 12 | [RCV1 rota aberta](12-rcv1.md) | P2 | [x] resultados regenerados no KBS/tese; JSON em `output/protocol_rcv1/rcv1.json` (9 runs, 3 seeds) | — |

> **Rota atual:** IJCNN 2027 é a submissão principal (31/01/2027; notificação 15/03/2027).
> A versão completa da KBS fica **congelada** em `docs/hmc-paper/` e será submetida em março,
> após a decisão do IJCNN, para o requisito de submissão a periódico.
>
> **Next steps:** a revisão editorial de 10/11/2026 corrigiu a tabela fatorial, separou
> AUPRC (sem DepthCal) de F1 (com DepthCal), padronizou H-MLCL+H-KNN e os desvios amostrais,
> e regenerou o RCV1 (JSON em `output/protocol_rcv1/rcv1.json`, 9 runs; números atualizados
> no KBS/tese). O pacote de submissão foi regenerado com `make submission`; faltam os
> metadados do spec 06. O spec 10 (prompt encoder) tem go/no-go em 15/01/2027 para a KBS de março.

Documento de apoio (não sequencial): [08 — Riscos de revisão na KBS](08-reviewer-risks.md) —
objeções prováveis e mitigações.

## Revisão 2026-11-10

- KBS: corrigida `tab:decomp_family` (valores não batiam com `output/protocol/*`); prosa
  dependente, atribuição AUPRC/F1, `tab:others`, deltas EUR-Lex/Diatoms, CD, memória,
  contagem de edges, root sintético, cosine, `L`, GBDT, Surj, contagens de famílias/Table 1,
  labels de “full model” e Figura 1.
- IJCNN: RCV1 passou a “rota aberta LYRL2004”; seeds qualificados; “never outperforms”
  suavizado; 6 páginas, 0 overfull/underfull; H-MLCL+H-KNN.
- Tese: abstract/resumo, cap. 1/5/6/7/8 alinhados; tabela fatorial corrigida; RCV1
  consistente; H-MLCL+H-KNN; Bessel em vez de Parseval; definições `w_i`/`\tilde{w}_{b,m}`;
  `L` por nível; título e siglas.
- RCV1: bateria completa em 10/11/2026 na GPU (pico de 7,25 GiB; cache de 2,4 GB
  reutilizado); JSON `output/protocol_rcv1/rcv1.json`; números do KBS/tese atualizados
  (`bce` 0,7634→0,7673; `hinge` 0,7656→0,7682; `mc` 0,7133→0,7629 AUPRC); GHV medido
  removido (não gravado pelo `run_protocol.py`; reconciliação garante GHV=0 por construção).
- DepthCal: corrigida a atribuição — o F1 26/27 é do pipeline completo (H-MLCL + DepthCal)
  contra o baseline, não do DepthCal isolado; dentro do H-MLCL, o DepthCal isolado melhora
  AUPRC em 11/27 e F1 em 18/27 (`output/sr_gae_depth`). A ressalva de independência dos 27
  datasets foi adicionada às limitações do KBS/tese, e `consolidated_results.json` ganhou a
  entrada `depthcal`.
- Tese: eliminados os 3 overfull hbox (equações do cap. 3 e tabela do cap. 4); restam 2
  overfull vbox pré-existentes (capa e quebra do cap. 3) e 7 underfull hbox. A tabela 3-seed
  de suporte ao 27/27 foi omitida do corpo porque gerava uma página em branco; o parágrafo de
  suporte já traz as médias por família, os p-values e o trade-off do DepthCal.
- Repositórios: KBS `c0fecaf` (tag `kbs-2026-11-10`); root `8656253` (tag
  `rcv1-open-2026-11-10`) e `e571096` (tag `ijcnn-2026-11-10`); tese arquivada em
  `/home/brunos/hmc-tese-2026-11-10.tar.gz`.
- Pendências: spec 06 (metadados do usuário).

Regras:

- Nenhum commit sem pedido.
- Os números do paper são **hardcoded** no `main.tex`; a atualização é manual.
- O `make_paper_artifacts.py` gera fragmentos de referência, mas o paper atual não os `\input`.
- Se um número mudar, atualize também `data/consolidated_results.json` e os scripts de figura.
