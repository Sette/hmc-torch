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
| 06 | [Metadata](06-metadata.md) | — | [~] nome/afiliação/financiamento/IA aplicados; faltam e-mail/ORCID do orientador, revisores, banca | usuário |
| 07 | [Verify](07-verify.md) | — | [~] revisão editorial concluída; RCV1 JSON concluído; falta spec 06 (metadados) | 01–06 |
| 08 | [Reviewer risks (KBS)](08-reviewer-risks.md) | P0/P1 | [x] mitigações aplicadas em `docs/hmc-paper/` | 01–07 |
| 09 | [IJCNN 2027](09-ijcnn.md) | P0 | [x] 6 p., 0 over/underfull após a revisão; falta apenas orientador | — |
| 10 | [Prompt hierarchy](10-prompt-hierarchy.md) | P1 (KBS 2027) | [x] M1 e M2 implementados e testados (`tests/test_hpt.py`) | 11 |
| 11 | [Baselines re-run](11-baselines-rerun.md) | P0 | [x] HPT reexecutado (0,8691 ± 0,0023); KBS 49 p., IJCNN 6 p. | — |
| 12 | [RCV1 rota aberta](12-rcv1.md) | P2 | [x] resultados regenerados no KBS/tese; JSON em `output/protocol_rcv1/rcv1.json` (9 runs, 3 seeds) | — |
| 13 | [Review IJCNN (2026-11-11)](13-review-ijcnn.md) | P0/P1/P2 | [ ] documento de apoio; 4 P0, 4 P1, 6 P2 | — |
| 14 | [HPT provenance + WOS table](14-hpt-provenance-wos.md) | P0 | [x] “reimplementation” nos três manuscritos; célula reconciliada do HPT (0,8712 ± 0,0029) | — |
| 15 | [FunCat/GO claims](15-funcat-go-claims.md) | P0 | [x] IJCNN corrigido; KBS/tese conferidas | — |
| 16 | [GO scaling claims](16-go-scaling-claims.md) | P0 | [x] teste legado removido dos três manuscritos; memória medida em `output/scaling/dense_r_memory.json` | — |
| 17 | [IJCNN editorial](17-ijcnn-editorial.md) | P2 | [x] 7 itens aplicados; 6 p., 0 over/underfull | 14–16 |

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
- Repositórios: KBS `b0d9f99` (tags `kbs-2026-11-10` e `kbs-metadata-2026-11-10`);
  root `a451c2d` (código RCV1), `8656253` (tag `rcv1-open-2026-11-10`), `5357cba`
  (tag `ijcnn-metadata-2026-11-10`) e `417b0c8` (tag `metadata-2026-11-10`); tese
  arquivada em `/home/brunos/hmc-tese-2026-11-10.tar.gz`.
- Pendências: spec 06 (metadados do usuário).

## Revisão 2026-11-11

- Revisão do IJCNN (spec 13) e correções dos specs 14–16 aplicadas a IJCNN, KBS e tese.
- HPT: os números são da reimplementação do hmc-torch, não do código oficial (que não roda
  com transformers 5.3 / PyG 2.7); texto e tabelas dizem “reimplementation” e trazem a célula
  reconciliada (0,8712 ± 0,0029; Δ −0,0008 contra o nosso E2E).
- GO: o teste batch-64 (+0,0055 AUPRC) vinha de `run_go_experiments.py` (legado: treino em
  train+valid, limiar no test, seed 42) e saiu dos três manuscritos. No lugar entram os ganhos
  do protocolo (8 de 9 datasets, média +0,0016) e a memória medida por
  `experiments/measure_dense_r_memory.py` (denso float64: 8,16 GiB na inferência e OOM no
  treino em batch 64; esparso: 4 MB / 43 MB).
- IJCNN: “within 0.003” atribuído ao MCLoss + rec. (BCE + rec. fica a 0,005); Nemenyi com os
  5 métodos; frase de F1 por nível removida; GAE suavizado. 6 páginas, 0 over/underfull.
- KBS: 48 páginas, sem referências indefinidas. Tese compila; os 2 overfull hbox restantes (cap. 3 e apêndice A) são
  anteriores a esta revisão.
- 13/11/2026: spec 17 aplicado no IJCNN; na KBS e na tese, GAE suavizado (+0,002 na média),
  vetores de arestas renomeados para `s`/`t`, e a tese passou a citar o par C-HMCNN × BCE + rec.
  do Nemenyi.
- Repositórios: KBS `0b24288` (branch `docs/paper-revision`, `make submission` regenerado);
  root `663892f` (script de memória), `5cd367a` (IJCNN) e o commit dos specs. Nada foi enviado
  (sem push); o gitlink `docs/hmc-paper` não foi atualizado.
- Pendências: spec 06 (metadados); regras do IJCNN 2027 (duplo-cego, limite do abstract).

Regras:

- Nenhum commit sem pedido.
- Os números do paper são **hardcoded** no `main.tex`; a atualização é manual.
- O `make_paper_artifacts.py` gera fragmentos de referência, mas o paper atual não os `\input`.
- Se um número mudar, atualize também `data/consolidated_results.json` e os scripts de figura.
