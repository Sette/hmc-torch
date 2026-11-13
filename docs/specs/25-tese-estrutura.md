# 25 — Posicionamento frente ao C-HMCNN (P0) e tese: título, revisão, RQs, ameaças, arguição (P1)

## Objetivo

Preparar `docs/tese/` para a banca nos pontos que são só da tese: alinhamento entre título,
perguntas de pesquisa e resultados; profundidade da revisão; ameaças à validade; resultados
que existem no repositório e não são reportados. Fecha os achados C1–C9 do
[relatório 18](18-audit-geral.md).

## Pré-requisitos

- Specs 20, 21 e 24 aplicados na tese.
- Backup da tese antes de editar (o diretório não é versionado no hmc-torch).
- Regras do PPGCC/UFSCar conferidas em 13/11/2026 (Normas Complementares 2024, Regimento
  Interno 2022): tese em inglês permitida (Art. 4); publicação ou aceite no Índice Restrito do
  Qualis + submissão a periódico do Índice Restrito (Art. 63), aluno primeiro autor e
  orientador coautor (Art. 64), comprovação no pedido de banca (Art. 66); versão original em
  até 1.460 dias e defesa em até 1.490 dias da matrícula. Falta confirmar com a secretaria o
  estrato atual do IJCNN e da KBS e os prazos contra a data de matrícula.

## Fatos a usar

- Tese: 80 páginas; capítulo 2 com 162 linhas; `refs.bib` com 24 entradas, todas citadas.
- Título (`main.tex:60`): “Scalable Knowledge-Constrained and Contrastive Representation
  Learning for Hierarchical Multi-Label Classification”.
- RQ4 como escrita (`01-introducao.tex:86-93`): reprodução do Surj, geometria dos embeddings,
  SR-GAE, H-MLCL+H-KNN, DepthCal. A resposta (`08-conclusao.tex:45-66`) abre com a comparação
  contra C-HMCNN e HPT.
- Resultados no repositório fora da tese:
  - `output/protein_lm/esm2_t33_650M_UR50D/` — ESM-2 congelado, protocolo completo, 3 seeds,
    ~5–6% das linhas sem ORF: `seq_FUN` 0,528 ± 0,004 (BCE + rec.) contra 0,290 com as features
    originais; `seq_GO` 0,601 ± 0,004 contra 0,446. O mapeamento linha → ORF
    (`experiments/recover_seq_ids.py`) não foi auditado.
  - `output/label_embeddings/mc/` — só `cellcycle_FUN`.
- **Ablação do C-HMCNN** (arXiv 2010.10151, Seção 5.5, Tabela 3; AUPRC na validação dos 8
  FunCat, BCE nos dois primeiros):

  | | `h+` (pós-processamento) | `h+MCM` | C-HMCNN (MCM + MCLoss) |
  |---|---|---|---|
  | Média (8) | 0,225 | 0,237 | 0,241 |
  | Ranking médio | 2,94 | 2,06 | 1,00 |

  O artigo conclui: “C-HMCNN(h), by exploiting both MCM and MCLoss, always outperforms h+ and
  h+MCM”. No nosso protocolo (test, 8 comuns) o sentido é o mesmo no FunCat: MCLoss + rec.
  0,2608 contra BCE + rec. 0,2593, e só o BCE + rec. é separado do C-HMCNN pelo Nemenyi.
- Protocolo do C-HMCNN: 10 execuções; dimensão oculta e learning rate ajustados por dataset na
  validação. O nosso: 3 seeds, learning rate fixo em 1e-4.
- ESM-2: `recover_seq_ids.py` foi auditado — usa só features, sem rótulos; 3 ORFs aparecem em
  test e em treino/validação (de ~3.700).
- Build: 4 overfull (um `\vbox` de 90 pt, um `\hbox` de 75 pt), 11 underfull.

## Passos

1. **Título (C1).** Decidir com o orientador entre:
   - (a) manter e reescrever o resumo para justificar “contrastive” como objeto de estudo com
     resultado negativo;
   - (b) trocar por um título que descreva a tese defendida, por exemplo “Hierarchical
     Constraints in Multi-Label Classification: Inference-Time Reconciliation, a Sparse
     Formulation, and the Limits of Ontology Embeddings” (e a versão em português).
   Conferir se o título registrado no programa pode mudar e até quando.
2. **Perguntas de pesquisa (C3).** Reescrever para que cada resultado tenha pergunta:
   - separar a comparação com o estado da arte (C-HMCNN, HMCN-F, HPT) numa pergunta própria
     (“RQ4: quão competitivo é um MLP com reconciliação sob protocolo só de validação?”) e
     deixar a parte exploratória como RQ5; ou
   - manter 4 RQs e mover a comparação para a RQ1/RQ2 como contexto, retirando “RQ4a”.
   Propagar para `01-introducao.tex:71-153`, `:155-191` e `08-conclusao.tex:13-66`.
3. **Revisão bibliográfica (C2).** Ampliar o capítulo 2 (meta: 12–15 páginas, 60+ referências).
   Tópicos ausentes, com trabalhos a localizar e conferir na fonte antes de citar:
   - correção de inconsistência após a predição em hierarquias: ensembles com a true-path rule
     em predição de função gênica; reconciliação por regressão isotônica e métodos bayesianos
     para saídas de GO; algoritmos gulosos para hierarquias em árvore e DAG; camadas de
     restrição por máximo anteriores ou contemporâneas ao C-HMCNN (por exemplo AWX);
   - reconciliação de previsões hierárquicas em séries temporais (o mesmo termo, outra área);
   - métricas hierárquicas (hF, hP, hR) e por que a tese usa micro-AUPRC/F1;
   - predição de função com redes profundas e modelos de linguagem de proteína;
   - aprendizado contrastivo supervisionado multirrótulo e k-NN em tempo de inferência (já há
     5 citações; falta a discussão);
   - HMC-LMLP e demais métodos locais brasileiros além de Cerri et al.
   Os nomes acima são pistas de busca, não referências: nenhuma entrada entra no `.bib` sem a
   referência conferida.
4. **Novidade da RQ1 (C9, P0 — vale para os três manuscritos e vem antes de qualquer
   submissão).** Retirar “as an indivisible package” (tese `01-introducao.tex:48-51`; KBS
   `main.tex:123`), “are always applied together” (tese `02-fundamentacao.tex:135-136`) e
   “bundle” (resumos: tese `frontmatter/abstract.tex:5`, KBS `:63`); no IJCNN (`main.tex:72-77`)
   trocar “typically evaluate the combination of both” por uma frase que cite a ablação.
   Texto-base: “Giunchiglia and Lukasiewicz ablate the two components on the validation splits
   of eight FunCat datasets and find that MCLoss adds to the max module on all of them; we
   extend that comparison to 27 datasets, a third objective, DAG ontologies, and test-set
   evaluation under a validation-only protocol.” Em seguida, discutir o resultado no lugar
   certo: no FunCat e no GO os nossos dados concordam em sentido com a ablação original
   (MCLoss + rec. é o melhor braço, por 0,001–0,002); a afirmação “training-time losses do not
   improve over BCE” vale para a média das quatro famílias e para cross-domain/texto, e precisa
   dessa ressalva em resumo, contribuições e conclusões.
5. **Ameaças à validade (C5).** Acrescentar em `07-discussao.tex:91-125`:
   - interna: época escolhida pelo score reconciliado (ou o controle do spec 23); teto de
     épocas no EUR-Lex; limiar transferido do modelo só-treino ao refeito; objetivo citado por
     família escolhido depois de ver o test;
   - externa: um único encoder de texto (SPECTER2); ArXiv limitado a 50.000 registros e sem
     split canônico; baselines de texto e HMCN-F copiados dos artigos (só C-HMCNN e HPT foram
     reexecutados); um único hardware;
   - conclusão estatística: 3 seeds; p-values sem correção (spec 24).
6. **Resultados não reportados (C6).** Decidir e executar:
   - ESM-2: (a) reportar como seção curta do capítulo 5 (o mapeamento já foi auditado: sem
     rótulos; 3 ORFs repetidos entre test e treino, a declarar) — o resultado
     mostra que as features pesam mais que qualquer restrição, o que reforça a leitura da tese
     e ajuda a responder “por que os ganhos são tão pequenos?”; ou (b) deixar fora e manter
     como trabalho futuro, sem dizer que não foi testado. Retirar “rather than end-to-end
     protein language models” das limitações se (a).
   - Embeddings com texto do `go.obo`: só um dataset; manter fora ou rodar o grupo `go`
     (`run_label_embeddings.py --group go`) antes de mencionar.
   - HPT + R (`globalHPT`): a tese já traz a célula reconciliada; conferir que o trabalho
     futuro nº 1 (`08-conclusao.tex:72-75`) não descreve algo que já foi feito.
7. **RQ2 com tempo (C4).** Reportar tempo por passo esparso × denso se o spec 21 (passo 6) o
   produzir; caso contrário, restringir a resposta da RQ2 a memória.
8. **Itens menores (C7, C8).** `05-funcat-go.tex:54-55`: sustentar “halves the epochs” com os
   `best_epoch` da ablação (`output/protocol_ablation/`) ou cortar. Corrigir os overfull
   (`grep -a -n "Overfull" main.log`): a `\vbox` de 90 pt é uma tabela ou figura maior que a
   página; a `\hbox` de 75 pt está num parágrafo das linhas 304–314 de um capítulo.
9. **Arguição.** Preparar respostas de até um parágrafo, com o número e a tabela, para:
   1. Se o ganho é de 0,002 AUPRC, por que isso é uma tese? (garantia de consistência a custo
      zero; decomposição; resultado negativo documentado)
   2. O C-HMCNN já não mostrou que o MCLoss ajuda? (passo 4)
   3. Por que BCE num lugar e MCLoss em outro? (spec 24)
   4. A época foi escolhida pelo score reconciliado — isso não favorece o rec? (spec 23)
   5. 27 datasets, 19 da mesma levedura: quantos são independentes?
   6. Por que micro-AUPRC e não métricas hierárquicas?
   7. O Surj não reproduz: os autores foram contatados? (`surj_repro/email_authors.md` existia
      no commit `42aadba^`)
   8. Se protótipos aleatórios dão o mesmo, o que sobra do SR-GAE como contribuição?
   9. O esparso é mais rápido ou só usa menos memória?
   10. O que muda com um encoder melhor (ESM-2, fine-tuning)? (passo 6)
10. `make check`; arquivar a tese (tarball datado).

## Saídas esperadas

- Título, RQs, contribuições e conclusões em correspondência um a um.
- Capítulo 2 ampliado, com cada referência conferida.
- Seção de ameaças à validade completa.
- Documento de respostas para a arguição (fora do texto da tese, por exemplo em
  `/home/brunos/git/phd/`).

## Critério de aceite

- [ ] Os três manuscritos citam a ablação da Seção 5.5 do C-HMCNN e nenhum diz que os
      mecanismos nunca foram separados.
- [ ] Cada RQ tem uma contribuição e uma resposta com o mesmo rótulo; nenhuma resposta traz
      resultado que a pergunta não pede.
- [ ] `refs.bib` ≥ 60 entradas, todas citadas e conferidas.
- [ ] As 8 ameaças do passo 5 aparecem no capítulo 7.
- [ ] Decisão registrada sobre ESM-2 (reportado com auditoria do mapeamento, ou fora).
- [ ] 0 overfull; 0 referências indefinidas.
- [ ] Lista de 10 perguntas com resposta escrita.

## Tempo/custo

Revisão bibliográfica: 2–3 semanas de leitura e escrita. Demais passos: ~2 dias. Sem compute,
salvo a opção de rodar `run_label_embeddings.py --group go` (~3 h de GPU).

## Falhas e rollback

- **O artigo do C-HMCNN já responde a RQ1 no FunCat:** reposicionar a contribuição como
  replicação ampliada com protocolo mais estrito, o que continua sendo defensável; não omitir.
- **A auditoria do `recover_seq_ids.py` encontra vazamento:** não reportar o ESM-2 e registrar
  o motivo.
- **Título não pode mudar no programa:** rota (a) do passo 1.
- Rollback: restaurar o tarball.

## Atualizações no paper/tese

- Tese: `main.tex:60`, `frontmatter/abstract.tex`, `frontmatter/resumo.tex`, capítulos 1, 2, 5,
  7 e 8, `refs.bib`.
- KBS: passo 4 em `hmc-paper/main.tex:63`, `:123` e na seção de trabalhos relacionados.
- IJCNN: passo 4 em `hmc-paper-ijcnn/main.tex:72-77` (uma frase; conferir as 6 páginas).
- `/home/brunos/git/phd/plano-defesa-2027.md`: cronograma da revisão bibliográfica.

## Checklist

- [ ] Título decidido.
- [ ] RQs e contribuições realinhadas.
- [ ] Capítulo 2 ampliado.
- [ ] Posicionamento frente à ablação do C-HMCNN corrigido em IJCNN, KBS e tese.
- [ ] Ameaças à validade.
- [ ] ESM-2 e embeddings de texto: decisão e execução.
- [ ] RQ2 com tempo ou restrita a memória.
- [ ] Itens menores e overfull.
- [ ] Respostas de arguição.
- [ ] `make check` e arquivo datado.
- [ ] Spec marcado como concluído no `README.md`.
