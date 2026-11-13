# 15 — Afirmações de FunCat/GO: “within 0.003”, Nemenyi, por nível e GAE (P0)

## Objetivo

Alinhar à Tabela 3 e aos JSONs quatro afirmações da Seção 5.2, da discussão e da conclusão do
paper IJCNN. Fecha os achados 2, 3, 6 e 7 do [relatório 13](13-review-ijcnn.md). Só edição de
texto.

## Pré-requisitos

- `output/protocol/*.json`, `docs/tese/data/protocol_summary.json` (`funcat_stats`),
  `output/autoencoder/final/gae_hybrid@64/` (existem).

## Fatos a usar

AUPRC nos 8 datasets comuns (reconciliado):

| | FunCat | Δ C-HMCNN (0,264) | GO | Δ C-HMCNN (0,420) | Δ HMCN-F (0,422) |
|---|---|---|---|---|---|
| BCE + rec. | 0,2593 | −0,005 | 0,4163 | −0,004 | −0,006 |
| Hinge + rec. | 0,2593 | −0,005 | 0,4166 | −0,003 | −0,005 |
| MCLoss + rec. | 0,2608 | −0,003 | 0,4177 | −0,002 | −0,004 |

Nemenyi no FunCat (5 métodos, CD = 2,157): C-HMCNN 1,00; HMCN-F 2,625; MCLoss + rec. 2,625;
**BCE + rec. 3,75**; GBDT 5,00. Pares significativos: C-HMCNN × GBDT, **C-HMCNN × BCE + rec.
(2,75)**, HMCN-F × GBDT, MCLoss × GBDT.

GAE (MCLoss + GAE + rec., 3 seeds): média 0,2625 vs 0,2608 do MCLoss + rec. (+0,0016);
derisi 0,1965 ± 0,0007; spo 0,2168 ± 0,0009; abaixo do MCLoss + rec. em expr, gasch2 e seq.

F1 por nível, BCE, Δ da reconciliação: Diatoms L1 +0,021, L2/L3 0,000; EUR-Lex L1–L3 ≈ +0,010;
WOS L1 +0,005, L2 0,000; `cellcycle_GO` L5–L9 +0,004 a +0,007.

## Passos

1. **Conclusão** (`main.tex:606-608`): o “within 0.003” é do **MCLoss + rec.** Reescrever, por
   exemplo: “On FunCat and GO, the three reconciled objectives are within $0.002$ AUPRC of each
   other; the best (MCLoss) is within $0.003$ of C-HMCNN, and BCE within $0.005$.”
2. **Discussão** (`main.tex:574-576`): restringir a “within $0.003$ AUPRC of C-HMCNN” (retirar
   “and HMCN-F”, que no GO fica a 0,004) e dizer que é o MCLoss + rec.
3. **Nemenyi** (`main.tex:426-431`): nomear os 5 métodos e reportar que BCE + rec. (rank 3,75)
   **é** separado do C-HMCNN, enquanto MCLoss + rec. (2,62) não é. Manter a ressalva de baixa
   potência com 8 datasets.
4. **Coerência da mensagem**: conferir que contribuição (`main.tex:99-103`), orientação prática
   (`main.tex:578-582`) e abstract (`main.tex:36-45`) não afirmam paridade do BCE + rec. com o
   C-HMCNN no FunCat. “BCE with reconciliation matches MCLoss in every family” vale na Tabela 2
   (≤ 0,001) e pode ficar; a paridade com o C-HMCNN é só do MCLoss + rec.
5. **Por nível** (`main.tex:440-442`): remover a frase. Se sobrar espaço, substituir por algo que
   o dado sustenta: “reconciliation changes only internal nodes, so its F1 effect concentrates on
   the upper levels of tree hierarchies (e.g., $+0.021$ at level 1 on Diatoms)”.
6. **GAE** (`main.tex:432-435`): trocar “raises … (beating C-HMCNN on …)” por “leaves the FunCat
   average at $0.262$, within seed noise of MCLoss + rec.”; ou cortar o GAE do IJCNN (o README do
   paper já o destina à KBS) e remover a linha `main.tex:415` da Tabela 3.
7. `make check`.

## Saídas esperadas

- Conclusão, discussão e Seção 5.2 consistentes com a Tabela 3.
- Nenhuma afirmação por nível sem dado no paper.

## Critério de aceite

- [ ] Toda ocorrência de “0.003” em `main.tex` tem como sujeito o MCLoss + rec. vs C-HMCNN, ou a
      paridade de pipeline do C-HMCNN (`:555`, `:596`).
- [ ] O parágrafo do Nemenyi cita os 5 métodos e o par C-HMCNN × BCE + rec.
- [ ] `main.tex:440-442` removida ou substituída por afirmação com número de `output/protocol/`.
- [ ] Nenhum “beating C-HMCNN” com margem ≤ 0,002.
- [ ] 6 páginas, 0 referências indefinidas.

## Tempo/custo

~1 h; sem compute.

## Falhas e rollback

- **Estouro de página:** o item 3 acrescenta ~2 linhas; compensar com o corte do item 5 ou do
  GAE no item 6.
- **Mensagem enfraquecida:** é o resultado medido. O enquadramento honesto é “a reconciliação
  torna os objetivos equivalentes entre si; a distância ao C-HMCNN é de 0,003 a 0,005”.

## Atualizações no paper/tese

- `docs/hmc-paper-ijcnn/main.tex:415`, `:426-436`, `:440-442`, `:574-576`, `:606-608`.
- KBS e tese atribuem o “within 0.003” ao modelo global com MCLoss (`hmc-paper/main.tex:191`,
  `:2176`; `tese/chapters/01-introducao.tex:129`, `08-conclusao.tex:47`) — conferir que o Nemenyi
  do BCE + rec. também está reportado lá.

## Checklist

- [x] Conclusão e discussão corrigidas (11/11/2026); o abstract passou a dizer qual objetivo sustenta cada número.
- [x] Nemenyi completo (5 métodos; BCE + rec. separado do C-HMCNN).
- [x] Frase por nível removida.
- [x] GAE suavizado (+0,002 na média; abaixo do MCLoss + rec. em 3 de 8).
- [x] KBS/tese conferidas: a KBS já reporta o par C-HMCNN × BCE (parágrafo “Statistical significance”). Em 13/11/2026 a tese passou a citar o par C-HMCNN × BCE + rec., e a frase do GAE foi suavizada na KBS e na tese.
- [x] Spec marcado como concluído no `README.md`.
