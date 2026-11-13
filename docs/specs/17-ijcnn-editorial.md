# 17 — Edição de texto do paper IJCNN (P2)

## Objetivo

Corrigir seis pontos de polimento em `docs/hmc-paper-ijcnn/main.tex`. Fecha os achados 9–14 do
[relatório 13](13-review-ijcnn.md). Executar depois dos specs 14–16, que mexem nos mesmos
parágrafos.

## Pré-requisitos

- Specs 14, 15 e 16 aplicados (os números de linha abaixo são os de 11/11/2026 e vão se deslocar).

## Passos

1. **Exemplo que mostra queda** (`main.tex:379-383`): em `cellcycle_GO` a reconciliação leva o
   BCE de 0,4333 para 0,4332. Trocar o verbo (“leaves BCE at $0.4332$”) ou usar um dataset em que
   o BCE melhora (por exemplo `seq_GO`: 0,4576 → 0,4607 em F1), mantendo a figura coerente.
2. **Referência errada** (`main.tex:545-546`): “($0.537$ on Diatoms and $0.649$ on ImCLEF07a;
   Table~\ref{tab:decomp})” — a Tabela 2 só tem médias por família. Retirar a referência ou citar
   a média da família (0,684), que está na tabela.
3. **Notação** (`main.tex:216-217`): os vetores de arestas $\mathbf{c}, \mathbf{p}$ colidem com
   as probabilidades $\mathbf{p}$ de `main.tex:167`. Renomear para, por exemplo,
   $\mathbf{s}, \mathbf{t}$ (origem/destino) e propagar para a Eq. `eq:sparse_step`
   (`main.tex:223`).
4. **MCLoss** (`main.tex:187-190`): reescrever a oração “while masking descendants that are
   positive only when the target itself is negative so gradients flow to the true active
   classes”. Forma sugerida: “for positive targets the maximum is taken only over descendants
   that are themselves positive, so the gradient reaches the truly active classes”.
5. **Surj** (`main.tex:562-564`): “they inflate both models by $1.09\times$--$1.45\times$” não é
   o que a KBS reporta (`hmc-paper/main.tex:1508-1512`). Forma sugerida: “the variants that bring
   our average closest to the reported values also inflate C-HMCNN to $1.40\times$--$1.45\times$
   its published numbers”.
6. **ArXiv** (`main.tex:264`): declarar o tamanho — subamostra de 50.000 documentos
   (32.000 / 8.000 / 10.000), conforme `output/protocol/arxiv.json`.
7. **Negrito em empate** (`main.tex:331`): na linha Text/AUPRC, pôr em negrito as duas células
   0,845 ou ajustar a legenda (“best before rounding”). Conferir as demais linhas da Tabela 2.
8. `make check`.

## Saídas esperadas

- `main.tex` com os sete trechos corrigidos, 6 páginas.

## Critério de aceite

- [ ] Nenhum exemplo ilustra “improves” com um número que cai.
- [ ] Toda referência a tabela aponta para um valor que a tabela contém.
- [ ] $\mathbf{p}$ tem um único significado no paper.
- [ ] A frase do Surj bate com a seção correspondente da KBS.
- [ ] 6 páginas, 0 referências indefinidas, 0 over/underfull.

## Atualizações no paper/tese

- `docs/hmc-paper-ijcnn/main.tex:187-190`, `:216-223`, `:264`, `:331`, `:379-383`, `:545-546`,
  `:562-564`.
- KBS/tese: conferir a mesma colisão de notação no `BlockDiagonalR` e o tamanho do ArXiv.

## Checklist

- [x] Itens 1–7 aplicados (13/11/2026). Vetores de arestas renomeados para $\mathbf{s}, \mathbf{t}$ (passo $k$) também na KBS e na tese; legenda da Tabela 2 diz “before rounding” (conferido contra `output/protocol/`).
- [x] Build em 6 páginas, 0 over/underfull.
- [x] Spec marcado como concluído no `README.md`.
