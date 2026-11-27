# 27 — KBS estendida: o que falta para submeter em março (P1)

## Objetivo

Submeter `docs/hmc-paper/` (KBS, ~49 páginas) logo depois da notificação do IJCNN (15/03/2027),
para cumprir o requisito (b) do PPGCC (submissão a periódico). Reúne as pendências dos specs
21–25 que só afetam a KBS e acrescenta a seção do ESM-2.

## Pré-requisitos

- IJCNN submetido (spec 26).
- Spec 23: `summarize_controls.py` rodado.
- Resposta do orientador sobre o título.

## Fatos a usar

- Rota: IJCNN primeiro; a KBS só é submetida depois de 15/03, porque a KBS exige que o
  manuscrito não esteja em avaliação em outro venue e os trabalhos se sobrepõem.
- KBS é o veículo (b); o aceite não é necessário para a defesa. Estrato Qualis do IJCNN e da
  KBS ainda a confirmar com a secretaria.
- Pendências abertas: spec 21 (`make_fig4_cross_domain.py` com valores fixos;
  `protocol_summary.json` com ddof = 0; `make submission`), spec 22 (publicar artefatos;
  `pyproject.toml` 0.0.11 × PyPI 0.1.1), spec 24 (título, negrito das tabelas, Holm opcional),
  spec 25 (título; ESM-2).
- ESM-2 (`output/protein_lm/esm2_t33_650M_UR50D/`, BCE + reconciliação, 3 seeds, protocolo
  completo): `seq_FUN` AUPRC 0,528 ± 0,004 contra 0,290 com as features do benchmark (C-HMCNN
  publicado 0,292); `seq_GO` 0,601 ± 0,004 contra 0,446. 4,8–6,5% das linhas sem ORF
  recuperado; `recover_seq_ids.py` auditado (só features, sem rótulos); 3 ORFs repetidos entre
  test e treino.
- Regra [[wos-no-sota]]: nenhuma afirmação de "SOTA" sem comparação de mesma entrada.

## Passos

1. **Spec 21, restos.** Gerar `fig4_cross_domain` a partir de `output/protocol/` em vez de
   valores fixos; `protocol_summary.json` com ddof = 1; `make submission`.
2. **Spec 23.** Levar o resultado de `summarize_controls.py` ao texto (protocolo e
   limitações); atualizar resumo/introdução/conclusão se a conclusão mudar; EUR-Lex com 200
   épocas: declarar se ainda toca o teto.
3. **Spec 24, restos.** Título (decisão do orientador; sugestão em 24): atualizar
   `main.tex`, `cover_letter.md` e o pacote; negrito das tabelas só quando a diferença exceder
   um desvio-padrão; Holm, se for reportar.
4. **Seção ESM-2** (curta, 1 tabela, capítulo de FunCat/GO):
   - reportar as duas linhas acima ao lado das features do benchmark;
   - dizer que a comparação troca o **encoder**, não o método; o C-HMCNN não foi rodado com
     ESM-2, então o ganho não se atribui ao nosso método;
   - declarar linhas sem ORF e os 3 ORFs repetidos;
   - tirar de Limitações "rather than end-to-end protein language models";
   - frase de conclusão: as features pesam mais que qualquer restrição hierárquica.
   - **Condicional:** rodar C-HMCNN (clone local em `./C-HMCNN/`, nunca copiar código) com as
     features do ESM-2 em `seq_FUN`/`seq_GO` (~2 h de GPU). Só então a seção pode comparar
     métodos; sem isso, não.
5. **Spec 10 (prompt hierarchy).** Go/no-go em **15/01/2027**: entra na KBS só se tiver
   resultado validado até lá; senão, fica em trabalhos futuros.
6. **Cover letter e highlights.** Divulgar o IJCNN (citar o paper e o que a KBS acrescenta:
   27 datasets, GO esparso, E2E, RCV1, análise de limitações; mirar ≥ 30% de conteúdo novo);
   `highlights.md` com 5 itens de até 85 caracteres.
7. **Artefatos.** Citar o DOI do Zenodo (spec 28) em "Data availability".
8. **Conferir os requisitos da KBS** (limite de páginas, formato, declaração de IA, destaque
   de conflito) no guia do periódico na data da submissão; ajustar `make submission`.
9. **Submeter entre 16 e 31/03** e guardar o comprovante para o pedido de banca.

## Saídas esperadas

- Pacote `docs/hmc-paper/submission/` regenerado; comprovante de submissão arquivado.
- Seção ESM-2 sem afirmação de SOTA.

## Critério de aceite

- [ ] `make check` e `make submission` sem erro nem referência indefinida.
- [ ] `grep -n -i "sota\|state of the art" ` na seção do ESM-2 não encontra afirmação de
      superioridade de método.
- [ ] `highlights.md`: 5 itens, ≤ 85 caracteres.
- [ ] Cover letter cita o IJCNN e a extensão.
- [ ] Comprovante de submissão guardado.

## Tempo/custo

~3 dias de trabalho em fevereiro; +2 h de GPU se o controle C-HMCNN + ESM-2 for feito.

## Falhas e rollback

- **IJCNN rejeitado em 15/03:** submeter a KBS no mesmo dia; decidir entre IEEE Access (A2) e
  defesa em agosto/setembro.
- **Orientador mantém o título:** registrar no spec 24 e seguir.
- **Controle C-HMCNN + ESM-2 sem tempo:** manter a seção sem comparação de métodos.
- Rollback: `git checkout` na KBS; tag `kbs-<data>` antes de submeter.

## Atualizações no paper/tese

- KBS: protocolo, limitações, capítulo FunCat/GO (ESM-2), título, `cover_letter.md`,
  `highlights.md`.
- Tese: seção equivalente do ESM-2 (spec 25, passo 6).

## Checklist

- [ ] Spec 21 (restos).
- [ ] Spec 23 no texto.
- [ ] Spec 24 (título, negrito).
- [ ] Seção ESM-2.
- [ ] Go/no-go spec 10 (15/01).
- [ ] Cover letter e highlights.
- [ ] DOI do Zenodo.
- [ ] Requisitos da KBS conferidos.
- [ ] Submetida (16–31/03).
- [ ] Spec marcado como concluído no `README.md`.
