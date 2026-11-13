# 14 — Proveniência do HPT e tabela do WOS (P0)

## Objetivo

Corrigir a afirmação de que o HPT foi reexecutado com o “código oficial” — os números vêm da
reimplementação do hmc-torch — e completar a tabela do WOS com a célula reconciliada do HPT.
Fecha os achados 1 e 4 do [relatório 13](13-review-ijcnn.md). Vale para os três manuscritos.

## Pré-requisitos

- `output/hpt/final/wos/summary.json` e os três `bert_none_down_hpt_raw_seed*.json` (existem).
- **Decisão (11/11/2026): reportar como reimplementação.** Rodar o clone `HPT/` foi testado e
  não é viável neste ambiente:
  - os dados no formato do HPT já existem (`data/wos/WebOfScience_{train,dev,test}.json`,
    `slot.pt`, `value_dict.pt`), então a entrada não é o obstáculo;
  - `python train.py --data WebOfScience --batch 8 --update 2 --seed 0` (cópia de `HPT/`, sem
    alterações) falha no carregamento do modelo com transformers 5.3:
    `AttributeError: 'Prompt' object has no attribute 'all_tied_weights_keys'`, e antes disso
    reporta `cls.predictions.decoder.weight | MISSING` (o tying da cabeça MLM não é reconhecido);
  - o repositório pede `torch-geometric == 1.7.2`, `torch-scatter == 2.0.8`,
    `torch-sparse == 0.6.12` (o ambiente tem PyG 2.7, torch 2.10);
  - qualquer correção para rodar aqui deixaria de ser “código oficial”; um ambiente legado
    separado custaria a montagem mais ~2,4 h de GPU por seed (≈ 7 h para 3 seeds).
  Se um revisor exigir o código original, abrir spec próprio com ambiente legado.

## Fatos a usar

| Célula (`summary.json`) | micro-F1 |
|---|---|
| `raw/thr0` (regra do HPT) | 0,8691 ± 0,0023 (seeds: 0,8672 / 0,8685 / 0,8716) |
| `raw/val` | 0,8697 ± 0,0023 |
| `rec/thr0` | 0,8702 ± 0,0027 |
| `rec/val` (nosso protocolo) | 0,8712 ± 0,0029 |
| Nosso E2E, raw / rec. (limiar na validação) | 0,8702 ± 0,0041 / 0,8704 ± 0,0040 |

Desvios da reimplementação em relação ao HPT de referência (spec 11, linha 64): inicialização dos
rótulos sem `[CLS]`/`[SEP]`; MLM só no vocabulário original; batch 8 × acumulação 2; avaliação
micro/macro-F1 padrão em vez do loop customizado.

## Passos

1. **IJCNN** (`docs/hmc-paper-ijcnn/main.tex`):
   - `:530-531` — trocar “Re-executing HPT's official code and selection rule” por
     “Our reimplementation of HPT, run with its published configuration and selection rule”.
   - `:47` (abstract), `:104-105` (contribuição), `:595` (limitações) — “re-execution of HPT” →
     “reimplementation of HPT”.
   - `:491` — rótulo “HPT (our reimpl.)”; preencher a célula reconciliada com 0,8712 ± 0,0029 e
     indicar o limiar de cada célula (raw@0 na primeira).
   - `:484-490` — os números publicados não são reconciliados: renomear as colunas (por exemplo
     “F1” / “+ Rec.”) e mover os publicados para a primeira; ajustar “like-for-like” na `:486`
     para “same split” (os limiares diferem).
   - `:532-533` — manter “within seed noise”; não afirmar que o nosso modelo fica acima.
2. **KBS** (`docs/hmc-paper/main.tex`): `:897` (“re-executed HPT's official code”), `:196`,
   `:854`, `:864`, `:2179`; a linha “Δ vs. HPT re-executed +0.0013” (`:864`) compara células de
   limiares diferentes — acrescentar o Δ em `rec/val` (−0,0008) ou qualificar.
3. **Tese**: `chapters/04-r-matrix-texto.tex:61`, `:71`, `:102-103` (“the official HPT
   repository”); `chapters/01-introducao.tex:127-134`, `:171`; `chapters/08-conclusao.tex:52`.
4. Registrar os desvios em uma frase no IJCNN (se couber) e por extenso na KBS/tese.
5. `make check` nos três projetos.

## Saídas esperadas

- Nenhuma ocorrência de “official code” / “official HPT repository” associada aos nossos números.
- Tabela do WOS com a célula reconciliada do HPT preenchida, nos três manuscritos.

## Critério de aceite

- [ ] `grep -n -i "official" docs/hmc-paper-ijcnn/main.tex docs/hmc-paper/main.tex docs/tese/chapters/*.tex`
      não retorna menção ao HPT executado por nós.
- [ ] As quatro células do HPT citadas batem com `output/hpt/final/wos/summary.json`.
- [ ] IJCNN continua em **6 páginas** (`make check`), sem over/underfull novos.
- [ ] KBS e tese compilam sem referências indefinidas.

## Tempo/custo

~1–2 h de edição; sem compute.

## Falhas e rollback

- **Estouro de página no IJCNN:** a frase dos desvios vai só para KBS/tese; no IJCNN basta
  “reimplementation”.
- **KBS congelada:** a edição é correção factual; registrar a mudança na seção de revisão do
  `README.md` dos specs e gerar nova tag após o commit (quando solicitado).

## Atualizações no paper/tese

- `docs/hmc-paper-ijcnn/main.tex:47`, `:104-105`, `:484-492`, `:530-533`, `:595`.
- `docs/hmc-paper/main.tex:196`, `:854`, `:864`, `:897`, `:2179`.
- `docs/tese/chapters/04-r-matrix-texto.tex:61`, `:71`, `:102-103`; `01-introducao.tex:127-134`,
  `:171`; `08-conclusao.tex:52`.
- `docs/hmc-paper/data/consolidated_results.json` (`text.wos.hpt_repro.note`): acrescentar
  `rec/val`.

## Checklist

- [x] IJCNN corrigido e em 6 páginas (11/11/2026; 0 over/underfull).
- [x] KBS corrigida (linha `rec/val` e Δ −0,0008 na `tab:text`; desvios por extenso).
- [x] Tese corrigida (cap. 1, 4, 7, 8 e apêndice D).
- [x] `consolidated_results.json` atualizado.
- [x] Spec marcado como concluído no `README.md`.
