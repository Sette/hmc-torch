# 16 — Afirmações de escala no GO: OOM medido e teste batch-64 legado (P0)

## Objetivo

Substituir as afirmações de memória e o teste de escala da R esparsa por números **medidos**, e
retirar do paper IJCNN o teste batch-64, que vem de um script legado fora do protocolo. Fecha o
achado 5 do [relatório 13](13-review-ijcnn.md). Subiu de P1 para P0 em 11/11/2026, depois que a
origem do teste foi localizada.

## O que foi apurado (11/11/2026)

### Origem do teste batch-64

Os números de `main.tex:463-468` (ΔAUPRC +0,0055, +0,0193 em `eisen_GO`) e da `tab:sparse` da
KBS vêm de `experiments/run_go_experiments.py`, que o `CLAUDE.md` lista como **legado** (“não use
os números deles”). O script:

- junta validação ao treino (`run_go_experiments.py:53-54`) e treina 20 épocas fixas;
- escolhe o limiar de F1 **no test** (`compute_metrics`, `:23-43`, chamado com `y_te` em `:135-136`);
- usa **seed 42** (`:183`), enquanto o paper diz “the inference-only GO ablation is seed 0”
  (`main.tex:291-292`);
- lista hoje 4 datasets (`:155`), contra os 9 reportados;
- nunca executa o R denso: treina com identidade e o OOM é um comentário (`:87`,
  “dense R would OOM for GO”);
- grava em `output/go_experiments/results.json`, que **não existe**.

Isso contradiz o abstract (“with every choice made on validation data”, `main.tex:35`). O ΔAUPRC
não depende de limiar, mas o treino em train+valid sem early stopping está fora do protocolo, e
o ΔF1 (−0,0051 na KBS) usa limiar do test.

### Memória medida

`experiments/measure_dense_r_memory.py` (novo) → `output/scaling/dense_r_memory.json`.
`seq_GO`, $C = 4.134$, $E = 5.839$, RTX 3060 (11,6 GiB utilizáveis), um passo de reconciliação,
pico de memória alocada:

| Operador | Modo | B=4 | B=32 | B=64 | B=128 | B=256 |
|---|---|---|---|---|---|---|
| Denso float64 (`get_constr_out`) | inferência | 0,51 GiB | 4,08 GiB | **8,16 GiB** | OOM | OOM |
| Denso float64 | com backward | 1,02 GiB | 8,15 GiB | **OOM** | OOM | OOM |
| Denso float32 | inferência | 0,25 GiB | 2,04 GiB | 4,08 GiB | 8,16 GiB | OOM |
| Denso float32 | com backward | 0,51 GiB | 4,08 GiB | 8,15 GiB | OOM | OOM |
| Esparso (`BlockDiagonalR`) | inferência | 0,3 MB | 2 MB | **4 MB** | 9 MB | 17 MB |
| Esparso | com backward | 3 MB | 21 MB | **43 MB** | 85 MB | 171 MB |

- Matriz R densa em float32: 65,2 MB.
- Saída densa × esparsa: diferença absoluta máxima **0,0** (sustenta “reproduces the dense one
  exactly”, `main.tex:41`).
- Consequência: a frase “causes an out-of-memory failure on a 12 GB GPU” em batch 64
  (`main.tex:463-465`) é **falsa para inferência** (8,16 GiB cabem) e para float32; o OOM em
  batch 64 só ocorre no **treino com o operador float64 de referência** (MCLoss).

## Passos

1. **Resultados** (`main.tex:462-468`): remover o teste batch-64 e pôr no lugar a medição. Forma
   sugerida: “On \texttt{seq\_GO} ($C = 4{,}134$) and a 12~GB GPU, the reference dense operator
   (float64) peaks at 8.2~GiB for one reconciliation at $B = 64$ and runs out of memory when
   back-propagating at $B = 64$ or reconciling at $B = 128$; the sparse operator needs 4~MB and
   43~MB at $B = 64$ and 171~MB at $B = 256$, with identical outputs.”
2. **Protocolo** (`main.tex:291-292`): retirar “the inference-only GO ablation is seed 0”.
3. **Método** (`main.tex:208-212`): trocar as estimativas pelos valores medidos; dizer que o
   float64 é o da implementação de referência do C-HMCNN
   (`src/hmc/models/global_classifier/constraint/utils.py:14`) e retirar “exhausting commodity GPU
   memory” do caso float32.
4. **Introdução** (`main.tex:77-79`): “materializing the dense $C\times C$ ancestor-closure matrix
   is prohibitive in GPU memory” → o gargalo é o intermediário $B\times C\times C$, que limita o
   batch (a matriz ocupa 65 MB).
5. **Orientação prática** (`main.tex:582-584`): “above $\sim 1{,}000$ classes” não tem medição;
   retirar o limiar ou medi-lo com o mesmo script (`--dataset` de outro tamanho).
6. **KBS** (`docs/hmc-paper/main.tex`): `tab:sparse` (`:1285-1312`) e a prosa dependente
   (`:1314-1340`) usam o script legado — remover a tabela ou trocá-la pelos ganhos por dataset do
   protocolo (já em `output/protocol/*_GO.json`: 8 de 9, até +0,0046) e pela tabela de memória
   acima; unificar `:486-488` (“4.3 GB … triggers OOM at larger batch sizes”) com a medição.
7. **Tese** (`chapters/03-plataforma.tex`, `05-funcat-go.tex`): mesmas trocas.
8. `make check` nos três projetos.

## Saídas esperadas

- Nenhum número do paper vem de `run_go_experiments.py`.
- Afirmações de memória citam valores de `output/scaling/dense_r_memory.json`.

## Critério de aceite

- [ ] `grep -n -E '0\.0193|0\.0055' docs/hmc-paper-ijcnn/main.tex` vazio.
- [ ] Toda afirmação de OOM diz operador (float64/float32), modo (inferência/treino) e batch, e
      bate com o JSON.
- [ ] “prohibitive” não se refere mais à matriz de 65 MB.
- [ ] IJCNN, KBS e tese dizem a mesma coisa.
- [ ] IJCNN em 6 páginas.

## Tempo/custo

~2 h de edição nos três manuscritos; a medição já foi feita (< 1 min de GPU, repetível).

## Falhas e rollback

- **Mensagem mais fraca:** em batch 4 (protocolo principal) o denso cabe (0,5–1,0 GiB). O
  enquadramento correto é “remove o teto de batch”: em batch 64, cerca de 190× (treino, contra o
  denso float32) a 2.000× (inferência, contra o float64) menos memória no passo de
  reconciliação, com saída idêntica.
- **KBS congelada:** é correção de proveniência; registrar na seção de revisão do `README.md` dos
  specs.

## Atualizações no paper/tese

- `docs/hmc-paper-ijcnn/main.tex:77-79`, `:208-212`, `:238-240`, `:291-292`, `:462-468`,
  `:582-584`.
- `docs/hmc-paper/main.tex:486-488`, `:1285-1340`.
- Tese cap. 3 e cap. 5.
- `CLAUDE.md`: incluir `measure_dense_r_memory.py` na tabela de scripts do paper.

## Checklist

- [x] Origem do teste batch-64 localizada (`run_go_experiments.py`, legado).
- [x] OOM medido (`output/scaling/dense_r_memory.json`).
- [x] IJCNN corrigido (11/11/2026): teste batch-64 removido; memória medida no método e na Seção 5.3.
- [x] KBS corrigida: `tab:sparse` refeita com os ganhos do protocolo (BCE, 3 seeds; 8 de 9, média +0,0016), nova `tab:sparse_mem`, figura `figures/fig3_go_reconciliation.pdf` (cópia da figura do IJCNN; a antiga `fig3_go_sparse.pdf` ficou no diretório, sem uso).
- [x] Tese corrigida (cap. 3 e 5; mesma figura nova).
- [x] `CLAUDE.md`: `measure_dense_r_memory.py` na tabela de scripts.
- [x] Spec marcado como concluído no `README.md`.
