# 21 — Números e figuras que não vêm do protocolo: Platt, F1 por nível, tempos (P0)

## Objetivo

Substituir, na KBS e na tese, três blocos de números que não saem de `output/protocol/` e as
figuras geradas antes dos JSONs atuais. Fecha os achados A7, A8, A9, A19 e C4 do
[relatório 18](18-audit-geral.md). O IJCNN não é afetado (a frase por nível saiu no spec 15).

## Pré-requisitos

- `output/protocol/*.json` (FunCat/GO de 04/11, texto de 03–05/11).
- `output/scaling/dense_r_memory.json`.
- `experiments/make_paper_artifacts.py` (gera `fig_per_level.pdf`, `fig_cd_diagram.pdf`).

## Fatos a usar

**Platt (8 datasets FunCat comuns, GBDT, micro-AUPRC):**

| Modelo | Sem Platt | Com Platt |
|---|---|---|
| Só treino (`trainonly_auprc` → `platt_auprc`) | 0,2023 | 0,2125 |
| Refeito em treino+validação (`raw.auprc`) | 0,2191 | não medido |
| Refeito + reconciliado (`rec.auprc`) | 0,2256 | não medido |

O texto compara 0,219 (refeito) com 0,212 (só-treino + Platt). Na comparação correta o Platt
**sobe** o AUPRC em 0,010.

**F1 por nível (BCE + rec., média de 3 seeds, `eval.rec.per_level_f1`):**

| Dataset | Texto atual | Protocolo |
|---|---|---|
| ArXiv | 0,93 → 0,59 (raiz → folhas) | 0,814 → 0,575 |
| WOS | 0,86 → 0,65 (raiz → folhas) | 0,837 → 0,645 |
| `cellcycle_FUN` | — | 0,42 / 0,267 / 0,20 / 0,20 / 0,088 / 0,237 |
| `cellcycle_GO` | “nível mais alto, um único nó, perto de zero; o seguinte 0,68” | começa no nível 2 (0,674); níveis 11–12 = 0,0 |

A figura atual (conferida renderizada) tem um nível L0 com n = 1 em ArXiv e WOS — o nó raiz,
que o protocolo não avalia — e um L1 com n = 1 no GO. O texto compara a raiz com as folhas.

**Arquivos consolidados (E1):** `consolidated_results.json` ainda traz `depthcal.full_pipeline`
com 27/27 e fonte `output/sr_gae_sota` (o pareado dá 25/27), `gpu_speedup` legado e desvios de
`wos_frozen`/`arxiv` diferentes do texto; `docs/tese/data/protocol_summary.json` tem as 324
médias corretas e desvios com ddof = 0 (o texto usa ddof = 1).

**Tempos (protocolo, `train_time_s`, RTX 3060):** FunCat MCLoss 254 s em média (99–518 s);
BCE 103 s; GBDT ~303 s. A tabela atual (15,7 s CPU / 5,5 s GPU; tabular MLP 5,0 s / 2,8 s) é
de uma configuração legada e não tem origem em `output/`.

## Passos

1. **Platt** — KBS `main.tex:2049-2056`; tese `chapters/06-cross-domain.tex:72-77`. Escolher:
   - (a) reescrever: “fitted on the validation scores of the train-only model, Platt scaling
     raises its raw Micro-AUPRC from 0.202 to 0.212; it was not applied to the refit model
     (0.219 raw, 0.226 reconciled), so we do not compare the two”, e retirar “We recommend raw
     scores over calibration” e a explicação causal;
   - (b) cortar o parágrafo e a menção em `hmc-paper/main.tex:688`.
   A variante com medida no modelo refeito está no spec 23.
2. **Conferir dependências do Platt**: `grep -n -i "platt" main.tex` na KBS e
   `grep -rn -i "platt" chapters/` na tese (por exemplo `07-discussao.tex:81-82`, “Per-node
   logistic calibrators … overfit rare leaves”, que perde o suporte).
3. **F1 por nível** — regenerar a figura a partir dos JSONs:
   `PYTHONPATH=src uv run python experiments/make_paper_artifacts.py --paper_dir docs/hmc-paper`
   e copiar `fig_per_level.pdf` sobre `figures/fig5_per_level_f1.pdf` (ou ajustar o
   `\includegraphics`). Atualizar os números em KBS `:1839-1847` e tese
   `04-r-matrix-texto.tex:165-170`; retirar a frase do “single node” do GO e “the drop spans an
   order of magnitude” se a figura nova não mostrar isso.
4. **Diagrama CD** — o mesmo comando gera `fig_cd_diagram.pdf`; substituir
   `figures/fig6_cd_diagram.pdf` e conferir os ranks (C-HMCNN 1,00; HMCN-F 2,625; MCLoss + rec.
   2,625; BCE + rec. 3,75; GBDT 5,00; CD = 2,157).
5. **Tabela de tempos** — KBS `:1257-1280` (`tab:gpu`) e frase em `:1250-1255`; tese
   `06-cross-domain.tex:78-80`. Escolher:
   - (a) remover a tabela e o speedup CPU/GPU;
   - (b) substituir por tempos do protocolo: por família e objetivo, `train_time_s` médio, com
     a ressalva de que inclui o early stopping e não o refit (conferir no código o que o campo
     mede antes de escrever).
6. **Tempo esparso × denso (RQ2)** — se `output/scaling/dense_r_memory.json` tiver tempo por
   passo, reportar ao lado da memória; se não tiver, dizer na tese que a comparação é de
   memória e não de tempo (`08-conclusao.tex:29-36`).
7. **Figura cross-domain** — `figures/make_fig4_cross_domain.py`: ler os valores dos JSONs em
   vez de hardcode; corrigir a legenda de KBS `:1222` (“Same architecture … throughout” não
   vale para a barra fine-tuned do WOS) e a da tese `06-cross-domain.tex:44-45`.
8. **Limpeza** — remover de `docs/hmc-paper/figures/` os arquivos sem uso
   (`fig2_ablation.pdf`, `fig3_go_sparse.pdf`, `fig_scaling_laws.pdf`) depois de confirmar com
   `grep -n "includegraphics" main.tex` e com os `\includegraphics` da tese (que lê a mesma
   pasta via `\graphicspath`).
9. Atualizar `docs/hmc-paper/data/consolidated_results.json`: números alterados aqui, bloco
   `depthcal` (trocar pelo resultado pareado), `gpu_speedup`, desvios de texto e a fonte do
   EUR-Lex. Em `make_paper_artifacts.py`, gravar desvio amostral (ddof = 1) no
   `protocol_summary.json`.
10. `make check` na KBS e na tese; `make submission` na KBS.

## Saídas esperadas

- Parágrafo do Platt correto ou removido.
- Figuras 5 e 6 regeneradas dos JSONs de novembro; números do texto iguais aos da figura.
- Tabela de tempos removida ou com origem em `output/protocol/`.

## Critério de aceite

- [ ] Nenhuma ocorrência de “0.93”, “0.86 $\rightarrow$” ou “15.7” nos parágrafos citados.
- [ ] Cada número de F1 por nível no texto confere com `per_level_f1` (3ª casa).
- [ ] `ls -l figures/fig5* figures/fig6*` com data posterior a 05/11/2026.
- [ ] Nenhuma frase diz que o Platt reduz o AUPRC.
- [ ] KBS e tese compilam sem referências indefinidas.

## Tempo/custo

~3 h; CPU apenas (geração de figuras em minutos). Sem GPU.

## Falhas e rollback

- **`make_paper_artifacts.py` falha por falta de JSON:** ele lê `output/protocol/` e
  `output/protocol/structure.json`; todos existem. Se faltar algum grupo, gerar só as figuras
  necessárias em vez de rodar o protocolo.
- **A figura nova muda a narrativa do GO:** descrever o que ela mostra; não ajustar a figura
  ao texto.
- Rollback: `git checkout` na KBS; backup da tese antes de editar.

## Atualizações no paper/tese

- KBS: `main.tex:688`, `:1222`, `:1250-1280`, `:1839-1847`, `:2049-2056`; `figures/`;
  `data/consolidated_results.json`.
- Tese: `chapters/04-r-matrix-texto.tex:157-170`, `06-cross-domain.tex:44-45`, `:64-80`,
  `07-discussao.tex:80-82`.

## Checklist

- [ ] Platt reescrito ou removido (KBS e tese) e dependências conferidas.
- [ ] Figura por nível regenerada e números atualizados.
- [ ] Diagrama CD regenerado.
- [ ] Tabela de tempos resolvida.
- [ ] Legenda e script da figura cross-domain.
- [x] Figuras órfãs removidas (KBS `b5fd3b8`, 16/11/2026; nenhuma figura incluída na KBS ou na tese ficou faltando).
- [ ] `consolidated_results.json` atualizado.
- [ ] `make check` e `make submission`.
- [ ] Spec marcado como concluído no `README.md`.

## Execução (16/11/2026)

- Platt: opção (a) — reescrito na KBS e na tese (0,202 → 0,212 no mesmo modelo; sem afirmação
  para o modelo refeito). As frases “overfits rare leaves” saíram.
- F1 por nível: `fig5_per_level_f1.pdf` regenerada de `output/protocol/` (KBS e tese), texto e
  legenda atualizados (ArXiv 0,81 → 0,58; WOS 0,84 → 0,65).
- Tempos: opção (b) — tabela da KBS trocada pelos `train_time_s` do protocolo por família e
  objetivo; a tese cita os mesmos números. RQ2 passou a dizer que a evidência é de memória.
- `consolidated_results.json` atualizado (KBS `9b84d4f`). Diagrama CD conferido, sem troca.
- Falta: `make_fig4_cross_domain.py` ainda tem valores fixos (só a legenda foi corrigida);
  `protocol_summary.json` ainda grava desvio com ddof = 0; `make submission` da KBS.
