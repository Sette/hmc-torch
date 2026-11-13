# 20 — Descrever o protocolo que o código executa (P0)

## Objetivo

Fazer as seções de método, protocolo e reprodutibilidade dos três manuscritos descreverem o que
`experiments/run_protocol.py` e `experiments/run_wos_e2e.py` realmente fazem. Fecha os achados
A1–A6, A10–A13, A15, D4 e D6 do [relatório 18](18-audit-geral.md). Só edição de texto: o código
e os números não mudam.

## Pré-requisitos

- `experiments/run_protocol.py`, `experiments/run_wos_e2e.py`, `experiments/run_sr_gae_sota.py`
  (versão commitada pelo spec 22).
- `output/protocol/*.json` (campos `hyperparams`, `n_train`, `best_epoch`).

## Fatos a usar (do código)

| Item | O que o código faz | Onde |
|---|---|---|
| BCE | `binary_cross_entropy` **sem pesos** por classe | `run_protocol.py` (laço de treino) |
| Hinge | `relu(p_desc − p_anc).mean()` sobre **todos os pares do fecho**; `LAMBDA_HIER = 1.0` | `run_protocol.py:18`, `:77` |
| Early stopping | AUPRC de validação dos scores **reconciliados**, nos três objetivos | `hp["select"] = "rec"` |
| Validação em datasets grandes | quando `n_val × n_eval > 10M` (EUR-Lex, RCV1): primeiras 1.500 linhas, a cada 2–4 épocas | `val_step` |
| Teto de épocas | EUR-Lex: `best_epoch` = 96–100 de 100 nos 9 runs | `output/protocol/eurlex.json` |
| Limiar | escolhido nos scores de validação do modelo só-treino; aplicado ao modelo refeito | refit |
| Refit | só ARFF (FunCat, GO, cross-domain); texto treina só no treino | `refit: False` |
| Sementes | `torch.manual_seed`, `np.random.seed`; sem `cudnn.deterministic` | — |
| Saída | um JSON por dataset com métricas por run; **sem** scores, **sem** `run-config.json` | — |
| MLP | 3 camadas lineares; oculto 512 (texto), 1000–2000 (cross-domain); dropout 0,7 (FunCat/GO) ou 0,3 | `hyperparams` |
| E2E WOS | `max_length=512`, dropout 0,3, batch 8, BCE sem pesos, época por micro-F1 reconciliado na validação | `run_wos_e2e.py:104` |
| GBDT | `HistGradientBoostingClassifier(max_iter=100)`, seed 0, nós com ≥ 5 positivos; raw/rec do modelo refeito | `run_protocol.py:455-474` |
| H-MLCL+H-KNN | projetor em ≤ 16.000 linhas de treino; índice k-NN ≤ 10.000; blend escolhido em ≤ 2.000 linhas de validação | `run_sr_gae_sota.py` |

Tamanho de treino efetivo no texto: WOS 30.070; AAPD 35.737; ArXiv 32.000; EUR-Lex 45.000.

## Passos

1. **λ** — KBS `main.tex:551`, tese `chapters/03-plataforma.tex:184`: trocar 0,1 por 1,0.
2. **BCE** — IJCNN `main.tex:176`; KBS `:528-530` (texto e Eq. com `w_i`), `:781-782`; tese
   `03-plataforma.tex:171-179`: remover “prevalence-weighted”/`w_i` e a frase que o define.
   A Eq. do BCE fica sem peso.
3. **Pares do hinge** — IJCNN `:187`; KBS `:406`; tese `03-plataforma.tex:95-97`, `:159-161`:
   dizer que os experimentos usam os pares do fecho transitivo; a variante sobre arestas pode
   ficar como observação (“zero violations on edges imply zero on the closure”), sem afirmar
   que foi a usada.
4. **Seleção de época** — na seção de protocolo dos três: “the epoch is selected by validation
   AUPRC of the reconciled scores for every objective; raw scores are evaluated at that same
   epoch”. Na tese, também em `05-funcat-go.tex:26-29`. Se o spec 23 for executado, citar o
   controle.
5. **EUR-Lex/RCV1** — KBS (parágrafo do EUR-Lex, perto de `:985-995`) e tese
   `04-r-matrix-texto.tex:28-38`: declarar o subconjunto de 1.500 documentos, a validação a
   cada 4 épocas e que o EUR-Lex atinge o teto de 100 épocas nos três seeds (mesmo formato da
   nota † do `spo`).
6. **Limiar e refit** — uma frase na descrição do protocolo ARFF (KBS perto de `:2150-2158`;
   tese `05-funcat-go.tex:26-29`, `07-discussao.tex:96-101`).
7. **Tabela de datasets** — KBS `:703-710` e tese `appendices/a-datasets.tex:11`: separar
   “train” e “validation” para texto ou acrescentar à legenda que o texto não é refeito. Tese
   `04-r-matrix-texto.tex:16`, `:21`: trocar “40,000 train+validation” por “32,000 / 8,000 /
   10,000” e “44,672 train+validation” por “35,737 / 8,935 / 11,168”.
8. **Reprodutibilidade** — KBS `:676-682`; tese `03-plataforma.tex:329-333` e
   `appendices/d-reprodutibilidade.tex:15-20`: descrever o que existe (um JSON por dataset com
   métricas por seed, hiperparâmetros, épocas e limiares) e retirar `scores_*.npz`,
   `run-config.json` e “deterministic cuDNN”. Alternativa: implementar a gravação (fora deste
   spec; exigiria rerodar).
9. **Componentes não avaliados** — KBS `:316-369`, `:778-782`; tese `03-plataforma.tex:19-66`:
   retirar `LocalLevelHead` e “Tabular MLP” de “methods compared” (ou dizer explicitamente que
   existem na biblioteca e não entram nos resultados), retirar o filtro de variância, e dizer
   que os números vêm dos scripts de `experiments/`.
10. **E2E e apêndice B** — IJCNN `:281`, KBS `:801`, tese `04-r-matrix-texto.tex:31`: “256-token
    limit” vale para as features congeladas; o E2E usa 512. Tese
    `appendices/b-hiperparametros.tex`: dropout do E2E 0,3; GBDT “refit on train+valid”; coluna
    “Layers” com a largura oculta.
11. **H-MLCL+H-KNN** — KBS (seção do método, perto de `:1712`) e tese
    `03-plataforma.tex:255-313`: uma frase com os três limites de subamostragem.
12. `make check` nos três.

## Saídas esperadas

- Nenhuma afirmação de método sem correspondência no código que gerou os números.

## Critério de aceite

- [ ] `grep -n "lambda = 0.1\|prevalence-weighted\|inverse prevalence\|run-config\|cuDNN\|npz" `
      nos três manuscritos não retorna nada.
- [ ] A regra de seleção de época aparece na seção de protocolo dos três.
- [ ] A nota do teto de épocas do EUR-Lex existe na KBS e na tese.
- [ ] Tamanhos de treino de texto batem com `n_train` dos JSONs.
- [ ] IJCNN: 6 páginas, 0 over/underfull; KBS e tese sem referências indefinidas.

## Tempo/custo

~3 h; sem compute.

## Falhas e rollback

- **IJCNN estoura a página:** os passos 2 e 3 encurtam o texto; o passo 4 acrescenta uma
  frase — compensar removendo a menção a P = E.
- **Revisor pede o BCE ponderado:** responder que o protocolo usa BCE sem pesos em todos os
  braços; não reintroduzir a frase.
- Rollback: `git checkout` dos `.tex` (KBS no repositório próprio; tese com backup em tarball).

## Atualizações no paper/tese

- IJCNN: `main.tex:176`, `:187`, `:281` e parágrafo de protocolo.
- KBS: `:316-369`, `:406`, `:528-530`, `:551`, `:676-682`, `:703-710`, `:778-782`, `:801`.
- Tese: `chapters/03-plataforma.tex`, `04-r-matrix-texto.tex`, `05-funcat-go.tex:26-29`,
  `07-discussao.tex:96-101`, apêndices A, B e D.

## Checklist

- [ ] λ e BCE corrigidos nos três.
- [ ] Pares do hinge.
- [ ] Seleção de época, EUR-Lex, limiar/refit declarados.
- [ ] Tabela de datasets e tamanhos de texto.
- [ ] Parágrafo de reprodutibilidade.
- [ ] Componentes não avaliados.
- [ ] E2E e apêndice B.
- [ ] Subamostragens do H-MLCL.
- [ ] `make check` ×3.
- [ ] Spec marcado como concluído no `README.md`.
