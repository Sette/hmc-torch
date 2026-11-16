# 23 — Controles experimentais para a comparação raw × rec (P1)

## Objetivo

Medir se três escolhas do protocolo, hoje não declaradas, alteram o resultado central (“a
reconciliação melhora o AUPRC em 26/27 datasets, mediana +0,0016”). Fecha a parte experimental
dos achados A4, A5, A7 e A14 do [relatório 18](18-audit-geral.md). É o único spec desta
auditoria que usa GPU. Se não for executado, o spec 20 declara as escolhas como limitação.

## Pré-requisitos

- Spec 22, passo 1 (scripts commitados).
- GPU livre; `data/` completo.
- Não sobrescrever `output/protocol/`: todo resultado novo vai para `output/protocol_controls/`.

## Fatos a usar

- `run_protocol.py` escolhe a época pelo AUPRC de validação dos scores reconciliados em todos
  os objetivos (`hp["select"] = "rec"`). O braço raw é avaliado numa época ótima para o rec.
- EUR-Lex: `best_epoch` ∈ {96, …, 100} com `max_epochs = 100` nos 9 runs; a validação do early
  stopping usa 1.500 documentos a cada 4 épocas.
- Platt: aplicado só ao GBDT treinado sem a validação.
- Tamanho do efeito a proteger: mediana +0,0016 AUPRC, IQR [+0,0008, +0,0044]; ganhos por
  família de +0,0016 a +0,0039.

## Passos

1. **Controle C1 — seleção simétrica.** Acrescentar ao `run_protocol.py` uma opção
   `--select {rec,raw,own}` (default `rec`, comportamento atual). Com `own`, cada braço escolhe
   a própria época: raw pelo AUPRC de validação cru, rec pelo reconciliado (dois checkpoints
   por run, um treino só). Rodar BCE, 3 seeds, 27 datasets, gravando em
   `output/protocol_controls/select_own/`.
   - Reportar: Δ(rec − raw) por dataset, contagem de vitórias, Wilcoxon, mediana e IQR, ao lado
     dos valores atuais.
   - Para reduzir custo: BCE apenas (é o braço do resultado central); FunCat/GO dominam o
     tempo (batch 4).
2. **Controle C2 — teto do EUR-Lex.** Rodar `eurlex` com `max_epochs 200` (e o mesmo
   `val_step`), 3 seeds, 3 objetivos, em `output/protocol_controls/eurlex_200/`. Reportar
   `best_epoch`, AUPRC e F1 raw/rec e a diferença para os números atuais (0,7812 → 0,7888 F1).
   Opcional: validação completa em vez de 1.500 linhas, se couber em memória.
3. **Controle C3 — Platt no modelo refeito.** No bloco do GBDT de `run_protocol.py`, ajustar o
   Platt com predições fora da amostra (por exemplo, validação cruzada interna no
   treino+validação) e aplicá-lo ao modelo refeito; gravar `platt_refit_auprc` e
   `platt_refit_rec_auprc`. CPU apenas (~40 min para os 10 FunCat). Só vale a pena se o
   parágrafo do Platt for mantido (spec 21, passo 1a).
4. **Objetivo fixo por família (A14)** — não exige compute: reportar no texto o BCE + rec. como
   braço principal em todas as famílias e o MCLoss + rec. como segundo, com as duas distâncias
   ao C-HMCNN (0,005 e 0,003 no FunCat; 0,004 e 0,002 no GO). Executado no spec 24.
5. **Agregação.** Script curto em `experiments/` (ou extensão de `make_paper_artifacts.py`) que
   lê `output/protocol_controls/` e imprime a tabela de comparação; `json.dump(…,
   default=float)`.
6. **Texto.** Uma frase por controle na seção de protocolo ou nas ameaças à validade da KBS e
   da tese; no IJCNN só se couber (uma oração: “selecting the epoch separately for raw and
   reconciled scores changes the gain by …”).

## Saídas esperadas

- `output/protocol_controls/select_own/<dataset>.json` (27), `eurlex_200/eurlex.json`, e os
  campos novos do GBDT.
- Tabela: Δ atual × Δ com seleção simétrica, por família.

## Critério de aceite

- [ ] Os 27 JSONs de C1 existem, com 3 seeds e os dois checkpoints por run.
- [ ] O resultado central é reportado nas duas regras de seleção; se a contagem de vitórias
      ou o sinal da mediana mudar, o resumo, a introdução e a conclusão dos três manuscritos são
      atualizados.
- [ ] O EUR-Lex com 200 épocas não para no teto, ou a limitação é declarada.
- [ ] Nenhum arquivo de `output/protocol/` foi modificado (`git`/`ls -l` com as datas de
      03–05/11).

## Tempo/custo

RTX 3060: C1 ≈ 5–6 h (FunCat ~1 h, GO ~3 h, cross-domain ~30 min, texto ~1 h com o cache de
embeddings); C2 ≈ 1 h; C3 ≈ 40 min de CPU. Total ~7–8 h, uma noite.

## Falhas e rollback

- **C1 reduz o ganho da reconciliação:** é o resultado a reportar. A garantia de zero
  violações não depende disso; a frase “improves on 26/27” passa a citar os dois números.
- **Sem tempo de GPU antes de 31/01:** rodar C1 só em texto + cross-domain (~1h30) e declarar
  o restante como limitação.
- **OOM no EUR-Lex com validação completa:** manter as 1.500 linhas.
- Rollback: apagar `output/protocol_controls/`; a opção `--select` tem default igual ao
  comportamento atual.

## Atualizações no paper/tese

- KBS: seção de protocolo e “Limitations”.
- Tese: `chapters/05-funcat-go.tex:26-29`, `06-cross-domain.tex:137-144`,
  `07-discussao.tex:96-101`.
- IJCNN: uma oração no protocolo, se couber.
- `CLAUDE.md`: linha para `output/protocol_controls/`.

## Checklist

- [ ] Opção `--select` implementada e testada num dataset pequeno.
- [ ] C1 rodado (27 datasets ou subconjunto declarado).
- [ ] C2 rodado.
- [ ] C3 rodado ou dispensado (Platt removido no spec 21).
- [ ] Tabela de comparação gerada.
- [ ] Texto atualizado.
- [ ] Spec marcado como concluído no `README.md`.

## Execução (16/11/2026)

- `run_protocol.py --select {rec,raw}` e `experiments/summarize_controls.py` commitados
  (`2f503d8`). O controle roda o BCE com a época escolhida pelo score cru; a comparação
  simétrica usa o braço raw desse run e o braço rec do protocolo.
- Bateria disparada em 16/11: `output/protocol_controls/run_controls.sh` (C1 nos 27 datasets,
  C2 = EUR-Lex com 200 épocas); log em `output/protocol_controls/run.log`.
- C3 (Platt no modelo refeito) dispensado: o parágrafo do Platt não faz afirmação sobre o
  modelo refeito.
- Falta: rodar `summarize_controls.py` ao fim da bateria e levar o resultado ao texto.
