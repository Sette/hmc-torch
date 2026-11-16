# 22 — Proveniência: script não commitado, `surj_repro/` removido, artefatos “públicos” (P0)

## Objetivo

Garantir que cada número dos manuscritos seja reproduzível a partir do que está versionado e
que as declarações de disponibilidade sejam verdadeiras. Fecha os achados A16, A17 e A18 do
[relatório 18](18-audit-geral.md).

## Pré-requisitos

- Acesso de escrita ao repositório hmc-torch.
- Decisão do usuário sobre onde publicar os artefatos (passo 3). Para o IJCNN, qualquer link
  precisa ser anônimo (spec 19).

## Fatos a usar

- `experiments/run_sr_gae_sota.py` está modificado e não commitado (114 inserções, 196
  remoções). A versão em disco grava `output/sr_gae_paired/` (11/11) e é a que sustenta a
  tabela pareada; a versão commitada grava `output/sr_gae_sota/` com outra lógica.
- `surj_repro/` foi removido no commit `42aadba` (02/11/2026). Os números citados — FunCat
  0,232 (10 seeds), one-hot 0,156, GO 0,352 (3 seeds), busca de 25 configurações — estão em
  `git show 42aadba^:surj_repro/results.md`.
- `experiments/run_surj.py --experiment literal` (o caminho documentado no `CLAUDE.md`) dá, em
  `output/surj/literal/`, Surj 0,202 e one-hot 0,194 na média de 8 datasets (10 seeds, Adam
  1e-3, BCE, sem refit): outra receita, outros números. `output/surj/main/` só tem
  `cellcycle_FUN`.
- `output/` é gitignored: 0 arquivos versionados. O `run_protocol.py` não grava scores de test.

## Passos

1. **Commitar o script pareado.** Revisar o diff de `experiments/run_sr_gae_sota.py`, rodar um
   smoke test num dataset pequeno (`enron_others`, seed 0) e comparar com
   `output/sr_gae_paired/enron_others*.json`; commitar
   (`feat(experiments): ✨ paired evaluation of H-MLCL+H-KNN with random-prototype control`).
   Acrescentar `run_sr_gae_sota.py` e `run_sr_gae_depth.py` à tabela de scripts do `CLAUDE.md`.
2. **Surj.** Escolher uma rota e aplicá-la aos dois manuscritos:
   - (a) restaurar `surj_repro/` (`git checkout 42aadba^ -- surj_repro`) e manter o texto;
   - (b) manter removido e criar uma tag no commit anterior (`surj-repro-2026-11`), citando-a
     no texto em vez de “in the repository”;
   - (c) rodar `run_surj.py --experiment main` nos 8 FunCat + GO e trocar os números do texto
     pelos de `output/surj/` (exige GPU; os números mudam).
   Em qualquer rota, corrigir o comentário de `experiments/make_paper_artifacts.py` que cita
   `surj_repro/` e explicar no texto a relação entre as duas reproduções (a receita
   selecionada na validação e a “literal”).
3. **Artefatos por seed.** Publicar o que o texto promete ou ajustar o texto:
   - (a) empacotar `output/protocol/`, `output/protocol_rcv1/`, `output/sr_gae_paired/`,
     `output/sr_gae_depth/`, `output/hpt/final/`, `output/scaling/`, `output/chmcnn_parity/` e
     os JSONs do Surj num arquivo com DOI (Zenodo) ou num release do GitHub, e citar o
     identificador; para o IJCNN, enviar como material suplementar anônimo;
   - (b) reescrever KBS `main.tex:2261-2266` e IJCNN `main.tex:108` para “per-seed metric files
     are available from the authors on request / as supplementary material” e retirar “test
     score archives”.
   Antes de publicar, conferir que os JSONs não contêm caminhos locais nem dados sob licença
   (RCV1 é restrito: só métricas).
4. **PyPI.** Conferir que `pip install hmc-torch` instala a versão descrita (KBS `:2238`,
   `:2264`; tese `appendices/d-reprodutibilidade.tex:9`); se não, retirar a menção.
5. **Tabela script → saída.** Atualizar a tese (`appendices/d-reprodutibilidade.tex`) e o
   `CLAUDE.md` com a lista final: tabela pareada → `run_sr_gae_sota.py`; DepthCal →
   `run_sr_gae_depth.py`; Surj → rota do passo 2; memória → `measure_dense_r_memory.py`.

## Saídas esperadas

- `git status` limpo para `experiments/`.
- Cada tabela dos manuscritos com um script versionado e um diretório de saída nomeados.
- Declaração de disponibilidade verdadeira na data da submissão.

## Critério de aceite

- [ ] `git status --short experiments/` vazio.
- [ ] `grep -rn "surj\\\\_repro" docs/hmc-paper/main.tex docs/tese` aponta para algo que existe
      (diretório restaurado ou tag).
- [ ] O link ou identificador citado em “Data availability” abre e contém os JSONs por seed.
- [ ] O `CLAUDE.md` lista os scripts da avaliação pareada.

## Tempo/custo

~2 h sem a rota 2(c); a rota 2(c) custa ~4 h de GPU (10 seeds × 8 FunCat + 3 seeds × GO).

## Falhas e rollback

- **O smoke test do passo 1 não reproduz o JSON de 11/11:** não commitar; investigar a
  diferença antes de qualquer submissão, porque a tabela pareada depende desse script.
- **Rota 2(c) muda a conclusão do Surj:** improvável (as duas reproduções ficam abaixo do
  C-HMCNN), mas se mudar, atualizar o spec 24 e a memória `surj-gae-negative-result`.
- Nada aqui é destrutivo; publicar em serviço externo exige confirmação do usuário.

## Atualizações no paper/tese

- IJCNN: `main.tex:108` (e o link, conforme o spec 19).
- KBS: `main.tex:1471`, `:2238`, `:2261-2266`.
- Tese: `chapters/05-funcat-go.tex:188`, `appendices/c-surj.tex:6`,
  `appendices/d-reprodutibilidade.tex`.
- `CLAUDE.md`, `experiments/make_paper_artifacts.py` (comentário).

## Checklist

- [ ] Script pareado revisado, testado e commitado.
- [ ] Rota do Surj escolhida e aplicada.
- [ ] Artefatos publicados ou texto ajustado.
- [ ] PyPI conferido.
- [ ] `CLAUDE.md` e apêndice D atualizados.
- [ ] Spec marcado como concluído no `README.md`.

## Execução (16/11/2026)

- Passo 1: `run_sr_gae_sota.py` testado em `enron_others` (0,7624 → 0,7636; aleatório 0,7642,
  igual ao JSON de 11/11) e commitado (`0cba69f`); `CLAUDE.md` lista os dois scripts.
- Passo 2: rota (b) — tag local `surj-repro-2026-11` em `42aadba^`; KBS e tese citam a tag.
  **Atualização (2026-11-16):** em vez de enviar a tag ao hmc-torch, o código foi extraído para o
  repositório próprio `github.com/Sette/surj-repro` (mesma tag, `surj-repro-2026-11`); KBS e tese
  citam esse repositório.
- Passo 3: rota (b) — KBS diz “available from the corresponding author on request”; IJCNN diz
  “to be released upon acceptance”. Publicar o arquivo (Zenodo/release) continua em aberto.
- Falta: decidir a publicação dos artefatos; alinhar a versão do `pyproject.toml` (0.0.11) com
  o PyPI (0.1.1); corrigir o comentário de `make_paper_artifacts.py`.
