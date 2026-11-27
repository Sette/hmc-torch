# 28 — Artefatos no Zenodo (P1)

## Objetivo

Publicar os resultados por seed e os scripts com um DOI citável, para que "Data availability"
do IJCNN (camera-ready) e da KBS seja verdadeira. Fecha o passo 3 do [spec 22](22-proveniencia-artefatos.md).

## Pré-requisitos

- Spec 22, passo 1 (scripts commitados) e passo 2 (repositório `Sette/surj-repro`, tag
  `surj-repro-2026-11`).
- Bateria do spec 23 concluída.
- Conta no Zenodo (ligada ao GitHub, se for usar o release automático).

## Fatos a usar

- `output/` não é público e é gitignored.
- IJCNN submetido sem link; o DOI entra no camera-ready (12/04) e na KBS.
- A versão do `pyproject.toml` (0.0.11) não bate com o PyPI (0.1.1): corrigir via
  release-please, nunca à mão (ver `CLAUDE.md`).
- Licença do hmc-torch: MIT. Nada copiado do C-HMCNN (GPL-3.0) entra no depósito.

## Passos

1. **Escolher o conteúdo** (somente JSONs de resultado e logs; sem dados brutos, checkpoints,
   caches de embeddings nem `data/`):
   - `output/protocol/`, `output/protocol_controls/`, `output/protocol_ablation/`,
     `output/protocol_rcv1/`, `output/sr_gae_paired/`, `output/sr_gae_depth/`,
     `output/scaling/`, `output/chmcnn_parity/`, `output/protein_lm/`, `output/hpt/`;
   - `docs/hmc-paper/data/consolidated_results.json`;
   - um `MANIFEST.md` com o script que gerou cada diretório, a data e o commit.
2. **Tamanho e privacidade.** `du -sh` do pacote; `grep -r "/home/brunos"` nos JSONs e
   remover caminhos absolutos; nada de e-mail ou token.
3. **Código.** Um release do hmc-torch (tag) e o `surj-repro-2026-11`; o Zenodo arquiva o
   release do GitHub. Garantir que os dois estejam em tags **empurradas** (a tag do Surj estava
   só local no spec 22).
4. **Rascunho no Zenodo** até 15/01: reservar o DOI, preencher título genérico, licença
   (CC-BY-4.0 para os dados; MIT para o código), autores e afiliações. **Não publicar** antes
   da notificação, porque o depósito é identificador.
5. **Se o CMT aceitar suplemento**, enviar um zip anônimo (JSONs + MANIFEST sem nomes).
6. **Publicar** depois de 15/03 (aceite) ou junto da submissão da KBS; colocar o DOI em
   `CAMERA_READY.md`, na KBS ("Data availability") e na tese.

## Saídas esperadas

- Depósito com DOI e `MANIFEST.md`.
- DOI citado nos três manuscritos.

## Critério de aceite

- [ ] `MANIFEST.md` relaciona cada diretório a um script versionado.
- [ ] Nenhum caminho absoluto, e-mail ou chave nos arquivos.
- [ ] Tags do hmc-torch e do surj-repro existem no remoto.
- [ ] O DOI abre e lista os JSONs por seed.
- [ ] "Data availability" nos manuscritos cita o DOI.

## Tempo/custo

~1 dia; sem compute.

## Falhas e rollback

- **Pacote grande demais (> 50 GB é o limite do Zenodo; pretendido < 1 GB):** cortar logs.
- **Depósito publicado antes da hora:** o Zenodo não apaga; criar nova versão, e como o
  conteúdo não identifica o autor, o risco à anonimização é só o metadado — por isso o
  rascunho.
- Rollback: apagar o rascunho.

## Atualizações no paper/tese

- IJCNN: item 3 de `CAMERA_READY.md`. KBS e tese: "Data availability" e apêndice D.
- `CLAUDE.md`: linha com o DOI quando publicado.

## Checklist

- [ ] Conteúdo e `MANIFEST.md`.
- [ ] Caminhos e dados sensíveis limpos.
- [ ] Tags no remoto.
- [ ] Rascunho com DOI reservado.
- [ ] Publicação e citação do DOI.
- [ ] Spec marcado como concluído no `README.md`.
