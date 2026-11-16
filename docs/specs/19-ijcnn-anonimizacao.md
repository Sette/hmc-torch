# 19 — IJCNN 2027: anonimização e declaração de IA (P0)

## Objetivo

Deixar `docs/hmc-paper-ijcnn/main.tex` conforme as regras de submissão do IJCNN 2027 (lidas em
13/11/2026): revisão duplo-cega obrigatória e declaração de texto gerado por IA. Fecha os
achados D1–D3 do [relatório 18](18-audit-geral.md). Sem isso o paper pode ser rejeitado sem
revisão.

## Pré-requisitos

- Reconferir as regras antes de submeter (as datas podem mudar):
  `https://ijcnn.org/2027/authors/initial-author-instructions` e
  `https://ijcnn.org/2027/authors/call-for-papers`.
- Specs 14–17 aplicados (estão).

## Fatos a usar

- Autor: `\author{\IEEEauthorblockN{Anonymous Authors}}`; trabalho próprio citado em terceira
  pessoa; preprints são permitidos.
- 4 a 6 páginas incluindo referências; estilo IEEE conference, Letter, sem número de página.
- Texto gerado por IA: declarar nos agradecimentos e citar a ferramenta.
- Camera-ready em 12/04/2027: os dados reais voltam só nessa versão.

## Passos

1. **Autores** (`main.tex:20-22`): trocar o bloco por
   `\author{\IEEEauthorblockN{Anonymous Authors}}`. Guardar o bloco original num comentário
   `% CAMERA-READY:` logo abaixo.
2. **URL no abstract** (`main.tex:50`): remover a frase “Code: \url{https://github.com/…}”.
   Alternativa: “Code and per-run metrics are provided as anonymized supplementary material”,
   desde que o material exista (ver passo 5).
3. **URL na conclusão** (`main.tex:617`): mesma troca do passo 2.
4. **Outras pistas de identidade**: procurar e remover
   `grep -n -i "sette\|ufscar\|são carlos\|sao carlos\|our previous\|our earlier" main.tex`.
   Conferir os metadados do PDF (`pdfinfo main.pdf`: campos Author, Title, Creator) e
   `\hypersetup{pdfauthor=…}`, se houver.
5. **Material suplementar anônimo** (opcional): se o código for enviado, usar um zip sem
   `.git`, sem `pyproject.toml` com nome/e-mail, sem `CLAUDE.md`/`README` com identificação, ou
   um espelho anônimo. Não apontar para o GitHub pessoal.
6. **Declaração de IA**: acrescentar uma seção `\section*{Acknowledgment}` curta, na versão de
   submissão sem agradecer pessoas ou instituições (identificariam os autores), com a
   declaração de uso de IA. Adaptar o texto que já está na KBS
   (`docs/hmc-paper/main.tex:2273-2277`) e citar a ferramenta.
7. **Espaço**: a seção do passo 6 custa 3–4 linhas; os passos 2 e 3 devolvem 2. Se passar de 6
   páginas, cortar da orientação prática ou encurtar uma legenda.
8. `make check` (impõe ≤ 6 páginas).
9. Criar `docs/hmc-paper-ijcnn/CAMERA_READY.md` com a lista do que restaurar em abril: autores,
   afiliação, URL, agradecimentos reais.

## Saídas esperadas

- PDF de 6 páginas sem nome, afiliação, e-mail ou URL identificável.
- Lista de restauração para o camera-ready.

## Critério de aceite

- [ ] `grep -n -i "sette\|ufscar\|github.com" main.tex` só retorna linhas comentadas.
- [ ] `pdfinfo main.pdf` sem autor identificável.
- [ ] Seção de agradecimento contém a declaração de IA.
- [ ] 6 páginas, 0 referências indefinidas, 0 over/underfull.

## Tempo/custo

~30 min; sem compute.

## Falhas e rollback

- **Estouro de página:** ver passo 7.
- **Regra alterada no site:** seguir o site e atualizar a memória `ijcnn-2027-rules`.
- Rollback: `git checkout -- docs/hmc-paper-ijcnn/main.tex`.

## Atualizações no paper/tese

- Só `docs/hmc-paper-ijcnn/main.tex`. KBS (single-blind) e tese não mudam.

## Checklist

- [ ] Autores anonimizados e bloco original preservado em comentário.
- [ ] URLs removidas do abstract e da conclusão.
- [ ] Metadados do PDF conferidos.
- [ ] Declaração de IA incluída.
- [ ] `CAMERA_READY.md` criado.
- [ ] `make check` passa.
- [ ] Spec marcado como concluído no `README.md`.

## Execução (16/11/2026)

- Feito: autores anônimos (bloco original em comentários `% CAMERA-READY:`), URLs removidas do
  abstract e da conclusão, seção Acknowledgment com a declaração de IA, `CAMERA_READY.md`,
  metadados do PDF sem autor. 6 páginas, 0 over/underfull (`ae112ff`).
- Falta: material suplementar anônimo, se for enviar código (passo 5); reconferir as regras no
  site antes de submeter.
