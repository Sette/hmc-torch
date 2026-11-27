# 26 — IJCNN 2027: submissão e camera-ready (P0)

## Objetivo

Submeter o paper de 6 páginas (`docs/hmc-paper-ijcnn/`) ao IJCNN 2027 até **27/01/2027**
(prazo oficial 31/01, domingo, AoE) e deixar o camera-ready (12/04) pronto para restaurar.
Consolida os specs 09, 17 e 19 num checklist por data e fecha o item "orientador" do spec 06.

## Pré-requisitos

- Specs 19, 20, 22 (passo 1) e 25 (passo 4) já aplicados no IJCNN (estado em 24/11/2026:
  anônimo, 6 páginas, 0 overfull/underfull, `make check` OK).
- Bateria do spec 23 concluída (`CONTROLS_DONE` em `output/protocol_controls/run.log`).

## Fatos a usar

- Fonte das regras: https://ijcnn.org/2027/authors/initial-author-instructions (lida em
  13/11/2026; reconferir antes de submeter, pode haver extensão).
- Duplo-cego obrigatório; 4–6 páginas incluindo referências; IEEE conference, Letter, duas
  colunas, 10 pt, sem número de página, só PDF.
- Uso de IA declarado nos agradecimentos e citado.
- Submissão abre **01/12/2026**; regular papers **31/01/2027**; notificação **15/03/2027**;
  camera-ready **12/04/2027** (AoE).
- A plataforma é o **CMT**, não o EDAS (o `plano-defesa-2027.md` dizia EDAS; corrigido).
- Páginas extras custam US$ 100 cada, pela maior versão: não passar de 6 na submissão.

## Passos

1. **Controles (spec 23).** Rodar `python experiments/summarize_controls.py` e ler a tabela
   (Δ rec − raw com seleção simétrica × atual). Se a contagem de vitórias ou o sinal da mediana
   mudar, ajustar resumo, introdução e conclusão. Se não mudar, tentar **uma oração** no
   protocolo ("selecting the epoch separately for raw and reconciled scores changes the gain
   by …"). Se estourar 6 páginas, cortar de outra parte; não pagar página extra.
2. **Conferir o site** do IJCNN (datas, limite, CMT, regra de IA); anotar a data da conferência
   neste spec.
3. **Artefatos (spec 28).** Rascunho no Zenodo. Na submissão, nenhum link identificador: o
   texto fica "to be released upon acceptance"; suplemento anônimo só se o CMT permitir.
4. **Revisão cruzada** com o orientador até 17/01 (nome, e-mail, ORCID para o camera-ready;
   revisão do texto).
5. **Congelar em 24/01.** `make -C docs/hmc-paper-ijcnn check`; abrir o PDF e conferir:
   - sem nome, afiliação, agradecimento de financiamento ou URL do GitHub; metadados do PDF
     (`pdfinfo`, `exiftool`) sem autor;
   - "Anonymous Authors"; trabalhos próprios em terceira pessoa;
   - declaração de IA presente;
   - 6 páginas, Letter, 0 referências indefinidas.
6. **Submeter em 25–27/01 no CMT**; guardar o comprovante (e-mail do CMT, PDF enviado,
   número do paper) em `/home/brunos/git/phd/`.
7. **Depois de 15/03.** Aceito: restaurar os itens de `docs/hmc-paper-ijcnn/CAMERA_READY.md`
   (bloco de autores, URL, DOI do Zenodo, agradecimentos), conferir 6 páginas, enviar até
   12/04, registrar um participante para a apresentação (14–18/06, Cape Town) — pode ser o
   orientador. Rejeitado: ver spec 27 (KBS imediata) e o plano de defesa.

## Saídas esperadas

- PDF anônimo de 6 páginas submetido; comprovante arquivado.
- `CAMERA_READY.md` seguido item a item em abril.

## Critério de aceite

- [ ] `make check` imprime `pages: 6 (limit 6)` e `OK`.
- [ ] `pdfinfo main.pdf` sem nome do autor; `grep -i "sette\|furtado\|github.com/Sette"` no
      PDF extraído não retorna nada.
- [ ] Declaração de IA nos agradecimentos.
- [ ] Comprovante de submissão guardado.
- [ ] Datas conferidas no site em data registrada.

## Tempo/custo

~1 dia de trabalho espalhado; sem compute.

## Falhas e rollback

- **Frase do spec 23 estoura a página:** não incluir; a KBS e a tese carregam o controle.
- **Controle muda a conclusão:** atualizar o resumo e a conclusão nos três manuscritos antes
  de congelar; se faltar tempo, submeter com a limitação declarada.
- **Prazo estendido pelo IJCNN:** manter 27/01; a extensão só serve de reserva.
- **CMT recusa o PDF:** fonte/margem — recompilar com `make pdf`; nunca reduzir fonte.
- Rollback: tag git do estado submetido (`ijcnn-submitted-2027-01`), criada localmente.

## Atualizações no paper/tese

- IJCNN: uma oração no protocolo (opcional). KBS e tese: spec 27.

## Checklist

- [ ] `summarize_controls.py` rodado e lido.
- [ ] Site conferido.
- [ ] Zenodo em rascunho.
- [ ] Revisão do orientador.
- [ ] Texto congelado (24/01).
- [ ] Submetido (≤ 27/01) e comprovante guardado.
- [ ] Camera-ready (12/04), se aceito.
- [ ] Spec marcado como concluído no `README.md`.
