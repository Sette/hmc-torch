# 08 — Riscos de revisão na KBS e mitigações

Documento de apoio (não é uma tarefa sequencial). Consolida os pontos que revisores da
*Knowledge-Based Systems* tendem a apertar e o que fazer a respeito. Atualizar quando o paper
mudar de estado.

## Veredito

A contribuição é **plausível para a KBS, mas não é uma aposta segura**. O encaixe depende de
vender a *injeção de conhecimento hierárquico* e o estudo sistemático, não SOTA. A novidade é
incremental-mas-defensável: a R-matrix é herdada do C-HMCNN; o que é novo é a formulação
esparsa, o estudo controlado e a atribuição do ganho à reconciliação na inferência. Sem os P0/P1
fechados, o risco de desk-reject por escopo/novidade é real.

## Por que encaixa na KBS

- O objeto é **conhecimento estruturado** (taxonomia/ontologia) injetado no classificador; a
  R-matrix é o operador de consistência desse conhecimento.
- A pergunta *“quando a restrição ajuda?”* é científica e útil; o achado de que o ganho vem da
  **reconciliação na inferência**, não da loss de treino, é conhecimento aplicado.
- A formulação **esparsa O(C+E)** é uma contribuição algorítmica concreta e escalável (GO com
  4.134 classes).
- O estudo de reprodutibilidade do Surj e os resultados negativos (autoencoders, calibração)
  reforçam o rigor; a KBS publica estudos empíricos.

## Objeções prováveis e respostas

### 1. “Qual é a novidade, se a R-matrix é do C-HMCNN (2020)?”
**Risco:** alto; é a primeira pergunta de um revisor.
**Resposta:** creditar explicitamente o C-HMCNN e abrir a introdução com a lacuna: a restrição
nunca foi isolada da arquitetura/receita, nem formulada de forma escalável. A contribuição é
(i) a formulação esparsa O(C+E), (ii) o estudo sistemático de 3 objetivos × raw/reconciliado,
(iii) a atribuição do ganho à inferência, (iv) a evidência de reprodutibilidade.
**Ação:** reescrever introdução/contribuições nesse enquadramento (spec 05).

### 2. “Os ganhos são pequenos.”
**Risco:** alto.
**Resposta:** não vender SOTA; vender **análise e orientação**. FunCat/GO empatam; cross-domain
ganha +0,008 (diatoms +0,027); o valor está em saber *quando* e *onde* a restrição ajuda. A
seção de orientação prática (“use R na inferência; não espere ganho da loss; use a forma esparsa
acima de X classes”) é a contribuição para o leitor da KBS.
**Ação:** adicionar seção de orientação prática e ajustar o abstract (spec 05).

### 3. “Faltam baselines recentes (2023–2025).”
**Risco:** médio-alto.
**Resposta:** o estudo isola consistência/reconciliação; incluir uma discussão explícita dos
métodos recentes e por que não entram no desenho (dados/split/compute), ou rodar um baseline
recente se houver tempo. Pelo menos ampliar a Related Work (~300 palavras hoje) com HMC/HTC e
injeção de conhecimento recentes.
**Ação:** ampliar Related Work; decidir se roda algum baseline extra (fora do escopo atual).

### 4. “Isso é um paper de engenharia/plataforma.”
**Risco:** médio.
**Resposta:** a plataforma é **veículo**, não protagonista; o título novo e a introdução devem
levar a pergunta científica. A R esparsa e o estudo sistemático são a contribuição; a plataforma
é o que torna o estudo reproduzível.
**Ação:** revisar título/abstract/introdução; não liderar com features da CLI.

### 5. “O paper é longo e parece um benchmark dump.”
**Risco:** médio.
**Resposta:** condensar Results (~4.900 palavras), mover ablações redundantes (`tab:ablation`,
`fig:ablation`, `fig:decomp`) para material suplementar; manter no corpo o que sustenta a tese
(paridade FunCat/GO, cross-domain, R esparsa, reconciliação vs. loss).
**Ação:** edição de estrutura no spec 05.

### 6. “O protocolo tem limiar no test.”
**Risco:** alto (eliminatório se não corrigido).
**Resposta:** refazer os números de texto validation-only (spec 01), rodar hinge em FunCat/GO
(spec 02) e WOS 3 seeds (spec 04); remover o caveat e afirmar um protocolo único.
**Ação:** specs 01, 02, 04 e 05.

### 7. “A afirmação sobre o Surj é forte demais.”
**Risco:** médio.
**Resposta:** manter a tabela por dataset e o protocolo; redigir como “não reproduzível a partir
da descrição publicada”, com o pipeline do C-HMCNN reproduzindo os números do C-HMCNN; não
atribuir má-fé aos autores.
**Ação:** seção Surj restaurada (já feita); revisar a redação no spec 05.

### 8. “O abstract fala em três objetivos, mas FunCat/GO só têm dois.”
**Risco:** alto se não corrigido.
**Resposta:** rodar hinge (spec 02) e incluir a coluna nas tabelas; ou reduzir a afirmação.
**Ação:** spec 02 + spec 05.

### 9. “EUR-Lex usa subamostra / hierarquia rala.”
**Risco:** médio.
**Resposta:** a release completa (45k/6k/6k) tem 4.271 conceitos do dataset, 4.520 nós de
hierarquia, 5.237 arestas e profundidade 8; os resultados novos são sobre essa release. A
subamostra de 10k/2.901 era a versão antiga. Declarar que 29% dos rótulos avaliados são internos,
mas a reconciliação muda 10,7% dos scores (ganho +0,0076).
**Ação:** spec 01 + spec 05.

### 10. “WOS está abaixo do HPT e era seed única.”
**Risco:** médio.
**Resposta:** o paper não reivindica SOTA; rodar as 3 seeds (spec 04) e reportar média ± desvio,
com a diferença para o HPT explicitada.
**Ação:** spec 04 + spec 05.

### 11. “ArXiv/AAPD não são like-for-like.”
**Risco:** médio.
**Resposta:** já declarado no paper; manter a ressalva, não usar como comparação direta e, se o
ArXiv for incluído, explicar o split próprio. O ArXiv foi baixado (spec 01), então dá para rodar
o protocolo validation-only e reportar com a ressalva.
**Ação:** spec 01 + spec 05.

## Mapa objeção → spec/ação

| Objeção | Spec | Ação principal |
|---|---|---|
| Novidade (R-matrix herdada) | 05 | reescrever intro/contribuições |
| Ganhos pequenos | 05 | seção de orientação prática; não vender SOTA |
| Baselines recentes | 05 | ampliar Related Work; discutir escopo |
| “Paper de engenharia” | 05 | plataforma como veículo; título/abstract |
| Longo/benchmark dump | 05 | condensar Results; mover ablações |
| Limiar no test | 01, 02, 04, 05 | reruns validation-only |
| Surj forte demais | 05 | redação cuidadosa + tabela |
| Hinge ausente | 02, 05 | rodar hinge; atualizar tabelas |
| EUR-Lex subamostra/hierarquia | 01, 05 | release completa (5.237 arestas, depth 8); ganho +0.0076 |
| WOS abaixo do HPT/seed única | 04, 05 | 3 seeds; sem SOTA |
| ArXiv/AAPD não comparáveis | 01, 05 | ressalva explícita |

## Checklist de mitigação

- [ ] Spec 01 (texto validation-only, incluindo ArXiv).
- [ ] Spec 02 (hinge FunCat/GO; tabelas com hinge).
- [ ] Spec 03 (structure.json; análise de estrutura).
- [ ] Spec 04 (WOS 3 seeds).
- [ ] Spec 05: introdução reenquadrada em “injeção de conhecimento hierárquico”.
- [ ] Spec 05: seção de orientação prática.
- [ ] Spec 05: Related Work ampliada (8–12 referências recentes).
- [ ] Spec 05: Results condensado; ablações para suplementar.
- [ ] Spec 05: redação do Surj revisada.
- [ ] Spec 06: cover letter destacando o encaixe na KBS.
