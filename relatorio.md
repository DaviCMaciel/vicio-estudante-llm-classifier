# Relatório: classificação de tópicos (Penal, Processo Penal, Legislação Especial, Criminologia)

**Resumo.** O sistema decide sem nenhum modelo de linguagem. Na folha-verdade ele acerta o tema (nível 1 da árvore) em 53% das questões publicando tudo. Com o limiar mais conservador, a precisão estimada fora da amostra é **85%** (intervalo 0,78–0,92), abaixo da régua de 95%, e só **6%** das questões fora da folha são publicadas. As demais vão para revisão com sugestão. Os números vêm de `regenerar.py` e são recalculados a cada execução.

## 1. O sistema, peça por peça

O comando `python regenerar.py` lê os anexos e escreve `publicacao.csv`, `previsao.txt` e `artifacts/calibracao.json`. Roda em cerca de 35 s para 659 questões (a primeira execução monta um cache de correspondência de termos; depois leva poucos segundos).

1. **Limpeza.** `html.unescape` duas vezes (as questões Q0360 em diante vêm com entidades), remoção de tags, minúsculas, sem acentos. Só temas com `status = active` e nível ≥ 1 são candidatos.
2. **Dicionário de termos por tema** (`data/dicionario_temas_otimizado.txt`). Termos, leis e artigos típicos de cada tema. Cada termo pesa pelo seu IDF nas 659 questões, e termos que aparecem em mais de 4% delas são descartados por serem genéricos.
3. **TF-IDF** (palavras, 1–2 gramas) da questão contra `caminho + título` do tema.
4. **kNN** contra as 100 questões da folha-verdade (K=5): cada vizinho vota no tema rotulado e, com metade do peso, nos seus ancestrais.
5. **Combinação.** Pontuação = 1,0·dicionário + 2,0·TF-IDF + 0,5·kNN, com softmax (temperatura 0,15). As probabilidades dos nós de um mesmo tema (nível 1) são somadas, e a questão é atribuída ao tema mais provável.
6. **Recuo para o tema-pai.** A regra de acerto do case aceita pai ou filho do tema correto. Por isso o sistema publica no nível 1: é o ponto mais seguro da árvore. O custo é perder o subtema.
7. **Confiança e limiar.** A coluna `confianca` é a probabilidade calibrada por regressão isotônica sobre as previsões leave-one-out da folha. O limiar de publicação é o maior conjunto, de pelo menos 8 questões, em que a precisão na folha atinge o alvo interno. O alvo é escolhido por validação cruzada aninhada (seção 2).
8. **Regras de saída.** `publicar` acima do limiar. `fora` quando a disciplina de origem não é penal (Constitucional, Tributário, Ambiental, Civil etc.) e o sistema não está seguro. O restante é `revisar`, com o motivo (confiança e tema alternativo) e `extras_ids` com os temas seguintes quando têm probabilidade ≥ 0,25.

**Proveniência do dicionário.** O dicionário foi proposto por um LLM **antes** da decisão, a partir da árvore de temas e do conhecimento geral de Direito, e depois revisado e editado por mim. É um dado versionado, e nenhum LLM é chamado ao classificar. O LLM não classificou, votou nem validou o tema de nenhuma questão. Ele viu a folha-verdade e a lista de erros ao propor e ajustar termos, o que é uma fonte de otimismo na avaliação (seção 2).

**Testado e descartado.**

| Tentativa | Resultado na folha | Decisão |
|---|---|---|
| TF-IDF puro contra título e caminho | acerto 38%; só 9 questões a 100% de precisão | insuficiente sozinho |
| `assunto_origem` como sinal | bate com um título de tema em 65 das 100; acerta 22 de 58 não-fora (38%) | descartado: é ruidoso e puxa o erro para dentro |
| Dicionário + TF-IDF + kNN | 53% | mantido |
| Dicionário original × dicionário revisado | empate (50–53% em ambos) | revisado mantido (proveniência humana) |
| Peso IDF e corte de termos genéricos | 51% → 53% | mantido |
| Confiança por regressão logística com margem e concordância entre sinais | teto de ~90% de precisão com 20–25 questões em 100 | descartado: sem ganho real |
| Busca em grade de pesos e limiares na própria folha | 95,2% em 21 questões; com escolha em metade e teste na outra, 81% | descartada como estimativa (ver 2) |

Não usei embeddings neurais: o ambiente de execução só tinha scikit-learn. Gerá-los antes da decisão seria permitido e é a evolução mais provável (seção 3).

## 2. Como as 100 questões foram usadas

- **Previsões sem vazamento.** Cada questão da folha é prevista com o kNN sem o próprio rótulo (leave-one-out). Isso produz 100 pares (confiança, acertou?).
- **Calibração.** A regressão isotônica usa esses 100 pares.
- **Teste.** Para o limiar não ser otimista, repeti 300 vezes: escolher o limiar em metade da folha e medir na outra metade. O alvo interno foi o menor entre 0,95, 0,97 e 1,0 cujo resultado fora da amostra batesse 95%. **Nenhum bateu** (0,83, 0,85 e 0,85), então o sistema usa o mais conservador, e a previsão fica abaixo da régua.
- **Limite da avaliação.** Com 100 rótulos, e cerca de 10 questões publicadas por metade, os intervalos são largos. A folha também orientou escolhas de pesos e do dicionário, então resta otimismo.

**O que a folha mostrou sobre a convenção da equipe**
- Os rótulos estão quase sempre no nível de tema, não de subtema, e muitas questões têm 1 a 3 extras. Isso torna o nível 1 a escolha natural.
- Rótulos pelo assunto principal, não pelo termo que aparece: Q0313 fala de ação civil ex delicto e foi rotulada em "Efeitos da Condenação"; Q0230 menciona concurso de pessoas e foi rotulada em Culpabilidade.
- Cerca de 8% da folha é `fora`, e a maior parte parece penal: Q0087 (prisão domiciliar na execução) e Q0252 (fixação de pena) são `fora`. Nesses casos, só o olhar humano separa.
- "Lei de Execução Penal" existe em duas disciplinas e a equipe usou os dois nós (4703 e 6734).

## 3. Diagnóstico

Dos 49 erros da folha com a versão anterior do dicionário, os tipos que restam são:

1. **Sem sinal.** Cerca da metade dos erros tem confiança abaixo de 0,15 (nenhum termo reconhecido) e cai num tema por acaso. São inofensivos para a precisão (vão para `revisar`) e mostram a falta de cobertura do dicionário. *Correção:* mais termos por tema a partir dos erros da fila de revisão, e embeddings de frases.
2. **Vocabulário compartilhado entre temas vizinhos.** Teoria da Pena × Execução Penal (Q0091, confiança 0,84), Procedimento Ordinário × Sentença Penal (Q0090, Q0339), Inquérito Policial e Provas em Espécie × Teoria Geral das Provas (Q0147, Q0276). *Correção:* termos exclusivos por tema, mais exemplos rotulados desses pares.
3. **Assunto principal × termo incidental.** Q0208 e Q0230 acima. É o erro mais perigoso: acontece com confiança alta (0,73 a 0,99). *Correção:* um sinal sobre o comando da questão (o que se pergunta) e não só sobre as palavras do enunciado, mais rótulos.
4. **`fora` dentro do penal.** O sistema só detecta `fora` por disciplina de origem. Q0043, Q0087, Q0252, Q0291, Q0303 passam. *Correção:* um exemplo `fora` explícito no kNN e a decisão do professor sobre os temas que faltam (seção 5).
5. **Problemas na árvore.** Duplicatas ("Parte 1", "Parte 2"), a LEP em duas disciplinas, dois temas inativos e erros de digitação ("Estaturo de Idoso", "Comeptência Territorial"). Esses nós dividem o mesmo conteúdo e confundem qualquer classificador.

**Para a revisão.** Entre as não-`fora`, o tema certo é a 1ª sugestão em 58% dos casos e está entre as 3 primeiras em **72%**. É esse número, e não a cobertura de publicação, que sustenta o ganho da fila de revisão.

## 4. Produção: 50 mil questões por semana

- **Custo e tempo.** Sem API de modelo de linguagem, o custo é o de uma CPU. A execução de 659 questões levou cerca de 35 s, incluindo o cache. Extrapolando de forma linear, 50 mil questões levam algumas dezenas de minutos em um núcleo (estimativa, não medida), e paralelizam por lote.
- **Volume de revisão.** Hoje 94% das questões novas iriam para revisão. Isso só é viável se a revisão for rápida (sugestão pronta, 72% de acerto nas 3 primeiras) e se a cobertura crescer com o uso.
- **Correções viram dado de treino.** Cada questão confirmada ou corrigida entra como novo exemplo rotulado no kNN, sem retreino. Além disso, ela alimenta a calibração e a análise de erros por tema.
- **Cadência sugerida.** Semanal: incorporar as correções, recalibrar a isotônica e o limiar. Mensal: revisar o dicionário e medir de novo a precisão em uma amostra auditada das publicadas (cerca de 2%). Se a precisão auditada cair abaixo de 95%, o limiar sobe automaticamente.
- **Evolução técnica.** Embeddings gerados antes da decisão e guardados em cache; extração explícita de lei e artigo citados; subtemas (nível 2) quando houver rótulos suficientes.

## 5. Memo para o professor

Ver `memo_professor.md`.

## 6. Limitações

A precisão prevista (0,85) está abaixo de 0,95 e a cobertura é baixa (0,06). A amostra de 100 rótulos limita o que dá para medir. O sistema decide no nível de tema, não de subtema, e não detecta `fora` dentro do penal. Os números acima foram impressos pelo próprio `regenerar.py`.
