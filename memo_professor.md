# Memo ao professor: o que o teste de classificação mostrou

**Contexto.** Testei um sistema que sugere o tema de cada questão. Usei 659 questões e as 100 que a equipe já classificou como gabarito. O sistema não usa inteligência artificial generativa na hora de decidir: compara o texto da questão com palavras típicas de cada tema e com as 100 questões já classificadas.

## O que encontrei

- **Ele acerta o tema em cerca de metade dos casos** (53%). Nas 3 primeiras sugestões, o tema certo aparece em 72%.
- **Só publicaria sozinho cerca de 6% das questões** com segurança. Mesmo assim, estimo que uns 15% dessas poucas estariam erradas, abaixo dos 95% que vocês exigem. Por isso a maioria vai para a sua fila de revisão, **com sugestão de tema e o motivo da dúvida**.
- **Hoje ele é uma ajuda para a triagem, não um substituto.** Ele vai melhorar à medida que vocês confirmarem ou corrigirem sugestões.
- **Parte dos erros vem da própria árvore de temas**, e é aí que o seu olhar mais ajuda.

## Mudanças que recomendo na árvore

1. **Temas repetidos em dois lugares.** Lei de Execução Penal (em Processo Penal e em Legislação Especial) e, entre Penal e Legislação Especial: Abuso de Autoridade, Crimes contra o Meio Ambiente, Estatuto do Idoso, Lavagem de Dinheiro, Lei Maria da Penha, Organizações Criminosas e Tráfico de Drogas. A equipe usou as duas versões na classificação, então o sistema não tem como saber qual é a certa. *Sugestão:* escolher um lugar para cada tema.
2. **Nós "(Parte 1)" e "(Parte 2)"** (por exemplo, Competência, Teoria da Pena, Sentença Penal, Crimes contra o Patrimônio). Parecem restos de importação. *Sugestão:* juntar ou apagar.
3. **Juizado Especial Criminal** aparece em Legislação Especial e em Procedimento Sumaríssimo, e as duas versões foram usadas. *Sugestão:* definir qual é a principal.
4. **Dois temas inativos** (Direito Penal Econômico e Tratados e Convenções). Já estão fora do uso, mas convém confirmar se ficam assim.
5. **Erros de digitação nos títulos**, como "Estaturo de Idoso", "Comeptência Territorial", "Escolas Crimonológicas" e "Dos Crimescontra a Propriedade Imaterial", e números soltos em "12 Disposições Gerais…". Atrapalham a busca e a leitura.

## O que preciso de você

1. **Decidir os itens 1 a 3 acima.**
2. **Esclarecer o critério de "fora da taxonomia".** Por exemplo, uma questão sobre prisão domiciliar na execução penal foi marcada como fora. Qual é a regra para esses casos?
3. **Confirmar a convenção de tema principal.** Parece que a equipe classifica pelo assunto central, mesmo quando outro termo aparece no enunciado (uma questão que cita concurso de pessoas, mas trata de culpabilidade). Está correto?
4. **Rotular mais questões nos pares que o sistema mais confunde:** Procedimento Ordinário × Sentença Penal, Provas em Espécie × Teoria Geral das Provas, Teoria da Pena × Execução Penal. Cerca de 15 questões por tema já ajudam bastante.
5. **Revisar uma amostra** das primeiras sugestões, para medirmos o ganho de tempo real em vez de estimá-lo.
