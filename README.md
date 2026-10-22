# Classificação de tópicos: Vício (Método VDE)

Sistema que decide **publicar / revisar / fora** para 659 questões de Penal, Processo Penal, Legislação Penal Especial e Criminologia, contra uma árvore de 377 temas. **A decisão não chama nenhum modelo de linguagem.** Tudo roda localmente com pandas, numpy e scikit-learn.

**Resultado (calculado por `regenerar.py`, ver `previsao.txt`):** acerto de 53% no tema na folha-verdade; precisão prevista de 85% (0,78–0,92) e cobertura prevista de 6% (0,06–0,21) nas questões fora da folha. A precisão prevista **fica abaixo da régua de 95%**. Detalhes e limitações em `relatorio.md`.

## Como rodar

```bash
pip install -r requirements.txt
python3 src/regenerar.py --data data/raw --dic dicionario_temas_otimizado.txt
```

Gera na raiz: `publicacao.csv` (659 linhas), `previsao.txt` e `artifacts/calibracao.json`.

- **Tempo:** cerca de 35 s na primeira execução (monta um cache de correspondência de termos em `artifacts/match_*.npz`) e poucos segundos depois, em CPU comum. O cache é regenerável e pode ser apagado.
- **Dependências:** Python 3.9+, `pandas`, `numpy`, `scikit-learn`.
- **Determinismo:** a semente da validação aninhada é fixa (`SEED = 0`); duas execuções geram os mesmos arquivos.
- **Entradas** (em `data/raw/`): `questoes.jsonl`, `temas.csv`, `folha_verdade.csv` e o dicionário.

## Estrutura

```
data/raw/            anexos do case + dicionario_temas_otimizado.txt
src/regenerar.py     pipeline completo (um comando)
artifacts/           calibracao.json (pesos, limiar, resultados da validação) e cache do dicionário
publicacao.csv       saída: id, decisao, tema_central_id, extras_ids, confianca, motivo
previsao.txt         previsão calculada, com intervalos e o método
relatorio.md         relatório técnico
memo_professor.md    memo de uma página para o professor
```

## Como o sistema decide

1. **Limpeza** do HTML (com `html.unescape`), minúsculas e sem acentos. Só temas ativos, de nível ≥ 1.
2. **Três sinais:** dicionário de termos por tema (com peso IDF e corte de termos genéricos), TF-IDF da questão contra o caminho do tema e kNN contra as 100 questões da folha-verdade.
3. **Combinação** 1,0 × dicionário + 2,0 × TF-IDF + 0,5 × kNN, em softmax; as probabilidades são somadas no **tema (nível 1)**, que é seguro pela regra do case (pai ou filho conta como certo).
4. **Confiança** calibrada por regressão isotônica sobre as previsões leave-one-out da folha. **Limiar** escolhido por validação cruzada aninhada (300 divisões meia-meia).
5. **Saídas:** `publicar` acima do limiar; `fora` para disciplinas de origem não penais sem confiança; `revisar` no resto, com o motivo e um tema alternativo.

## Proveniência dos artefatos

| Artefato | Origem |
|---|---|
| `dicionario_temas_otimizado.txt` | Proposto por um LLM **antes** da decisão, a partir da árvore de temas e de conhecimento geral de Direito. Depois revisado e editado manualmente por mim (o histórico do git registra as diferenças). O LLM viu a folha-verdade e a lista de erros ao propor e ajustar termos, mas **não classificou, votou nem validou o tema de nenhuma questão**. |
| `temas.csv`, `questoes.jsonl`, `folha_verdade.csv` | Anexos do case, sem alteração. |
| `artifacts/calibracao.json` | Gerado por `regenerar.py` (pesos, limiar, precisão e cobertura da validação aninhada). |
| `artifacts/match_*.npz` | Cache gerado por `regenerar.py` (matriz termo × questão). Não é fonte de verdade. |

Não há embeddings nem modelo treinado salvo: o kNN e a isotônica são recalculados a cada execução a partir da folha.

## Decisões manuais e seus motivos

- **Dicionário:** edições pontuais minhas sobre a versão proposta pelo LLM, para tirar termos ambíguos e acrescentar jargões que eu conhecia. Mantido o revisado porque, na folha, empatou com o original (50–53%) e tem proveniência humana.
- **Pesos 1,0 / 2,0 / 0,5:** vieram de uma busca em grade que fiz na própria folha. Essa escolha é, portanto, **otimista**; por isso a precisão prevista usa validação aninhada para o limiar, mas os pesos em si não foram validados fora da amostra.
- **Nível 1 da árvore:** o sistema publica no tema e não no subtema, para aproveitar a regra de pai/filho e proteger a precisão. Custo: perde o subtema.
- **Regra de `fora`:** só marca `fora` quando a disciplina de origem não é penal (lista `NAO_PENAL` no código). `fora` dentro do penal (cerca de 8% da folha) não é detectado e vai para `revisar`.
- **Limiar conservador:** nenhum alvo interno de precisão bateu 95% fora da amostra; o sistema usa o mais conservador e declara isso no `previsao.txt`.
- **Temas inativos** (5078 e 5234) e as quatro raízes de disciplina nunca são usados.

## Limitações

Amostra de 100 rótulos; precisão prevista abaixo de 95% e cobertura baixa; sem subtemas; sem extração explícita de lei e artigo além dos termos do dicionário; sem embeddings. Evoluções propostas no `relatorio.md`.
