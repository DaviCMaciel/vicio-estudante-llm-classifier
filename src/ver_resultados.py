"""Mostra, em formato legível, o que o modelo respondeu — SEM revelar as questões da sua amostra rotulada.

  python3 src/ver_resultados.py --modelo gemini-3.8-flash --passada A
  python3 src/ver_resultados.py --modelo gemini-3.8-flash --passada A --n 5 --seed 1
Use --mostrar-amostra só DEPOIS de terminar a planilha de rotulagem.
"""
import argparse
import random
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--modelo", required=True)
ap.add_argument("--passada", default="A")
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--mostrar-amostra", action="store_true")
a = ap.parse_args()

res = pd.read_json(f"data/resultados_{a.modelo}_{a.passada}.jsonl", lines=True)
q = pd.read_json("data/questoes_proc.jsonl", lines=True).set_index("id")
t = pd.read_csv("data/temas_proc.csv").set_index("tema_id")
amostra = set(open("data/amostra_ids.txt").read().split())

print(f"Resultados: {len(res)} questões | confiança média {res['confianca'].mean():.2f} | "
      f"conf>=0.9: {(res['confianca'] >= 0.9).sum()} | fora: {res['fora'].sum()} | "
      f"dúvida entre subtemas: {res['duvida_subtemas'].sum()}")
if not a.mostrar_amostra:
    oculto = res["id"].isin(amostra).sum()
    res = res[~res["id"].isin(amostra)]
    print(f"({oculto} questões da sua amostra de rotulagem foram ocultadas)")
ids = list(res["id"]); random.Random(a.seed).shuffle(ids)
nome = lambda i: t.loc[i, "caminho"] if i in t.index else "(nenhum)"
for qid in ids[:a.n]:
    r = res[res["id"] == qid].iloc[0]
    print(f"\n[{qid}] {q.loc[qid, 'enunciado'][:260].replace(chr(10), ' ')}")
    print(f"   -> {'FORA' if r['fora'] else nome(r['tema_central_id'])}   (conf {r['confianca']:.2f}"
          f"{', dúvida subtemas' if r['duvida_subtemas'] else ''})")
    if r["alternativa_tema_id"]:
        print(f"   alt: {nome(r['alternativa_tema_id'])}")
    print(f"   motivo: {r['motivo']}")