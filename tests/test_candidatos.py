"""Testes do passo 2. Rode depois de src/02_candidatos.py:

    python3 tests/test_02_candidatos.py            # sanidade + métrica proxy + amostra para olhar
    python3 -m pytest tests/ -q                    # só as checagens (se tiver pytest)

IMPORTANTE: a métrica proxy usa o assunto_origem como rótulo FRACO. Ele pode estar errado
(o enunciado avisa), então a recall aqui é uma estimativa aproximada, NÃO a precisão do case.
"""
import json
import random
import re
import unicodedata
import pandas as pd

K = 10


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split())


def carregar():
    q = pd.read_json("data/questoes_proc.jsonl", lines=True)
    t = pd.read_csv("data/temas_proc.csv", dtype={"parent_id": "Int64"})
    c = {}
    for l in open("data/candidatos.jsonl", encoding="utf-8"):
        d = json.loads(l); c[d["id"]] = d["candidatos"]
    sombra = set(pd.read_csv("data/gemeos.csv")["tema_id"])
    return q, t, c, sombra


# ---------- checagens de sanidade (têm que passar sempre) ----------
def test_sanidade():
    q, t, c, sombra = carregar()
    ativos = set(t["tema_id"])                       # temas_proc.csv já só tem ativos
    nivel0 = set(t.loc[t["nivel"] == 0, "tema_id"])
    assert set(c) == set(q["id"]), "faltam ou sobram questões em candidatos.jsonl"
    for qid, lista in c.items():
        ids = [x["tema_id"] for x in lista]
        assert len(ids) == K, f"{qid}: esperava {K} candidatos"
        assert len(set(ids)) == K, f"{qid}: candidatos repetidos"
        assert all(i in ativos for i in ids), f"{qid}: tema inativo/inexistente"
        assert not (set(ids) & nivel0), f"{qid}: disciplina (nível 0) como candidato"
        assert not (set(ids) & sombra), f"{qid}: nó gêmeo-sombra como candidato"
        scores = [x["score"] for x in lista]
        assert scores == sorted(scores, reverse=True), f"{qid}: candidatos fora de ordem"
        assert all(0 <= s <= 1 for s in scores), f"{qid}: score fora de [0,1]"


# ---------- métrica proxy (informativa) ----------
def recall_proxy():
    q, t, c, sombra = carregar()
    pai = dict(zip(t["tema_id"], t["parent_id"]))
    por_titulo = {}
    for _, r in t[t["candidato"] & ~t["tema_id"].isin(sombra)].iterrows():
        por_titulo.setdefault(norm(r["titulo"]), set()).add(int(r["tema_id"]))

    def mesmo_ramo(a, b):
        return a == b or pai.get(a) == b or pai.get(b) == a

    n_com_rotulo, hits = 0, {1: 0, 3: 0, 5: 0, 10: 0}
    for _, r in q.iterrows():
        alvos = por_titulo.get(norm(r["assunto_origem"]), set())
        if not alvos:
            continue
        n_com_rotulo += 1
        ids = [x["tema_id"] for x in c[r["id"]]]
        for k in hits:
            if any(mesmo_ramo(i, a) for i in ids[:k] for a in alvos):
                hits[k] += 1
    return len(q), n_com_rotulo, hits


def amostra(n=8, seed=7):
    q, t, c, _ = carregar()
    random.seed(seed)
    for _, r in q.sample(n, random_state=seed).iterrows():
        print(f"\n[{r['id']}] origem: {r['disciplina_origem']} / {r['assunto_origem']}")
        print("  " + r["enunciado"][:220].replace("\n", " "))
        for x in c[r["id"]][:3]:
            print(f"    {x['score']:.3f}  {x['caminho']}")


if __name__ == "__main__":
    test_sanidade()
    print("OK - checagens de sanidade passaram")
    total, com, hits = recall_proxy()
    print(f"\nMétrica proxy: {com}/{total} questões têm assunto_origem = título de um tema")
    for k, h in hits.items():
        print(f"  recall@{k:<2}: {h}/{com} = {h / com:.1%}")
    amostra()