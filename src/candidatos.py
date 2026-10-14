"""Passo 2: para cada questão, lista os K temas mais prováveis (candidatos) com TF-IDF.

Entradas (data/): questoes_proc.jsonl, temas_proc.csv, gemeos.csv
Saída   (data/): candidatos.jsonl  -> {"id", "candidatos": [{tema_id, titulo, caminho, score}, ...]}

Este passo NÃO decide nada: só reduz 371 temas para K plausíveis, para o passo 3 (LLM) escolher entre eles.
Obs.: o assunto_origem NÃO entra no score de propósito; ele fica livre para servir de
verificação independente (tests/test_02_candidatos.py) e de feature no passo 4.
"""
import json
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

K = 10
STOP_PT = [
    "a", "o", "as", "os", "de", "da", "do", "das", "dos", "e", "em", "no", "na", "nos", "nas",
    "um", "uma", "uns", "umas", "para", "por", "com", "sem", "que", "se", "ao", "aos", "ou",
    "como", "mais", "mas", "ser", "foi", "sua", "seu", "suas", "seus", "ja", "nao", "sao",
    "pelo", "pela", "pelos", "pelas", "entre", "sobre", "quando", "onde", "qual", "quais",
    "assinale", "alternativa", "correta", "incorreta", "afirmativa", "assertiva", "item",
    "considerando", "acordo", "segundo", "conforme", "correto", "incorreto", "julgue",
]
# frases de escopo negativo nas descrições ("Não se confunde com...") atrairiam o tema errado
ESCOPO_NEGATIVO = re.compile(r"[^.]*(não se confunde|sem adentrar|não abrange|não trata|exclui)[^.]*\.", re.I)
BOILERPLATE = re.compile(r"(Subtema de .*?Path:.*|Disciplina \(root\).*)", re.S)


def texto_do_tema(r, titulos: dict) -> str:
    """Texto que representa o tema: título (3x, é o sinal mais forte), caminho e descrição limpa."""
    desc = "" if pd.isna(r["descricao"]) else BOILERPLATE.sub("", r["descricao"])
    desc = ESCOPO_NEGATIVO.sub("", desc)
    pai = titulos.get(r["parent_id"], "") if pd.notna(r["parent_id"]) else ""
    return f"{r['titulo']}. " * 3 + f"{pai}. {r['caminho']}. {desc}"


def carregar():
    q = pd.read_json("data/questoes_proc.jsonl", lines=True)
    t = pd.read_csv("data/temas_proc.csv", dtype={"parent_id": "Int64"})
    sombra = set(pd.read_csv("data/gemeos.csv")["tema_id"])
    cand = t[t["candidato"] & ~t["tema_id"].isin(sombra)].reset_index(drop=True)
    return q, t, cand


def gerar_candidatos(q, t, cand, k=K):
    titulos = dict(zip(t["tema_id"], t["titulo"]))
    textos_temas = [texto_do_tema(r, titulos) for _, r in cand.iterrows()]
    textos_q = (q["enunciado"] + " " + q["comando_limpo"] + " " + q["alternativas_texto"]).tolist()

    # o vocabulário é aprendido com questões + temas juntos, para o IDF refletir o domínio
    vec = TfidfVectorizer(strip_accents="unicode", lowercase=True, ngram_range=(1, 2),
                          sublinear_tf=True, stop_words=[s for s in STOP_PT], min_df=1)
    vec.fit(textos_q + textos_temas)
    Xq, Xt = vec.transform(textos_q), vec.transform(textos_temas)
    sim = (Xq @ Xt.T).toarray()  # vetores já normalizados -> produto = cosseno

    saida = []
    for i, qid in enumerate(q["id"]):
        top = np.argsort(-sim[i])[:k]
        saida.append({"id": qid, "candidatos": [
            {"tema_id": int(cand.loc[j, "tema_id"]), "titulo": cand.loc[j, "titulo"],
             "caminho": cand.loc[j, "caminho"], "score": round(float(sim[i, j]), 4)} for j in top]})
    return saida


if __name__ == "__main__":
    q, t, cand = carregar()
    print(f"Questões: {len(q)} | temas candidatos (sem inativos, nível 0 e gêmeos-sombra): {len(cand)}")
    saida = gerar_candidatos(q, t, cand)
    with open("data/candidatos.jsonl", "w", encoding="utf-8") as f:
        for s in saida:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Gravado data/candidatos.jsonl ({len(saida)} linhas, K={K})")