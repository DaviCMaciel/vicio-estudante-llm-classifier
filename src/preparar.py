"""Passo 1: limpa as questões e prepara a árvore de temas.

Gera em data/:
  questoes_proc.jsonl  - questões com texto limpo (enunciado, comando, alternativas, texto_completo)
  temas_proc.csv       - só temas ativos, com pai, nível e caminho; nível 0 marcado como não candidato
"""
import json
import re
import html
import pandas as pd
from bs4 import BeautifulSoup, Comment

BLOCOS = ["p", "div", "li", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6"]


def limpar_html(s) -> str:
    """HTML cru -> texto puro. Preserva quebras de parágrafo, remove lixo do Word."""
    if not isinstance(s, str) or not s.strip():
        return ""
    soup = BeautifulSoup(s, "lxml")
    # comentários condicionais do Word (<!--[if gte mso 9]>...) e blocos sem texto útil
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()
    for tag in soup(["style", "script", "xml"]):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag in soup.find_all(BLOCOS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    texto = html.unescape(soup.get_text()).replace("\xa0", " ")
    linhas = [re.sub(r"[ \t]+", " ", l).strip() for l in texto.splitlines()]
    return "\n".join(l for l in linhas if l)


def processar_questoes(caminho_jsonl: str) -> pd.DataFrame:
    with open(caminho_jsonl, encoding="utf-8") as f:
        df = pd.DataFrame([json.loads(l) for l in f])

    df["enunciado"] = df["enunciado_html"].apply(limpar_html)
    df["comando_limpo"] = df["comando"].apply(limpar_html)

    def fmt_alts(alts):
        if not isinstance(alts, dict):
            return ""
        return "\n".join(f"{k}) {limpar_html(v)}" for k, v in alts.items())

    df["alternativas_texto"] = df["alternativas"].apply(fmt_alts)
    df["n_alternativas"] = df["alternativas"].apply(lambda a: len(a) if isinstance(a, dict) else 0)

    partes = df["enunciado"]
    df["texto_completo"] = (
        "ENUNCIADO:\n" + partes
        + "\n\nCOMANDO:\n" + df["comando_limpo"]
        + "\n\nALTERNATIVAS:\n" + df["alternativas_texto"]
    )
    return df


def processar_temas(caminho_csv: str) -> pd.DataFrame:
    t = pd.read_csv(caminho_csv, dtype={"tema_id": "Int64", "parent_id": "Int64"})
    t = t[t["status"] == "active"].copy()
    # disciplinas (nível 0) são genéricas demais para publicar
    t["candidato"] = t["nivel"] > 0
    # id do tema de nível 1 a que cada nó pertence (útil para "subir" na hierarquia)
    pai = dict(zip(t["tema_id"], t["parent_id"]))
    nivel = dict(zip(t["tema_id"], t["nivel"]))

    def tema_n1(i):
        while nivel.get(i, 0) > 1:
            i = pai[i]
        return i

    t["tema_n1_id"] = t["tema_id"].apply(tema_n1)
    return t


if __name__ == "__main__":
    dq = processar_questoes("data/questoes.jsonl")
    cols = ["id", "disciplina_origem", "assunto_origem", "enunciado", "comando_limpo",
            "alternativas_texto", "n_alternativas", "texto_completo"]
    dq[cols].to_json("data/questoes_proc.jsonl", orient="records", lines=True, force_ascii=False)
    print(f"Questões: {len(dq)} | sem comando: {(dq['comando_limpo'] == '').sum()}")

    dt = processar_temas("data/temas.csv")
    dt.to_csv("data/temas_proc.csv", index=False)
    print(f"Temas ativos: {len(dt)} | candidatos (nível>0): {int(dt['candidato'].sum())}")