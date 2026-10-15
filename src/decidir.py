"""Passo 5: combina as passadas do LLM, decide publicar / revisar / fora e (se houver gabarito) mede a precisão.

Uso básico (gera publicacao.csv):
  python3 src/05_decidir.py
Com a sua planilha rotulada (mede precisão, testa variações da regra, escreve rascunho do previsao.txt):
  python3 src/05_decidir.py --gabarito data/amostra_rotulada.xlsx

REGRA (v1), por questão, usando as passadas B e C (A é opcional: --passada-a):
  fora     : todas as passadas dizem que nenhum tema serve.
  revisar  : (a) passadas discordam sobre fora/não-fora; (b) discordam no TEMA (nível 1, o "pai");
             (c) confiança mínima < --tau; (d) tema em ramo com homônimo (ex.: "Ação Penal" existe em
             Penal e em Processo Penal); (e) menos de --min-passadas passadas disponíveis.
  publicar : o resto. Publica o subtema só se TODAS as passadas apontaram exatamente o mesmo subtema e
             nenhuma marcou dúvida entre subtemas; caso contrário publica o tema PAI (que cobre os filhos
             pela regra do case: acertar pai ou filho conta como certo).
A "confianca" escrita é a MENOR confiança informada pelos modelos (não calibrada).
"""
import argparse
import glob
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

Z = 1.96


def wilson(k: int, n: int):
    """Intervalo de Wilson de 95% para uma proporção k/n."""
    if n == 0:
        return 0.0, 1.0
    p, d = k / n, 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"^\d+\s+", "", s)
    s = re.sub(r"\(parte \d+\)", "", s)
    return re.sub(r"[^a-z ]", "", s).strip()


class Arvore:
    def __init__(self, temas_csv="data/temas_proc.csv", gemeos_csv="data/gemeos.csv"):
        t = pd.read_csv(temas_csv, dtype={"parent_id": "Int64"})
        self.pai = {int(i): (int(p) if pd.notna(p) else None) for i, p in zip(t["tema_id"], t["parent_id"])}
        self.n1 = {int(i): int(n) for i, n in zip(t["tema_id"], t["tema_n1_id"])}
        self.caminho = dict(zip(t["tema_id"].astype(int), t["caminho"]))
        sombra = set(pd.read_csv(gemeos_csv)["tema_id"])
        cand = t[t["candidato"] & ~t["tema_id"].isin(sombra)]
        self.validos = set(int(i) for i in cand["tema_id"])
        grupos = {}
        for r in cand.itertuples():
            grupos.setdefault(norm(r.titulo), []).append(int(r.tema_id))
        # ramos (n1) que têm tema homônimo em outro ramo: publicar neles é arriscado
        self.homonimos_n1 = set()
        for ids in grupos.values():
            if len({self.n1[i] for i in ids}) > 1:
                self.homonimos_n1 |= {self.n1[i] for i in ids}

    def mesmo_ramo(self, a, b):
        """Regra do case: certo = mesmo tema, ou um é pai do outro."""
        return a == b or self.pai.get(a) == b or self.pai.get(b) == a


# ------------------------------------------------------------------ decisão
def decidir(rows, arv: Arvore, tau=0.85, so_pai=False, min_passadas=2):
    """rows: lista de dicts (uma por passada) com fora, tema_central_id, confianca, duvida_subtemas, extras_ids, passada."""
    def sug(rs):  # sugestão = tema da passada mais confiante entre as que não disseram fora
        cand = [r for r in rs if not r["fora"]]
        return max(cand, key=lambda r: r["confianca"])["tema_central_id"] if cand else None

    nomes = ",".join(r["passada"] for r in rows)
    if len(rows) < min_passadas:
        return dict(decisao="revisar", tema=sug(rows), extras=[], conf=min([r["confianca"] for r in rows] or [0]),
                    cat="poucas_passadas", motivo=f"só {len(rows)} passada(s) disponível(is) ({nomes})")
    foras = [r["fora"] for r in rows]
    cmin = min(r["confianca"] for r in rows)
    if all(foras):
        return dict(decisao="fora", tema=None, extras=[], conf=cmin, cat="fora_unanime",
                    motivo="todas as passadas indicam que nenhum tema da lista serve")
    if any(foras):
        return dict(decisao="revisar", tema=sug(rows), extras=[], conf=cmin, cat="fora_discordante",
                    motivo="passadas discordam sobre a questão estar fora da árvore")
    n1s = {arv.n1[r["tema_central_id"]] for r in rows}
    detalhe = "; ".join(f"{r['passada']}: {arv.caminho[r['tema_central_id']]}" for r in rows)
    if len(n1s) > 1:
        return dict(decisao="revisar", tema=sug(rows), extras=[], conf=cmin, cat="discordancia_tema",
                    motivo="passadas discordam no tema - " + detalhe)
    n1 = n1s.pop()
    if n1 in arv.homonimos_n1:
        return dict(decisao="revisar", tema=sug(rows), extras=[], conf=cmin, cat="homonimo",
                    motivo=f"tema com homônimo em outro ramo ({arv.caminho[n1]})")
    if cmin < tau:
        return dict(decisao="revisar", tema=sug(rows), extras=[], conf=cmin, cat="confianca_baixa",
                    motivo=f"passadas concordam, mas confiança mínima {cmin:.2f} < {tau}")
    temas = {r["tema_central_id"] for r in rows}
    exato = len(temas) == 1 and not any(r["duvida_subtemas"] for r in rows) and not so_pai
    tema = temas.pop() if exato else n1
    cont = Counter(i for r in rows for i in set(r["extras_ids"]))
    extras = sorted(i for i, c in cont.items() if c >= min(2, len(rows)) and i != tema)
    return dict(decisao="publicar", tema=tema, extras=extras, conf=cmin,
                cat="publicar_subtema" if (exato and arv.pai.get(tema) is not None) else "publicar_tema",
                motivo="passadas concordam: " + arv.caminho[tema])


# ------------------------------------------------------------------ dados
def ler_passada(caminho, nome):
    df = pd.read_json(caminho, lines=True)
    df["passada"] = nome
    return df.set_index("id", drop=False)


def linhas_por_id(passadas: dict, ids):
    return {qid: [{**p.loc[qid].to_dict(), "passada": nome} for nome, p in passadas.items() if qid in p.index]
            for qid in ids}


def decidir_todas(por_id, arv, **kw):
    return {qid: decidir(rows, arv, **kw) for qid, rows in por_id.items()}


# ------------------------------------------------------------------ avaliação contra o seu gabarito
def ler_gabarito(caminho, arv):
    g = pd.read_excel(caminho, sheet_name="Rotulagem")
    g["gab_fora"] = g["fora"].astype(str).str.strip().str.lower().eq("sim")
    g["gab_tema"] = pd.to_numeric(g["tema_id"], errors="coerce")
    g = g[g["gab_fora"] | g["gab_tema"].notna()].copy()
    g["gab_tema"] = g["gab_tema"].apply(lambda x: int(x) if pd.notna(x) else None)
    ruim = g[(~g["gab_fora"]) & (~g["gab_tema"].isin(set(arv.pai)))]
    if len(ruim):
        print(f"ATENÇÃO: {len(ruim)} rótulo(s) com tema_id inexistente: {list(ruim['id'])}")
    return g.set_index("id", drop=False)


def acertou(tema, fora_pred, gab, arv):
    """tema=None e fora_pred=True significa que a decisão foi 'fora'."""
    if gab["gab_fora"]:
        return fora_pred
    return (not fora_pred) and tema is not None and arv.mesmo_ramo(int(tema), int(gab["gab_tema"]))


def avaliar(por_id, gab, arv, params):
    linhas = []
    for tau in (0.0, 0.8, 0.85, 0.9, 0.95):
        for so_pai in (False, True):
            dec = decidir_todas(por_id, arv, tau=tau, so_pai=so_pai, min_passadas=params["min_passadas"])
            pub = [q for q, d in dec.items() if d["decisao"] == "publicar"]
            amostra_pub = [q for q in pub if q in gab.index]
            k = sum(acertou(dec[q]["tema"], False, gab.loc[q], arv) for q in amostra_pub)
            lo, hi = wilson(k, len(amostra_pub))
            linhas.append(dict(tau=tau, so_pai=so_pai, cobertura_359=len(pub) / len(dec), n_pub_amostra=len(amostra_pub),
                               acertos=k, precisao=(k / len(amostra_pub)) if amostra_pub else float("nan"), ic_inf=lo, ic_sup=hi))
    return pd.DataFrame(linhas)


def relatorio_gabarito(por_id, dec, gab, arv, params):
    ids_g = [q for q in gab.index if q in dec]
    out = [f"# Avaliação na amostra rotulada ({len(ids_g)} questões)\n"]
    # 1) cada passada sozinha
    out.append("## Acerto de cada passada sozinha (tema direto, pai/filho conta)\n")
    for nome in sorted({r["passada"] for rs in por_id.values() for r in rs}):
        ok = n = 0
        for q in ids_g:
            r = next((x for x in por_id[q] if x["passada"] == nome), None)
            if r is None:
                continue
            n += 1
            ok += acertou(None if r["fora"] else r["tema_central_id"], r["fora"], gab.loc[q], arv)
        lo, hi = wilson(ok, n)
        out.append(f"- passada {nome}: {ok}/{n} = {ok / max(n, 1):.1%} (IC95% {lo:.1%}-{hi:.1%})")
    # 2) regra escolhida
    pub = [q for q in ids_g if dec[q]["decisao"] == "publicar"]
    k = sum(acertou(dec[q]["tema"], False, gab.loc[q], arv) for q in pub)
    lo, hi = wilson(k, len(pub))
    out.append(f"\n## Regra escolhida (tau={params['tau']}, so_pai={params['so_pai']})\n")
    out.append(f"- publicadas na amostra: {len(pub)}/{len(ids_g)} ({len(pub) / max(len(ids_g), 1):.1%})")
    out.append(f"- precisão nas publicadas: {k}/{len(pub)} = {k / max(len(pub), 1):.1%} (IC95% {lo:.1%}-{hi:.1%})")
    erradas = [q for q in pub if not acertou(dec[q]["tema"], False, gab.loc[q], arv)]
    out.append(f"- publicadas ERRADAS (leia uma a uma!): {erradas}")
    rev = [q for q in ids_g if dec[q]["decisao"] == "revisar"]
    k2 = sum(acertou(dec[q]["tema"], False, gab.loc[q], arv) for q in rev)
    out.append(f"- em revisar: {len(rev)} questões; a sugestão estaria certa em {k2} (custo de revisar o que já estava certo)")
    fora = [q for q in ids_g if dec[q]["decisao"] == "fora"]
    out.append(f"- em fora: {len(fora)}; certas: {sum(acertou(None, True, gab.loc[q], arv) for q in fora)}")
    # 3) por certeza do rotulador
    if "certeza" in gab.columns:
        out.append("\n## Publicadas por certeza do rotulador\n")
        for c in ("alta", "media", "baixa"):
            qs = [q for q in pub if str(gab.loc[q, "certeza"]).strip().lower() == c]
            if qs:
                kk = sum(acertou(dec[q]["tema"], False, gab.loc[q], arv) for q in qs)
                out.append(f"- certeza {c}: {kk}/{len(qs)}")
    return "\n".join(out), (k, len(pub))


# ------------------------------------------------------------------ main
def achar(padrao, nome):
    c = [p for p in glob.glob(padrao) if "_dry" not in p]
    if len(c) != 1:
        raise SystemExit(f"Não achei exatamente 1 arquivo da passada {nome} com '{padrao}' (achei {c}). Use --passada-{nome.lower()}.")
    return c[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--passada-b", default=None)
    ap.add_argument("--passada-c", default=None)
    ap.add_argument("--passada-a", default=None, help="opcional (ex.: a parcial do gemini-3.8-flash)")
    ap.add_argument("--tau", type=float, default=0.85, help="confiança mínima (menor entre as passadas) para publicar")
    ap.add_argument("--so-pai", action="store_true", help="publica sempre o tema pai (mais seguro, menos específico)")
    ap.add_argument("--min-passadas", type=int, default=2)
    ap.add_argument("--gabarito", default=None, help="planilha rotulada (xlsx) para medir precisão")
    ap.add_argument("--saida", default="data")
    a = ap.parse_args(argv)

    arv = Arvore()
    passadas = {"B": ler_passada(a.passada_b or achar("data/resultados_*_B.jsonl", "B"), "B"),
                "C": ler_passada(a.passada_c or achar("data/resultados_*_C.jsonl", "C"), "C")}
    if a.passada_a:
        passadas["A"] = ler_passada(a.passada_a, "A")
    ids = list(pd.read_json("data/questoes_proc.jsonl", lines=True)["id"])
    for nome, p in passadas.items():
        print(f"passada {nome}: {len(p)} questões" + ("" if len(p) == len(ids) else "  <-- INCOMPLETA"))

    por_id = linhas_por_id(passadas, ids)
    params = dict(tau=a.tau, so_pai=a.so_pai, min_passadas=a.min_passadas)
    dec = decidir_todas(por_id, arv, **params)

    # publicacao.csv: exatamente as 6 colunas do LEIA-ME
    pub = pd.DataFrame([dict(
        id=q, decisao=d["decisao"],
        tema_central_id="" if d["tema"] is None else int(d["tema"]),
        extras_ids=";".join(map(str, d["extras"])), confianca=round(float(d["conf"]), 3), motivo=d["motivo"][:240])
        for q, d in dec.items()])
    saida = Path(a.saida)
    pub.to_csv(saida / "publicacao.csv", index=False, encoding="utf-8")
    det = pd.DataFrame([dict(id=q, categoria=d["cat"], **{f"pred_{r['passada']}": ("fora" if r["fora"] else r["tema_central_id"]) for r in por_id[q]},
                             **{f"conf_{r['passada']}": r["confianca"] for r in por_id[q]}) for q, d in dec.items()])
    det.to_csv(saida / "decisoes_detalhe.csv", index=False, encoding="utf-8")

    cont = Counter(d["decisao"] for d in dec.values())
    n = len(dec)
    print(f"\npublicar: {cont['publicar']} ({cont['publicar'] / n:.1%}) | revisar: {cont['revisar']} | fora: {cont['fora']}")
    print("motivos:", dict(Counter(d["cat"] for d in dec.values())))
    b, c = passadas["B"], passadas["C"]
    comuns = [q for q in ids if q in b.index and q in c.index]
    ig = sum(b.loc[q, "tema_central_id"] == c.loc[q, "tema_central_id"] for q in comuns)
    ig1 = sum(arv.n1.get(int(b.loc[q, "tema_central_id"]), -1) == arv.n1.get(int(c.loc[q, "tema_central_id"]), -2) for q in comuns)
    print(f"concordância B x C: mesmo tema exato {ig}/{len(comuns)} ({ig / max(len(comuns), 1):.1%}); mesmo tema pai {ig1}/{len(comuns)} ({ig1 / max(len(comuns), 1):.1%})")
    print(f"Gravado {saida / 'publicacao.csv'} e {saida / 'decisoes_detalhe.csv'}")

    if a.gabarito:
        gab = ler_gabarito(a.gabarito, arv)
        texto, (k, npub) = relatorio_gabarito(por_id, dec, gab, arv, params)
        grade = avaliar(por_id, gab, arv, params)
        texto += "\n\n## Variações da regra (precisão medida na amostra; cobertura em todas as 359)\n\n"
        texto += grade.to_markdown(index=False, floatfmt=".3f") if _tem_tabulate() else grade.round(3).to_string(index=False)
        texto += ("\n\nCUIDADO: se você escolher a regra olhando esta tabela, a precisão da regra escolhida fica otimista "
                  "(escolheu-se o melhor resultado na própria amostra). Registre isso no previsao.txt.")
        (saida / "avaliacao_amostra.md").write_text(texto, encoding="utf-8")
        print("\n" + texto)
        lo, hi = wilson(k, npub)
        cob = cont["publicar"] / n
        (saida / "previsao_rascunho.txt").write_text(
            f"precisao_prevista: {k / max(npub, 1):.2f}\nprecisao_intervalo: {lo:.2f}-{hi:.2f}\n"
            f"cobertura_prevista: {cob:.2f}\ncobertura_intervalo: {cob:.2f}-{cob:.2f}\n\n"
            "RASCUNHO - reescreva com as suas palavras. Pontos que precisam estar no texto:\n"
            f"- precisão: {k}/{npub} acertos nas publicadas da amostra rotulada à mão ({a.gabarito}); intervalo = Wilson 95%.\n"
            "- esse número é otimista se a regra (tau, so_pai) foi escolhida olhando a própria amostra; e o rótulo é seu, não da equipe.\n"
            "- cobertura: é a fração que o meu pipeline publica nas 359; não tem incerteza (é contagem), por isso o intervalo é degenerado.\n",
            encoding="utf-8")
        print(f"\nGravado {saida / 'avaliacao_amostra.md'} e {saida / 'previsao_rascunho.txt'}")


def _tem_tabulate():
    try:
        import tabulate  # noqa
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()