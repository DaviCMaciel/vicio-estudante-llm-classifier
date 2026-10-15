"""Testes do passo 5 (offline). Rode da raiz do projeto:

    python3 tests/test_05_decidir.py
    python3 -m pytest tests/test_05_decidir.py -q
Precisam de: data/temas_proc.csv, data/gemeos.csv, data/questoes_proc.jsonl, data/candidatos.jsonl.
O teste monta a própria planilha de gabarito sintética: não lê nem altera a sua planilha real.
"""
import csv
import importlib.util
import json
import tempfile
from pathlib import Path

import pandas as pd
from openpyxl import Workbook

spec = importlib.util.spec_from_file_location(
    "d5", next(p for p in ("src/05_decidir.py", "src/decidir.py") if Path(p).exists()))
d5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d5)
ARV = d5.Arvore()

# nós reais da árvore para montar casos
filhos = {}
for i, p in ARV.pai.items():
    if p is not None and ARV.pai.get(p) is not None and i in ARV.validos and p in ARV.validos and p not in ARV.homonimos_n1:
        filhos.setdefault(p, []).append(i)
PAI = next(p for p, f in filhos.items() if len(f) >= 2)
F1, F2 = filhos[PAI][:2]
OUTRO_N1 = next(p for p in filhos if p != PAI)
OUTRO_F = filhos[OUTRO_N1][0]
HOMO = sorted(ARV.homonimos_n1)[0]


def r(nome, tema, conf=0.9, fora=False, duvida=False, extras=()):
    return dict(passada=nome, tema_central_id=0 if fora else tema, confianca=conf, fora=fora,
                duvida_subtemas=duvida, extras_ids=list(extras))


def test_wilson_e_ramo():
    lo, hi = d5.wilson(38, 40)
    assert abs(lo - 0.835) < 0.003 and abs(hi - 0.986) < 0.003
    assert d5.wilson(0, 0) == (0.0, 1.0)
    assert ARV.mesmo_ramo(F1, PAI) and ARV.mesmo_ramo(PAI, F1) and ARV.mesmo_ramo(F1, F1)
    assert not ARV.mesmo_ramo(F1, F2), "subtemas irmãos NÃO contam como mesmo ramo"


def test_concordam_exato_publica_subtema():
    d = d5.decidir([r("B", F1), r("C", F1)], ARV)
    assert d["decisao"] == "publicar" and d["tema"] == F1 and d["cat"] == "publicar_subtema"


def test_irmaos_publica_pai():
    d = d5.decidir([r("B", F1), r("C", F2)], ARV)
    assert d["decisao"] == "publicar" and d["tema"] == PAI


def test_pai_e_filho_publica_pai():
    d = d5.decidir([r("B", PAI), r("C", F1)], ARV)
    assert d["decisao"] == "publicar" and d["tema"] == PAI


def test_duvida_de_subtema_sobe_para_pai():
    d = d5.decidir([r("B", F1, duvida=True), r("C", F1)], ARV)
    assert d["tema"] == PAI


def test_so_pai_forca_o_pai():
    d = d5.decidir([r("B", F1), r("C", F1)], ARV, so_pai=True)
    assert d["tema"] == PAI


def test_discordancia_de_tema_vai_para_revisar_com_sugestao():
    d = d5.decidir([r("B", F1, conf=0.8), r("C", OUTRO_F, conf=0.95)], ARV)
    assert d["decisao"] == "revisar" and d["cat"] == "discordancia_tema"
    assert d["tema"] == OUTRO_F, "sugestão deve vir da passada mais confiante"
    assert "B:" in d["motivo"] and "C:" in d["motivo"]


def test_confianca_baixa_vai_para_revisar():
    d = d5.decidir([r("B", F1, conf=0.7), r("C", F1, conf=0.95)], ARV, tau=0.85)
    assert d["decisao"] == "revisar" and d["cat"] == "confianca_baixa" and d["tema"] == F1


def test_homonimo_vai_para_revisar():
    d = d5.decidir([r("B", HOMO), r("C", HOMO)], ARV)
    assert d["decisao"] == "revisar" and d["cat"] == "homonimo"


def test_fora_unanime_e_discordante():
    d = d5.decidir([r("B", 0, fora=True), r("C", 0, fora=True)], ARV)
    assert d["decisao"] == "fora" and d["tema"] is None
    d = d5.decidir([r("B", 0, fora=True), r("C", F1)], ARV)
    assert d["decisao"] == "revisar" and d["tema"] == F1


def test_poucas_passadas_nao_publica():
    d = d5.decidir([r("B", F1)], ARV, min_passadas=2)
    assert d["decisao"] == "revisar" and d["cat"] == "poucas_passadas"


def test_extras_so_com_apoio_de_duas_passadas():
    d = d5.decidir([r("B", F1, extras=[OUTRO_F, F2]), r("C", F1, extras=[OUTRO_F])], ARV)
    assert d["extras"] == [OUTRO_F]


# ---------------- ponta a ponta com dados sintéticos
def _sintetico(tmp: Path):
    cand = [json.loads(l) for l in open("data/candidatos.jsonl", encoding="utf-8")]
    linhas_b, linhas_c = [], []
    for i, c in enumerate(cand):
        t1, t2 = c["candidatos"][0]["tema_id"], c["candidatos"][1]["tema_id"]
        base = dict(extras_ids=[], alternativa_tema_id=0, duvida_subtemas=False, motivo="x")
        b = dict(id=c["id"], fora=(i % 37 == 0), tema_central_id=0 if i % 37 == 0 else t1, confianca=0.9, **base)
        cc = dict(id=c["id"], fora=(i % 37 == 0), tema_central_id=0 if i % 37 == 0 else (t2 if i % 5 == 0 else t1),
                  confianca=0.7 if i % 7 == 0 else 0.92, **base)
        linhas_b.append(b); linhas_c.append(cc)
    for nome, ls in (("B", linhas_b), ("C", linhas_c)):
        (tmp / f"res_{nome}.jsonl").write_text("\n".join(json.dumps(l) for l in ls) + "\n", encoding="utf-8")
    return linhas_b


def test_ponta_a_ponta_publicacao_csv_no_formato_do_case():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _sintetico(tmp)
        d5.main(["--passada-b", str(tmp / "res_B.jsonl"), "--passada-c", str(tmp / "res_C.jsonl"), "--saida", str(tmp)])
        df = pd.read_csv(tmp / "publicacao.csv", dtype=str, keep_default_na=False)
    assert list(df.columns) == ["id", "decisao", "tema_central_id", "extras_ids", "confianca", "motivo"]
    ids = [json.loads(l)["id"] for l in open("data/questoes_proc.jsonl", encoding="utf-8")]
    assert list(df["id"]) == ids and len(df) == 359
    assert set(df["decisao"]) <= {"publicar", "revisar", "fora"}
    for _, x in df.iterrows():
        if x["decisao"] == "fora":
            assert x["tema_central_id"] == "" and x["motivo"], x["id"]
        else:
            assert int(x["tema_central_id"]) in ARV.validos, f"{x['id']}: tema inválido/inativo/sombra"
        if x["decisao"] == "revisar":
            assert x["motivo"], x["id"]
        assert 0 <= float(x["confianca"]) <= 1
    assert (df["decisao"] == "publicar").sum() > 0 and (df["decisao"] == "revisar").sum() > 0


def test_ponta_a_ponta_com_gabarito_confere_a_conta():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        b = _sintetico(tmp)
        pred_b = {x["id"]: x for x in b}
        # planilha de gabarito SINTÉTICA, com o mesmo formato da aba "Rotulagem": 30 linhas iguais à predição de B,
        # 4 linhas com tema de OUTRO ramo; onde B disse "fora", o gabarito também é fora
        ids = [json.loads(l)["id"] for l in open("data/questoes_proc.jsonl", encoding="utf-8")][:34]
        wb = Workbook(); ws = wb.active; ws.title = "Rotulagem"
        ws.append(["id", "Questão", "Alternativas", "tema_id", "Caminho", "tema_alternativo_id", "extras_ids", "certeza", "fora", "observação"])
        esperado_b = 0
        for k, qid in enumerate(ids):
            p = pred_b[qid]
            tema, fora = None, None
            if p["fora"]:
                fora = "sim"; esperado_b += 1                     # B disse fora e o gabarito é fora: certo
            elif k >= 30:
                tema = next(i for i in sorted(ARV.validos) if ARV.n1[i] != ARV.n1[p["tema_central_id"]])
            else:
                tema = p["tema_central_id"]; esperado_b += 1
            ws.append([qid, "", "", tema, "", None, None, "alta", fora, None])
        gx = tmp / "gab.xlsx"; wb.save(gx)
        d5.main(["--passada-b", str(tmp / "res_B.jsonl"), "--passada-c", str(tmp / "res_C.jsonl"),
                 "--saida", str(tmp), "--gabarito", str(gx)])
        md = (tmp / "avaliacao_amostra.md").read_text(encoding="utf-8")
        prev = (tmp / "previsao_rascunho.txt").read_text(encoding="utf-8")
    assert "Avaliação na amostra rotulada (34 questões)" in md
    assert f"- passada B: {esperado_b}/34" in md, f"esperava {esperado_b}/34 em:\n{md[:600]}"
    assert "Variações da regra" in md and "cobertura_359" in md
    assert all(k in prev for k in ("precisao_prevista:", "precisao_intervalo:", "cobertura_prevista:", "cobertura_intervalo:"))


if __name__ == "__main__":
    ts = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in ts:
        t(); print("ok ", t.__name__)
    print(f"\n{len(ts)} testes passaram")