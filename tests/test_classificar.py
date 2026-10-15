"""Testes do passo 3 (rodam offline, sem gastar cota e sem chave de API).

    python3 tests/test_03_classificar.py
    python3 -m pytest tests/test_03_classificar.py -q
"""
import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path

spec = importlib.util.spec_from_file_location("c3", next(p for p in ("src/03_classificar.py", "src/classificar.py") if __import__("os").path.exists(p)))
c3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c3)

q, cand = c3.carregar_dados()
VALIDOS = set(int(i) for i in cand["tema_id"])
IDS = list(q["id"][:3])
TEMA_OK = sorted(VALIDOS)[10]


def item(qid, **kw):
    base = dict(id=qid, fora=False, tema_central_id=TEMA_OK, extras_ids=[], alternativa_tema_id=0,
                duvida_subtemas=False, confianca=0.9, motivo="x")
    base.update(kw)
    return base


def lote(itens):
    return json.dumps({"questoes": itens})


def test_resposta_valida_passa():
    itens, erro = c3.validar_lote(lote([item(i) for i in IDS]), IDS, VALIDOS)
    assert erro is None and len(itens) == 3


def test_id_faltando_e_rejeitado():
    _, erro = c3.validar_lote(lote([item(i) for i in IDS[:2]]), IDS, VALIDOS)
    assert erro and "IDs" in erro


def test_id_repetido_e_rejeitado():
    _, erro = c3.validar_lote(lote([item(IDS[0]), item(IDS[0]), item(IDS[1])]), IDS, VALIDOS)
    assert erro


def test_tema_inexistente_inativo_ou_sombra_e_rejeitado():
    sombra = int(__import__("pandas").read_csv("data/gemeos.csv")["tema_id"].iloc[0])
    for ruim in (999999, sombra, 4365):  # inexistente, nó-sombra, disciplina (nível 0)
        _, erro = c3.validar_lote(lote([item(i, tema_central_id=ruim) for i in IDS]), IDS, VALIDOS)
        assert erro, f"tema {ruim} deveria ser rejeitado"


def test_fora_exige_tema_zero():
    _, erro = c3.validar_lote(lote([item(i, fora=True) for i in IDS]), IDS, VALIDOS)
    assert erro
    itens, erro = c3.validar_lote(lote([item(i, fora=True, tema_central_id=0) for i in IDS]), IDS, VALIDOS)
    assert erro is None


def test_confianca_fora_do_intervalo_e_rejeitada():
    _, erro = c3.validar_lote(lote([item(i, confianca=1.5) for i in IDS]), IDS, VALIDOS)
    assert erro


def test_json_quebrado_e_campo_ausente():
    _, e1 = c3.validar_lote("isso não é json", IDS, VALIDOS)
    sem_motivo = [{k: v for k, v in item(i).items() if k != "motivo"} for i in IDS]
    _, e2 = c3.validar_lote(lote(sem_motivo), IDS, VALIDOS)
    assert e1 and e2


def test_cota_diaria_para_e_retoma():
    """Com --rpd 2 devem ocorrer exatamente 2 chamadas; a 2ª execução não passa disso no mesmo dia."""
    chamadas = []

    def fake(modelo, prompt):
        chamadas.append(1)
        ids = re.findall(r"^### (Q\d+)", prompt, re.M)
        return lote([item(i) for i in ids])

    c3.chamar_gemini = fake
    c3.time.sleep = lambda s: None
    with tempfile.TemporaryDirectory() as tmp:
        c3.CACHE = Path(tmp)
        argv = ["x", "--modelo", "fake", "--passada", "Z", "--rpd", "2"]
        sys.argv = argv
        c3.main()
        assert len(chamadas) == 2, f"esperava 2 chamadas, houve {len(chamadas)}"
        sys.argv = argv
        c3.main()  # mesmo dia: cota já esgotada
        assert len(chamadas) == 2, "gastou cota além do limite"
        feitos = list((Path(tmp) / "fake_Z").glob("lote_*.json"))
        assert len(feitos) == 2
        # chamada que falha também conta na cota
        assert c3.chamadas_hoje("fake") == 2
    Path("data/resultados_fake_Z.jsonl").unlink(missing_ok=True)  # limpa o consolidado do teste


def _rodar(fake, extra, passada):
    c3.chamar_gemini = fake
    c3.time.sleep = lambda s: None
    with tempfile.TemporaryDirectory() as tmp:
        c3.CACHE = Path(tmp)
        sys.argv = ["x", "--modelo", "fake", "--passada", passada] + extra
        c3.main()
        feitos = len(list((Path(tmp) / f"fake_{passada}").glob("lote_*.json")))
        gasto = c3.chamadas_hoje("fake")
    Path(f"data/resultados_fake_{passada}.jsonl").unlink(missing_ok=True)
    return feitos, gasto


def test_503_tenta_o_mesmo_lote_de_novo():
    n = []

    def fake(modelo, prompt):
        n.append(1)
        if len(n) == 1:
            raise RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand")
        ids = re.findall(r"^### (Q\d+)", prompt, re.M)
        return lote([item(i) for i in ids])

    feitos, gasto = _rodar(fake, ["--rpd", "5", "--max-chamadas", "1"], "R")
    assert feitos == 1 and len(n) == 2, f"lotes {feitos}, chamadas {len(n)}"


def test_429_para_tudo():
    n = []

    def fake(modelo, prompt):
        n.append(1)
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    feitos, gasto = _rodar(fake, ["--rpd", "10"], "E")
    assert feitos == 0 and len(n) == 1, "deveria parar na primeira resposta 429"


def test_ajustar_contador():
    chamadas = []

    def fake(modelo, prompt):
        chamadas.append(1)
        ids = re.findall(r"^### (Q\d+)", prompt, re.M)
        return lote([item(i) for i in ids])

    feitos, gasto = _rodar(fake, ["--rpd", "3", "--ajustar-contador", "2"], "C")
    assert len(chamadas) == 1 and gasto == 3, "com 2 gastos e rpd=3, só 1 chamada cabe"


def test_chamada_pendurada_vira_timeout_e_tenta_de_novo():
    import threading
    n = []

    def fake(modelo, prompt):
        n.append(1)
        if len(n) == 1:
            threading.Event().wait(2)  # simula o servidor que nunca responde
        ids = re.findall(r"^### (Q\d+)", prompt, re.M)
        return lote([item(i) for i in ids])

    c3.TIMEOUT_CHAMADA = 0.3
    try:
        feitos, gasto = _rodar(fake, ["--rpd", "5", "--max-chamadas", "1"], "T")
    finally:
        c3.TIMEOUT_CHAMADA = 180
    assert feitos == 1 and len(n) == 2, f"lotes {feitos}, chamadas {len(n)}"


def test_extras_invalidos_sao_removidos_sem_perder_o_lote():
    avisos = []
    itens, erro = c3.validar_lote(
        lote([item(IDS[0], extras_ids=[484, TEMA_OK]), item(IDS[1], alternativa_tema_id=484), item(IDS[2])]),
        IDS, VALIDOS, avisos)
    assert erro is None
    assert itens[0].extras_ids == [], "484 (inexistente) e o próprio tema central deveriam sair"
    assert itens[1].alternativa_tema_id == 0
    assert len(avisos) == 2


def test_tema_central_invalido_continua_derrubando_o_lote():
    _, erro = c3.validar_lote(lote([item(IDS[0], tema_central_id=484), item(IDS[1]), item(IDS[2])]), IDS, VALIDOS, [])
    assert erro


def test_raw_antigo_e_recuperado_sem_chamar_a_api():
    chamadas = []

    def fake(modelo, prompt):
        chamadas.append(1)
        ids = re.findall(r"^### (Q\d+)", prompt, re.M)
        return lote([item(i) for i in ids])

    c3.chamar_gemini = fake
    c3.time.sleep = lambda s: None
    ids1 = list(q["id"][:10])
    with tempfile.TemporaryDirectory() as tmp:
        c3.CACHE = Path(tmp)
        pasta = Path(tmp) / "fake_V"
        pasta.mkdir(parents=True)
        chave = c3.chave_lote("fake", "V", ids1)
        # resposta crua que antes foi rejeitada por causa de um extras_ids inexistente
        (pasta / f"lote_{chave}.raw.txt").write_text(lote([item(i, extras_ids=[484]) for i in ids1]), encoding="utf-8")
        sys.argv = ["x", "--modelo", "fake", "--passada", "V", "--rpd", "1", "--max-chamadas", "0"]
        c3.main()
        assert (pasta / f"lote_{chave}.json").exists(), "deveria ter recuperado o lote do .raw.txt"
        assert len(chamadas) == 0, "recuperar do raw não pode gastar chamada"
    Path("data/resultados_fake_V.jsonl").unlink(missing_ok=True)


if __name__ == "__main__":
    testes = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in testes:
        t(); print(f"ok  {t.__name__}")
    print(f"\n{len(testes)} testes passaram")