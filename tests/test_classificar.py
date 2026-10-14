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

spec = importlib.util.spec_from_file_location("c3", "src/classificar.py")
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


if __name__ == "__main__":
    testes = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in testes:
        t(); print(f"ok  {t.__name__}")
    print(f"\n{len(testes)} testes passaram")