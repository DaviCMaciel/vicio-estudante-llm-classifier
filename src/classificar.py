"""Passo 3: classifica as questões com um LLM, em lotes, protegendo a cota diária.

Uso:
  python3 src/03_classificar.py --modelo gemini-2.5-flash --passada A --dry-run      # offline, não gasta cota
  python3 src/03_classificar.py --modelo gemini-2.5-flash --passada A --max-chamadas 1 # 1 chamada real (teste)
  python3 src/03_classificar.py --modelo gemini-2.5-flash --passada A                 # roda até o limite do dia
  python3 src/03_classificar.py --status

Chave: variável de ambiente GEMINI_API_KEY (só é lida fora do --dry-run).
Saída: data/resultados_<modelo>_<passada>.jsonl  (+ cache por lote em data/cache_llm/)
"""
import argparse
import functools
import hashlib
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

print = functools.partial(print, flush=True)  # mostra cada linha na hora, mesmo sem quebra de linha

PROMPT_VERSAO = "v1"
TAM_LOTE = 10
RPD_PADRAO = 20          # requisições/dia do plano gratuito (conferido no AI Studio)
RPM_PADRAO = 5           # requisições/minuto do gemini-2.5-flash (flash-lite: 10)
RETRIES_INDISPONIVEL = 2  # tentativas extras quando a API responde 503 (sobrecarga)
TIMEOUT_CHAMADA = 180      # segundos: se o modelo não responder, desiste e trata como falha
ESPERA_503 = 45            # segundos de espera antes de repetir
MAX_CHARS_QUESTAO = 2500  # corta questões gigantes para controlar tokens
CACHE = Path("data/cache_llm")


# ------------------------------------------------------------------ schema de saída
class ResultadoItem(BaseModel):
    id: str = Field(description="ID exato da questão, como no lote (ex: Q0001)")
    fora: bool = Field(description="true se NENHUM tema da lista serve para a questão")
    tema_central_id: int = Field(description="ID do tema central da lista; 0 se fora=true")
    extras_ids: List[int] = Field(description="IDs de outros temas que a questão também cobre (até 3); lista vazia se nenhum")
    alternativa_tema_id: int = Field(description="ID do 2º melhor tema central; 0 se não houver dúvida relevante")
    duvida_subtemas: bool = Field(description="true se há dúvida entre subtemas irmãos do mesmo tema")
    confianca: float = Field(description="Probabilidade (0 a 1) de o tema central estar certo")
    motivo: str = Field(description="Justificativa jurídica em UMA frase")


class LoteClassificacao(BaseModel):
    questoes: List[ResultadoItem]


# ------------------------------------------------------------------ dados e prompt
SISTEMA = """Você é um professor especialista em concursos jurídicos brasileiros (Direito Penal, Processo Penal, \
Legislação Penal Especial e Criminologia). Sua tarefa é classificar questões na árvore de temas fornecida.

REGRAS
1. Escolha UM tema central por questão, usando somente IDs que aparecem na lista de temas.
2. Classifique pelo assunto principal que o COMANDO cobra, não pelo cenário do caso hipotético.
3. Use o subtema mais específico apenas quando o conteúdo o identifica com clareza. Se a questão cobre
   vários subtemas do mesmo tema, ou você hesita entre subtemas irmãos, escolha o TEMA PAI e marque
   duvida_subtemas=true.
4. Se a matéria da questão não existe na lista (ex.: Direito Constitucional, Tributário, Civil),
   marque fora=true e use tema_central_id=0.
5. extras_ids: outros temas da lista que a questão também cobre de forma relevante (até 3).
6. confianca: seja calibrado. Use 0.95 ou mais só quando o tema for inequívoco; questões curtas,
   genéricas ou que misturam matérias merecem valores bem menores.
7. alternativa_tema_id: o segundo melhor tema central, ou 0 se não houver dúvida relevante.
8. Devolva exatamente uma entrada por questão do lote, com o mesmo id. Não invente IDs."""


def carregar_dados():
    q = pd.read_json("data/questoes_proc.jsonl", lines=True)
    t = pd.read_csv("data/temas_proc.csv", dtype={"parent_id": "Int64"})
    sombra = set(pd.read_csv("data/gemeos.csv")["tema_id"])
    cand = t[t["candidato"] & ~t["tema_id"].isin(sombra)]
    return q, cand


def texto_arvore(cand) -> str:
    return "\n".join(f"[{r.tema_id}] {r.caminho}" for r in cand.itertuples())


def texto_questao(r, com_origem: bool) -> str:
    corpo = r["texto_completo"][:MAX_CHARS_QUESTAO]
    pista = f"\n(pista, pode estar errada: origem = {r['disciplina_origem']} / {r['assunto_origem']})" if com_origem else ""
    return f"### {r['id']}\n{corpo}{pista}"


def montar_prompt(lote_df, arvore: str, com_origem: bool) -> str:
    questoes = "\n\n".join(texto_questao(r, com_origem) for _, r in lote_df.iterrows())
    return f"TEMAS DISPONÍVEIS (formato [id] caminho):\n{arvore}\n\nQUESTÕES DO LOTE:\n\n{questoes}"


# ------------------------------------------------------------------ validação
def validar_lote(texto_json: str, ids_esperados: list, ids_validos: set, avisos=None):
    """Devolve (lista_de_ResultadoItem, None) se ok, ou (None, motivo_do_erro).

    tema_central_id inválido invalida o lote (é o dado que importa). extras_ids e alternativa_tema_id
    são opcionais: IDs inexistentes são REMOVIDOS e registrados em `avisos`, em vez de jogar fora o lote.
    """
    try:
        lote = LoteClassificacao.model_validate_json(texto_json)
    except ValidationError as e:
        return None, f"JSON fora do schema: {str(e)[:200]}"
    ids_resp = [x.id for x in lote.questoes]
    if sorted(ids_resp) != sorted(ids_esperados):
        faltam, sobram = set(ids_esperados) - set(ids_resp), set(ids_resp) - set(ids_esperados)
        return None, f"IDs diferentes do lote (faltam {sorted(faltam)}, sobram {sorted(sobram)}, ou repetidos)"
    for x in lote.questoes:
        if not 0 <= x.confianca <= 1:
            return None, f"{x.id}: confianca fora de [0,1]"
        if x.fora:
            if x.tema_central_id != 0:
                return None, f"{x.id}: fora=true exige tema_central_id=0"
        elif x.tema_central_id not in ids_validos:
            return None, f"{x.id}: tema_central_id {x.tema_central_id} não existe na lista (ou é inativo/sombra)"
        ruins = [i for i in x.extras_ids if i not in ids_validos]
        if ruins and avisos is not None:
            avisos.append(f"{x.id}: extras_ids inexistentes removidos {ruins}")
        x.extras_ids = list(dict.fromkeys(i for i in x.extras_ids if i in ids_validos and i != x.tema_central_id))
        if x.alternativa_tema_id not in ids_validos and x.alternativa_tema_id != 0:
            if avisos is not None:
                avisos.append(f"{x.id}: alternativa_tema_id inexistente removido {x.alternativa_tema_id}")
            x.alternativa_tema_id = 0
    return lote.questoes, None


# ------------------------------------------------------------------ cota diária
def hoje_pacifico() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")


def chamadas_hoje(modelo: str) -> int:
    f = CACHE / "contador.json"
    return json.loads(f.read_text()).get(hoje_pacifico(), {}).get(modelo, 0) if f.exists() else 0


def registrar_chamada(modelo: str):
    f = CACHE / "contador.json"
    d = json.loads(f.read_text()) if f.exists() else {}
    dia = d.setdefault(hoje_pacifico(), {})
    dia[modelo] = dia.get(modelo, 0) + 1
    f.write_text(json.dumps(d, indent=1))


# ------------------------------------------------------------------ chamada ao modelo
def chamar_gemini(modelo: str, prompt: str) -> str:
    from google import genai  # import tardio: o --dry-run não precisa do SDK
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.generate_content(
        model=modelo, contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SISTEMA, temperature=0,
            response_mime_type="application/json", response_schema=LoteClassificacao))
    return resp.text


def chamar_com_timeout(modelo: str, prompt: str, segundos=None) -> str:
    """Roda chamar_gemini numa thread e desiste se passar do tempo limite (o SDK pode ficar pendurado)."""
    segundos = segundos or TIMEOUT_CHAMADA
    res = {}

    def alvo():
        try:
            res["ok"] = chamar_gemini(modelo, prompt)
        except Exception as e:  # noqa
            res["erro"] = e

    th = threading.Thread(target=alvo, daemon=True)
    th.start()
    th.join(segundos)
    if th.is_alive():
        raise TimeoutError(f"timeout: sem resposta em {segundos}s")
    if "erro" in res:
        raise res["erro"]
    return res["ok"]


def registrar_erro(modelo: str, msg: str):
    with open(CACHE / "erros.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} | {modelo} | {msg[:300]}\n")


def chamar_simulado(lote_df, cand_ids: set) -> str:
    """--dry-run: responde com o 1º candidato do TF-IDF (passo 2). Só serve para testar o fluxo."""
    cands = {}
    for l in open("data/candidatos.jsonl", encoding="utf-8"):
        d = json.loads(l); cands[d["id"]] = d["candidatos"]
    itens = []
    for qid in lote_df["id"]:
        top = cands[qid]
        itens.append(dict(id=qid, fora=False, tema_central_id=top[0]["tema_id"], extras_ids=[],
                          alternativa_tema_id=top[1]["tema_id"], duvida_subtemas=False,
                          confianca=min(1.0, top[0]["score"] * 4), motivo="simulado (tf-idf)"))
    return json.dumps({"questoes": itens}, ensure_ascii=False)


# ------------------------------------------------------------------ orquestração
def chave_lote(modelo, passada, ids) -> str:
    return hashlib.sha1(f"{PROMPT_VERSAO}|{modelo}|{passada}|{','.join(ids)}".encode()).hexdigest()[:10]


def salvar_lote(pasta, chave, ids, itens):
    (pasta / f"lote_{chave}.json").write_text(
        json.dumps({"ids": ids, "itens": [i.model_dump() for i in itens]}, ensure_ascii=False, indent=1), encoding="utf-8")


def fazer_lotes(q, ids_arquivo):
    ids = [l.strip() for l in open(ids_arquivo)] if ids_arquivo else list(q["id"])
    return [ids[i:i + TAM_LOTE] for i in range(0, len(ids), TAM_LOTE)]


def consolidar(modelo, passada, pasta):
    linhas = []
    for f in sorted(pasta.glob("lote_*.json")):
        d = json.loads(f.read_text())
        for x in d["itens"]:
            linhas.append({**x, "modelo": modelo, "passada": passada, "lote": f.stem})
    saida = Path(f"data/resultados_{modelo}_{passada}.jsonl")
    saida.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in linhas) + ("\n" if linhas else ""), encoding="utf-8")
    return saida, len(linhas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="gemini-2.5-flash")
    ap.add_argument("--passada", default="A", help="rótulo da passada (A, B...); passadas diferentes não compartilham cache")
    ap.add_argument("--rpd", type=int, default=RPD_PADRAO)
    ap.add_argument("--rpm", type=int, default=RPM_PADRAO)
    ap.add_argument("--max-chamadas", type=int, default=None, help="teto de chamadas reais NESTA execução")
    ap.add_argument("--ids-arquivo", default=None, help="arquivo com 1 id por linha (ex.: sua amostra rotulada)")
    ap.add_argument("--com-origem", action="store_true", help="inclui assunto_origem como pista (experimento)")
    ap.add_argument("--ajustar-contador", type=int, default=None, help="define quantas chamadas já foram gastas HOJE (use o valor do AI Studio)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    if a.ajustar_contador is not None:
        cf = CACHE / "contador.json"
        d = json.loads(cf.read_text()) if cf.exists() else {}
        d.setdefault(hoje_pacifico(), {})[a.modelo] = a.ajustar_contador
        cf.write_text(json.dumps(d, indent=1))
        print(f"Contador de hoje para {a.modelo} ajustado para {a.ajustar_contador}.")
    q, cand = carregar_dados()
    ids_validos = set(int(i) for i in cand["tema_id"])
    arvore = texto_arvore(cand)
    lotes = fazer_lotes(q, a.ids_arquivo)
    tag = f"{a.modelo}_{a.passada}" + ("_dry" if a.dry_run else "")
    pasta = CACHE / tag
    pasta.mkdir(parents=True, exist_ok=True)

    feitos = [l for l in lotes if (pasta / f"lote_{chave_lote(a.modelo, a.passada, l)}.json").exists()]
    pend = [l for l in lotes if l not in feitos]
    usadas = chamadas_hoje(a.modelo)
    print(f"[{tag}] lotes: {len(lotes)} | prontos: {len(feitos)} | pendentes: {len(pend)} | "
          f"chamadas hoje (Pacífico {hoje_pacifico()}): {usadas}/{a.rpd}")
    if a.status:
        return

    chamadas_feitas, falhas_seguidas = 0, 0
    q_idx = q.set_index("id", drop=False)
    for ids in pend:
        # 0) resposta crua guardada de uma execução anterior? Revalida SEM gastar chamada.
        chave0 = chave_lote(a.modelo, a.passada, ids)
        raw0 = pasta / f"lote_{chave0}.raw.txt"
        if not a.dry_run and raw0.exists():
            avisos0 = []
            itens0, erro0 = validar_lote(raw0.read_text(encoding="utf-8"), ids, ids_validos, avisos0)
            if not erro0:
                salvar_lote(pasta, chave0, ids, itens0)
                for av in avisos0: registrar_erro(a.modelo, "aviso (recuperado do raw): " + av)
                print(f"  lote {chave0}: recuperado do .raw.txt sem gastar chamada ({len(avisos0)} avisos)")
                continue
        if not a.dry_run:
            if chamadas_hoje(a.modelo) >= a.rpd:
                print("Cota diária local atingida. Rode de novo depois do reset (meia-noite Pacífico)."); break
            if a.max_chamadas is not None and chamadas_feitas >= a.max_chamadas:
                print("Teto de --max-chamadas atingido."); break
        lote_df = q_idx.loc[ids]
        chave = chave_lote(a.modelo, a.passada, ids)
        prompt = montar_prompt(lote_df, arvore, a.com_origem)
        print(f"  lote {chave}: {len(ids)} questões, prompt ~{len(prompt)//3} tokens", end=" ... ")

        parar_tudo = False
        if a.dry_run:
            bruto = chamar_simulado(lote_df, ids_validos)
        else:
            bruto = None
            for tentativa in range(1 + RETRIES_INDISPONIVEL):
                if chamadas_hoje(a.modelo) >= a.rpd:
                    print("cota diária local atingida."); parar_tudo = True; break
                registrar_chamada(a.modelo)  # conta ANTES (conservador); confira o valor real no AI Studio
                chamadas_feitas += 1
                try:
                    bruto = chamar_com_timeout(a.modelo, prompt)
                    time.sleep(60 / a.rpm + 1)
                    break
                except Exception as e:
                    msg = str(e)
                    print(f"ERRO da API: {msg[:160]}")
                    registrar_erro(a.modelo, msg)
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                        parar_tudo = True; break
                    if ("503" in msg or "UNAVAILABLE" in msg or "timeout" in msg.lower()) and tentativa < RETRIES_INDISPONIVEL:
                        print(f"  sem resposta/servidor sobrecarregado; esperando {ESPERA_503}s para tentar o mesmo lote de novo...", end=" ")
                        time.sleep(ESPERA_503)
                        continue
                    time.sleep(60 / a.rpm + 1)
                    break
            if parar_tudo:
                print("Parando para não gastar cota à toa."); break
            if bruto is None:
                falhas_seguidas += 1
                if falhas_seguidas >= 2:
                    print("2 lotes seguidos sem resposta, parando."); break
                continue

        (pasta / f"lote_{chave}.raw.txt").write_text(bruto, encoding="utf-8")  # sempre guarda a resposta crua
        avisos = []
        itens, erro = validar_lote(bruto, ids, ids_validos, avisos)
        if erro:
            print(f"INVÁLIDO: {erro}")
            falhas_seguidas += 1
            if falhas_seguidas >= 2 and not a.dry_run:
                print("2 falhas seguidas, parando."); break
            continue
        falhas_seguidas = 0
        salvar_lote(pasta, chave, ids, itens)
        for av in avisos: registrar_erro(a.modelo, "aviso: " + av)
        print("ok" + (f" ({len(avisos)} ids inválidos removidos de extras/alternativa)" if avisos else ""))

    saida, n = consolidar(a.modelo, a.passada + ("_dry" if a.dry_run else ""), pasta)
    print(f"Consolidado: {n} questões em {saida}")


if __name__ == "__main__":
    main()