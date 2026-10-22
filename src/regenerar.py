#!/usr/bin/env python3
"""Regenera publicacao.csv e previsao.txt a partir dos anexos. NENHUM modelo de linguagem é chamado.
Uso:  python regenerar.py [--data data] [--dic dicionario_temas_otimizado.txt] [--out .]
Sinais (todos locais): (1) dicionário de termos por tema (dado versionado, revisado por humano);
(2) TF-IDF 1-2gram da questão contra caminho+título do tema; (3) kNN contra as 100 questões da folha-verdade.
Decisão no nível 1 da árvore (recuo para o tema-pai); confiança calibrada por regressão isotônica (leave-one-out na folha);
limiar e previsão por validação cruzada aninhada (meia-meia repetida)."""
import argparse, hashlib, html, json, os, re, unicodedata
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics.pairwise import cosine_similarity

W_DIC, W_TFIDF, W_KNN, TEMP, K = 1.0, 2.0, 0.5, 0.15, 5      # pesos escolhidos em busca na folha (otimista; ver README)
DF_MAX = 0.04
N_MIN, ALVOS, SEED, NSPLIT = 8, [0.95, 0.97, 1.0], 0, 300
NAO_PENAL = {'Direito Constitucional', 'Direito Civil - Legislação Especial', 'Direito Tributário',
             'Direito Ambiental', 'Disciplina Unificada', 'Direito Civil', 'Direito Administrativo'}

def norm(t):
    t = html.unescape(html.unescape(t or ""))
    t = re.sub(r'<[^>]+>', ' ', t)
    t = unicodedata.normalize('NFD', t.lower())
    return re.sub(r'\s+', ' ', ''.join(c for c in t if unicodedata.category(c) != 'Mn')).strip()

def wilson(x, n, z=1.645):
    if n == 0: return (0.0, 1.0)
    p = x / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data'); ap.add_argument('--dic', default='dicionario_temas_otimizado.txt')
    ap.add_argument('--out', default='.'); a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'artifacts'), exist_ok=True)
    T = pd.read_csv(f'{a.data}/temas.csv')
    par = {int(x): (int(y) if y == y else None) for x, y in zip(T.tema_id, T.parent_id)}
    niv = dict(zip(T.tema_id.astype(int), T.nivel)); cam = dict(zip(T.tema_id.astype(int), T.caminho))
    A = T[(T.status == 'active') & (T.nivel >= 1)].reset_index(drop=True)      # inativos nunca são usados
    ids = A.tema_id.astype(int).tolist(); pos = {t: i for i, t in enumerate(ids)}
    def anc(t):
        o = [t]
        while par.get(t) is not None: t = par[t]; o.append(t)
        return set(o)
    def n1(t):
        while niv[t] > 1: t = par[t]
        return t
    ok = lambda p, gold: any(p in anc(g) or g in anc(p) for g in gold)
    Q = pd.DataFrame([json.loads(l) for l in open(f'{a.data}/questoes.jsonl', encoding='utf-8') if l.strip()])
    Q['txt'] = (Q.comando.fillna('') + ' ' + Q.enunciado_html).map(norm)
    F = pd.read_csv(f'{a.data}/folha_verdade.csv').fillna('')
    F['qi'] = F.id.map({i: k for k, i in enumerate(Q.id)}); F['fora'] = F.fora_da_taxonomia.astype(int)
    F['gold'] = [set() if f else {int(c)} | {int(x) for x in str(e).split(';') if x}
                 for f, c, e in zip(F.fora, F.tema_central_id, F.extras_ids)]
    # --- sinal 1: dicionário (matriz termo x questão em cache; chave = hash do dicionário + questões)
    raw = {}
    for l in open(f'{a.data}/{a.dic}', encoding='utf-8'):
        if l.startswith('#') or ':' not in l: continue
        k, v = l.split(':', 1); raw[int(k)] = [norm(x) for x in v.split('|') if len(norm(x)) > 2]
    terms = sorted({t for v in raw.values() for t in v})
    key = hashlib.md5((open(f'{a.data}/{a.dic}', 'rb').read() + open(f'{a.data}/questoes.jsonl', 'rb').read())).hexdigest()
    cp = os.path.join(a.out, 'artifacts', f'match_{key[:10]}.npz')
    if os.path.exists(cp): M = np.load(cp)['M']
    else:
        print('calculando matriz termo x questão (uma vez; fica em cache)...')
        M = np.zeros((len(terms), len(Q)), dtype=bool)
        for i, t in enumerate(terms):
            p = re.compile(r'\b' + re.escape(t)); M[i] = [bool(p.search(x)) for x in Q.txt]
        np.savez_compressed(cp, M=M)
    tix = {t: i for i, t in enumerate(terms)}; DS = np.zeros((len(Q), len(ids))); dfc = M.sum(1); NQ = len(Q)
    for k, v in raw.items():
        if k not in pos: continue
        for t in dict.fromkeys(v):
            d = dfc[tix[t]]
            if d / NQ > DF_MAX: continue                      # termo genérico (aparece em >4% das questões): descartado
            DS[:, pos[k]] += (1 + .5 * (len(t.split()) - 1)) * (np.log(NQ / (1 + d)) / np.log(NQ)) * M[tix[t]]   # peso por IDF
    DS = np.tanh(DS / 2)
    # --- sinais 2 e 3
    A['txt'] = (A.caminho + ' ' + A.titulo).map(norm)
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2).fit(pd.concat([A.txt, Q.txt]))
    XQ = vec.transform(Q.txt); TS = cosine_similarity(XQ, vec.transform(A.txt)); SQF = cosine_similarity(XQ, XQ[F.qi.values])
    def knn(i, excl=None):
        s = np.zeros(len(ids)); sims = SQF[i].copy()
        if excl is not None: sims[excl] = 0
        for j in np.argsort(-sims)[:K]:
            if F.fora[j] or sims[j] <= 0: continue
            g = int(F.tema_central_id[j]); s[pos[g]] += sims[j]
            for x in anc(g):
                if x in pos: s[pos[x]] += sims[j] * .5
        return s
    def probs(i, excl=None):
        s = W_DIC * DS[i] + W_TFIDF * TS[i] + W_KNN * knn(i, excl)
        e = np.exp((s - s.max()) / TEMP); e /= e.sum(); o = {}
        for t, p in zip(ids, e): o[n1(t)] = o.get(n1(t), 0) + p
        return o
    # --- avaliação leave-one-out na folha
    loo = []
    for k, r in F.iterrows():
        pr = probs(r.qi, excl=k); b = max(pr, key=pr.get)
        loo.append((pr[b], bool((not r.fora) and ok(b, r.gold))))
    conf = np.array([c for c, _ in loo]); certo = np.array([c for _, c in loo])
    print(f'folha (leave-one-out): acerto top1 no nível 1 = {certo.mean():.2f}')
    def limiar(cf, ct, alvo):
        o = np.argsort(-cf); c = ct[o]; best = None
        for n in range(N_MIN, len(c) + 1):
            if c[:n].mean() >= alvo: best = cf[o][n - 1]
        return best if best is not None else np.inf
    # --- validação cruzada aninhada: limiar escolhido em metade, medido na outra metade
    rng = np.random.default_rng(SEED); aninhado = {}
    for alvo in ALVOS:
        pc = pn = 0; precs = []; covs = []; ths = []
        for _ in range(NSPLIT):
            perm = rng.permutation(len(conf)); tr, te = perm[:len(conf) // 2], perm[len(conf) // 2:]
            th = limiar(conf[tr], certo[tr], alvo); ths.append(th); sel = conf[te] >= th
            covs.append(sel.mean())
            if sel.sum(): precs.append(certo[te][sel].mean()); pc += certo[te][sel].sum(); pn += sel.sum()
        aninhado[alvo] = dict(prec=pc / pn if pn else np.nan, precs=precs, covs=covs, ths=ths)
        print(f'  alvo interno {alvo:.2f}: precisão fora da amostra {aninhado[alvo]["prec"]:.2f} | cobertura {np.mean(covs):.2f}')
    ok_alvos = [al for al in ALVOS if aninhado[al]['prec'] >= 0.95]
    alvo = ok_alvos[0] if ok_alvos else ALVOS[-1]            # menor alvo cujo resultado fora da amostra bate 95%; senão o mais conservador
    TH = limiar(conf, certo, alvo)
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(conf, certo)
    # --- decisões para as 659 questões
    fset = {int(k): kk for kk, k in zip(F.index, F.qi)}
    rows, raws = [], []
    for i in range(len(Q)):
        pr = probs(i, excl=fset.get(i)); rank = sorted(pr, key=pr.get, reverse=True); b = rank[0]; c = pr[b]
        cal = float(min(aninhado[alvo]['prec'], iso.predict([c])[0])); disc = Q.disciplina_origem[i]
        extras = [str(t) for t in rank[1:3] if pr[t] >= 0.25]
        if c >= TH: dec, mot = 'publicar', ''
        elif disc in NAO_PENAL:
            dec, mot = 'fora', f'origem "{disc}" e sem correspondência confiável na taxonomia penal (conf {c:.2f})'
        else:
            dec = 'revisar'; mot = f'confiança {c:.2f} abaixo do limiar {TH:.2f}; alternativa: {cam[rank[1]].split(" > ")[-1]} ({pr[rank[1]]:.2f})'
        rows.append(dict(id=Q.id[i], decisao=dec, tema_central_id=('' if dec == 'fora' else b), extras_ids=';'.join(extras),
                         confianca=round(cal, 3), motivo=mot)); raws.append(c)
    pub = pd.DataFrame(rows); pub.to_csv(os.path.join(a.out, 'publicacao.csv'), index=False)
    # --- previsao.txt (calculada; ver método abaixo)
    nf = ~Q.id.isin(F.id).values; cob = (pub.decisao[nf] == 'publicar').mean(); R = aninhado[alvo]
    p_lo, p_hi = np.percentile(R['precs'], [10, 90]); p_pt = R['prec']
    thf = [t for t in R['ths'] if np.isfinite(t)] or [TH]
    rawnf = np.array(raws)[nf]; cobs = [(rawnf >= t).mean() for t in thf]
    c_lo, c_hi = np.percentile(cobs, [10, 90])
    lo_w, _ = wilson(int(certo[conf >= TH].sum()), int((conf >= TH).sum()))
    txt = f"""precisao_prevista: {p_pt:.2f}
precisao_intervalo: {min(p_lo, p_pt):.2f}-{max(p_hi, p_pt):.2f}
cobertura_prevista: {cob:.2f}
cobertura_intervalo: {min(c_lo, cob):.2f}-{max(c_hi, cob):.2f}

Como os números foram obtidos (nenhum foi digitado à mão):
- A folha-verdade tem 100 questões. Cada uma recebeu a previsão do sistema sem usar o próprio rótulo (kNN leave-one-out).
  Acerto no nível 1 da árvore, publicando tudo: {certo.mean():.2f}.
- O limiar de publicação (confiança bruta >= {TH:.3f}) é o maior conjunto, com no mínimo {N_MIN} questões, cuja precisão na folha é >= {alvo:.2f}.
  O alvo interno ({alvo:.2f}) foi o menor entre {ALVOS} cuja precisão FORA DA AMOSTRA bateu 0.95{'' if ok_alvos else ' (nenhum bateu: usei o mais conservador, e a precisão prevista fica abaixo de 0.95)'}.
- Precisão prevista = validação cruzada aninhada: {NSPLIT} divisões meia-meia da folha; o limiar é escolhido em uma metade e medido na outra.
  Valor pontual = acertos/publicadas somados nas divisões; intervalo = percentis 10-90 das divisões. Wilson 90% do conjunto publicado na folha (in-sample, otimista): {lo_w:.2f}-1.00.
- Cobertura prevista = fração das 559 questões fora da folha que o sistema publica com o limiar final. Intervalo = o mesmo cálculo com os limiares escolhidos nas divisões.
- Cuidados: a amostra é pequena (n=100), então os intervalos são largos; a folha também ajustou pesos/dicionário, então há otimismo residual.
  Erros de 'fora' publicados contam como erro. Classes de nível 2 não são distinguidas: o sistema publica no tema (nível 1), que conta como certo se a questão está no tema ou em um filho.
"""
    open(os.path.join(a.out, 'previsao.txt'), 'w', encoding='utf-8').write(txt)
    json.dump(dict(pesos=dict(dic=W_DIC, tfidf=W_TFIDF, knn=W_KNN), temp=TEMP, k=K, alvo_interno=alvo, limiar=float(TH) if np.isfinite(TH) else None,
                   acerto_top1_loo=float(certo.mean()), nested={str(k): dict(prec=v['prec'], cob=float(np.mean(v['covs']))) for k, v in aninhado.items()}),
              open(os.path.join(a.out, 'artifacts', 'calibracao.json'), 'w'), indent=1)
    print(pub.decisao.value_counts().to_dict()); print(txt.split('\n\n')[0])

if __name__ == '__main__': main()
