"""
07 — (Ponto extra) Embeddings Skip-Gram com a nossa própria MLP.

O ENUNCIADO
-----------
"A partir da implementação feita, gere uma representação Embedding SkipGram
das palavras do texto, e substitua a representação TF-IDF com a nova
representação concatenada. Utilize a melhor configuração de parâmetros e veja
se a acurácia melhorou. DICA: diferente da rede treinada antes (com duas
camadas ocultas), aqui existe somente uma, então retire uma camada."

TEORIA — Skip-Gram (word2vec, Mikolov et al., 2013)
---------------------------------------------------
Ideia: palavras que aparecem nos mesmos contextos têm significados parecidos.
O Skip-Gram treina uma rede para, dada uma palavra central, prever as
palavras vizinhas (a janela de contexto):

    "i feel so [happy] and grateful today"      janela = 2
    pares (central, contexto): (happy, feel) (happy, so) (happy, and) (happy, grateful)

A rede é uma MLP com UMA camada oculta LINEAR — exatamente a nossa classe MLP
com camadas_ocultas=(d,) e ativacao="linear" (a "camada retirada" da dica):

    entrada x : one-hot da palavra central (V posições, um único 1)
    oculta  h = x W1 + b1    -> como x tem um só 1, h é a LINHA de W1 daquela palavra
    saída   P = softmax(h W2 + b2)   probabilidade de cada palavra do vocabulário
                                     ser vizinha
    custo     = entropia cruzada com a palavra de contexto verdadeira

Depois do treino, jogamos a saída fora e ficamos com W1: a linha i de W1 é o
EMBEDDING (vetor denso de d números) da palavra i. Palavras usadas em
contextos parecidos acabam com vetores parecidos (cosseno alto).

Dois truques do word2vec que usamos:
  * subamostragem: palavras frequentíssimas ("i", "feel") são descartadas com
    probabilidade 1 - sqrt(t / f), t = 1e-3, f = frequência relativa. Elas
    aparecem em quase todo tweet e não ensinam nada sobre o vizinho;
  * o softmax é sobre o vocabulário inteiro (3.000 palavras) — viável aqui
    porque o vocabulário é pequeno. (Em vocabulários enormes usa-se negative
    sampling.) Treinamos com Adam: os embeddings de palavras raras recebem
    gradiente poucas vezes, e o passo adaptativo ajuda.

REPRESENTAÇÕES DOS TWEETS (substituindo o TF-IDF)
------------------------------------------------
"Representação concatenada" admite leituras diferentes; testamos três:
  1. CONCATENAÇÃO das embeddings das palavras, em ordem: as 40 primeiras
     palavras do tweet (40 cobre 95% dos tweets), cada uma com d = 100
     números -> vetor de 4.000; tweets mais curtos são completados com zeros.
  2. [MÉDIA ; MÁXIMO] das embeddings: a média e o máximo (coordenada a
     coordenada) das embeddings do tweet, concatenados -> 200 números.
  3. [TF-IDF ; MÉDIA]: o TF-IDF original concatenado com a média das
     embeddings -> 3.100 números.
Todas são avaliadas com a MELHOR configuração do script 03 (mesma arquitetura,
ativação, L2, otimizador e épocas), nos mesmos 3 folds e no teste.

DIAGNÓSTICO LINEAR: para cada representação também treinamos uma regressão
logística. Se nem ela separa as classes, a informação não está na
representação — e não adianta culpar (ou ajustar) a MLP.

Observação: o Skip-Gram usa só os textos do treino (sem rótulos). Na
validação cruzada ele viu os textos dos folds de validação, mas não os seus
rótulos — o teste nunca é visto.

Saídas: resultados/07_skipgram.json, 07_vizinhos.csv, figuras/07_*.png
"""
import json
import time
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src import config, dados, graficos
from src.mlp import MLP

DIM = 100            # tamanho do embedding (neurônios da camada oculta)
JANELA = 2           # palavras de contexto de cada lado
T_SUBAMOSTRA = 1e-3
EPOCAS_SG = 5
LOTE_SG = 512
TAXA_SG = 1e-3       # Adam
MAX_PALAVRAS_CONCAT = 40
PALAVRAS_TESTE = ["happy", "sad", "angry", "scared", "love", "surprised", "amazing",
                  "lonely", "nervous", "grateful"]


# ---------------------------------------------------------------------------
# 1) Dados do Skip-Gram
# ---------------------------------------------------------------------------
def montar_vocabulario(textos, n=config.MAX_PALAVRAS):
    contagem = Counter(p for t in textos for p in t.split())
    vocab = [p for p, _ in contagem.most_common(n)]
    return vocab, {p: i for i, p in enumerate(vocab)}, contagem


def pares_skipgram(textos, indice, contagem, rng):
    """Lista de pares (central, contexto) com subamostragem das palavras frequentes."""
    total = sum(contagem.values())
    p_manter = {p: min(1.0, np.sqrt(T_SUBAMOSTRA / (c / total))) for p, c in contagem.items()}
    centrais, contextos = [], []
    for t in textos:
        ids = [indice[p] for p in t.split() if p in indice and rng.random() < p_manter[p]]
        for i, c in enumerate(ids):
            for j in range(max(0, i - JANELA), min(len(ids), i + JANELA + 1)):
                if j != i:
                    centrais.append(c)
                    contextos.append(ids[j])
    return np.array(centrais), np.array(contextos)


def one_hot_esparso(ids, V):
    """Matriz esparsa (N x V) com um único 1 por linha — a entrada da rede."""
    n = len(ids)
    return sp.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), ids)), shape=(n, V))


def treinar_skipgram(textos):
    rng = np.random.default_rng(config.SEMENTE)
    vocab, indice, contagem = montar_vocabulario(textos)
    V = len(vocab)
    centrais, contextos = pares_skipgram(textos, indice, contagem, rng)
    print(f"  vocabulário {V} palavras, {len(centrais):,} pares (central, contexto)".replace(",", "."))

    # A MLP com UMA camada oculta linear — "retire uma camada" da dica
    rede = MLP(V, (DIM,), V, ativacao="linear", l2=0.0, semente=config.SEMENTE)
    X = one_hot_esparso(centrais, V)
    t0 = time.perf_counter()
    hist = rede.treinar(X, contextos, epocas=EPOCAS_SG, tamanho_lote=LOTE_SG, taxa=TAXA_SG,
                        otimizador="adam", semente=config.SEMENTE, avaliar_cada_epoca=False,
                        verbose=True)
    print(f"  Skip-Gram treinado em {time.perf_counter() - t0:.0f}s")
    embeddings = rede.W[0]            # V x DIM: linha i = embedding da palavra i
    return vocab, indice, embeddings, hist, len(centrais)


def vizinhos(palavra, vocab, indice, E, k=6):
    """As k palavras de maior similaridade de cosseno com `palavra`."""
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    sims = En @ En[indice[palavra]]
    ordem = [i for i in np.argsort(sims)[::-1] if i != indice[palavra]][:k]
    return [(vocab[i], float(sims[i])) for i in ordem]


# ---------------------------------------------------------------------------
# 2) Representações dos tweets
# ---------------------------------------------------------------------------
def rep_concatenada(textos, indice, E):
    """[e(w1) ; e(w2) ; ... ; e(w40)] — preenche com zeros depois do fim do tweet."""
    X = np.zeros((len(textos), MAX_PALAVRAS_CONCAT * DIM), dtype=np.float32)
    for n, t in enumerate(textos):
        ids = [indice[p] for p in t.split() if p in indice][:MAX_PALAVRAS_CONCAT]
        for pos, i in enumerate(ids):
            X[n, pos * DIM:(pos + 1) * DIM] = E[i]
    return X


def rep_media_max(textos, indice, E):
    X = np.zeros((len(textos), 2 * DIM), dtype=np.float32)
    for n, t in enumerate(textos):
        ids = [indice[p] for p in t.split() if p in indice]
        if ids:
            X[n, :DIM] = E[ids].mean(axis=0)
            X[n, DIM:] = E[ids].max(axis=0)
    return X


def media(textos, indice, E):
    X = np.zeros((len(textos), DIM), dtype=np.float32)
    for n, t in enumerate(textos):
        ids = [indice[p] for p in t.split() if p in indice]
        if ids:
            X[n] = E[ids].mean(axis=0)
    return X


def norma_1(X):
    """Divide cada linha pela sua norma L2 (linhas zeradas continuam zeradas).

    É o mesmo que o TF-IDF faz com cada tweet. Com todas as representações na
    mesma escala (norma 1), a taxa de aprendizado da melhor configuração
    continua adequada — sem isso, a comparação mediria a escala, não a
    representação.
    """
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.where(n > 0, n, 1)).astype(np.float32)


def construir(nome, textos, indice, E, X_tfidf):
    """Representação `nome` de uma lista de tweets.

    E já vem centralizado (cada coordenada com média 0 no vocabulário), para
    que o "zero" do preenchimento da concatenação seja uma palavra média.
    X_tfidf = a matriz TF-IDF desses mesmos tweets (usada em "tfidf" e "tfidf_media").
    """
    if nome == "tfidf":
        return X_tfidf
    if nome == "concatenacao":
        return norma_1(rep_concatenada(textos, indice, E))
    if nome == "media_max":
        return norma_1(rep_media_max(textos, indice, E))
    if nome == "tfidf_media":
        return sp.hstack([X_tfidf, sp.csr_matrix(norma_1(media(textos, indice, E)))]).tocsr()
    raise ValueError(nome)


REPRESENTACOES = {
    "tfidf": "TF-IDF (3.000)",
    "concatenacao": f"Concatenação de {MAX_PALAVRAS_CONCAT} embeddings ({MAX_PALAVRAS_CONCAT * DIM:,})".replace(",", "."),
    "media_max": f"[média ; máximo] ({2 * DIM})",
    "tfidf_media": f"[TF-IDF ; média] ({config.MAX_PALAVRAS + DIM:,})".replace(",", "."),
}


def avaliar(X_tr, y_tr, X_ou, y_ou, melhor, semente):
    """Treina a melhor configuração do script 03 numa representação.

    As sementes são as mesmas dos scripts 03 (SEMENTE + fold) e 06 (SEMENTE), e o
    TF-IDF é montado do mesmo jeito: a linha "TF-IDF" desta comparação reproduz
    exatamente os números da validação cruzada e do teste.
    """
    modelo = MLP(X_tr.shape[1], tuple(melhor["camadas"]), config.N_CLASSES,
                 ativacao=melhor["ativacao"], l2=melhor["l2"], semente=semente)
    hist = modelo.treinar(X_tr, y_tr, X_ou, y_ou, epocas=config.EPOCAS,
                          tamanho_lote=config.TAMANHO_LOTE, taxa=config.TAXA_APRENDIZADO,
                          momento=config.MOMENTO, semente=semente)
    return hist


def diagnostico_linear(X_tr, y_tr, X_te, y_te):
    """Regressão logística na mesma representação.

    Responde: a informação está na representação? Se nem um modelo linear bem
    ajustado separa as classes, o problema é a representação, não a MLP.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    m = LogisticRegression(C=10, max_iter=3000).fit(X_tr, y_tr)
    p = m.predict(X_te)
    return dict(acc_treino=float((m.predict(X_tr) == y_tr).mean()),
                acc_teste=float((p == y_te).mean()),
                f1_teste=float(f1_score(y_te, p, average="macro")))


def fig_comparacao(tab):
    def desenhar(P):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        fig, eixos = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
        nomes = [REPRESENTACOES[r] for r in tab.representacao]
        y = np.arange(len(nomes))
        for ax, (col_cv, col_dp, col_te, titulo) in zip(
                eixos, (("acc_cv", "acc_cv_dp", "acc_teste", "Acurácia"),
                        ("f1_cv", "f1_cv_dp", "f1_teste", "F1 macro"))):
            ax.barh(y - 0.17, tab[col_cv], height=0.32, xerr=tab[col_dp], color=P.serie(0),
                    error_kw=dict(ecolor=P.texto2, lw=1.2, capsize=3))
            ax.barh(y + 0.17, tab[col_te], height=0.32, color=P.serie(2))
            for yi, (a, d, b) in enumerate(zip(tab[col_cv], tab[col_dp], tab[col_te])):
                # rótulo depois da ponta da barra de erro, para não cobrir nada
                ax.text(a + d + 0.015, yi - 0.17, graficos.num(a), va="center", fontsize=8.5,
                        color=P.texto2)
                ax.text(b + 0.015, yi + 0.17, graficos.num(b), va="center", fontsize=8.5,
                        color=P.texto2)
            ax.set_xlim(0, 1.08)
            ax.set_title(titulo, loc="left")
            ax.grid(axis="y", visible=False)
        eixos[0].set_yticks(y, nomes)
        eixos[0].invert_yaxis()
        fig.legend(handles=[Patch(color=P.serie(0), label="validação cruzada 3-fold (±1 desvio)"),
                            Patch(color=P.serie(2), label="teste (2.000 tweets)")],
                   loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle("Skip-Gram no lugar do TF-IDF — mesma MLP (melhor configuração)",
                     x=0.01, ha="left", fontsize=12.5, color=P.texto)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        return fig

    graficos.salvar(desenhar, "07_skipgram_comparacao")


if __name__ == "__main__":
    with open(config.RESULTADOS / "melhor_config.json", encoding="utf-8") as f:
        melhor = json.load(f)
    treino, validacao, teste = dados.carregar_emotion()
    tx_tr, tx_te = treino.text.values, teste.text.values
    y_tr, y_te = treino.label.values, teste.label.values

    print("1) Treinando o Skip-Gram com a classe MLP (1 camada oculta linear)...")
    vocab, indice, E, hist_sg, n_pares = treinar_skipgram(tx_tr)
    viz = {p: vizinhos(p, vocab, indice, E) for p in PALAVRAS_TESTE if p in indice}
    pd.DataFrame([dict(palavra=p, vizinhos=", ".join(f"{w} ({s:.2f})" for w, s in v))
                  for p, v in viz.items()]).to_csv(config.RESULTADOS / "07_vizinhos.csv", index=False)
    for p, v in viz.items():
        print(f"   {p:10s} -> " + ", ".join(w for w, _ in v))

    np.savez(config.RESULTADOS / "07_embeddings.npz", embeddings=E, vocabulario=np.array(vocab))

    print("\n2) Melhor configuração com cada representação (3 folds + teste)")
    Ec = E - E.mean(axis=0)           # embeddings centralizados
    folds = dados.montar_folds_tfidf(tx_tr, y_tr)      # os mesmos folds do script 03
    vet = dados.criar_tfidf()
    T_tr = vet.fit_transform(tx_tr)                     # o mesmo TF-IDF do script 06
    T_te = vet.transform(tx_te)
    linhas = []
    for rep in REPRESENTACOES:
        accs, f1s = [], []
        for f in folds:
            Xa = construir(rep, tx_tr[f["idx_tr"]], indice, Ec, f["X_tr"])
            Xb = construir(rep, tx_tr[f["idx_va"]], indice, Ec, f["X_va"])
            h = avaliar(Xa, f["y_tr"], Xb, f["y_va"], melhor, config.SEMENTE + f["fold"])
            accs.append(h["acc_val"][-1])
            f1s.append(h["f1_val"][-1])
        Xa, Xb = construir(rep, tx_tr, indice, Ec, T_tr), construir(rep, tx_te, indice, Ec, T_te)
        h = avaliar(Xa, y_tr, Xb, y_te, melhor, config.SEMENTE)
        lin = diagnostico_linear(Xa, y_tr, Xb, y_te)
        linhas.append(dict(representacao=rep, dimensao=int(Xa.shape[1]),
                           acc_cv=float(np.mean(accs)), acc_cv_dp=float(np.std(accs)),
                           f1_cv=float(np.mean(f1s)), f1_cv_dp=float(np.std(f1s)),
                           acc_teste=h["acc_val"][-1], f1_teste=h["f1_val"][-1],
                           acc_treino_mlp=h["acc_treino"][-1],
                           reg_logistica_acc_treino=lin["acc_treino"],
                           reg_logistica_acc_teste=lin["acc_teste"],
                           reg_logistica_f1_teste=lin["f1_teste"]))
        r = linhas[-1]
        print(f"   {REPRESENTACOES[rep]:34s} CV: acc {r['acc_cv']:.4f} F1 {r['f1_cv']:.4f} | "
              f"teste: acc {r['acc_teste']:.4f} F1 {r['f1_teste']:.4f} | "
              f"MLP treino acc {r['acc_treino_mlp']:.3f} | reg. logística teste acc "
              f"{r['reg_logistica_acc_teste']:.3f}", flush=True)

    tab = pd.DataFrame(linhas)
    tab.to_csv(config.RESULTADOS / "07_skipgram_comparacao.csv", index=False)
    fig_comparacao(tab)
    with open(config.RESULTADOS / "07_skipgram.json", "w", encoding="utf-8") as f:
        json.dump(dict(dim=DIM, janela=JANELA, epocas=EPOCAS_SG, lote=LOTE_SG, taxa=TAXA_SG,
                       subamostragem_t=T_SUBAMOSTRA, n_pares=n_pares, vocabulario=len(vocab),
                       perda_por_epoca=hist_sg["perda_media_lotes"],
                       vizinhos={p: v for p, v in viz.items()},
                       comparacao=linhas), f, indent=1, ensure_ascii=False)
