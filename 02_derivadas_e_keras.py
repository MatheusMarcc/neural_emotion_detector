"""
02 — A regra da cadeia está certa? Comparação com o Keras.

O enunciado pede para implementar as derivadas (pesos e bias) e comparar com
o Keras "com mesmos parâmetros". Fazemos três conferências, da mais local para
a mais global:

1. DIFERENÇAS FINITAS (float64)
   A derivada é o limite de [L(w + ε) - L(w - ε)] / 2ε quando ε -> 0. Se
   perturbarmos um único peso por ε = 1e-6 e medirmos a variação do custo,
   temos uma estimativa numérica de ∂L/∂w que NÃO usa a regra da cadeia. Se o
   backward estiver certo, as duas devem bater até ~1e-7 de erro relativo.
   (Diferença central: o erro da aproximação é O(ε²), bem menor que a
   diferença progressiva [L(w+ε) - L(w)] / ε, que é O(ε).)

2. MESMO LOTE NO KERAS (float32)
   Copiamos os pesos da rede NumPy para um modelo Keras idêntico e pedimos ao
   TensorFlow (autograd, tf.GradientTape) o gradiente no mesmo lote. Comparamos
   tensor a tensor: dW1, db1, dW2, db2, dW3, db3.

3. TREINO LADO A LADO
   Mesmos pesos iniciais, mesmo otimizador, mesmos lotes na mesma ordem. Se as
   derivadas forem iguais, as duas redes percorrem o mesmo caminho e as curvas
   de custo e acurácia se sobrepõem.

Saídas:
  resultados/02_derivadas.json      todos os números
  resultados/02_derivadas.csv       tabela das conferências 1 e 2
  figuras/02_numpy_vs_keras.png     curvas do item 3
"""
import json
import time

import numpy as np
import pandas as pd

from src import config, dados, graficos
from src.keras_mlp import (compilar, construir_keras, copiar_pesos,
                           gradientes_keras, pesos_keras, prever_proba_keras)
from src.mlp import MLP

CAMADAS_REF = (64, 32)          # arquitetura de referência para as conferências
NOMES = ["W1", "b1", "W2", "b2", "W3", "b3"]


def erro_relativo(a, b):
    """||a - b|| / (||a|| + ||b||): 0 = idênticos, 1 = sem relação nenhuma.

    Calculado sobre o VETOR de entradas sorteadas de cada tensor (e não entrada
    por entrada), porque uma entrada com gradiente minúsculo, perto de 1e-9,
    teria erro relativo dominado pelo arredondamento da própria diferença finita.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-30))


# ---------------------------------------------------------------------------
# 1) Diferenças finitas
# ---------------------------------------------------------------------------
def conferir_diferencas_finitas(X, y, ativacao, l2, n_amostras=25, eps=1e-6):
    rng = np.random.default_rng(0)
    modelo = MLP(X.shape[1], CAMADAS_REF, config.N_CLASSES, ativacao=ativacao,
                 l2=l2, semente=config.SEMENTE, dtype=np.float64)
    # dá uma "sacudida" nos bias (que começam em zero) para o teste valer
    # também longe da inicialização
    for b in modelo.b:
        b += rng.normal(0, 0.1, size=b.shape)

    grad_W, grad_b = modelo.gradientes(X, y)          # backward (regra da cadeia)
    linhas_ativas = np.unique(X.nonzero()[1])          # palavras presentes no lote
    resultado = {}
    for l in range(modelo.n_camadas):
        for tipo, P, G in (("W", modelo.W, grad_W), ("b", modelo.b, grad_b)):
            analiticos, numericos = [], []
            for _ in range(n_amostras):
                if tipo == "W":
                    # em W1 sorteamos linhas de palavras que aparecem no lote;
                    # as outras têm gradiente (quase) zero e o teste seria trivial
                    i = rng.choice(linhas_ativas) if l == 0 else rng.integers(P[l].shape[0])
                    j = rng.integers(P[l].shape[1])
                    pos = (i, j)
                else:
                    pos = (rng.integers(P[l].shape[0]),)
                original = P[l][pos]
                P[l][pos] = original + eps
                l_mais = modelo.perda_total(X, y)
                P[l][pos] = original - eps
                l_menos = modelo.perda_total(X, y)
                P[l][pos] = original                         # devolve o peso
                numericos.append((l_mais - l_menos) / (2 * eps))
                analiticos.append(G[l][pos])
            resultado[f"{tipo}{l + 1}"] = erro_relativo(analiticos, numericos)
    return resultado


# ---------------------------------------------------------------------------
# 2) Gradiente do Keras no mesmo lote
# ---------------------------------------------------------------------------
def conferir_keras_mesmo_lote(X, y, ativacao, l2):
    modelo = MLP(X.shape[1], CAMADAS_REF, config.N_CLASSES, ativacao=ativacao,
                 l2=l2, semente=config.SEMENTE)
    keras_m = construir_keras(X.shape[1], CAMADAS_REF, config.N_CLASSES, ativacao, l2)
    copiar_pesos(modelo, keras_m)

    gW_np, gb_np = modelo.gradientes(X, y)
    gW_k, gb_k = gradientes_keras(keras_m, X.toarray(), y)
    resultado = {}
    for l in range(modelo.n_camadas):
        for nome, a, b in ((f"W{l + 1}", gW_np[l], gW_k[l]), (f"b{l + 1}", gb_np[l], gb_k[l])):
            # erro relativo pela norma: ||a - b|| / ||b||
            resultado[nome] = dict(
                dif_max=float(np.max(np.abs(a - b))),
                rel_norma=float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)),
                norma=float(np.linalg.norm(b)),
            )
    return resultado


# ---------------------------------------------------------------------------
# 3) Treino lado a lado
# ---------------------------------------------------------------------------
def treinar_lado_a_lado(X_tr, y_tr, X_va, y_va, ativacao, l2):
    modelo = MLP(X_tr.shape[1], CAMADAS_REF, config.N_CLASSES, ativacao=ativacao,
                 l2=l2, semente=config.SEMENTE)
    keras_m = compilar(construir_keras(X_tr.shape[1], CAMADAS_REF, config.N_CLASSES,
                                       ativacao, l2),
                       config.TAXA_APRENDIZADO, config.MOMENTO)
    copiar_pesos(modelo, keras_m)                 # mesmos pesos iniciais

    # --- NumPy --- (guardamos uma cópia dos pesos no fim de cada época para
    # medir, depois, o quanto as duas implementações se afastam)
    fotos = []
    hist_np = modelo.treinar(X_tr, y_tr, X_va, y_va, epocas=config.EPOCAS,
                             tamanho_lote=config.TAMANHO_LOTE,
                             taxa=config.TAXA_APRENDIZADO, momento=config.MOMENTO,
                             semente=config.SEMENTE,
                             ao_fim_da_epoca=lambda m, ep: fotos.append([W.copy() for W in m.W]))

    # --- Keras, alimentado com os MESMOS lotes ---
    # A classe MLP embaralha com np.random.default_rng(semente).permutation(n)
    # a cada época; recriamos a mesma sequência e passamos os dados já
    # embaralhados com shuffle=False, então o lote k do Keras é o lote k do NumPy.
    Xd_tr, Xd_va = X_tr.toarray(), X_va.toarray()
    rng = np.random.default_rng(config.SEMENTE)
    hist_k = {k: [] for k in ["perda_treino", "acc_treino", "f1_treino",
                              "perda_val", "acc_val", "f1_val", "tempo", "dif_pesos"]}
    for epoca in range(config.EPOCAS):
        perm = rng.permutation(len(y_tr))
        t0 = time.perf_counter()
        keras_m.fit(Xd_tr[perm], y_tr[perm], batch_size=config.TAMANHO_LOTE,
                    epochs=1, shuffle=False, verbose=0)
        hist_k["tempo"].append(time.perf_counter() - t0)
        # maior diferença, peso a peso, entre as duas redes depois desta época
        W_keras = [W for W, _ in pesos_keras(keras_m)]
        hist_k["dif_pesos"].append(float(max(np.max(np.abs(a - b))
                                             for a, b in zip(fotos[epoca], W_keras))))
        for parte, Xd, yy in (("treino", Xd_tr, y_tr), ("val", Xd_va, y_va)):
            m = avaliar_probs(prever_proba_keras(keras_m, Xd), yy)
            hist_k[f"perda_{parte}"].append(m["perda"])
            hist_k[f"acc_{parte}"].append(m["acc"])
            hist_k[f"f1_{parte}"].append(m["f1"])

    # diferença final entre os pesos das duas implementações
    dif_final = {}
    for l, (Wk, bk) in enumerate(pesos_keras(keras_m)):
        dif_final[f"W{l + 1}"] = float(np.max(np.abs(modelo.W[l] - Wk)))
        dif_final[f"b{l + 1}"] = float(np.max(np.abs(modelo.b[l] - bk)))
    return hist_np, hist_k, dif_final


def avaliar_probs(P, y):
    from sklearn.metrics import f1_score
    P = P.astype(np.float64)
    pred = P.argmax(axis=1)
    return dict(perda=float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None)).mean()),
                acc=float((pred == y).mean()),
                f1=float(f1_score(y, pred, average="macro")))


def figura_lado_a_lado(curvas):
    epocas = np.arange(1, config.EPOCAS + 1)

    def desenhar(P):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        fig, eixos = plt.subplots(1, 3, figsize=(15, 4.3))
        for ativ, (h_np, h_k) in curvas.items():
            cor = P.cor(ativ)
            for ax, chave in ((eixos[0], "perda_val"), (eixos[1], "acc_val")):
                ax.plot(epocas, h_np[chave], color=cor, lw=2.2)
                ax.plot(epocas[::3], np.asarray(h_k[chave])[::3], "o", ms=7,
                        mfc="none", mec=cor, mew=1.6)
            eixos[2].plot(epocas, h_k["dif_pesos"], color=cor, lw=2.2)
        eixos[0].set(title="Custo na validação", xlabel="época",
                     ylabel="entropia cruzada")
        eixos[1].set(title="Acurácia na validação", xlabel="época", ylabel="acurácia")
        eixos[2].set(title="Maior |W_NumPy − W_Keras|", xlabel="época",
                     ylabel="diferença (escala log)", yscale="log")
        legenda = [Line2D([], [], color=P.cor("relu"), lw=2.2, label="ReLU"),
                   Line2D([], [], color=P.cor("sigmoid"), lw=2.2, label="Sigmoid"),
                   Line2D([], [], color=P.texto2, lw=2.2, label="linha = NumPy (nossa)"),
                   Line2D([], [], color=P.texto2, ls="none", marker="o", mfc="none",
                          ms=7, label="círculo = Keras")]
        fig.legend(handles=legenda, loc="lower center", ncol=4,
                   bbox_to_anchor=(0.5, 0.0))
        fig.suptitle(f"Mesmos pesos iniciais e mesmos lotes: NumPy × Keras  "
                     f"(camadas {CAMADAS_REF}, L2 = {L2_REF:g}, η = {config.TAXA_APRENDIZADO:g})",
                     fontsize=12.5, color=P.texto, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        return fig

    graficos.salvar(desenhar, "02_numpy_vs_keras")


L2_REF = 1e-4

if __name__ == "__main__":
    treino, validacao, _ = dados.carregar_emotion()
    vet = dados.criar_tfidf()
    X_tr = vet.fit_transform(treino.text)
    X_va = vet.transform(validacao.text)
    y_tr, y_va = treino.label.values, validacao.label.values

    # lote fixo para as conferências 1 e 2
    idx = np.random.default_rng(1).choice(len(y_tr), size=64, replace=False)
    X_lote, y_lote = X_tr[idx], y_tr[idx]

    resumo = {"camadas": CAMADAS_REF, "diferencas_finitas": {}, "keras_mesmo_lote": {}}
    linhas = []
    for ativ in config.ATIVACOES:
        for l2 in config.L2_VALORES:
            chave = f"{ativ}_l2={l2:g}"
            df_ = conferir_diferencas_finitas(X_lote[:32].astype(np.float64), y_lote[:32],
                                              ativ, l2)
            kr = conferir_keras_mesmo_lote(X_lote, y_lote, ativ, l2)
            resumo["diferencas_finitas"][chave] = df_
            resumo["keras_mesmo_lote"][chave] = kr
            for nome in NOMES:
                linhas.append(dict(ativacao=ativ, l2=l2, parametro=nome,
                                   erro_rel_dif_finitas=df_[nome],
                                   erro_rel_keras=kr[nome]["rel_norma"],
                                   dif_max_keras=kr[nome]["dif_max"]))
            print(f"{chave:16s} dif. finitas: máx erro rel = {max(df_.values()):.2e}   "
                  f"Keras: máx erro rel = {max(v['rel_norma'] for v in kr.values()):.2e}")
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(config.RESULTADOS / "02_derivadas.csv", index=False)

    print("\nTreino lado a lado (pode levar alguns minutos)...")
    curvas, resumo["lado_a_lado"] = {}, {}
    for ativ in config.ATIVACOES:
        h_np, h_k, dif = treinar_lado_a_lado(X_tr, y_tr, X_va, y_va, ativ, L2_REF)
        curvas[ativ] = (h_np, h_k)
        resumo["lado_a_lado"][ativ] = dict(
            numpy=h_np, keras=h_k, dif_pesos_final=dif,
            dif_max_acc_val=float(np.max(np.abs(np.subtract(h_np["acc_val"], h_k["acc_val"])))),
            dif_max_perda_val=float(np.max(np.abs(np.subtract(h_np["perda_val"], h_k["perda_val"])))),
            tempo_epoca_numpy=float(np.mean(h_np["tempo"])),
            tempo_epoca_keras=float(np.mean(h_k["tempo"])),
        )
        r = resumo["lado_a_lado"][ativ]
        print(f"{ativ:8s} acc_val final  NumPy {h_np['acc_val'][-1]:.4f}  Keras {h_k['acc_val'][-1]:.4f} | "
              f"maior diferença de acc_val em {config.EPOCAS} épocas: {r['dif_max_acc_val']:.4f} | "
              f"maior |W_numpy - W_keras| no fim: {max(dif.values()):.2e} | "
              f"s/época NumPy {r['tempo_epoca_numpy']:.2f}  Keras {r['tempo_epoca_keras']:.2f}")

    with open(config.RESULTADOS / "02_derivadas.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=1, ensure_ascii=False)
    figura_lado_a_lado(curvas)
    print("\nFiguras e tabelas salvas em figuras/ e resultados/.")
