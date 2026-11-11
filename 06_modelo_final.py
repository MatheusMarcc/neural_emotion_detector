"""
06 — Modelo final: a melhor configuração da validação cruzada no conjunto de teste.

Agora, e só agora, usamos o conjunto de teste (2.000 tweets que nenhuma
decisão anterior viu). A configuração vencedora do script 03 é treinada no
treino completo (16.000 tweets), com a validação oficial (2.000) acompanhando
a curva de aprendizado, e avaliada uma única vez no teste.

MÉTRICAS (teoria completa em src/metricas.py)
---------------------------------------------
  acurácia                  acertos / total (dominada pelas classes grandes)
  precisão, revocação, F1   por classe, e resumidas de dois jeitos:
     macro = média das 6 classes (cada emoção pesa igual)
     micro = contas somadas sobre todos os tweets (= acurácia, com 1 rótulo por tweet)
  curva ROC e AUROC         uma curva por emoção (uma contra as outras): mede se a
                            rede ORDENA bem os tweets pela probabilidade, sem depender
                            do limiar. 0,5 = sorteio; 1,0 = perfeito.

Referências para comparar (todas no mesmo teste):
  * Random Stratified Baseline: sorteia a classe na proporção do treino (o piso
    pedido no enunciado; acerta ~24% e tem AUROC ~0,5);
  * classe majoritária: chuta sempre "alegria" (34,8% de acurácia, F1 macro 0,086);
  * regressão logística no mesmo TF-IDF: um modelo LINEAR. Se a MLP não for
    melhor que ela, as camadas ocultas não estão ajudando nesta tarefa.

Saídas:
  resultados/06_modelo_final.json, 06_metricas_resumo.csv, 06_metricas_por_classe.csv,
             06_erros_exemplos.csv
  figuras/06_curva_modelo_final.png, 06_matriz_confusao.png, 06_curva_roc.png
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

from src import backend, config, dados, graficos, metricas
from src.mlp import MLP

# cor de cada emoção na curva ROC (índices da paleta; azul e laranja ficam para ReLU/Sigmoid)
COR_CLASSE = [6, 3, 4, 7, 5, 2]


def regressao_logistica(X_tr, y_tr, X_va, y_va):
    """Escolhe o C (inverso da regularização) na validação; devolve o C e o modelo."""
    melhor = None
    for C in (0.1, 1, 10, 100):
        m = LogisticRegression(C=C, max_iter=3000).fit(X_tr, y_tr)
        f1 = f1_score(y_va, m.predict(X_va), average="macro")
        if melhor is None or f1 > melhor[0]:
            melhor = (f1, C, m)
    return melhor[1], melhor[2]


def fig_curva(hist):
    ep = np.arange(1, config.EPOCAS + 1)

    def desenhar(P):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        fig, eixos = plt.subplots(1, 2, figsize=(12, 4.2))
        cor = P.cor("relu")
        for ax, (tr, va, titulo) in zip(eixos, (("perda_treino", "perda_val", "Custo (entropia cruzada)"),
                                                ("f1_treino", "f1_val", "F1 macro"))):
            ax.plot(ep, hist[tr], color=cor, ls="--")
            ax.plot(ep, hist[va], color=cor)
            ax.set(title=titulo, xlabel="época")
        k = int(np.argmin(hist["perda_val"]))
        eixos[0].plot(ep[k], hist["perda_val"][k], "o", color=cor, ms=8, mec=P.fundo, mew=2)
        eixos[0].annotate(f"menor custo de validação: época {k + 1}", (ep[k], hist["perda_val"][k]),
                          xytext=(10, 18), textcoords="offset points", fontsize=9, color=P.texto2)
        eixos[1].legend(handles=[Line2D([], [], color=P.texto2, ls="--", label="treino (16.000)"),
                                 Line2D([], [], color=P.texto2, label="validação (2.000)")],
                        loc="lower right")
        fig.tight_layout()
        return fig

    graficos.salvar(desenhar, "06_curva_modelo_final")


def fig_confusao(cm):
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    nomes = config.CLASSES_PT

    def desenhar(P):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.4, 6.2))
        im = ax.imshow(cm_norm, cmap=P.cmap_seq(), vmin=0, vmax=1)
        for i in range(len(nomes)):
            for j in range(len(nomes)):
                v = cm_norm[i, j]
                claro_no_escuro = (v > 0.5) if P.nome == "claro" else (v > 0.55)
                cor_txt = "#ffffff" if (claro_no_escuro and P.nome == "claro") else (
                    "#0b0b0b" if claro_no_escuro else P.texto)
                ax.text(j, i, f"{v:.0%}\n({cm[i, j]})", ha="center", va="center",
                        fontsize=8.5, color=cor_txt)
        ax.set_xticks(range(len(nomes)), nomes, rotation=30, ha="right")
        ax.set_yticks(range(len(nomes)), nomes)
        ax.set(xlabel="classe prevista", ylabel="classe verdadeira")
        ax.grid(False)
        ax.set_title("Matriz de confusão no teste (% da linha e contagem)", loc="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.outline.set_visible(False)
        fig.tight_layout()
        return fig

    graficos.salvar(desenhar, "06_matriz_confusao")


def fig_roc(curvas_mlp, auc_classes, auc_mlp, curvas_ref):
    """Esquerda: ROC da MLP por emoção. Direita: média macro da MLP × referências."""

    def desenhar(P):
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.4))
        for ax in (ax1, ax2):
            ax.plot([0, 1], [0, 1], color=P.mudo, ls=":", lw=1.2)
            ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="taxa de falsos positivos (FPR)",
                   ylabel="taxa de verdadeiros positivos (TPR = revocação)")
            ax.set_aspect("equal")
        for k, nome in enumerate(config.CLASSES_PT):
            fpr, tpr = curvas_mlp[k]
            ax1.plot(fpr, tpr, color=P.serie(COR_CLASSE[k]), lw=1.8,
                     label=f"{nome}  (AUC {graficos.num(auc_classes[k])})")
        fpr, tpr = curvas_mlp["micro"]
        ax1.plot(fpr, tpr, color=P.texto, lw=1.6, ls="--",
                 label=f"média micro  (AUC {graficos.num(auc_mlp['auc_micro'])})")
        ax1.legend(loc="lower right", fontsize=9)
        ax1.set_title("MLP no teste: uma emoção contra as outras", loc="left")

        fpr, tpr = curvas_mlp["macro"]
        ax2.plot(fpr, tpr, color=P.cor("relu"), lw=2.2,
                 label=f"MLP (nossa)  —  AUROC macro {graficos.num(auc_mlp['auc_macro'])}")
        estilos = {"Regressão logística": dict(color=P.texto2, lw=1.8, ls="--"),
                   "Random Stratified Baseline": dict(color=P.mudo, lw=1.8, ls="-.")}
        for nome, (curva, auc) in curvas_ref.items():
            ax2.plot(*curva, label=f"{nome}  —  {graficos.num(auc)}", **estilos[nome])
        ax2.legend(loc="lower right", fontsize=9)
        ax2.set_title("Média macro: MLP × referências", loc="left")
        ax1.text(0.62, 0.36, "diagonal = sorteio", color=P.mudo, fontsize=8.5, rotation=45)
        fig.tight_layout()
        return fig

    graficos.salvar(desenhar, "06_curva_roc")


if __name__ == "__main__":
    print("Onde a MLP roda:", backend.descrever())
    with open(config.RESULTADOS / "melhor_config.json", encoding="utf-8") as f:
        melhor = json.load(f)
    camadas, l2, ativ = tuple(melhor["camadas"]), melhor["l2"], melhor["ativacao"]
    print(f"Configuração escolhida na validação cruzada: {melhor['config']}")

    treino, validacao, teste = dados.carregar_emotion()
    vet = dados.criar_tfidf()
    X_tr = vet.fit_transform(treino.text)     # o TF-IDF final aprende só com o treino
    X_va, X_te = vet.transform(validacao.text), vet.transform(teste.text)
    y_tr, y_va, y_te = treino.label.values, validacao.label.values, teste.label.values

    modelo = MLP(X_tr.shape[1], camadas, config.N_CLASSES, ativacao=ativ, l2=l2,
                 semente=config.SEMENTE)
    hist = modelo.treinar(X_tr, y_tr, X_va, y_va, epocas=config.EPOCAS,
                          tamanho_lote=config.TAMANHO_LOTE, taxa=config.TAXA_APRENDIZADO,
                          momento=config.MOMENTO, semente=config.SEMENTE)
    P_te = modelo.prever_proba(X_te)
    pred = P_te.argmax(axis=1)

    # referências — todas devolvem probabilidades no formato N x 6
    C, lr = regressao_logistica(X_tr, y_tr, X_va, y_va)
    P_lr = lr.predict_proba(X_te)
    P_estrat = metricas.baseline_estratificado(y_tr, len(y_te), semente=config.SEMENTE)
    P_maj = np.zeros_like(P_te)
    P_maj[:, np.bincount(y_tr).argmax()] = 1

    modelos = {"MLP (nossa)": P_te, "Regressão logística": P_lr,
               "Random Stratified Baseline": P_estrat, "Classe majoritária": P_maj}
    resumo = pd.DataFrame([dict(modelo=nome, **metricas.metricas_completas(y_te, Pm))
                           for nome, Pm in modelos.items()])
    resumo.to_csv(config.RESULTADOS / "06_metricas_resumo.csv", index=False)

    pc = metricas.metricas_por_classe(y_te, P_te)
    por_classe = pd.DataFrame(dict(classe=config.CLASSES_PT, classe_en=config.CLASSES,
                                   precisao=pc["precisao"], revocacao=pc["revocacao"], f1=pc["f1"],
                                   auc=pc["auc"], suporte=pc["suporte"].astype(int)))
    por_classe.to_csv(config.RESULTADOS / "06_metricas_por_classe.csv", index=False)
    cm = confusion_matrix(y_te, pred)

    resultado = dict(config=melhor["config"], camadas=camadas, l2=l2, ativacao=ativ,
                     dispositivo=modelo.dispositivo, n_parametros=modelo.n_parametros(),
                     validacao_cruzada=dict(n_folds=melhor["n_folds"],
                                            f1_macro_media=melhor["f1_val_media"],
                                            f1_macro_ic95=melhor["f1_val_ic"],
                                            acc_media=melhor["acc_val_media"],
                                            acc_ic95=melhor["acc_val_ic"],
                                            f1_por_fold=melhor["f1_por_fold"]),
                     teste={linha["modelo"]: {k: float(v) for k, v in linha.items() if k != "modelo"}
                            for linha in resumo.to_dict(orient="records")},
                     regressao_logistica_C=C,
                     validacao_final=dict(acc=hist["acc_val"][-1], f1=hist["f1_val"][-1],
                                          perda=hist["perda_val"][-1]),
                     epoca_menor_perda_val=int(np.argmin(hist["perda_val"])) + 1,
                     matriz_confusao=cm.tolist(), historico=hist)

    # erros mais "confiantes": onde a rede deu probabilidade alta para a classe errada
    erradas = np.flatnonzero(pred != y_te)
    conf = P_te[erradas, pred[erradas]]
    ordem = erradas[np.argsort(conf)[::-1][:25]]
    pd.DataFrame(dict(texto=teste.text.values[ordem],
                      verdadeira=[config.CLASSES_PT[k] for k in y_te[ordem]],
                      prevista=[config.CLASSES_PT[k] for k in pred[ordem]],
                      prob_prevista=P_te[ordem, pred[ordem]].round(3))
                 ).to_csv(config.RESULTADOS / "06_erros_exemplos.csv", index=False)

    with open(config.RESULTADOS / "06_modelo_final.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=1, ensure_ascii=False)

    curvas_mlp = metricas.curvas_roc(y_te, P_te)
    curvas_ref = {nome: (metricas.curvas_roc(y_te, modelos[nome])["macro"],
                         float(resumo.loc[resumo.modelo == nome, "auc_macro"].iloc[0]))
                  for nome in ("Regressão logística", "Random Stratified Baseline")}
    fig_curva(hist)
    fig_confusao(cm)
    fig_roc(curvas_mlp, pc["auc"], resumo.iloc[0], curvas_ref)

    vc = resultado["validacao_cruzada"]
    print(f"\nValidação cruzada ({vc['n_folds']} folds): F1 macro {vc['f1_macro_media']:.4f} "
          f"(IC 95%: {vc['f1_macro_media'] - vc['f1_macro_ic95']:.4f} a "
          f"{vc['f1_macro_media'] + vc['f1_macro_ic95']:.4f})")
    print(f"\nTeste ({len(y_te)} tweets)")
    pd.set_option("display.width", 200)
    print(resumo.set_index("modelo").round(3).to_string())
    print("\n", por_classe.round(3).to_string(index=False))
