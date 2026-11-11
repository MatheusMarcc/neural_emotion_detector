"""
06 — Modelo final: a melhor configuração da validação cruzada no conjunto de teste.

Agora, e só agora, usamos o conjunto de teste (2.000 tweets que nenhuma
decisão anterior viu). A configuração vencedora do script 03 é treinada no
treino completo (16.000 tweets), com a validação oficial (2.000) acompanhando
a curva de aprendizado, e avaliada uma única vez no teste.

TEORIA — métricas numa base desbalanceada
-----------------------------------------
  precisão(k)  = dos tweets que o modelo chamou de k, quantos eram k
  revocação(k) = dos tweets que eram k, quantos o modelo achou
  F1(k)        = média harmônica das duas = 2·P·R / (P + R)
  F1 macro     = média simples dos F1 das 6 classes (cada classe pesa igual)
  acurácia     = acertos / total (dominada pelas classes grandes)

Um modelo que chutasse sempre "alegria" teria 34,8% de acurácia e F1 macro
de 0,086 no teste — por isso olhamos as duas métricas.

Duas referências para comparar:
  * classe majoritária (o piso);
  * regressão logística no mesmo TF-IDF: um modelo LINEAR. Se a MLP não for
    melhor que ela, as camadas ocultas não estão ajudando nesta tarefa.

Saídas:
  resultados/06_modelo_final.json, 06_metricas_por_classe.csv, 06_erros_exemplos.csv
  figuras/06_curva_modelo_final.png, 06_matriz_confusao.png
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)

from src import config, dados, graficos
from src.mlp import MLP


def regressao_logistica(X_tr, y_tr, X_va, y_va, X_te):
    """Escolhe o C (inverso da regularização) na validação e prevê o teste."""
    melhor = None
    for C in (0.1, 1, 10, 100):
        m = LogisticRegression(C=C, max_iter=3000).fit(X_tr, y_tr)
        f1 = f1_score(y_va, m.predict(X_va), average="macro")
        if melhor is None or f1 > melhor[0]:
            melhor = (f1, C, m)
    return melhor[1], melhor[2].predict(X_te)


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


if __name__ == "__main__":
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

    # referências
    maj = np.full_like(y_te, np.bincount(y_tr).argmax())
    C, pred_lr = regressao_logistica(X_tr, y_tr, X_va, y_va, X_te)

    def metricas(p):
        return dict(acc=float(accuracy_score(y_te, p)),
                    f1_macro=float(f1_score(y_te, p, average="macro")),
                    f1_ponderado=float(f1_score(y_te, p, average="weighted")))

    resultado = dict(config=melhor["config"], camadas=camadas, l2=l2, ativacao=ativ,
                     n_parametros=modelo.n_parametros(),
                     mlp=metricas(pred), classe_majoritaria=metricas(maj),
                     regressao_logistica=dict(C=C, **metricas(pred_lr)),
                     validacao_final=dict(acc=hist["acc_val"][-1], f1=hist["f1_val"][-1],
                                          perda=hist["perda_val"][-1]),
                     epoca_menor_perda_val=int(np.argmin(hist["perda_val"])) + 1,
                     historico=hist)

    rel = classification_report(y_te, pred, target_names=config.CLASSES_PT, output_dict=True,
                                zero_division=0)
    por_classe = pd.DataFrame([dict(classe=c, classe_en=config.CLASSES[k],
                                    precisao=rel[c]["precision"], revocacao=rel[c]["recall"],
                                    f1=rel[c]["f1-score"], suporte=int(rel[c]["support"]))
                               for k, c in enumerate(config.CLASSES_PT)])
    por_classe.to_csv(config.RESULTADOS / "06_metricas_por_classe.csv", index=False)
    cm = confusion_matrix(y_te, pred)
    resultado["matriz_confusao"] = cm.tolist()

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
    fig_curva(hist)
    fig_confusao(cm)

    print(f"\nTeste ({len(y_te)} tweets)")
    for nome, chave in (("MLP (nossa)", "mlp"), ("Regressão logística", "regressao_logistica"),
                        ("Classe majoritária", "classe_majoritaria")):
        r = resultado[chave]
        print(f"  {nome:22s} acurácia {r['acc']:.4f}   F1 macro {r['f1_macro']:.4f}")
    print("\n", por_classe.round(3).to_string(index=False))
