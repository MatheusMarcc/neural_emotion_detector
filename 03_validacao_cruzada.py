"""
03 — Validação cruzada 3-fold para escolher os hiperparâmetros.

TEORIA — por que validação cruzada?
-----------------------------------
Se escolhermos os hiperparâmetros olhando o desempenho no conjunto de teste,
o teste deixa de ser uma estimativa honesta (escolhemos justamente o que foi
bem nele). A validação cruzada estima o desempenho só com o treino:

  1. divide o treino (16.000 tweets) em k = 3 partes (folds) estratificadas;
  2. para cada fold i: treina nos outros 2 folds (~10.667 tweets) e avalia
     no fold i (~5.333 tweets);
  3. a nota da configuração é a MÉDIA dos 3 folds, e o desvio padrão entre
     eles mostra o quanto essa nota depende de quais dados caíram em cada parte.

Todos os 3 folds são usados uma vez como validação, então todo exemplo do
treino ajuda a avaliar. O conjunto de teste só é usado no script 06, uma única
vez, com a configuração já escolhida.

Grade (o que o enunciado pede para variar), 2 × 2 × 3 = 12 configurações:
  ativação  : ReLU, Sigmoid
  L2        : sem regularização (λ = 0) e λ = 1e-4
  neurônios : (16, 8), (64, 32), (256, 128)

Fixos: SGD com momentum 0,9, η = 0,05, lote 64, 50 épocas (ver src/config.py).
A nota de cada fold é a da ÚLTIMA época (orçamento fixo de 50 épocas, sem
parada antecipada), para que overfitting e underfitting pesem na escolha.

Saídas:
  resultados/03_cv_resultados.csv   uma linha por configuração (média ± desvio)
  resultados/03_cv_por_fold.csv     uma linha por configuração × fold
  resultados/03_historicos.json     curvas época a época (usadas no script 04)
  resultados/melhor_config.json     a configuração escolhida
  figuras/03_validacao_cruzada.png
"""
import itertools
import json
import time

import numpy as np
import pandas as pd

from src import config, dados, graficos
from src.mlp import MLP


def nome_config(ativacao, l2, camadas):
    return f"{ativacao} | L2={'sem' if l2 == 0 else f'{l2:g}'} | {camadas[0]}-{camadas[1]}"


def rodar_grade(folds):
    grade = list(itertools.product(config.ATIVACOES, config.L2_VALORES, config.NEURONIOS))
    linhas, historicos = [], {}
    for k, (ativ, l2, camadas) in enumerate(grade, 1):
        nome = nome_config(ativ, l2, camadas)
        historicos[nome] = []
        t0 = time.perf_counter()
        for fold in folds:
            modelo = MLP(fold["X_tr"].shape[1], camadas, config.N_CLASSES,
                         ativacao=ativ, l2=l2, semente=config.SEMENTE + fold["fold"])
            hist = modelo.treinar(fold["X_tr"], fold["y_tr"], fold["X_va"], fold["y_va"],
                                  epocas=config.EPOCAS, tamanho_lote=config.TAMANHO_LOTE,
                                  taxa=config.TAXA_APRENDIZADO, momento=config.MOMENTO,
                                  semente=config.SEMENTE + fold["fold"])
            historicos[nome].append(hist)
            melhor_ep = int(np.argmax(hist["f1_val"])) + 1
            linhas.append(dict(
                config=nome, ativacao=ativ, l2=l2, n1=camadas[0], n2=camadas[1],
                fold=fold["fold"],
                acc_treino=hist["acc_treino"][-1], f1_treino=hist["f1_treino"][-1],
                perda_treino=hist["perda_treino"][-1],
                acc_val=hist["acc_val"][-1], f1_val=hist["f1_val"][-1],
                perda_val=hist["perda_val"][-1],
                melhor_epoca_f1=melhor_ep, melhor_f1_val=max(hist["f1_val"]),
                epoca_menor_perda_val=int(np.argmin(hist["perda_val"])) + 1,
            ))
        dt = time.perf_counter() - t0
        f1s = [l["f1_val"] for l in linhas[-len(folds):]]
        print(f"[{k:2d}/12] {nome:32s} F1 val = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}"
              f"   ({dt:.0f}s)", flush=True)
    return pd.DataFrame(linhas), historicos


def resumir(por_fold):
    metricas = ["acc_treino", "f1_treino", "perda_treino", "acc_val", "f1_val",
                "perda_val", "melhor_epoca_f1", "melhor_f1_val", "epoca_menor_perda_val"]
    g = por_fold.groupby(["config", "ativacao", "l2", "n1", "n2"], sort=False)[metricas]
    resumo = g.mean().add_suffix("_media").join(g.std(ddof=0).add_suffix("_desvio"))
    resumo = resumo.reset_index()
    # diferença treino - validação: o "gap" usado na análise de variância
    resumo["gap_f1"] = resumo["f1_treino_media"] - resumo["f1_val_media"]
    resumo["gap_acc"] = resumo["acc_treino_media"] - resumo["acc_val_media"]
    return resumo.sort_values("f1_val_media", ascending=False).reset_index(drop=True)


def figura(resumo):
    def desenhar(P):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        camadas_txt = [f"{a}-{b}" for a, b in config.NEURONIOS]
        deslocamentos = {("relu", 1e-4): -0.27, ("relu", 0.0): -0.09,
                         ("sigmoid", 1e-4): 0.09, ("sigmoid", 0.0): 0.27}
        for _, r in resumo.iterrows():
            yb = camadas_txt.index(f"{r.n1}-{r.n2}") + deslocamentos[(r.ativacao, r.l2)]
            cor = P.cor(r.ativacao)
            cheio = r.l2 > 0
            ax.errorbar(r.f1_val_media, yb, xerr=r.f1_val_desvio, fmt="o", ms=8,
                        color=cor, mfc=cor if cheio else P.fundo, mec=cor, mew=1.8,
                        ecolor=cor, elinewidth=1.5, capsize=3)
            # rótulo depois da ponta da barra de erro, para não cobrir nada
            ax.annotate(graficos.num(r.f1_val_media), (r.f1_val_media + r.f1_val_desvio, yb),
                        xytext=(7, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=P.texto2)
        ax.set_yticks(range(len(camadas_txt)), [f"neurônios\n{c}" for c in camadas_txt])
        ax.invert_yaxis()
        ax.set_xlabel("F1 macro na validação (média dos 3 folds, barra = ±1 desvio)")
        ax.set_xlim(0, 1)
        ax.grid(axis="y", visible=False)
        legenda = [Line2D([], [], color=P.cor("relu"), marker="o", ls="none", ms=8, label="ReLU"),
                   Line2D([], [], color=P.cor("sigmoid"), marker="o", ls="none", ms=8, label="Sigmoid"),
                   Line2D([], [], color=P.texto2, marker="o", ls="none", ms=8, label="com L2 (cheio)"),
                   Line2D([], [], color=P.texto2, marker="o", ls="none", ms=8, mfc=P.fundo,
                          mew=1.8, label="sem L2 (vazado)")]
        ax.legend(handles=legenda, loc="upper left", ncol=2)
        ax.set_title("Validação cruzada 3-fold: 12 configurações", loc="left")
        fig.tight_layout()
        return fig

    graficos.salvar(desenhar, "03_validacao_cruzada")


if __name__ == "__main__":
    treino, _, _ = dados.carregar_emotion()
    print("Montando os 3 folds (TF-IDF ajustado dentro de cada fold)...")
    folds = dados.montar_folds_tfidf(treino.text, treino.label.values)
    for f in folds:
        print(f"  fold {f['fold']}: treino {f['X_tr'].shape[0]} | validação {f['X_va'].shape[0]}"
              f" | vocabulário {f['X_tr'].shape[1]}")

    por_fold, historicos = rodar_grade(folds)
    resumo = resumir(por_fold)

    por_fold.to_csv(config.RESULTADOS / "03_cv_por_fold.csv", index=False)
    resumo.to_csv(config.RESULTADOS / "03_cv_resultados.csv", index=False)
    with open(config.RESULTADOS / "03_historicos.json", "w", encoding="utf-8") as f:
        json.dump(historicos, f)

    melhor = resumo.iloc[0]
    escolha = dict(config=melhor.config, ativacao=melhor.ativacao, l2=float(melhor.l2),
                   camadas=[int(melhor.n1), int(melhor.n2)],
                   f1_val_media=float(melhor.f1_val_media),
                   f1_val_desvio=float(melhor.f1_val_desvio),
                   acc_val_media=float(melhor.acc_val_media),
                   acc_val_desvio=float(melhor.acc_val_desvio),
                   criterio="maior F1 macro média nos 3 folds (última época)")
    with open(config.RESULTADOS / "melhor_config.json", "w", encoding="utf-8") as f:
        json.dump(escolha, f, indent=1, ensure_ascii=False)

    figura(resumo)
    cols = ["config", "acc_val_media", "acc_val_desvio", "f1_val_media", "f1_val_desvio",
            "f1_treino_media", "gap_f1", "perda_val_media"]
    print("\n", resumo[cols].round(4).to_string(index=False))
    print(f"\nMelhor configuração: {melhor.config}")
