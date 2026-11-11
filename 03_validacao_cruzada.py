"""
03 — Validação cruzada 5-fold para escolher os hiperparâmetros.

TEORIA — por que validação cruzada?
-----------------------------------
Se escolhermos os hiperparâmetros olhando o desempenho no conjunto de teste,
o teste deixa de ser uma estimativa honesta (escolhemos justamente o que foi
bem nele). A validação cruzada estima o desempenho só com o treino:

  1. divide o treino (16.000 tweets) em k = 5 partes (folds) estratificadas
     (cada parte mantém a proporção das 6 emoções);
  2. para cada fold i: treina nos outros 4 folds (12.800 tweets) e avalia no
     fold i (3.200 tweets);
  3. a nota da configuração é a MÉDIA dos 5 folds. O intervalo de confiança de
     95% dessa média (t de Student, ver src/metricas.py) mostra o quanto a nota
     depende de quais dados caíram em cada parte: duas configurações cujos
     intervalos se sobrepõem bastante estão, na prática, empatadas.

Todos os 5 folds são usados uma vez como validação, então todo exemplo do
treino ajuda a avaliar. O conjunto de teste só é usado no script 06, uma única
vez, com a configuração já escolhida.

Grade (o que o enunciado pede para variar), 2 × 2 × 3 = 12 configurações:
  ativação  : ReLU, Sigmoid
  L2        : sem regularização (λ = 0) e λ = 1e-4
  neurônios : (16, 8), (64, 32), (256, 128)

Fixos: SGD com momentum 0,9, η = 0,05, lote 64, 50 épocas (ver src/config.py).
A nota de cada fold é a da ÚLTIMA época (orçamento fixo de 50 épocas, sem
parada antecipada), para que overfitting e underfitting pesem na escolha.

Métricas em cada fold (src/metricas.py): acurácia (= F1 micro), F1 macro,
precisão e revocação macro e micro, AUROC macro e micro. O critério de escolha
continua sendo o F1 macro (a base é desbalanceada). Nos mesmos 5 folds avaliamos
o Random Stratified Baseline (sorteia a classe na proporção do treino): é o piso
com que as 12 configurações são comparadas.

Saídas:
  resultados/03_cv_resultados.csv        uma linha por configuração (média, desvio, IC)
  resultados/03_cv_por_fold.csv          uma linha por configuração × fold
  resultados/03_baseline_estratificado.csv  o baseline em cada fold e a média com IC
  resultados/03_historicos.json          curvas época a época (usadas no script 04)
  resultados/melhor_config.json          a configuração escolhida
  figuras/03_validacao_cruzada.png
"""
import itertools
import json
import time

import numpy as np
import pandas as pd

from src import backend, config, dados, graficos, metricas
from src.mlp import MLP

# métricas resumidas por média, desvio e intervalo de confiança
METRICAS_VAL = ["acc_val", "f1_val", "f1_micro_val", "precisao_macro_val", "revocacao_macro_val",
                "precisao_micro_val", "revocacao_micro_val", "auc_macro_val", "auc_micro_val",
                "perda_val"]
METRICAS_OUTRAS = ["acc_treino", "f1_treino", "perda_treino", "melhor_epoca_f1", "melhor_f1_val",
                   "epoca_menor_perda_val"]


def nome_config(ativacao, l2, camadas):
    return f"{ativacao} | L2={'sem' if l2 == 0 else f'{l2:g}'} | {camadas[0]}-{camadas[1]}"


def metricas_validacao(y_va, P):
    """As métricas do enunciado no fold de validação, com sufixo _val."""
    m = metricas.metricas_completas(y_va, P, config.N_CLASSES)
    return {"f1_micro_val": m["f1_micro"],
            "precisao_macro_val": m["precisao_macro"], "revocacao_macro_val": m["revocacao_macro"],
            "precisao_micro_val": m["precisao_micro"], "revocacao_micro_val": m["revocacao_micro"],
            "auc_macro_val": m["auc_macro"], "auc_micro_val": m["auc_micro"]}


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
            P_va = modelo.prever_proba(fold["X_va"])          # probabilidades na última época
            melhor_ep = int(np.argmax(hist["f1_val"])) + 1
            linhas.append(dict(
                config=nome, ativacao=ativ, l2=l2, n1=camadas[0], n2=camadas[1],
                fold=fold["fold"], dispositivo=modelo.dispositivo,
                acc_treino=hist["acc_treino"][-1], f1_treino=hist["f1_treino"][-1],
                perda_treino=hist["perda_treino"][-1],
                acc_val=hist["acc_val"][-1], f1_val=hist["f1_val"][-1],
                perda_val=hist["perda_val"][-1],
                **metricas_validacao(fold["y_va"], P_va),
                melhor_epoca_f1=melhor_ep, melhor_f1_val=max(hist["f1_val"]),
                epoca_menor_perda_val=int(np.argmin(hist["perda_val"])) + 1,
            ))
        dt = time.perf_counter() - t0
        f1s = [l["f1_val"] for l in linhas[-len(folds):]]
        media, ic = metricas.intervalo_confianca(f1s)
        onde = "GPU" if linhas[-1]["dispositivo"] == "cuda" else linhas[-1]["dispositivo"].upper()
        print(f"[{k:2d}/12] {nome:32s} F1 val = {media:.4f} ± {ic:.4f} (IC 95%)"
              f"   ({dt:.0f}s, {onde})", flush=True)
    return pd.DataFrame(linhas), historicos


def rodar_baseline(folds):
    """Random Stratified Baseline nos mesmos 5 folds, com as mesmas métricas."""
    linhas = []
    for fold in folds:
        P = metricas.baseline_estratificado(fold["y_tr"], len(fold["y_va"]),
                                            semente=config.SEMENTE + fold["fold"])
        m = metricas.metricas_completas(fold["y_va"], P, config.N_CLASSES)
        linhas.append(dict(config="baseline estratificado", fold=fold["fold"],
                           acc_val=m["acc"], f1_val=m["f1_macro"],
                           **metricas_validacao(fold["y_va"], P)))
    return pd.DataFrame(linhas)


def resumir(por_fold, chaves=("config", "ativacao", "l2", "n1", "n2"), ordenar=True):
    """Média, desvio padrão (amostral) e meia-largura do IC 95% de cada métrica."""
    cols = [c for c in METRICAS_VAL + METRICAS_OUTRAS if c in por_fold.columns]
    g = por_fold.groupby(list(chaves), sort=False)[cols]
    resumo = g.mean().add_suffix("_media").join(g.std(ddof=1).add_suffix("_desvio"))
    ic = g.agg(lambda v: metricas.intervalo_confianca(v, config.NIVEL_CONFIANCA)[1])
    resumo = resumo.join(ic.add_suffix("_ic")).reset_index()
    if "f1_treino_media" in resumo:
        # diferença treino - validação: o "gap" usado na análise de variância
        resumo["gap_f1"] = resumo["f1_treino_media"] - resumo["f1_val_media"]
        resumo["gap_acc"] = resumo["acc_treino_media"] - resumo["acc_val_media"]
    if ordenar:
        resumo = resumo.sort_values("f1_val_media", ascending=False).reset_index(drop=True)
    return resumo


def figura(resumo, base):
    def desenhar(P):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        camadas_txt = [f"{a}-{b}" for a, b in config.NEURONIOS]
        deslocamentos = {("relu", 1e-4): -0.27, ("relu", 0.0): -0.09,
                         ("sigmoid", 1e-4): 0.09, ("sigmoid", 0.0): 0.27}
        f1_base = float(base["f1_val_media"].iloc[0])
        ax.axvline(f1_base, color=P.texto2, ls="--", lw=1.2)
        ax.text(f1_base + 0.01, 2.58, f"Random Stratified Baseline (sorteio): {graficos.num(f1_base)}",
                fontsize=8.5, color=P.texto2, va="center")
        for _, r in resumo.iterrows():
            yb = camadas_txt.index(f"{r.n1}-{r.n2}") + deslocamentos[(r.ativacao, r.l2)]
            cor = P.cor(r.ativacao)
            cheio = r.l2 > 0
            ax.errorbar(r.f1_val_media, yb, xerr=r.f1_val_ic, fmt="o", ms=8,
                        color=cor, mfc=cor if cheio else P.fundo, mec=cor, mew=1.8,
                        ecolor=cor, elinewidth=1.5, capsize=3)
            # rótulo depois da ponta da barra de erro, para não cobrir nada
            ax.annotate(graficos.num(r.f1_val_media), (r.f1_val_media + r.f1_val_ic, yb),
                        xytext=(7, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=P.texto2)
        ax.set_yticks(range(len(camadas_txt)), [f"neurônios\n{c}" for c in camadas_txt])
        ax.set_ylim(2.8, -0.95)      # eixo invertido, com espaço para a legenda e o baseline
        ax.set_xlabel(f"F1 macro na validação (média dos {config.N_FOLDS} folds, "
                      f"barra = IC {config.NIVEL_CONFIANCA:.0%})")
        ax.set_xlim(0, 1)
        ax.grid(axis="y", visible=False)
        legenda = [Line2D([], [], color=P.cor("relu"), marker="o", ls="none", ms=8, label="ReLU"),
                   Line2D([], [], color=P.cor("sigmoid"), marker="o", ls="none", ms=8, label="Sigmoid"),
                   Line2D([], [], color=P.texto2, marker="o", ls="none", ms=8, label="com L2 (cheio)"),
                   Line2D([], [], color=P.texto2, marker="o", ls="none", ms=8, mfc=P.fundo,
                          mew=1.8, label="sem L2 (vazado)")]
        ax.legend(handles=legenda, loc="upper left", ncol=2)
        ax.set_title(f"Validação cruzada {config.N_FOLDS}-fold: 12 configurações", loc="left")
        fig.tight_layout()
        return fig

    graficos.salvar(desenhar, "03_validacao_cruzada")


if __name__ == "__main__":
    print("Onde a MLP roda:", backend.descrever())
    treino, _, _ = dados.carregar_emotion()
    print(f"Montando os {config.N_FOLDS} folds (TF-IDF ajustado dentro de cada fold)...")
    folds = dados.montar_folds_tfidf(treino.text, treino.label.values)
    for f in folds:
        print(f"  fold {f['fold']}: treino {f['X_tr'].shape[0]} | validação {f['X_va'].shape[0]}"
              f" | vocabulário {f['X_tr'].shape[1]}")

    por_fold, historicos = rodar_grade(folds)
    resumo = resumir(por_fold)
    base_fold = rodar_baseline(folds)
    base = resumir(base_fold, chaves=("config",), ordenar=False)

    por_fold.to_csv(config.RESULTADOS / "03_cv_por_fold.csv", index=False)
    resumo.to_csv(config.RESULTADOS / "03_cv_resultados.csv", index=False)
    pd.concat([base_fold, base], ignore_index=True).to_csv(
        config.RESULTADOS / "03_baseline_estratificado.csv", index=False)
    with open(config.RESULTADOS / "03_historicos.json", "w", encoding="utf-8") as f:
        json.dump(historicos, f)

    melhor = resumo.iloc[0]
    escolha = dict(config=melhor.config, ativacao=melhor.ativacao, l2=float(melhor.l2),
                   camadas=[int(melhor.n1), int(melhor.n2)],
                   n_folds=config.N_FOLDS, nivel_confianca=config.NIVEL_CONFIANCA,
                   f1_val_media=float(melhor.f1_val_media),
                   f1_val_desvio=float(melhor.f1_val_desvio),
                   f1_val_ic=float(melhor.f1_val_ic),
                   acc_val_media=float(melhor.acc_val_media),
                   acc_val_desvio=float(melhor.acc_val_desvio),
                   acc_val_ic=float(melhor.acc_val_ic),
                   auc_macro_val_media=float(melhor.auc_macro_val_media),
                   auc_macro_val_ic=float(melhor.auc_macro_val_ic),
                   f1_por_fold=[float(v) for v in por_fold.loc[por_fold.config == melhor.config, "f1_val"]],
                   baseline_f1_val_media=float(base.f1_val_media.iloc[0]),
                   baseline_acc_val_media=float(base.acc_val_media.iloc[0]),
                   criterio=f"maior F1 macro média nos {config.N_FOLDS} folds (última época)")
    with open(config.RESULTADOS / "melhor_config.json", "w", encoding="utf-8") as f:
        json.dump(escolha, f, indent=1, ensure_ascii=False)

    figura(resumo, base)
    pd.set_option("display.width", 200)
    cols = ["config", "acc_val_media", "f1_val_media", "f1_val_ic", "auc_macro_val_media",
            "f1_treino_media", "gap_f1", "perda_val_media"]
    print("\n", resumo[cols].round(4).to_string(index=False))
    b = base.iloc[0]
    print(f"\nRandom Stratified Baseline: acurácia {b.acc_val_media:.4f} ± {b.acc_val_ic:.4f} | "
          f"F1 macro {b.f1_val_media:.4f} ± {b.f1_val_ic:.4f} | AUROC macro {b.auc_macro_val_media:.4f}")
    print(f"\nMelhor configuração: {melhor.config} — F1 macro {melhor.f1_val_media:.4f} "
          f"(IC 95%: {melhor.f1_val_media - melhor.f1_val_ic:.4f} a {melhor.f1_val_media + melhor.f1_val_ic:.4f})")
