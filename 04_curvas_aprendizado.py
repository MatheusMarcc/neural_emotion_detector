"""
04 — Curvas de aprendizado: overfitting, underfitting, viés e variância.

TEORIA
------
Erro de um modelo = viés² + variância + ruído irredutível.

  * VIÉS alto (UNDERFITTING): o modelo não consegue nem ajustar o treino.
    Sintoma: desempenho ruim no treino E na validação, com os dois próximos.
    Mais dados não ajudam; é preciso um modelo mais flexível ou um
    treinamento que funcione.
  * VARIÂNCIA alta (OVERFITTING): o modelo decora o treino, inclusive o ruído.
    Sintoma: treino muito melhor que a validação (gap grande) e custo de
    validação que SOBE depois de certa época, enquanto o de treino segue
    caindo. Mais dados ou regularização (L2) ajudam.
  * RUÍDO: na Emotion os rótulos vieram de hashtags, então parte deles é
    discutível — nenhum modelo chega a 100% na validação.

Dois tipos de curva respondem a perguntas diferentes:

  A) CURVA POR ÉPOCA (custo e F1 de treino × validação ao longo do treino).
     Mostra QUANDO o overfitting começa: o ponto em que o custo de validação
     para de cair e passa a subir. Usamos os históricos salvos pelo script 03
     (média dos 3 folds, faixa = ±1 desvio).

  B) CURVA POR TAMANHO DO TREINO (F1 final × número de exemplos de treino).
     Treinamos com 10%, 20%, 40%, 70% e 100% de cada fold de treino e
     avaliamos sempre no mesmo fold de validação.
       - gap grande que diminui com mais dados  -> variância alta
       - curvas juntas e baixas                 -> viés alto
     Para comparar só o efeito da QUANTIDADE de dados, todas as rodadas fazem
     o mesmo número de atualizações do SGD (50 épocas do fold completo); com
     10% dos dados isso dá ~500 épocas. Se fixássemos 50 épocas, os conjuntos
     pequenos teriam 10× menos atualizações e misturaríamos dois efeitos.
     (O TF-IDF de cada fold é o do script 03, ajustado no treino do fold.)

Saídas:
  figuras/04_curvas_perda.png, 04_curvas_f1.png    (as 12 configurações)
  figuras/04_curvas_representativas.png             (4 casos típicos)
  figuras/04_curva_tamanho_treino.png               (curva B)
  resultados/04_vies_variancia.csv, 04_curva_tamanho.csv
"""
import json
import math

import numpy as np
import pandas as pd

from src import config, dados, graficos
from src.mlp import MLP

FRACOES = [0.1, 0.2, 0.4, 0.7, 1.0]
# casos típicos para a curva B: (ativação, L2, camadas, papel)
CASOS_TAMANHO = [
    ("relu", 0.0, (256, 128), "maior capacidade, sem L2"),
    ("relu", 1e-4, (256, 128), "melhor configuração"),
    ("sigmoid", 1e-4, (16, 8), "sigmoid pequena, com L2"),
]
ORDEM_COLUNAS = [("relu", 1e-4), ("relu", 0.0), ("sigmoid", 1e-4), ("sigmoid", 0.0)]


def nome_config(ativacao, l2, camadas):
    return f"{ativacao} | L2={'sem' if l2 == 0 else f'{l2:g}'} | {camadas[0]}-{camadas[1]}"


def media_desvio(historicos, chave):
    m = np.array([h[chave] for h in historicos])
    return m.mean(axis=0), m.std(axis=0)


# ---------------------------------------------------------------------------
# A) Curvas por época (a partir do script 03)
# ---------------------------------------------------------------------------
def fig_grade(historicos, chave_tr, chave_va, nome_arquivo, titulo, ylabel, ylim):
    ep = np.arange(1, config.EPOCAS + 1)

    def desenhar(P):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        fig, eixos = plt.subplots(3, 4, figsize=(15, 9.5), sharex=True, sharey=True)
        for i, camadas in enumerate(config.NEURONIOS):
            for j, (ativ, l2) in enumerate(ORDEM_COLUNAS):
                ax = eixos[i, j]
                h = historicos[nome_config(ativ, l2, camadas)]
                cor = P.cor(ativ)
                for chave, estilo, alfa in ((chave_tr, "--", 0.12), (chave_va, "-", 0.18)):
                    m, d = media_desvio(h, chave)
                    ax.plot(ep, m, color=cor, ls=estilo, lw=1.8)
                    ax.fill_between(ep, m - d, m + d, color=cor, alpha=alfa, lw=0)
                if i == 0:
                    ax.set_title(f"{graficos.NOME_ATIVACAO[ativ]} · "
                                 f"{'com L2' if l2 > 0 else 'sem L2'}", fontsize=11)
                if j == 0:
                    ax.set_ylabel(f"{camadas[0]}-{camadas[1]}\n{ylabel}")
                if i == 2:
                    ax.set_xlabel("época")
                ax.set_ylim(*ylim)
        legenda = [Line2D([], [], color=P.texto2, ls="--", lw=1.8, label="treino"),
                   Line2D([], [], color=P.texto2, ls="-", lw=1.8, label="validação"),
                   Line2D([], [], color=P.cor("relu"), lw=1.8, label="ReLU"),
                   Line2D([], [], color=P.cor("sigmoid"), lw=1.8, label="Sigmoid")]
        fig.legend(handles=legenda, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle(titulo, x=0.01, ha="left", fontsize=13, color=P.texto)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        return fig

    graficos.salvar(desenhar, nome_arquivo)


def fig_representativas(historicos):
    casos = [(("relu", 1e-4, (256, 128)), "melhor configuração"),
             (("relu", 0.0, (256, 128)), "sem L2: custo de validação sobe"),
             (("sigmoid", 0.0, (16, 8)), "sigmoid pequena: aprende devagar"),
             (("sigmoid", 0.0, (256, 128)), "sigmoid larga: presa no platô")]
    ep = np.arange(1, config.EPOCAS + 1)

    def desenhar(P):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        fig, eixos = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
        for ax, ((ativ, l2, camadas), legenda_caso) in zip(eixos, casos):
            h = historicos[nome_config(ativ, l2, camadas)]
            cor = P.cor(ativ)
            for chave, estilo in (("perda_treino", "--"), ("perda_val", "-")):
                m, d = media_desvio(h, chave)
                ax.plot(ep, m, color=cor, ls=estilo)
                ax.fill_between(ep, m - d, m + d, color=cor, alpha=0.15, lw=0)
            mv, _ = media_desvio(h, "perda_val")
            k = int(np.argmin(mv))
            if ativ == "relu":
                ax.plot(ep[k], mv[k], "o", color=cor, ms=8, mec=P.fundo, mew=2)
                ax.annotate(f"mínimo na época {k + 1}", (ep[k], mv[k]), xytext=(8, -16),
                            textcoords="offset points", fontsize=8.5, color=P.texto2)
            ax.set_title(f"{graficos.NOME_ATIVACAO[ativ]} {camadas[0]}-{camadas[1]} · "
                         f"{'L2' if l2 > 0 else 'sem L2'}\n{legenda_caso}", fontsize=10.5,
                         loc="left")
            ax.set_xlabel("época")
            ax.set_ylim(0, 1.75)
        eixos[0].set_ylabel("custo (entropia cruzada)")
        legenda = [Line2D([], [], color=P.texto2, ls="--", label="treino"),
                   Line2D([], [], color=P.texto2, ls="-", label="validação")]
        eixos[0].legend(handles=legenda, loc="upper right")
        fig.tight_layout()
        return fig

    graficos.salvar(desenhar, "04_curvas_representativas")


def tabela_vies_variancia(historicos):
    linhas = []
    for nome, h in historicos.items():
        f1_tr, _ = media_desvio(h, "f1_treino")
        f1_va, f1_va_d = media_desvio(h, "f1_val")
        pv, _ = media_desvio(h, "perda_val")
        pt, _ = media_desvio(h, "perda_treino")
        k = int(np.argmin(pv))
        linhas.append(dict(
            config=nome, f1_treino=f1_tr[-1], f1_val=f1_va[-1], f1_val_desvio_folds=f1_va_d[-1],
            gap_f1=f1_tr[-1] - f1_va[-1], perda_treino=pt[-1], perda_val=pv[-1],
            perda_val_minima=pv[k], epoca_perda_val_minima=k + 1,
            aumento_perda_val_pct=100 * (pv[-1] / pv[k] - 1),
            melhor_f1_val=float(f1_va.max()), epoca_melhor_f1_val=int(np.argmax(f1_va)) + 1,
        ))
    t = pd.DataFrame(linhas)
    # Diagnóstico com regras explícitas (limiares escolhidos olhando as curvas):
    #   F1 de treino < 0,90                     -> não ajusta nem o treino: viés alto
    #   custo de validação termina > 50% acima   -> variância alta: o modelo ficou
    #   do seu mínimo                              superconfiante nos erros
    #   caso contrário (treino ~1, custo de       -> variância moderada: decora o
    #   validação sobe pouco)                      treino, mas a L2 segura o custo
    def diagnostico(r):
        if r.f1_treino < 0.90:
            return "underfitting (viés alto)"
        if r.aumento_perda_val_pct > 50:
            return "overfitting (variância alta)"
        return "overfitting moderado (variância contida)"
    t["diagnostico"] = t.apply(diagnostico, axis=1)
    return t.sort_values("f1_val", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# B) Curva por tamanho do treino
# ---------------------------------------------------------------------------
def subamostra_estratificada(y, fracao, rng):
    """Sorteia `fracao` dos índices mantendo a proporção de cada classe."""
    idx = []
    for k in range(config.N_CLASSES):
        da_classe = np.flatnonzero(y == k)
        n = max(1, int(round(fracao * len(da_classe))))
        idx.append(rng.choice(da_classe, size=n, replace=False))
    return np.sort(np.concatenate(idx))


def curva_tamanho(folds):
    linhas = []
    for ativ, l2, camadas, papel in CASOS_TAMANHO:
        for fold in folds:
            n_full = fold["X_tr"].shape[0]
            passos_alvo = config.EPOCAS * math.ceil(n_full / config.TAMANHO_LOTE)
            for fr in FRACOES:
                rng = np.random.default_rng(config.SEMENTE + fold["fold"])
                idx = subamostra_estratificada(fold["y_tr"], fr, rng)
                X_sub, y_sub = fold["X_tr"][idx], fold["y_tr"][idx]
                epocas = round(passos_alvo / math.ceil(len(idx) / config.TAMANHO_LOTE))
                modelo = MLP(X_sub.shape[1], camadas, config.N_CLASSES, ativacao=ativ, l2=l2,
                             semente=config.SEMENTE + fold["fold"])
                modelo.treinar(X_sub, y_sub, epocas=epocas, tamanho_lote=config.TAMANHO_LOTE,
                               taxa=config.TAXA_APRENDIZADO, momento=config.MOMENTO,
                               semente=config.SEMENTE + fold["fold"], avaliar_cada_epoca=False)
                m_tr = modelo.avaliar(X_sub, y_sub)
                m_va = modelo.avaliar(fold["X_va"], fold["y_va"])
                linhas.append(dict(config=nome_config(ativ, l2, camadas), papel=papel,
                                   ativacao=ativ, fold=fold["fold"], fracao=fr,
                                   n_treino=len(idx), epocas=epocas,
                                   f1_treino=m_tr["f1"], f1_val=m_va["f1"],
                                   acc_treino=m_tr["acc"], acc_val=m_va["acc"],
                                   perda_treino=m_tr["perda"], perda_val=m_va["perda"]))
                print(f"  {nome_config(ativ, l2, camadas):30s} fold {fold['fold']} "
                      f"{fr:>4.0%} ({len(idx):5d} ex., {epocas:3d} épocas): "
                      f"F1 treino {m_tr['f1']:.3f}  F1 val {m_va['f1']:.3f}", flush=True)
    return pd.DataFrame(linhas)


def fig_tamanho(df):
    def desenhar(P):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        fig, eixos = plt.subplots(1, len(CASOS_TAMANHO), figsize=(15, 4.4), sharey=True)
        for ax, (ativ, l2, camadas, papel) in zip(eixos, CASOS_TAMANHO):
            d = df[df.config == nome_config(ativ, l2, camadas)]
            g = d.groupby("fracao").agg(n=("n_treino", "mean"),
                                        tr=("f1_treino", "mean"), tr_d=("f1_treino", "std"),
                                        va=("f1_val", "mean"), va_d=("f1_val", "std"))
            cor = P.cor(ativ)
            for col, estilo in (("tr", "--"), ("va", "-")):
                ax.plot(g.n, g[col], color=cor, ls=estilo, marker="o", ms=6,
                        mec=P.fundo, mew=1.5)
                ax.fill_between(g.n, g[col] - g[f"{col}_d"], g[col] + g[f"{col}_d"],
                                color=cor, alpha=0.15, lw=0)
            ultimo = g.iloc[-1]
            ax.annotate(f"gap = {ultimo.tr - ultimo.va:.2f}".replace(".", ","),
                        (ultimo.n, (ultimo.tr + ultimo.va) / 2), xytext=(-8, 0),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=9, color=P.texto2)
            ax.set_title(f"{graficos.NOME_ATIVACAO[ativ]} {camadas[0]}-{camadas[1]} · "
                         f"{'L2' if l2 > 0 else 'sem L2'}\n{papel}", loc="left", fontsize=10.5)
            ax.set_xlabel("exemplos de treino")
            ax.set_ylim(0, 1.05)
        eixos[0].set_ylabel("F1 macro")
        legenda = [Line2D([], [], color=P.texto2, ls="--", label="treino"),
                   Line2D([], [], color=P.texto2, ls="-", label="validação")]
        eixos[0].legend(handles=legenda, loc="lower right")
        fig.suptitle("Curva de aprendizado pelo tamanho do treino (mesmo nº de atualizações do SGD)",
                     x=0.01, ha="left", fontsize=12.5, color=P.texto)
        fig.tight_layout()
        return fig

    graficos.salvar(desenhar, "04_curva_tamanho_treino")


if __name__ == "__main__":
    with open(config.RESULTADOS / "03_historicos.json", encoding="utf-8") as f:
        historicos = json.load(f)

    print("A) Curvas por época das 12 configurações (históricos do script 03)")
    fig_grade(historicos, "perda_treino", "perda_val", "04_curvas_perda",
              "Custo (entropia cruzada) por época — treino tracejado, validação contínua",
              "custo", (0, 1.75))
    fig_grade(historicos, "f1_treino", "f1_val", "04_curvas_f1",
              "F1 macro por época — treino tracejado, validação contínua", "F1 macro", (0, 1.02))
    fig_representativas(historicos)
    tabela = tabela_vies_variancia(historicos)
    tabela.to_csv(config.RESULTADOS / "04_vies_variancia.csv", index=False)
    print(tabela[["config", "f1_treino", "f1_val", "gap_f1", "perda_val",
                  "epoca_perda_val_minima", "aumento_perda_val_pct", "diagnostico"]]
          .round(3).to_string(index=False))

    print("\nB) Curva pelo tamanho do treino (pode levar alguns minutos)")
    treino, _, _ = dados.carregar_emotion()
    folds = dados.montar_folds_tfidf(treino.text, treino.label.values)
    df = curva_tamanho(folds)
    df.to_csv(config.RESULTADOS / "04_curva_tamanho.csv", index=False)
    fig_tamanho(df)
    resumo = (df.groupby(["config", "fracao"])[["n_treino", "f1_treino", "f1_val"]]
              .mean().round(3))
    print(resumo.to_string())
