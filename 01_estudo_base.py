"""
01 — Estudo da base Emotion (antes de qualquer modelo).

A base dair-ai/emotion tem 20.000 tweets em inglês, cada um rotulado com uma
de 6 emoções. Os rótulos vieram de hashtags (#sad, #angry, ...), que foram
removidas do texto — então parte dos rótulos é ruidosa: um tweet com #joy
pode ser irônico. Esse ruído impõe um teto ao desempenho de qualquer modelo.

O que este script mede:
  1. DESBALANCEAMENTO: quantos exemplos por classe (o enunciado pede base
     desbalanceada e com várias classes);
  2. TAMANHO dos tweets (em palavras);
  3. VOCABULÁRIO: quantas palavras distintas e quanto do texto as 3.000 mais
     frequentes (o vocabulário do TF-IDF) cobrem;
  4. PALAVRAS MAIS ASSOCIADAS a cada emoção (teste qui-quadrado), que ajudam a
     entender a tarefa e são usadas como linhas no mapa de calor do script 05.

Saídas: figuras 01_*.png e resultados/01_*.csv / 01_estudo_base.json
"""
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

from src import config, dados, graficos


def distribuicao(treino, validacao, teste):
    linhas = []
    for nome, df in (("treino", treino), ("validação", validacao), ("teste", teste)):
        cont = df.label.value_counts().reindex(range(config.N_CLASSES), fill_value=0)
        for k in range(config.N_CLASSES):
            linhas.append(dict(conjunto=nome, classe=config.CLASSES[k],
                               classe_pt=config.CLASSES_PT[k], n=int(cont[k]),
                               pct=float(cont[k] / len(df))))
    return pd.DataFrame(linhas)


def palavras_por_classe(textos, y, n=8):
    """Qui-quadrado (um contra todos) sobre presença/ausência de cada palavra.

    TEORIA: o χ² compara quantas vezes a palavra aparece na classe com quantas
    apareceria se palavra e classe fossem independentes. Quanto maior, mais a
    palavra "denuncia" a classe.
    """
    vet = CountVectorizer(max_features=config.MAX_PALAVRAS, token_pattern=r"(?u)\b\w+\b",
                          binary=True)
    X = vet.fit_transform(textos)
    vocab = np.array(vet.get_feature_names_out())
    linhas = []
    for k in range(config.N_CLASSES):
        alvo = (y == k).astype(int)
        score, _ = chi2(X, alvo)
        # só palavras MAIS frequentes na classe do que fora dela (associação positiva)
        freq_in = np.asarray(X[alvo == 1].mean(axis=0)).ravel()
        freq_out = np.asarray(X[alvo == 0].mean(axis=0)).ravel()
        score = np.where(freq_in > freq_out, score, 0)
        for pos in np.argsort(score)[::-1][:n]:
            linhas.append(dict(classe=config.CLASSES[k], classe_pt=config.CLASSES_PT[k],
                               palavra=vocab[pos], chi2=float(score[pos]),
                               freq_na_classe=float(freq_in[pos]),
                               freq_fora=float(freq_out[pos])))
    return pd.DataFrame(linhas)


def fig_distribuicao(dist):
    tr = dist[dist.conjunto == "treino"].sort_values("n")

    def desenhar(P):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 3.8))
        rotulos = [f"{a}  ({b})" for a, b in zip(tr.classe_pt, tr.classe)]
        ax.barh(rotulos, tr.n, height=0.6, color=P.serie(0))
        for i, (n, p) in enumerate(zip(tr.n, tr.pct)):
            ax.text(n + 60, i, f"{n:,}".replace(",", ".") + f"  ({p:.1%})".replace(".", ","),
                    va="center", fontsize=9.5, color=P.texto2)
        ax.set_xlim(0, tr.n.max() * 1.25)
        ax.set_xlabel("tweets no conjunto de treino")
        ax.grid(axis="y", visible=False)
        razao = tr.n.max() / tr.n.min()
        ax.set_title(f"Distribuição das classes — a maior tem {razao:.1f}× mais exemplos que a menor",
                     loc="left")
        fig.tight_layout()
        return fig

    graficos.salvar(desenhar, "01_distribuicao_classes")


def fig_tamanho(tamanhos):
    def desenhar(P):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.hist(tamanhos, bins=np.arange(0, tamanhos.max() + 2) - 0.5, color=P.serie(0),
                edgecolor=P.fundo, linewidth=0.6)
        med = np.median(tamanhos)
        ax.axvline(med, color=P.texto2, lw=1.2)
        ax.text(med + 1, ax.get_ylim()[1] * 0.92, f"mediana = {med:.0f} palavras",
                color=P.texto2, fontsize=9.5)
        ax.set(xlabel="palavras por tweet", ylabel="tweets")
        ax.set_title("Tamanho dos tweets (treino)", loc="left")
        fig.tight_layout()
        return fig

    graficos.salvar(desenhar, "01_tamanho_tweets")


def fig_palavras(pal):
    def desenhar(P):
        import matplotlib.pyplot as plt
        fig, eixos = plt.subplots(2, 3, figsize=(12, 6.2))
        for k, ax in enumerate(eixos.ravel()):
            d = pal[pal.classe == config.CLASSES[k]].iloc[::-1]
            ax.barh(d.palavra, d.chi2, height=0.62, color=P.serie(0))
            ax.set_title(f"{config.CLASSES_PT[k]} ({config.CLASSES[k]})", loc="left", fontsize=11)
            ax.grid(axis="y", visible=False)
            ax.tick_params(axis="y", labelsize=10)
            ax.tick_params(axis="x", labelsize=8.5)
        fig.supxlabel("χ² (associação da palavra com a classe)", fontsize=10, color=P.texto2)
        fig.suptitle("Palavras mais associadas a cada emoção", x=0.01, ha="left",
                     fontsize=12.5, color=P.texto)
        fig.tight_layout()
        return fig

    graficos.salvar(desenhar, "01_palavras_por_classe")


if __name__ == "__main__":
    treino, validacao, teste = dados.carregar_emotion()
    dist = distribuicao(treino, validacao, teste)
    dist.to_csv(config.RESULTADOS / "01_distribuicao_classes.csv", index=False)

    tamanhos = treino.text.str.split().str.len().values
    contagem = Counter(" ".join(treino.text).split())
    total_tokens = sum(contagem.values())
    top = contagem.most_common(config.MAX_PALAVRAS)
    cobertura = sum(c for _, c in top) / total_tokens

    pal = palavras_por_classe(treino.text, treino.label.values)
    pal.to_csv(config.RESULTADOS / "01_palavras_por_classe.csv", index=False)

    tr = dist[dist.conjunto == "treino"]
    resumo = dict(
        n_treino=len(treino), n_validacao=len(validacao), n_teste=len(teste),
        razao_desbalanceamento=float(tr.n.max() / tr.n.min()),
        classe_maior=tr.loc[tr.n.idxmax(), "classe"], classe_menor=tr.loc[tr.n.idxmin(), "classe"],
        acuracia_classe_majoritaria_teste=float((teste.label == treino.label.mode()[0]).mean()),
        palavras_por_tweet=dict(media=float(tamanhos.mean()), mediana=float(np.median(tamanhos)),
                                p95=float(np.percentile(tamanhos, 95)), max=int(tamanhos.max())),
        vocabulario_total=len(contagem),
        palavras_com_freq_ao_menos_5=int(sum(1 for c in contagem.values() if c >= 5)),
        cobertura_tokens_top_3000=float(cobertura),
        textos_duplicados_treino=int(treino.text.duplicated().sum()),
        textos_em_treino_e_teste=len(set(treino.text) & set(teste.text)),
        palavras_mais_frequentes=contagem.most_common(10),
    )
    with open(config.RESULTADOS / "01_estudo_base.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=1, ensure_ascii=False)

    fig_distribuicao(dist)
    fig_tamanho(tamanhos)
    fig_palavras(pal)

    print(dist.pivot(index="classe_pt", columns="conjunto", values="pct").round(3))
    print(json.dumps({k: v for k, v in resumo.items() if k != "palavras_mais_frequentes"},
                     indent=1, ensure_ascii=False))
    print(pal.groupby("classe_pt").palavra.apply(lambda s: ", ".join(s)).to_string())
