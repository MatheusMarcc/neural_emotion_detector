"""
Estilo e utilidades de gráficos.

Toda figura é salva duas vezes:
  figuras/<nome>.png          tema claro  (relatório)
  figuras/escuro/<nome>.png   tema escuro (slides, padrão preto neutro)

As cores seguem a ENTIDADE, nunca a posição: ReLU é sempre azul e Sigmoid é
sempre laranja, em todas as figuras do projeto.
"""
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")   # gera arquivos sem abrir janela (funciona em qualquer máquina)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from . import config

TEMAS = {
    "claro": dict(
        fundo="#fcfcfb", texto="#0b0b0b", texto2="#52514e", mudo="#898781",
        grade="#e1e0d9", eixo="#c3c2b7",
        series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#6250d6", "#e34948"],
        # rampa sequencial azul: perto de zero some no fundo, valores altos escurecem
        seq=["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
             "#256abf", "#184f95", "#0d366b"],
        div=["#256abf", "#86b6ef", "#f0efec", "#f19b9a", "#c23535"],
    ),
    "escuro": dict(
        fundo="#1a1a19", texto="#f0efec", texto2="#c3c2b7", mudo="#898781",
        grade="#2c2c2a", eixo="#383835",
        series=["#3987e5", "#d95926", "#199e70", "#c98500",
                "#d55181", "#008300", "#9085e9", "#e66767"],
        # no escuro a rampa se inverte: zero some no fundo escuro, alto fica claro
        seq=["#1a1a19", "#0d366b", "#184f95", "#256abf", "#3987e5",
             "#6da7ec", "#9ec5f4", "#cde2fb"],
        div=["#3987e5", "#1c5cab", "#383835", "#a83a3a", "#e66767"],
    ),
}

# Cor fixa de cada entidade (índice na lista "series")
COR_ATIVACAO = {"relu": 0, "sigmoid": 1}
NOME_ATIVACAO = {"relu": "ReLU", "sigmoid": "Sigmoid"}


class Paleta(dict):
    """Dicionário com atalhos: P.cor('relu'), P.serie(2), P.cmap_seq()."""

    def __getattr__(self, k):
        return self[k]

    def cor(self, ativacao):
        return self["series"][COR_ATIVACAO[ativacao]]

    def serie(self, i):
        return self["series"][i]

    def cmap_seq(self):
        return LinearSegmentedColormap.from_list("seq", self["seq"])

    def cmap_div(self):
        return LinearSegmentedColormap.from_list("div", self["div"])


@contextmanager
def tema(nome):
    P = Paleta(TEMAS[nome], nome=nome)
    rc = {
        "figure.facecolor": P.fundo, "axes.facecolor": P.fundo,
        "savefig.facecolor": P.fundo,
        "text.color": P.texto, "axes.labelcolor": P.texto2,
        "axes.titlecolor": P.texto, "axes.edgecolor": P.eixo,
        "xtick.color": P.mudo, "ytick.color": P.mudo,
        "xtick.labelcolor": P.texto2, "ytick.labelcolor": P.texto2,
        "axes.grid": True, "grid.color": P.grade, "grid.linewidth": 0.8,
        "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "axes.titleweight": "semibold",
        "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5, "legend.frameon": False,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
        "lines.markersize": 6,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "figure.dpi": 100, "savefig.dpi": 160, "savefig.bbox": "tight",
    }
    with plt.rc_context(rc):
        yield P


def num(x, casas=3):
    """Número com vírgula decimal: num(0.8209) -> '0,821'.

    Arredonda "meio para cima" (0,8795 -> 0,880). Sem o 1e-9, o Python
    arredondaria o valor binário 0,87949999… para 0,879 — e acurácias de 2.000
    tweets caem muito em meios exatos (múltiplos de 0,0005).
    """
    return f"{x + 1e-9:.{casas}f}".replace(".", ",")


def _virgula_nos_eixos(fig):
    """Troca o ponto decimal por vírgula nos eixos lineares (padrão brasileiro)."""
    from matplotlib.ticker import FuncFormatter, ScalarFormatter
    fmt = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    for ax in fig.axes:
        for eixo in (ax.xaxis, ax.yaxis):
            if eixo.get_scale() == "linear" and isinstance(eixo.get_major_formatter(),
                                                           ScalarFormatter):
                eixo.set_major_formatter(fmt)


def salvar(desenhar, nome, temas=("claro", "escuro")):
    """Chama desenhar(P) -> Figure uma vez por tema e salva os PNGs."""
    (config.FIGURAS / "escuro").mkdir(exist_ok=True)
    caminhos = []
    for t in temas:
        with tema(t) as P:
            fig = desenhar(P)
            _virgula_nos_eixos(fig)
            pasta = config.FIGURAS if t == "claro" else config.FIGURAS / "escuro"
            caminho = pasta / f"{nome}.png"
            fig.savefig(caminho)
            plt.close(fig)
            caminhos.append(caminho)
    return caminhos
