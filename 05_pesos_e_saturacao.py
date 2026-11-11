"""
05 — Mapas de calor da variação dos pesos e análise de saturação (ReLU × Sigmoid).

O QUE O ENUNCIADO PEDE
----------------------
"Em cada época você tem duas matrizes [W1 e W2, as das duas camadas ocultas]
e a cada época esses valores vão mudando. Então acumule esses números para
mostrar a variação dos parâmetros" — e comparar ReLU com Sigmoid. Depois,
"a partir do experimento anterior, faça uma análise sobre a saturação dos pesos".

COMO MEDIMOS
------------
Treinamos duas redes iguais em tudo (arquitetura e L2 da melhor configuração
do script 03, MESMOS pesos iniciais, mesmos lotes), mudando só a ativação. No
fim de cada época t guardamos

      ΔW(t) = W(t) - W(t-1)                    (o quanto cada peso andou)
      V     = Σ_t |ΔW(t)|                      (variação ACUMULADA, peso a peso)

V é uma matriz do mesmo tamanho de W: o mapa de calor de V mostra quais pesos
se mexeram ao longo do treino. Um peso com V ≈ 0 ficou "congelado".

TEORIA — por que um peso para de mudar?
---------------------------------------
No SGD, ΔW ∝ gradiente, e o gradiente de W_l contém o fator f'(Z_l) de cada
camada por onde ele passou (ver src/mlp.py, passo 3 do backward):

  * SIGMOID: f'(z) = σ(z)(1 - σ(z)) ≤ 0,25. Duas camadas -> o sinal que chega
    em W1 já foi multiplicado por, no máximo, 0,25 × 0,25 = 1/16. Se |z| fica
    grande, σ(z) encosta em 0 ou 1 e f'(z) -> 0: o neurônio SATURA e os pesos
    que chegam nele param de receber gradiente.
    Saturado aqui: σ(z) < 0,05 ou σ(z) > 0,95  (|z| > 2,94), onde f'(z) < 0,048.
  * RELU: f'(z) = 1 se z > 0 e 0 se z ≤ 0. Não satura para z positivo (por
    isso o gradiente não encolhe), mas um neurônio com z ≤ 0 para TODOS os
    exemplos "morre": f' = 0 sempre, e seus pesos nunca mais mudam.

Também medimos, a cada época, num conjunto fixo (os 2.000 tweets da validação):
  - a derivada média E[f'(z)] de cada camada (o "fator de atenuação" do gradiente);
  - a fração de ativações na zona de derivada ~0 (saturadas / desligadas);
  - a fração de NEURÔNIOS saturados (sigmoid, em ≥ 99% dos exemplos) ou mortos
    (ReLU, desligados em 100% dos exemplos);
  - o quanto a saída de cada neurônio varia entre exemplos (se não varia, o
    neurônio não carrega informação, mesmo sem estar saturado);
  - a norma do gradiente de cada camada e a razão ||∂L/∂W1|| / ||∂L/∂W3||:
    quanto o gradiente encolhe da saída até a primeira camada.

EXPERIMENTO DE CONTROLE: a mesma sigmoid treinada com Adam (ver src/config.py).
Aumentar a taxa do SGD não resolve — a saída passa a oscilar antes de a
primeira camada receber sinal. O Adam dá a cada peso um passo proporcional
ao seu próprio gradiente normalizado; se assim a rede aprende, o problema era
a escala do gradiente nas primeiras camadas, e não a capacidade da rede.

Saídas: figuras 05_*.png e resultados/05_pesos_saturacao.json
"""
import json

import numpy as np
from sklearn.feature_selection import chi2

from src import config, dados, graficos
from src.mlp import MLP

LIMIAR_SIGMOID = 0.05        # σ < 0,05 ou σ > 0,95  =>  saturado

# Arquitetura: a da melhor configuração do script 03 (256-128).
# L2: DESLIGADA neste experimento. Com L2, todo peso encolhe um pouco a cada
# passo (o termo 2λW do gradiente), mesmo que os dados não peçam mudança
# nenhuma. Num teste com L2, quase toda a "variação" da W1 da sigmoid era esse
# encolhimento, e o mapa de calor ficaria medindo a L2, não o aprendizado. Sem
# L2, tudo que ΔW mostra vem do gradiente dos dados. (Na validação cruzada a
# versão sem L2 da 256-128 empatou com a melhor: F1 0,8204 × 0,8209.)
L2_EXPERIMENTO = 0.0


class Monitor:
    """Chamado no fim de cada época (parâmetro ao_fim_da_epoca de MLP.treinar)."""

    def __init__(self, modelo, X_monitor):
        self.X = X_monitor
        self.anterior = [W.copy() for W in modelo.W[:2]]          # W1, W2 no início
        self.acumulado = [np.zeros_like(W, dtype=np.float64) for W in self.anterior]
        self.var_neuronio = [[], []]   # por época: média de |ΔW| de cada neurônio
        self.norma_delta = [[], []]    # por época: ||ΔW|| (norma de Frobenius)
        self.norma_W = [[], []]        # por época: ||W||
        self.estado = []               # estatísticas de saturação por época
        self.z_inicio = None
        self.registrar_estado(modelo)  # "época 0" = pesos iniciais

    def __call__(self, modelo, epoca):
        for l in range(2):
            delta = modelo.W[l] - self.anterior[l]
            self.acumulado[l] += np.abs(delta)                  # V += |ΔW|
            self.var_neuronio[l].append(np.abs(delta).mean(axis=0))  # uma coluna = um neurônio
            self.norma_delta[l].append(float(np.linalg.norm(delta)))
            self.norma_W[l].append(float(np.linalg.norm(modelo.W[l])))
            self.anterior[l] = modelo.W[l].copy()
        self.registrar_estado(modelo)

    def registrar_estado(self, modelo):
        Z, A = modelo.forward(self.X)
        if self.z_inicio is None:
            self.z_inicio = [Z[0].copy(), Z[1].copy()]
        self.z_fim = [Z[0], Z[1]]
        linha = {}
        for l in range(2):
            z, a = Z[l], A[l + 1]
            if modelo.ativacao == "sigmoid":
                derivada = a * (1 - a)
                zona_zero = (a < LIMIAR_SIGMOID) | (a > 1 - LIMIAR_SIGMOID)
                travado = zona_zero.mean(axis=0) >= 0.99       # saturado em ≥99% dos exemplos
            else:
                derivada = (z > 0).astype(np.float64)
                zona_zero = z <= 0
                travado = zona_zero.all(axis=0)                 # morto: desligado sempre
            linha[f"derivada_media_c{l + 1}"] = float(derivada.mean())
            linha[f"frac_zona_zero_c{l + 1}"] = float(zona_zero.mean())
            linha[f"frac_neuronios_travados_c{l + 1}"] = float(travado.mean())
            linha[f"variacao_saida_c{l + 1}"] = float(a.std(axis=0).mean())
            linha[f"z_abs_medio_c{l + 1}"] = float(np.abs(z).mean())
        self.travados_fim = []
        for l in range(2):
            z, a = Z[l], A[l + 1]
            if modelo.ativacao == "sigmoid":
                zz = (a < LIMIAR_SIGMOID) | (a > 1 - LIMIAR_SIGMOID)
                self.travados_fim.append(zz.mean(axis=0) >= 0.99)
            else:
                self.travados_fim.append((z <= 0).all(axis=0))
        self.estado.append(linha)


def treinar_monitorado(X_tr, y_tr, X_va, y_va, ativacao, camadas, l2, otimizador, taxa):
    modelo = MLP(X_tr.shape[1], camadas, config.N_CLASSES, ativacao=ativacao, l2=l2,
                 semente=config.SEMENTE)       # mesma semente -> mesmos pesos iniciais
    monitor = Monitor(modelo, X_va)
    hist = modelo.treinar(X_tr, y_tr, X_va, y_va, epocas=config.EPOCAS,
                          tamanho_lote=config.TAMANHO_LOTE, taxa=taxa,
                          momento=config.MOMENTO, semente=config.SEMENTE,
                          otimizador=otimizador,
                          ao_fim_da_epoca=monitor, registrar_gradientes=True)
    return modelo, monitor, hist


def palavras_marcantes(X_tr, y_tr, vocab, por_classe=5):
    """As 5 palavras com maior χ² de cada emoção -> linhas do mapa de calor de W1."""
    linhas, rotulos = [], []
    for k in range(config.N_CLASSES):
        score, _ = chi2(X_tr, (y_tr == k).astype(int))
        score = np.nan_to_num(score)
        for pos in np.argsort(score)[::-1][:por_classe]:
            if pos not in linhas:
                linhas.append(int(pos))
                rotulos.append(f"{vocab[pos]}  ·  {config.CLASSES_PT[k]}")
    return linhas, rotulos


# ===========================================================================
# Figuras
# ===========================================================================
def _limites_log(*matrizes):
    vals = np.concatenate([m.ravel() for m in matrizes])
    vals = vals[vals > 0]
    return np.percentile(vals, 1), np.percentile(vals, 99.5)


def _milhar(x):
    """33887.97 -> '33.888' (separador de milhar brasileiro)."""
    return f"{x:,.0f}".replace(",", ".")


def fig_mapa_W1(mon, linhas, rotulos, camadas):
    # ESCALA LINEAR e a MESMA nas duas redes: a cor compara magnitudes
    # diretamente (um peso com a mesma cor mudou o mesmo tanto nas duas).
    V = {a: mon[a].acumulado[0][linhas] for a in ("relu", "sigmoid")}
    vmax = np.percentile(np.concatenate([v.ravel() for v in V.values()]), 99)

    def desenhar(P):
        import matplotlib.pyplot as plt
        fig, eixos = plt.subplots(1, 2, figsize=(14, 7.2), sharey=True)
        for ax, a in zip(eixos, ("relu", "sigmoid")):
            im = ax.imshow(V[a], aspect="auto", cmap=P.cmap_seq(), vmin=0, vmax=vmax,
                           interpolation="nearest")
            total = mon[a].acumulado[0].sum()
            ax.set_title(f"{graficos.NOME_ATIVACAO[a]}   ·   Σ de toda a W1 = {_milhar(total)}",
                         loc="left")
            ax.set_xlabel(f"neurônio da camada oculta 1 (0 a {camadas[0] - 1})")
            ax.grid(False)
            for s in range(5, len(linhas), 5):
                ax.axhline(s - 0.5, color=P.fundo, lw=1.5)
        eixos[0].set_yticks(range(len(rotulos)), rotulos, fontsize=8.5)
        cb = fig.colorbar(im, ax=eixos, fraction=0.025, pad=0.01, extend="max")
        cb.set_label("variação acumulada Σ|ΔW| em 50 épocas", color=P.texto2)
        cb.outline.set_visible(False)
        fig.suptitle(f"W1: quanto cada peso palavra → neurônio mudou em {config.EPOCAS} épocas "
                     "(5 palavras mais típicas de cada emoção)", x=0.01, ha="left",
                     fontsize=12.5, color=P.texto)
        return fig

    graficos.salvar(desenhar, "05_mapa_calor_W1")


def fig_mapa_W2(mon, camadas):
    V = {a: mon[a].acumulado[1] for a in ("relu", "sigmoid")}
    vmax = np.percentile(np.concatenate([v.ravel() for v in V.values()]), 99)

    def desenhar(P):
        import matplotlib.pyplot as plt
        fig, eixos = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)
        for ax, a in zip(eixos, ("relu", "sigmoid")):
            im = ax.imshow(V[a], aspect="auto", cmap=P.cmap_seq(), vmin=0, vmax=vmax,
                           interpolation="nearest")
            ax.set_title(f"{graficos.NOME_ATIVACAO[a]}   ·   Σ de toda a W2 = {_milhar(V[a].sum())}",
                         loc="left")
            ax.set_xlabel(f"neurônio da camada oculta 2 (0 a {camadas[1] - 1})")
            ax.grid(False)
        eixos[0].set_ylabel(f"neurônio da camada oculta 1 (0 a {camadas[0] - 1})")
        cb = fig.colorbar(im, ax=eixos, fraction=0.025, pad=0.01, extend="max")
        cb.set_label("variação acumulada Σ|ΔW| em 50 épocas", color=P.texto2)
        cb.outline.set_visible(False)
        fig.suptitle(f"W2: variação acumulada de cada peso em {config.EPOCAS} épocas",
                     x=0.01, ha="left", fontsize=12.5, color=P.texto)
        return fig

    graficos.salvar(desenhar, "05_mapa_calor_W2")


def fig_variacao_por_epoca(mon):
    from matplotlib.colors import LogNorm
    M = {(a, l): np.array(mon[a].var_neuronio[l]).T for a in ("relu", "sigmoid") for l in (0, 1)}
    vmin, vmax = _limites_log(*M.values())

    def desenhar(P):
        import matplotlib.pyplot as plt
        fig, eixos = plt.subplots(2, 2, figsize=(13, 7.4), sharex=True)
        for j, a in enumerate(("relu", "sigmoid")):
            for l in (0, 1):
                ax = eixos[l, j]
                im = ax.imshow(np.clip(M[(a, l)], vmin, None), aspect="auto", cmap=P.cmap_seq(),
                               norm=LogNorm(vmin=vmin, vmax=vmax), interpolation="nearest",
                               extent=(0.5, config.EPOCAS + 0.5, M[(a, l)].shape[0] - 0.5, -0.5))
                ax.set_title(f"{graficos.NOME_ATIVACAO[a]} · W{l + 1}", loc="left")
                ax.set_ylabel(f"neurônio da camada {l + 1}")
                ax.grid(False)
        for ax in eixos[1]:
            ax.set_xlabel("época")
        cb = fig.colorbar(im, ax=eixos, fraction=0.025, pad=0.01)
        cb.set_label("média de |ΔW| dos pesos que chegam no neurônio (log)", color=P.texto2)
        cb.outline.set_visible(False)
        fig.suptitle("Quanto os pesos de cada neurônio mudaram em cada época",
                     x=0.01, ha="left", fontsize=12.5, color=P.texto)
        return fig

    graficos.salvar(desenhar, "05_variacao_por_epoca")


def fig_saturacao(mon, hist, rotulo_controle):
    ep0 = np.arange(0, config.EPOCAS + 1)
    ep1 = np.arange(1, config.EPOCAS + 1)

    def serie(a, chave):
        return [e[chave] for e in mon[a].estado]

    def desenhar(P):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        fig, eixos = plt.subplots(2, 2, figsize=(13, 8.2))
        cores = {"relu": P.cor("relu"), "sigmoid": P.cor("sigmoid"), "controle": P.serie(2)}
        estilos = {1: "-", 2: "--"}
        rodadas = ("relu", "sigmoid", "controle")
        # (a) derivada média
        ax = eixos[0, 0]
        for a in rodadas:
            for l in (1, 2):
                ax.plot(ep0, serie(a, f"derivada_media_c{l}"), color=cores[a], ls=estilos[l])
        ax.axhline(0.25, color=P.mudo, lw=1)
        ax.text(config.EPOCAS, 0.27, "máximo possível da sigmoid = 0,25", ha="right",
                fontsize=8.5, color=P.texto2)
        ax.set(title="Derivada média da ativação  E[f′(z)]", xlabel="época", ylim=(0, 1.02))
        # (b) neurônios travados
        ax = eixos[0, 1]
        for a in rodadas:
            for l in (1, 2):
                ax.plot(ep0, np.array(serie(a, f"frac_neuronios_travados_c{l}")) * 100,
                        color=cores[a], ls=estilos[l])
        ax.set(title="Neurônios travados (%): ReLU morto / Sigmoid saturado",
               xlabel="época", ylabel="% dos neurônios da camada")
        # (c) quanto o gradiente encolhe da saída até a 1ª camada
        ax = eixos[1, 0]
        for a in rodadas:
            g = np.array(hist[a]["norma_grad"])
            ax.plot(ep1, g[:, 0] / g[:, -1], color=cores[a])
        ax.set(title="Gradiente que chega em W1 ÷ gradiente da saída  (||∂L/∂W1|| / ||∂L/∂W3||)",
               xlabel="época", yscale="log")
        ax.title.set_fontsize(10.5)
        # (d) acurácia
        ax = eixos[1, 1]
        for a in rodadas:
            ax.plot(ep1, hist[a]["acc_val"], color=cores[a])
        ax.set(title="Acurácia na validação", xlabel="época", ylim=(0, 1))
        legenda = [Line2D([], [], color=cores["relu"], label="ReLU (SGD, η = 0,05)"),
                   Line2D([], [], color=cores["sigmoid"], label="Sigmoid (SGD, η = 0,05)"),
                   Line2D([], [], color=cores["controle"], label=rotulo_controle),
                   Line2D([], [], color=P.texto2, ls="-", label="camada oculta 1"),
                   Line2D([], [], color=P.texto2, ls="--", label="camada oculta 2")]
        fig.legend(handles=legenda, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle("Saturação: por que os pesos da sigmoid quase não mudam",
                     x=0.01, ha="left", fontsize=12.5, color=P.texto)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        return fig

    graficos.salvar(desenhar, "05_saturacao")


def fig_histograma_z(mon, rotulo_controle):
    titulos = {"relu": "ReLU (SGD)", "sigmoid": "Sigmoid (SGD)", "controle": rotulo_controle}

    def desenhar(P):
        import matplotlib.pyplot as plt
        fig, eixos = plt.subplots(2, 3, figsize=(16, 7))
        cores = {"relu": P.cor("relu"), "sigmoid": P.cor("sigmoid"), "controle": P.serie(2)}
        for j, a in enumerate(("relu", "sigmoid", "controle")):
            for l in (0, 1):
                ax = eixos[l, j]
                z0 = mon[a].z_inicio[l].ravel()
                z1 = mon[a].z_fim[l].ravel()
                lim = np.percentile(np.abs(np.concatenate([z0, z1])), 99.9) * 1.05
                if a != "relu":
                    lim = max(lim, 3.6)      # deixa a zona saturada sempre à vista
                bins = np.linspace(-lim, lim, 81)
                ax.hist(z0, bins=bins, color=P.mudo, alpha=0.55, label="início (época 0)")
                ax.hist(z1, bins=bins, color=cores[a], alpha=0.85,
                        label=f"fim (época {config.EPOCAS})")
                if a != "relu":
                    zs = np.log(1 / LIMIAR_SIGMOID - 1)          # |z| > 2,94 -> saturado
                    for x0, x1 in ((-lim, -zs), (zs, lim)):
                        ax.axvspan(x0, x1, color=P.texto2, alpha=0.10, lw=0)
                else:
                    ax.axvspan(-lim, 0, color=P.texto2, alpha=0.10, lw=0)
                ax.set_xlim(-lim, lim)
                ax.set_yscale("log")
                ax.set_title(f"{titulos[a]} · z da camada {l + 1}", loc="left", fontsize=10.5)
                ax.set_xlabel("pré-ativação z")
                if l == 0:
                    ax.legend(loc="upper left", fontsize=8.5)
        fig.suptitle("Distribuição das pré-ativações z (validação). Faixa sombreada = derivada da "
                     "ativação ≈ 0  (ReLU: z ≤ 0;  Sigmoid: |z| > 2,94)",
                     x=0.01, ha="left", fontsize=12, color=P.texto)
        fig.tight_layout()
        return fig

    graficos.salvar(desenhar, "05_histograma_z")


# ===========================================================================
if __name__ == "__main__":
    with open(config.RESULTADOS / "melhor_config.json", encoding="utf-8") as f:
        melhor = json.load(f)
    camadas, l2 = tuple(melhor["camadas"]), L2_EXPERIMENTO
    print(f"Arquitetura da melhor configuração: camadas {camadas}; L2 neste experimento = {l2:g}")

    treino, validacao, _ = dados.carregar_emotion()
    vet = dados.criar_tfidf()
    X_tr = vet.fit_transform(treino.text)
    X_va = vet.transform(validacao.text)
    y_tr, y_va = treino.label.values, validacao.label.values
    vocab = vet.get_feature_names_out()

    mon, hist, modelos = {}, {}, {}
    ctrl = config.CONTROLE_SIGMOID
    rodadas = [("relu", "relu", "sgd", config.TAXA_APRENDIZADO),
               ("sigmoid", "sigmoid", "sgd", config.TAXA_APRENDIZADO),
               ("controle", "sigmoid", ctrl["otimizador"], ctrl["taxa"])]
    for chave, ativ, otim, taxa in rodadas:
        print(f"Treinando {chave} ({otim}, η = {taxa:g})...", flush=True)
        modelos[chave], mon[chave], hist[chave] = treinar_monitorado(
            X_tr, y_tr, X_va, y_va, ativ, camadas, l2, otim, taxa)
        print(f"   acc_val final = {hist[chave]['acc_val'][-1]:.4f}")

    linhas, rotulos = palavras_marcantes(X_tr, y_tr, vocab)
    rot_controle = (f"Sigmoid ({ctrl['otimizador'].capitalize()}, η = {ctrl['taxa']:g}) — controle"
                    .replace(".", ","))
    fig_mapa_W1(mon, linhas, rotulos, camadas)
    fig_mapa_W2(mon, camadas)
    fig_variacao_por_epoca(mon)
    fig_saturacao(mon, hist, rot_controle)
    fig_histograma_z(mon, rot_controle)

    # ---------------- números para o relatório ----------------
    resumo = dict(camadas=camadas, l2=l2, taxa=config.TAXA_APRENDIZADO,
                  controle=ctrl, rodadas={})
    for chave, ativ, otim, taxa in rodadas:
        m = mon[chave]
        acc = np.array(hist[chave]["acc_val"])
        saida_plato = int(np.argmax(acc > 0.5)) + 1 if (acc > 0.5).any() else None
        g = np.array(hist[chave]["norma_grad"])
        r = dict(ativacao=ativ, otimizador=otim, taxa=taxa,
                 acc_val_final=float(acc[-1]), f1_val_final=hist[chave]["f1_val"][-1],
                 razao_grad_W1_sobre_W3_inicio=float(g[0, 0] / g[0, -1]),
                 razao_grad_W1_sobre_W3_fim=float(g[-1, 0] / g[-1, -1]),
                 primeira_epoca_acc_val_maior_que_0_5=saida_plato,
                 variacao_total_W1=float(m.acumulado[0].sum()),
                 variacao_total_W2=float(m.acumulado[1].sum()),
                 variacao_media_por_peso_W1=float(m.acumulado[0].mean()),
                 variacao_media_por_peso_W2=float(m.acumulado[1].mean()),
                 norma_grad_inicio=hist[chave]["norma_grad"][0],
                 norma_grad_fim=hist[chave]["norma_grad"][-1],
                 estado_inicio=m.estado[0], estado_fim=m.estado[-1])
        # variação acumulada dos neurônios travados x dos demais
        for l in range(2):
            por_neuronio = m.acumulado[l].sum(axis=0)
            trav = m.travados_fim[l]
            r[f"neuronios_travados_c{l + 1}"] = int(trav.sum())
            r[f"variacao_media_travados_c{l + 1}"] = float(por_neuronio[trav].mean()) if trav.any() else None
            r[f"variacao_media_ativos_c{l + 1}"] = float(por_neuronio[~trav].mean()) if (~trav).any() else None
        resumo["rodadas"][chave] = r
    resumo["razao_variacao_W1_relu_sobre_sigmoid"] = (
        resumo["rodadas"]["relu"]["variacao_total_W1"] / resumo["rodadas"]["sigmoid"]["variacao_total_W1"])
    resumo["razao_variacao_W2_relu_sobre_sigmoid"] = (
        resumo["rodadas"]["relu"]["variacao_total_W2"] / resumo["rodadas"]["sigmoid"]["variacao_total_W2"])
    resumo["series"] = {chave: dict(estado=mon[chave].estado,
                                    norma_delta_W1=mon[chave].norma_delta[0],
                                    norma_delta_W2=mon[chave].norma_delta[1],
                                    norma_W1=mon[chave].norma_W[0], norma_W2=mon[chave].norma_W[1],
                                    norma_grad=hist[chave]["norma_grad"],
                                    acc_val=hist[chave]["acc_val"],
                                    var_neuronio_W1=np.array(mon[chave].var_neuronio[0]).tolist(),
                                    var_neuronio_W2=np.array(mon[chave].var_neuronio[1]).tolist())
                        for chave, _, _, _ in rodadas}
    with open(config.RESULTADOS / "05_pesos_saturacao.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=1, ensure_ascii=False)

    for chave in ("relu", "sigmoid", "controle"):
        r = resumo["rodadas"][chave]
        print(f"\n{chave}: Σ|ΔW1| = {r['variacao_total_W1']:.2f}  Σ|ΔW2| = {r['variacao_total_W2']:.2f}"
              f"  acc_val = {r['acc_val_final']:.4f}  saída do platô: época {r['primeira_epoca_acc_val_maior_que_0_5']}")
        print("   início:", {k: round(v, 4) for k, v in r["estado_inicio"].items()})
        print("   fim:   ", {k: round(v, 4) for k, v in r["estado_fim"].items()})
        print(f"   neurônios travados: c1 = {r['neuronios_travados_c1']}  c2 = {r['neuronios_travados_c2']}")
    print(f"\nW1 da ReLU variou {resumo['razao_variacao_W1_relu_sobre_sigmoid']:.1f}× mais que a da Sigmoid")
