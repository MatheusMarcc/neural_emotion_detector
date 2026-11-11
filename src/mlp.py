"""
MLP (Perceptron Multicamadas) implementada do zero com NumPy.

Nada de autograd aqui: o forward, a função de custo, as derivadas (regra da
cadeia) e a atualização dos pesos estão escritos à mão. O Keras só aparece no
script 02, para conferir que esta implementação calcula exatamente o mesmo.

CPU OU GPU: as contas passam pelas funções de src/backend.py, que fazem a
mesma operação com NumPy (CPU) ou com PyTorch (GPU). As equações abaixo são
as mesmas nos dois casos; na GPU só muda ONDE as matrizes moram.

TEORIA — MLP com duas camadas ocultas
=====================================

Arquitetura usada no projeto (exemplo com n1 = 64 e n2 = 32 neurônios):

    X (N x 3000) --W1,b1--> Z1 --f--> A1 (N x 64)
                 --W2,b2--> Z2 --f--> A2 (N x 32)
                 --W3,b3--> Z3 --softmax--> P (N x 6)

    f = ReLU ou Sigmoid (a mesma nas duas camadas ocultas)
    N = número de exemplos no lote (mini-batch), 6 = número de emoções

Convenção de formas (igual à do Keras): W tem forma (entradas, saídas) e o
produto é X @ W. Cada LINHA de X é um exemplo.

1) FORWARD (propagação para frente)
       Z1 = X  W1 + b1        A1 = f(Z1)
       Z2 = A1 W2 + b2        A2 = f(Z2)
       Z3 = A2 W3 + b3        P  = softmax(Z3)

   O softmax transforma os 6 números de cada linha (os "logits") em
   probabilidades que somam 1:
       P[i,k] = exp(Z3[i,k]) / Σ_j exp(Z3[i,j])

2) FUNÇÃO DE CUSTO (entropia cruzada + L2)
       L = -(1/N) Σ_i log P[i, y_i]  +  λ (||W1||² + ||W2||² + ||W3||²)

   O primeiro termo é pequeno quando a rede dá probabilidade alta para a
   classe certa. O segundo (regularização L2) penaliza pesos grandes; λ = 0 é
   "sem regularização". ||W||² = soma dos quadrados de todos os elementos.
   Mesma convenção do keras.regularizers.L2(λ): λ·Σw², sem o fator 1/2.
   O bias não é regularizado (também é o padrão do Keras).

3) BACKWARD (retropropagação = regra da cadeia)
   Queremos ∂L/∂W e ∂L/∂b de cada camada. A regra da cadeia diz que a
   derivada de uma composição é o produto das derivadas de cada etapa.
   Começamos na saída e voltamos camada por camada, reaproveitando o que já
   foi calculado na camada seguinte — daí o nome retropropagação.

   (a) Saída. Softmax + entropia cruzada juntos têm uma derivada simples:
           dZ3 = ∂L/∂Z3 = (P - Y) / N        (Y = rótulos em one-hot)
       Para um exemplo: L = -log P_y, com P_k = e^{z_k} / Σ_j e^{z_j}.
           ∂L/∂z_k = -(1/P_y)·∂P_y/∂z_k = P_k - 1[k = y]
       O 1/N aparece porque L é a MÉDIA sobre o lote.

   (b) Pesos e bias da saída. Como Z3 = A2 W3 + b3:
           dW3 = A2ᵀ dZ3 + 2λ W3     (2λW3 = derivada de λ||W3||²)
           db3 = Σ_i dZ3[i, :]       (o bias entra em todos os exemplos, soma)

   (c) Volta para a camada oculta 2:
           dA2 = dZ3 W3ᵀ             (quanto cada ativação de A2 influenciou L)
           dZ2 = dA2 ⊙ f'(Z2)        (⊙ = produto elemento a elemento)
           dW2 = A1ᵀ dZ2 + 2λ W2     db2 = Σ_i dZ2[i, :]

   (d) Camada oculta 1, mesmo padrão:
           dA1 = dZ2 W2ᵀ             dZ1 = dA1 ⊙ f'(Z1)
           dW1 = Xᵀ dZ1 + 2λ W1      db1 = Σ_i dZ1[i, :]

   Derivadas das ativações:
       ReLU:    f(z) = max(0, z)        f'(z) = 1 se z > 0, senão 0
       Sigmoid: f(z) = 1 / (1 + e^-z)   f'(z) = f(z)·(1 - f(z))  (máx. 0,25 em z = 0)

   Repare que a cada camada o gradiente é multiplicado por f'(Z). Na sigmoid
   esse fator é no máximo 0,25 e fica perto de 0 quando |z| é grande
   (neurônio SATURADO): o gradiente encolhe ao voltar pelas camadas
   ("vanishing gradient"). Na ReLU o fator é 1 para neurônios ativos e 0 para
   inativos; um neurônio inativo para todos os exemplos "morre" e nunca mais
   recebe gradiente. Esses dois efeitos são medidos no script 05.

4) ATUALIZAÇÃO — SGD com momentum (mesma fórmula do keras.optimizers.SGD)
       v ← μ·v - η·g        (g = gradiente do lote, η = taxa de aprendizado)
       W ← W + v
   v funciona como uma "velocidade": acumula a direção dos gradientes
   recentes, acelera em direções consistentes e amortece oscilações.
   Com μ = 0 vira o SGD puro (W ← W - η·g).

   Opção usada só no experimento de controle do script 05 — Adam (mesma
   fórmula do keras.optimizers.Adam):
       m ← m + (1-β1)(g - m)          média móvel do gradiente
       s ← s + (1-β2)(g² - s)         média móvel do gradiente ao quadrado
       W ← W - α · m / (√s + ε),      α = η·√(1-β2^t) / (1-β1^t)
   O passo m/√s tem tamanho ~1 qualquer que seja a escala de g: um peso
   com gradiente minúsculo anda quase tanto quanto um com gradiente grande.
   No SGD, ao contrário, quem tem gradiente pequeno quase não anda.

5) INICIALIZAÇÃO — Glorot uniforme (padrão do Keras)
       W ~ U(-a, a),  a = sqrt(6 / (fan_in + fan_out)),   b = 0
   Mantém a variância do sinal parecida de uma camada para a outra no início,
   para a rede não começar saturada nem com sinais minúsculos.
"""
import time

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score

from . import backend as B   # mesma conta na CPU (NumPy) ou na GPU (PyTorch)


# ===========================================================================
# Funções de ativação e suas derivadas
# ===========================================================================
# Cada derivada recebe Z (pré-ativação) e A = f(Z) (já calculada no forward).
# Usar A evita recalcular a função: na sigmoid, f'(z) = A·(1 - A).

def relu(z):
    return B.maximo_zero(z)              # max(z, 0)


def d_relu(z, a):
    # 1 onde o neurônio está ativo (z > 0), 0 onde está desligado
    return B.indicador_positivo(z)


def sigmoid(z):
    # 1 / (1 + e^-z), calculada de forma estável (sem estourar para |z| grande)
    return B.sigmoide(z)


def d_sigmoid(z, a):
    return a * (1 - a)


def linear(z):
    # usada só no Skip-Gram (script 07): a camada oculta é uma projeção linear
    return z


def d_linear(z, a):
    return B.uns_como(z)


ATIVACOES = {
    "relu": (relu, d_relu),
    "sigmoid": (sigmoid, d_sigmoid),
    "linear": (linear, d_linear),
}


# ===========================================================================
# Softmax, entropia cruzada e utilidades
# ===========================================================================
def softmax(z):
    # Subtrair o máximo de cada linha não muda o resultado (o fator e^-max
    # aparece em cima e embaixo da fração) e evita exp() de números enormes.
    z = z - B.max_por_linha(z)
    e = B.exp(z)
    return e / B.soma_por_linha(e)


def log_softmax(z):
    # log P = z - log(Σ e^z), calculado de forma estável (log-sum-exp)
    z = z - B.max_por_linha(z)
    return z - B.log(B.soma_por_linha(B.exp(z)))


def one_hot(y, n_classes, dtype=np.float32):
    """Rótulo 2 com 6 classes -> [0, 0, 1, 0, 0, 0]."""
    return B.one_hot(y, n_classes, dtype)


# ===========================================================================
# A rede
# ===========================================================================
class MLP:
    """MLP com qualquer número de camadas ocultas e saída softmax.

    No projeto usamos camadas_ocultas=(n1, n2) — duas camadas ocultas. O
    Skip-Gram (script 07) usa a mesma classe com UMA camada oculta linear, como
    sugere a dica do enunciado ("retire uma camada").

    Parâmetros
    ----------
    n_entrada       : tamanho do vetor de entrada (3.000 no TF-IDF)
    camadas_ocultas : tupla com o número de neurônios de cada camada oculta
    n_saida         : número de classes (6)
    ativacao        : "relu", "sigmoid" ou "linear" (nas camadas ocultas)
    l2              : λ da regularização L2 (0.0 = sem regularização)
    semente         : semente da inicialização dos pesos
    dtype           : float32 (padrão, igual ao Keras) ou float64 (para o
                      teste de derivadas por diferenças finitas)
    dispositivo     : "auto" (GPU para modelos grandes, se houver; ver
                      src/backend.py), "cpu" ou "gpu"
    """

    def __init__(self, n_entrada, camadas_ocultas, n_saida, ativacao="relu",
                 l2=0.0, semente=0, dtype=np.float32, dispositivo="auto"):
        self.tamanhos = [n_entrada, *camadas_ocultas, n_saida]
        self.ativacao = ativacao
        self.f, self.df = ATIVACOES[ativacao]
        self.l2 = l2
        self.dtype = dtype
        self.n_saida = n_saida

        rng = np.random.default_rng(semente)
        self.W, self.b = [], []
        for fan_in, fan_out in zip(self.tamanhos[:-1], self.tamanhos[1:]):
            a = np.sqrt(6.0 / (fan_in + fan_out))          # Glorot uniforme
            self.W.append(rng.uniform(-a, a, size=(fan_in, fan_out)).astype(dtype))
            self.b.append(np.zeros(fan_out, dtype=dtype))   # bias começa em zero

        # Os pesos são sorteados sempre na CPU, com a mesma semente: a rede começa
        # igual na CPU e na GPU. Se o modelo for para a GPU, as matrizes são copiadas.
        self.dispositivo = B.escolher(dispositivo, self.n_parametros())
        if self.na_gpu:
            self.W = [B.para_dispositivo(W, self.dispositivo, dtype) for W in self.W]
            self.b = [B.para_dispositivo(b, self.dispositivo, dtype) for b in self.b]

    @property
    def na_gpu(self):
        """True quando as contas rodam com PyTorch (GPU, ou o modo de teste torch-cpu)."""
        return self.dispositivo != "cpu"

    def pesos_numpy(self):
        """Cópia dos pesos e bias como arrays NumPy (para análises e figuras)."""
        return [B.para_numpy(W) for W in self.W], [B.para_numpy(b) for b in self.b]

    @property
    def n_camadas(self):
        return len(self.W)   # camadas com pesos (ocultas + saída)

    def n_parametros(self):
        return sum(int(np.prod(W.shape)) + int(np.prod(b.shape)) for W, b in zip(self.W, self.b))

    # -----------------------------------------------------------------------
    # 1) Forward
    # -----------------------------------------------------------------------
    def forward(self, X):
        """Propaga X pela rede e guarda o que o backward vai precisar.

        Devolve
          Z : lista com as pré-ativações de cada camada  [Z1, Z2, Z3]
          A : lista com as entradas de cada camada + saída [X, A1, A2, P]
              (A[l] é a ENTRADA da camada l; A[-1] são as probabilidades)
        X pode ser denso (numpy) ou esparso (scipy): X @ W funciona nos dois.
        Na GPU, X já chega como tensor denso na placa (ver treinar).
        """
        A = [X]
        Z = []
        for l in range(self.n_camadas):
            z = A[-1] @ self.W[l] + self.b[l]           # Z_l = A_{l-1} W_l + b_l
            Z.append(z)
            if l < self.n_camadas - 1:
                A.append(self.f(z))                     # camada oculta: A = f(Z)
            else:
                A.append(softmax(z))                    # saída: P = softmax(Z)
        return Z, A

    def forward_numpy(self, X):
        """forward() com o resultado em NumPy (usado pelas análises do script 05)."""
        Z, A = self.forward(self._no_dispositivo(X))
        return [B.para_numpy(z) for z in Z], [X] + [B.para_numpy(a) for a in A[1:]]

    def _no_dispositivo(self, X):
        """Entrada pronta para o forward: tensor na GPU, ou X como está na CPU."""
        return B.para_dispositivo(X, self.dispositivo, self.dtype) if self.na_gpu else X

    # -----------------------------------------------------------------------
    # 2) Função de custo
    # -----------------------------------------------------------------------
    def termo_l2(self):
        """λ · Σ_l ||W_l||²  (o bias fica de fora)."""
        if self.l2 == 0:
            return 0.0
        return self.l2 * sum(B.soma_quadrados(W) for W in self.W)

    def entropia_cruzada(self, X, y):
        """-(1/N) Σ log P[i, y_i], calculada pelos logits (estável)."""
        Z, _ = self.forward(self._no_dispositivo(X))
        z = Z[-1].double() if B.eh_tensor(Z[-1]) else Z[-1].astype(np.float64)
        logp = B.para_numpy(log_softmax(z))
        return float(-logp[np.arange(len(y)), y].mean())

    def perda_total(self, X, y):
        """Custo que o gradiente minimiza: entropia cruzada + L2."""
        return self.entropia_cruzada(X, y) + self.termo_l2()

    # -----------------------------------------------------------------------
    # 3) Backward — regra da cadeia
    # -----------------------------------------------------------------------
    def backward(self, Z, A, Y):
        """Calcula ∂L/∂W e ∂L/∂b de todas as camadas (equações do topo).

        Z, A : o que o forward guardou
        Y    : rótulos do lote em one-hot (N x 6)
        """
        N = Y.shape[0]
        grad_W = [None] * self.n_camadas
        grad_b = [None] * self.n_camadas

        # (a) saída: softmax + entropia cruzada -> dZ = (P - Y) / N
        dZ = (A[-1] - Y) / N

        # percorre as camadas de trás para frente: l = 2, 1, 0 (W3, W2, W1)
        for l in reversed(range(self.n_camadas)):
            # (b) gradiente dos pesos e do bias da camada l
            #     Z_l = A_{l-1} W_l + b_l  =>  dW_l = A_{l-1}ᵀ dZ_l ,  db_l = Σ dZ_l
            grad_W[l] = A[l].T @ dZ
            if self.l2 > 0:
                grad_W[l] += 2 * self.l2 * self.W[l]     # derivada de λ||W||²
            grad_b[l] = B.soma_por_coluna(dZ)

            if l > 0:
                # (c) propaga para a camada anterior
                #     dA_{l-1} = dZ_l W_lᵀ        (derivada de Z_l em relação a A_{l-1})
                #     dZ_{l-1} = dA_{l-1} ⊙ f'(Z_{l-1})
                dA = dZ @ self.W[l].T
                dZ = dA * self.df(Z[l - 1], A[l])
        return grad_W, grad_b

    def gradientes(self, X, y):
        """Atalho: forward + backward num lote (usado na conferência de derivadas)."""
        if self.na_gpu:
            Z, A = self.forward(self._no_dispositivo(X))
            Y = one_hot(B.vetor_indices(y, self.dispositivo), self.n_saida, self.dtype)
            gW, gb = self.backward(Z, A, Y)
            return [B.para_numpy(g) for g in gW], [B.para_numpy(g) for g in gb]
        Z, A = self.forward(X)
        return self.backward(Z, A, one_hot(y, self.n_saida, self.dtype))

    # -----------------------------------------------------------------------
    # Predição e métricas
    # -----------------------------------------------------------------------
    def prever_proba(self, X, lote=8192):
        """Probabilidades das classes (array NumPy), em pedaços para não estourar a memória.

        Na GPU, X pode já estar na placa (tensor) ou na CPU (cada pedaço é copiado).
        """
        saidas = []
        for i in range(0, X.shape[0], lote):
            pedaco = self._no_dispositivo(X[i:i + lote])
            saidas.append(B.para_numpy(self.forward(pedaco)[1][-1]))
        return np.vstack(saidas)

    def prever(self, X):
        return self.prever_proba(X).argmax(axis=1)   # classe de maior probabilidade

    def avaliar(self, X, y):
        """Entropia cruzada (sem o termo L2), acurácia e F1 macro."""
        P = self.prever_proba(X).astype(np.float64)
        perda = float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None)).mean())
        pred = P.argmax(axis=1)
        return dict(perda=perda,
                    acc=float((pred == y).mean()),
                    f1=float(f1_score(y, pred, average="macro")))

    # -----------------------------------------------------------------------
    # 4) Treinamento — SGD com momentum em mini-batches
    # -----------------------------------------------------------------------
    def treinar(self, X, y, X_val=None, y_val=None, epocas=30, tamanho_lote=64,
                taxa=0.1, momento=0.9, semente=0, ao_fim_da_epoca=None,
                registrar_gradientes=False, avaliar_cada_epoca=True, verbose=False,
                otimizador="sgd", beta1=0.9, beta2=0.999, eps=1e-7):
        """Treina a rede e devolve o histórico por época.

        A cada época os exemplos são embaralhados com np.random.default_rng(semente)
        — a mesma sequência é recriada no script 02 para alimentar o Keras com
        os MESMOS lotes, na mesma ordem.

        ao_fim_da_epoca(modelo, epoca): função opcional chamada no fim de cada
        época (o script 05 usa para guardar a variação dos pesos).
        otimizador: "sgd" (SGD com momentum, usado em todo o projeto) ou
        "adam" (o controle do script 05 e o Skip-Gram do script 07).

        Na GPU: X e X_val são copiados UMA vez para a placa (densos) e os lotes
        são recortados lá; se X não couber (o Skip-Gram tem 571 mil linhas),
        cada lote é recortado na CPU e copiado. A ordem dos lotes é a mesma da CPU.
        """
        rng = np.random.default_rng(semente)
        n = X.shape[0]
        gpu = self.na_gpu
        if gpu:
            X_disp = self._no_dispositivo(X) if B.cabe_na_gpu(X) else None
            y_disp = B.vetor_indices(y, self.dispositivo)
            if X_val is not None and B.cabe_na_gpu(X_val):
                X_val = self._no_dispositivo(X_val)
        # SGD: velocidades do momentum, uma por parâmetro, começando em zero
        # Adam: os mesmos vetores guardam a média do gradiente (m) e criamos s
        vel_W = [B.zeros_como(W) for W in self.W]
        vel_b = [B.zeros_como(b) for b in self.b]
        s_W = [B.zeros_como(W) for W in self.W] if otimizador == "adam" else None
        s_b = [B.zeros_como(b) for b in self.b] if otimizador == "adam" else None
        passo = 0

        hist = {k: [] for k in ["perda_treino", "acc_treino", "f1_treino",
                                "perda_val", "acc_val", "f1_val",
                                "perda_media_lotes", "tempo"]}
        if registrar_gradientes:
            hist["norma_grad"] = []   # norma média de dW por camada, por época

        for epoca in range(1, epocas + 1):
            t0 = time.perf_counter()
            perm = rng.permutation(n)             # nova ordem dos exemplos
            soma_perda, normas, n_lotes = 0.0, 0.0, 0
            if gpu:
                perm_disp = B.vetor_indices(perm, self.dispositivo)   # a mesma ordem, na placa

            for inicio in range(0, n, tamanho_lote):
                idx = perm[inicio:inicio + tamanho_lote]
                if gpu:
                    idx_disp = perm_disp[inicio:inicio + tamanho_lote]
                    Xb = X_disp[idx_disp] if X_disp is not None else self._no_dispositivo(X[idx])
                    yb = y_disp[idx_disp]
                else:
                    Xb, yb = X[idx], y[idx]
                Yb = one_hot(yb, self.n_saida, self.dtype)

                Z, A = self.forward(Xb)
                grad_W, grad_b = self.backward(Z, A, Yb)

                # perda do lote (só para acompanhar; igual ao "loss" que o Keras imprime)
                soma_perda += B.perda_do_lote(A[-1], yb)
                if registrar_gradientes:
                    normas += B.normas(grad_W)
                n_lotes += 1

                if otimizador == "sgd":
                    # atualização: v <- μv - ηg ; W <- W + v   (pesos e bias, "no lugar")
                    B.sgd_momentum(self.W + self.b, vel_W + vel_b, grad_W + grad_b,
                                   taxa, momento)
                else:
                    # Adam: passo de tamanho ~η em cada peso, normalizado por √s
                    passo += 1
                    alfa = taxa * np.sqrt(1 - beta2 ** passo) / (1 - beta1 ** passo)
                    for l in range(self.n_camadas):
                        for P, G, M, S in ((self.W, grad_W, vel_W, s_W),
                                           (self.b, grad_b, vel_b, s_b)):
                            M[l] += (1 - beta1) * (G[l] - M[l])
                            S[l] += (1 - beta2) * (G[l] ** 2 - S[l])
                            P[l] -= alfa * M[l] / (B.raiz(S[l]) + eps)

            # tempo só do treino (sem a avaliação abaixo) — usado na comparação com o Keras
            hist["tempo"].append(time.perf_counter() - t0)
            hist["perda_media_lotes"].append(float(soma_perda) / n_lotes)
            if registrar_gradientes:
                hist["norma_grad"].append((B.para_numpy(normas) / n_lotes).tolist())

            # métricas no fim da época (conjunto inteiro, sem dropout/ruído)
            if avaliar_cada_epoca:
                m_tr = self.avaliar(X_disp if gpu and X_disp is not None else X, y)
                hist["perda_treino"].append(m_tr["perda"])
                hist["acc_treino"].append(m_tr["acc"])
                hist["f1_treino"].append(m_tr["f1"])
                if X_val is not None:
                    m_va = self.avaliar(X_val, y_val)
                    hist["perda_val"].append(m_va["perda"])
                    hist["acc_val"].append(m_va["acc"])
                    hist["f1_val"].append(m_va["f1"])

            if ao_fim_da_epoca is not None:
                ao_fim_da_epoca(self, epoca)
            if verbose:
                msg = f"época {epoca:3d}  perda {hist['perda_media_lotes'][-1]:.4f}"
                if avaliar_cada_epoca and X_val is not None:
                    msg += f"  acc_val {hist['acc_val'][-1]:.4f}  f1_val {hist['f1_val'][-1]:.4f}"
                print(msg)
        return hist
