"""
Onde as contas da MLP rodam: CPU (NumPy) ou GPU (PyTorch com CUDA).

A REDE CONTINUA ESCRITA À MÃO
-----------------------------
Forward, regra da cadeia e atualização dos pesos são as mesmas equações de
src/mlp.py. O PyTorch entra só como um "NumPy da placa de vídeo": guarda as
matrizes na memória da GPU e faz lá as mesmas operações (produto de matrizes,
exp, soma, comparação...). O autograd do PyTorch NÃO é usado: nenhum tensor
pede gradiente (requires_grad=False), então nenhuma derivada é calculada
automaticamente — quem calcula ∂L/∂W é o nosso backward.

QUANDO A GPU COMPENSA
---------------------
Cada operação na GPU custa um tempo fixo de lançamento (~10 µs), qualquer que
seja o tamanho da matriz. Com lotes de 64 tweets, uma rede pequena faz contas
tão rápidas que esse custo fixo domina, e a CPU (com a entrada esparsa do
TF-IDF) ganha. Numa rede grande (256-128: 800 mil pesos) ou com entrada densa
(as representações do Skip-Gram), cada passo do SGD mexe em centenas de
milhares de números e a GPU ganha com folga. Por isso o modo "auto" manda para
a GPU só os modelos grandes.

ESCOLHA DO DISPOSITIVO (variável de ambiente USAR_GPU)
------------------------------------------------------
  auto (padrão) : modelos com >= 300 mil parâmetros na GPU (se houver PyTorch
                  com CUDA); os menores na CPU
  1             : todo modelo na GPU (erro se não houver GPU)
  0             : tudo na CPU (NumPy), como na primeira versão do projeto
  torch-cpu     : só para testar este código numa máquina sem GPU (lento)

Para ter a GPU: placa NVIDIA + PyTorch com CUDA (ver README).
"""
import os

import numpy as np
import scipy.sparse as sp
from scipy.special import expit

try:
    import torch
except Exception:      # PyTorch não instalado: tudo roda na CPU
    torch = None

MODO = os.environ.get("USAR_GPU", "auto").strip().lower()
MIN_PARAMETROS_GPU = 300_000
# entrada maior que isto (em bytes, densa) não vai inteira para a GPU:
# os lotes são copiados um a um (caso do Skip-Gram: 571 mil linhas x 3.000)
MAX_BYTES_ENTRADA_GPU = 1_500_000_000

if torch is not None:
    # contas em float32 de verdade (sem o TF32 das placas Ampere, que arredonda mais)
    torch.backends.cuda.matmul.allow_tf32 = False


def gpu_disponivel():
    return torch is not None and torch.cuda.is_available()


def descrever():
    """Texto curto dizendo onde os modelos vão rodar (impresso pelo rodar_tudo.py)."""
    if MODO in ("0", "cpu", "nao", "não", "false"):
        return "CPU (USAR_GPU=0)"
    if MODO == "torch-cpu":
        return "PyTorch na CPU (modo de teste)"
    if not gpu_disponivel():
        if torch is None:
            return "CPU — PyTorch não instalado (ver README para usar a GPU)"
        return f"CPU — PyTorch {torch.__version__} instalado, mas sem CUDA (versão só-CPU)"
    nome = torch.cuda.get_device_name(0)
    quais = "todos os modelos" if MODO in ("1", "sim", "true", "gpu") else \
        f"modelos com >= {MIN_PARAMETROS_GPU:,} parâmetros".replace(",", ".")
    return f"GPU {nome} (PyTorch {torch.__version__}, CUDA {torch.version.cuda}) para {quais}"


def escolher(pedido, n_parametros):
    """Decide onde um modelo roda: "cpu", "cuda" ou "torch-cpu".

    pedido: "auto" (segue USAR_GPU), "cpu" (força NumPy — usado no script 02,
    que compara com o Keras e confere derivadas em float64) ou "gpu".
    """
    if pedido == "cpu" or MODO in ("0", "cpu", "nao", "não", "false"):
        return "cpu"
    if MODO == "torch-cpu":
        if torch is None:
            raise RuntimeError("USAR_GPU=torch-cpu precisa do PyTorch instalado")
        return "torch-cpu" if (pedido == "gpu" or n_parametros >= MIN_PARAMETROS_GPU) else "cpu"
    forcar = pedido == "gpu" or MODO in ("1", "sim", "true", "gpu")
    if not gpu_disponivel():
        if forcar:
            raise RuntimeError("Pediu GPU, mas não há PyTorch com CUDA (ver README)")
        return "cpu"
    if forcar or n_parametros >= MIN_PARAMETROS_GPU:
        return "cuda"
    return "cpu"


def _dev(dispositivo):
    return "cpu" if dispositivo == "torch-cpu" else dispositivo


# ---------------------------------------------------------------------------
# Conversões CPU <-> GPU
# ---------------------------------------------------------------------------
def eh_tensor(x):
    return torch is not None and isinstance(x, torch.Tensor)


def para_numpy(x):
    """Tensor (GPU ou CPU) -> array NumPy; arrays NumPy passam direto."""
    if eh_tensor(x):
        return x.detach().cpu().numpy()
    return x


def para_dispositivo(x, dispositivo, dtype=np.float32):
    """Matriz NumPy ou esparsa (SciPy) -> tensor denso no dispositivo.

    Na GPU usamos a matriz densa: com ela cada lote é um produto de matrizes
    comum, a operação mais rápida da placa. (Na CPU a entrada continua esparsa.)
    """
    if eh_tensor(x):
        return x
    if sp.issparse(x):
        x = x.toarray()
    # torch.tensor copia os dados (o array original pode ser só-leitura, vindo do pandas)
    return torch.tensor(np.asarray(x, dtype=dtype), device=_dev(dispositivo))


def vetor_indices(y, dispositivo):
    return torch.tensor(np.asarray(y, dtype=np.int64), device=_dev(dispositivo))


def cabe_na_gpu(X):
    """A matriz X, densa em float32, cabe inteira na memória da GPU?"""
    return X.shape[0] * X.shape[1] * 4 <= MAX_BYTES_ENTRADA_GPU


# ---------------------------------------------------------------------------
# Operações usadas pela MLP — a mesma conta em NumPy (CPU) ou PyTorch (GPU)
# ---------------------------------------------------------------------------
def maximo_zero(z):                 # ReLU
    return torch.relu(z) if eh_tensor(z) else np.maximum(z, 0)


def indicador_positivo(z):          # derivada da ReLU: 1 onde z > 0
    return (z > 0).to(z.dtype) if eh_tensor(z) else (z > 0).astype(z.dtype)


def sigmoide(z):
    return torch.sigmoid(z) if eh_tensor(z) else expit(z)


def uns_como(z):
    return torch.ones_like(z) if eh_tensor(z) else np.ones_like(z)


def zeros_como(x):
    return torch.zeros_like(x) if eh_tensor(x) else np.zeros_like(x)


def exp(x):
    return torch.exp(x) if eh_tensor(x) else np.exp(x)


def log(x):
    return torch.log(x) if eh_tensor(x) else np.log(x)


def raiz(x):
    return torch.sqrt(x) if eh_tensor(x) else np.sqrt(x)


def max_por_linha(z):               # máximo de cada linha, mantendo a coluna
    return z.amax(dim=1, keepdim=True) if eh_tensor(z) else z.max(axis=1, keepdims=True)


def soma_por_linha(z):
    return z.sum(dim=1, keepdim=True) if eh_tensor(z) else z.sum(axis=1, keepdims=True)


def soma_por_coluna(z):             # soma no lote (usada no gradiente do bias)
    return z.sum(dim=0) if eh_tensor(z) else z.sum(axis=0)


def soma_quadrados(W):              # ||W||² em float64, devolvido como float
    if eh_tensor(W):
        return float((W.double() ** 2).sum())
    return float(np.sum(W.astype(np.float64) ** 2))


def normas(matrizes):
    """Norma de Frobenius de cada matriz (fica na GPU até o fim da época)."""
    if matrizes and eh_tensor(matrizes[0]):
        return torch.stack([torch.linalg.vector_norm(m) for m in matrizes]).double()
    return np.array([np.linalg.norm(m) for m in matrizes], dtype=np.float64)


def one_hot(y, n_classes, dtype):
    """Rótulo 2 com 6 classes -> [0, 0, 1, 0, 0, 0]."""
    if eh_tensor(y):
        return torch.nn.functional.one_hot(y, n_classes).to(torch.float32 if dtype == np.float32
                                                             else torch.float64)
    Y = np.zeros((len(y), n_classes), dtype=dtype)
    Y[np.arange(len(y)), y] = 1
    return Y


def perda_do_lote(P, y):
    """-média(log P[i, y_i]) — na GPU devolve um tensor (somado sem sincronizar)."""
    if eh_tensor(P):
        p = P.gather(1, y.view(-1, 1)).clamp_min(1e-12)
        return -torch.log(p).mean()
    p = np.clip(P[np.arange(len(y)), y], 1e-12, None)
    return float(-np.log(p).mean())


def sgd_momentum(params, velocidades, grads, taxa, momento):
    """v <- μv - ηg ;  W <- W + v   (para todos os pesos e bias).

    Na GPU a mesma conta é feita em três chamadas que atualizam todas as
    matrizes de uma vez (torch._foreach_*), em vez de três chamadas por matriz:
    menos lançamentos, o que conta muito com lotes pequenos.
    """
    if params and eh_tensor(params[0]) and hasattr(torch, "_foreach_mul_"):
        torch._foreach_mul_(velocidades, momento)
        torch._foreach_add_(velocidades, grads, alpha=-taxa)
        torch._foreach_add_(params, velocidades)
        return
    for P, V, G in zip(params, velocidades, grads):
        V *= momento
        V -= taxa * G
        P += V
