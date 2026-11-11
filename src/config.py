"""
Configurações compartilhadas por todos os scripts do Projeto 3.

Deixar tudo aqui evita "números mágicos" espalhados pelo código: se você quiser
mudar o número de épocas, a taxa de aprendizado ou a grade de neurônios, muda
só neste arquivo e todos os experimentos passam a usar o novo valor.

Dica: para testar o pipeline inteiro rapidamente, rode com menos épocas
definindo a variável de ambiente EPOCAS, por exemplo (PowerShell):
    $env:EPOCAS=3; python 05_pesos_e_saturacao.py
Os resultados oficiais usam o padrão de 50 épocas.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos (pathlib funciona igual no Windows e no Linux)
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados" / "emotion"
FIGURAS = RAIZ / "figuras"
RESULTADOS = RAIZ / "resultados"
FIGURAS.mkdir(exist_ok=True)
RESULTADOS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Base de dados
# ---------------------------------------------------------------------------
# Ordem dos rótulos da base dair-ai/emotion (o número de cada classe é a posição)
CLASSES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
CLASSES_PT = ["tristeza", "alegria", "amor", "raiva", "medo", "surpresa"]
N_CLASSES = len(CLASSES)

# TF-IDF: as 3.000 palavras mais frequentes do treino formam o vocabulário.
# Com 3.000 a regressão logística já chega no mesmo nível que com o vocabulário
# inteiro (15.212 palavras), e a primeira matriz de pesos fica com tamanho
# administrável (3.000 x n1).
MAX_PALAVRAS = 3000

# ---------------------------------------------------------------------------
# Treinamento (fixos em todos os experimentos — o que varia está na GRADE)
# ---------------------------------------------------------------------------
SEMENTE = 42          # reprodutibilidade: mesma semente -> mesmos resultados
EPOCAS = int(os.environ.get("EPOCAS", 50))
TAMANHO_LOTE = 64     # mini-batch
TAXA_APRENDIZADO = 0.05
MOMENTO = 0.9         # SGD com momentum (mesma fórmula do keras.optimizers.SGD)
N_FOLDS = 3           # 3-fold cross-validation, como pede o enunciado
# Por que SGD e não Adam? No SGD o tamanho do passo é proporcional ao
# gradiente (ΔW = -η·g, suavizado pelo momento). Assim, quando o gradiente
# "some" (sigmoid saturada), os pesos param de mudar — e isso aparece nos
# mapas de calor do script 05. O Adam normaliza cada passo pelo tamanho do
# gradiente e esconderia exatamente o fenômeno que o enunciado pede para
# analisar.
# Por que η = 0,05? Teste preliminar (treino -> validação, 50 épocas):
#   η = 0,3  -> a ReLU diverge (fica presa prevendo só a classe majoritária)
#   η = 0,1  -> a ReLU aprende, mas o custo tem picos (instável)
#   η = 0,05 -> a ReLU fica estável; a sigmoid aprende mais devagar (platô)
# O platô da sigmoid é justamente um dos fenômenos analisados.
#
# Experimento de controle do script 05: a mesma sigmoid treinada com Adam.
# Aumentar η no SGD não resolve (testado: com η = 0,5, 2 e 5 a sigmoid
# 256-128 continua presa; a camada de saída passa a oscilar antes de W1 receber
# sinal). O Adam ajusta o passo de cada peso pela escala do próprio gradiente
# e destrava a rede — evidência de que o problema é o gradiente que encolhe
# entre as camadas, não a capacidade da rede.
CONTROLE_SIGMOID = dict(otimizador="adam", taxa=1e-3)

# ---------------------------------------------------------------------------
# Grade de hiperparâmetros (o que o enunciado manda variar)
# ---------------------------------------------------------------------------
ATIVACOES = ["relu", "sigmoid"]
L2_VALORES = [0.0, 1e-4]                  # 0.0 = sem regularização
NEURONIOS = [(16, 8), (64, 32), (256, 128)]  # (camada oculta 1, camada oculta 2)

# Métrica usada para escolher a melhor configuração.
# A base é desbalanceada (surpresa tem 3,6% dos exemplos), então a acurácia
# sozinha favorece quem acerta as classes grandes. A F1 macro dá o mesmo peso
# a cada classe e por isso é o critério de escolha; a acurácia é reportada junto.
METRICA_SELECAO = "f1_val"
