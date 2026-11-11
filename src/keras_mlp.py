"""
A mesma MLP escrita em Keras — usada só para CONFERIR a implementação NumPy.

"Mesmos parâmetros" aqui significa tudo igual:
  * mesma arquitetura (n1, n2), mesma ativação, mesma saída com 6 classes;
  * os MESMOS pesos iniciais (copiamos os da rede NumPy para o Keras);
  * mesma função de custo (entropia cruzada média) e mesma L2: λ·Σw²;
  * mesmo otimizador (SGD com momentum), taxa, momento e tamanho de lote;
  * os MESMOS lotes na mesma ordem (embaralhamos com a mesma semente).
Se a regra da cadeia estiver certa, as duas redes têm de andar juntas.

Detalhe da saída: a última camada do Keras é linear (devolve logits) e a
função de custo usa from_logits=True, que aplica o softmax DENTRO do cálculo
da entropia cruzada (log-sum-exp, numericamente estável). É matematicamente
o mesmo que Dense(6, activation="softmax") + entropia cruzada, e é exatamente
o que a classe MLP faz: o gradiente da saída é (P - Y)/N nos dois casos.
"""
import os

# Desliga as otimizações oneDNN do TensorFlow, que reordenam somas e mudam os
# últimos dígitos dos resultados. Com isso a comparação fica reprodutível.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # menos mensagens no terminal

import numpy as np
import tensorflow as tf
from tensorflow import keras


def construir_keras(n_entrada, camadas_ocultas, n_saida, ativacao, l2):
    """Sequential: Dense(n1, f) -> Dense(n2, f) -> Dense(6) com L2 em todos os kernels."""
    reg = keras.regularizers.L2(l2) if l2 > 0 else None
    camadas = [keras.Input(shape=(n_entrada,))]
    for n in camadas_ocultas:
        camadas.append(keras.layers.Dense(n, activation=ativacao, kernel_regularizer=reg))
    camadas.append(keras.layers.Dense(n_saida, kernel_regularizer=reg))   # logits
    return keras.Sequential(camadas)


def compilar(modelo, taxa, momento):
    modelo.compile(
        optimizer=keras.optimizers.SGD(learning_rate=taxa, momentum=momento),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    return modelo


def copiar_pesos(mlp_numpy, modelo_keras):
    """Coloca no Keras exatamente os pesos atuais da rede NumPy.

    O Dense do Keras guarda o kernel com forma (entradas, saídas), a mesma
    convenção da classe MLP, então não é preciso transpor nada.
    """
    densas = [c for c in modelo_keras.layers if isinstance(c, keras.layers.Dense)]
    for camada, W, b in zip(densas, mlp_numpy.W, mlp_numpy.b):
        camada.set_weights([W.astype(np.float32), b.astype(np.float32)])


def pesos_keras(modelo_keras):
    """Lista [(W1, b1), (W2, b2), (W3, b3)] em numpy."""
    densas = [c for c in modelo_keras.layers if isinstance(c, keras.layers.Dense)]
    return [tuple(v.numpy() if hasattr(v, "numpy") else v for v in c.get_weights())
            for c in densas]


def gradientes_keras(modelo_keras, X, y):
    """Gradiente do custo total (entropia cruzada + L2) pelo autograd do TensorFlow.

    tf.GradientTape "grava" as operações do forward e depois aplica a regra da
    cadeia automaticamente — é a referência contra a qual comparamos o
    backward escrito à mão.
    Devolve listas (grad_W, grad_b) na mesma ordem da classe MLP.
    """
    perda_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    X = tf.constant(X, dtype=tf.float32)
    with tf.GradientTape() as fita:
        logits = modelo_keras(X, training=True)
        perda = perda_fn(y, logits)
        if modelo_keras.losses:                 # termos de regularização L2
            perda = perda + tf.add_n(modelo_keras.losses)
    variaveis = modelo_keras.trainable_variables   # [W1, b1, W2, b2, W3, b3]
    grads = fita.gradient(perda, variaveis)
    grads = [g.numpy() for g in grads]
    return grads[0::2], grads[1::2]


def prever_proba_keras(modelo_keras, X, lote=4096):
    logits = modelo_keras.predict(X, batch_size=lote, verbose=0)
    return tf.nn.softmax(logits).numpy()
