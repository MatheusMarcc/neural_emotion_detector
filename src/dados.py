"""
Carga da base Emotion e construção da representação TF-IDF.

TEORIA — TF-IDF
---------------
Uma rede neural só recebe números, então cada tweet precisa virar um vetor.
No TF-IDF cada posição do vetor corresponde a uma palavra do vocabulário:

    tfidf(t, d) = tf(t, d) * idf(t)

    tf(t, d) = quantas vezes a palavra t aparece no documento d
    idf(t)   = ln((1 + N) / (1 + df(t))) + 1     (fórmula "suavizada" do sklearn)
               N = número de documentos, df(t) = em quantos documentos t aparece

A ideia: uma palavra pesa mais quando aparece no tweet (tf alto) e é rara na
base (idf alto). Palavras que aparecem em quase todo tweet, como "i" e "feel",
recebem idf baixo e quase não pesam. No fim, cada vetor é dividido pela sua
norma (norma L2 = 1), para tweets longos e curtos ficarem na mesma escala.

O vetor resultante é ESPARSO: das 3.000 posições, um tweet típico tem só ~16
diferentes de zero. Por isso guardamos a matriz no formato esparso do scipy
(só os valores não nulos), e a multiplicação X @ W continua funcionando.

CUIDADO COM VAZAMENTO DE DADOS
-------------------------------
O vocabulário e o idf são "aprendidos" dos dados. Se fossem calculados com a
base inteira, o modelo veria informação do fold de validação antes da hora.
Por isso o TF-IDF é ajustado (fit) só nos dados de treino de cada fold e
apenas aplicado (transform) nos de validação.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

from . import config


def carregar_emotion():
    """Lê os três CSVs oficiais (treino 16.000 / validação 2.000 / teste 2.000).

    Cada CSV tem as colunas: text (o tweet), label (0..5) e label_name.
    Os textos já vêm em minúsculas e sem pontuação.
    """
    treino = pd.read_csv(config.DADOS / "emotion_train.csv")
    validacao = pd.read_csv(config.DADOS / "emotion_validation.csv")
    teste = pd.read_csv(config.DADOS / "emotion_test.csv")
    return treino, validacao, teste


def criar_tfidf(max_palavras=config.MAX_PALAVRAS):
    """Cria (sem ajustar) o vetorizador TF-IDF usado em todo o projeto.

    token_pattern=r"\\b\\w+\\b" mantém palavras de uma letra (o padrão do
    sklearn descartaria "i"). Não removemos stopwords: palavras como "not"
    mudam a emoção da frase ("not happy").
    dtype float32: mesma precisão que o Keras usa, e metade da memória.
    """
    return TfidfVectorizer(
        max_features=max_palavras,
        token_pattern=r"(?u)\b\w+\b",
        dtype=np.float32,
    )


def folds_estratificados(y, n_folds=config.N_FOLDS, semente=config.SEMENTE):
    """Índices do k-fold ESTRATIFICADO.

    Estratificado = cada fold mantém a proporção original das classes. Numa
    base desbalanceada isso importa: um fold sem quase nenhuma "surpresa"
    daria uma estimativa distorcida.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=semente)
    return list(skf.split(np.zeros(len(y)), y))


def montar_folds_tfidf(textos, y, n_folds=config.N_FOLDS, semente=config.SEMENTE):
    """Para cada fold: ajusta o TF-IDF no treino do fold e transforma os dois lados.

    Devolve uma lista de dicionários com X_tr, y_tr, X_va, y_va (X esparso) e o
    vetorizador. Os folds são calculados uma vez e reaproveitados por todas as
    12 configurações, então todas são avaliadas exatamente nos mesmos dados.
    """
    textos = np.asarray(textos)
    y = np.asarray(y)
    folds = []
    for i, (idx_tr, idx_va) in enumerate(folds_estratificados(y, n_folds, semente)):
        vet = criar_tfidf()
        X_tr = vet.fit_transform(textos[idx_tr])   # fit só no treino do fold
        X_va = vet.transform(textos[idx_va])        # validação só é transformada
        folds.append(dict(fold=i + 1, X_tr=X_tr, y_tr=y[idx_tr],
                          X_va=X_va, y_va=y[idx_va], vetorizador=vet,
                          idx_tr=idx_tr, idx_va=idx_va))
    return folds
