"""
Métricas de classificação, intervalo de confiança e o baseline estratificado.

TEORIA — precisão, revocação e F1 (por classe)
----------------------------------------------
Para uma classe k (ex.: "medo"), olhando só "é medo" × "não é medo":
  precisão  = VP / (VP + FP)   dos tweets que a rede chamou de medo, quantos eram medo
  revocação = VP / (VP + FN)   dos tweets de medo, quantos a rede encontrou
  F1        = 2·P·R / (P + R)  média harmônica: só é alta se as duas forem altas

MACRO × MICRO (como juntar as 6 classes num número só)
------------------------------------------------------
  macro : calcula a métrica em cada classe e tira a média simples. Cada emoção
          pesa igual — surpresa (66 tweets no teste) vale tanto quanto alegria
          (695). É a métrica certa para base desbalanceada, e é o nosso critério.
  micro : soma VP, FP e FN de todas as classes e calcula uma vez. Cada TWEET
          pesa igual, então as classes grandes dominam. Quando cada tweet tem
          exatamente um rótulo, todo erro é um FP de uma classe e um FN de outra:
          precisão micro = revocação micro = F1 micro = ACURÁCIA.
  A diferença macro − micro mostra quanto as classes pequenas puxam a nota para baixo.

TEORIA — curva ROC e AUROC
--------------------------
A rede devolve probabilidades; a classe prevista é a de maior probabilidade.
A curva ROC olha a probabilidade de UMA classe como um placar e varre todos os
limiares: com o limiar t, chamamos de "medo" todo tweet com P(medo) >= t e
medimos
  TPR (taxa de verdadeiros positivos) = revocação
  FPR (taxa de falsos positivos)      = FP / (FP + VN)
Limiar alto: poucos positivos (canto (0, 0)); limiar baixo: todos positivos
(canto (1, 1)). A AUROC é a área sob essa curva e tem uma leitura direta: a
probabilidade de um tweet de medo sorteado receber P(medo) maior que um tweet
de outra emoção sorteado. 0,5 = sorteio; 1,0 = separação perfeita.
Com 6 classes usamos "uma contra as outras" (one-vs-rest): uma curva por
emoção. A AUROC macro é a média das 6 áreas; a micro junta todos os pares
(tweet, classe) numa curva só.
A AUROC não depende do limiar nem da proporção das classes: mede se a rede
ORDENA bem, mesmo quando o argmax erra.

TEORIA — Random Stratified Baseline
-----------------------------------
Um classificador que ignora o texto e sorteia a classe de cada tweet na
proporção do treino (33,5% alegria, 29,2% tristeza, ...). É o piso honesto:
um modelo só aprendeu alguma coisa se ficar claramente acima dele.
Esperado: acurácia ≈ Σ p_k² ≈ 0,24 e, em cada classe, precisão ≈ revocação ≈ p_k
(então F1 macro ≈ média dos p_k = 1/6 ≈ 0,17); AUROC ≈ 0,5.

TEORIA — intervalo de confiança para a média dos k folds
--------------------------------------------------------
Os k = 5 folds dão 5 notas x_1..x_5. A média x̄ é a estimativa; o intervalo de
95% é
        x̄ ± t_{0,975; k−1} · s / √k
  s           = desvio padrão amostral das 5 notas (divide por k − 1)
  s / √k      = erro padrão da média
  t_{0,975;4} = 2,776, o quantil da t de Student com 4 graus de liberdade (com
                só 5 notas, a t é mais larga que a normal, cujo valor seria 1,96)
Ressalva: a fórmula supõe notas independentes, e os folds compartilham boa
parte do treino (cada par de folds treina com 75% dos mesmos tweets). Por isso
o intervalo é um pouco otimista (Nadeau e Bengio, 2003); ele serve para
comparar configurações, não como garantia exata.
"""
import numpy as np
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from sklearn.preprocessing import label_binarize

NIVEL_CONFIANCA = 0.95


def intervalo_confianca(valores, nivel=NIVEL_CONFIANCA):
    """(média, meia-largura) do intervalo de confiança t de Student."""
    v = np.asarray(valores, dtype=float)
    k = len(v)
    if k < 2:
        return float(v.mean()), float("nan")
    t = stats.t.ppf(0.5 + nivel / 2, k - 1)
    return float(v.mean()), float(t * v.std(ddof=1) / np.sqrt(k))


def metricas_completas(y, P, n_classes=6):
    """Acurácia, precisão/revocação/F1 (macro e micro) e AUROC (macro e micro).

    y : rótulos verdadeiros (N)
    P : probabilidades previstas (N x n_classes)
    """
    y = np.asarray(y)
    pred = P.argmax(axis=1)
    m = dict(acc=float(accuracy_score(y, pred)))
    for media in ("macro", "micro"):
        p, r, f, _ = precision_recall_fscore_support(y, pred, average=media, zero_division=0)
        m[f"precisao_{media}"], m[f"revocacao_{media}"], m[f"f1_{media}"] = float(p), float(r), float(f)
    Y = label_binarize(y, classes=range(n_classes))
    m["auc_macro"] = float(roc_auc_score(Y, P, average="macro"))
    m["auc_micro"] = float(roc_auc_score(Y, P, average="micro"))
    return m


def metricas_por_classe(y, P, n_classes=6):
    """Precisão, revocação, F1, AUROC e suporte de cada classe."""
    pred = P.argmax(axis=1)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(n_classes), zero_division=0)
    Y = label_binarize(y, classes=range(n_classes))
    auc = [roc_auc_score(Y[:, k], P[:, k]) for k in range(n_classes)]
    return dict(precisao=p, revocacao=r, f1=f, auc=np.array(auc), suporte=s)


def curvas_roc(y, P, n_classes=6):
    """Pontos (FPR, TPR) de cada classe e das médias micro e macro.

    macro: interpola as 6 curvas na mesma grade de FPR e tira a média do TPR.
    """
    Y = label_binarize(y, classes=range(n_classes))
    curvas = {}
    for k in range(n_classes):
        fpr, tpr, _ = roc_curve(Y[:, k], P[:, k])
        curvas[k] = (fpr, tpr)
    fpr, tpr, _ = roc_curve(Y.ravel(), P.ravel())
    curvas["micro"] = (fpr, tpr)
    grade = np.linspace(0, 1, 501)
    curvas["macro"] = (grade, np.mean([np.interp(grade, *curvas[k]) for k in range(n_classes)], axis=0))
    return curvas


def baseline_estratificado(y_treino, n_amostras, semente):
    """Random Stratified Baseline: sorteia a classe na proporção do treino.

    Devolve "probabilidades" no formato da rede (N x 6): cada linha é o
    one-hot da classe sorteada (é o que o DummyClassifier do scikit-learn faz).
    """
    modelo = DummyClassifier(strategy="stratified", random_state=semente)
    modelo.fit(np.zeros((len(y_treino), 1)), y_treino)   # o texto é ignorado
    return modelo.predict_proba(np.zeros((n_amostras, 1)))
