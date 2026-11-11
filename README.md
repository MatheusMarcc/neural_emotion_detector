# Projeto 3 — Redes Neurais

MLP com duas camadas ocultas **implementada do zero** (forward, custo, regra da
cadeia para pesos e bias, SGD com momentum) e comparada com a mesma rede em
**Keras**, numa base de texto difícil: a **Emotion** (20.000 tweets, 6 emoções,
desbalanceada 9,4 : 1).

O Keras só aparece no script 02, para conferir as derivadas. O scikit-learn entra
apenas no TF-IDF, na divisão em folds, nas métricas e nas referências (regressão
logística e baseline estratificado). O PyTorch, quando instalado, é só o "NumPy da
GPU": a rede continua escrita à mão, sem autograd (ver `src/backend.py`).

## Como rodar (com GPU)

Num terminal aberto nesta pasta:

```
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python rodar_tudo.py
```

A terceira linha deve mostrar `True NVIDIA GeForce RTX 3050 ...`. Se mostrar
`False`, o PyTorch instalado é o só-CPU: desinstale (`python -m pip uninstall torch`)
e repita a segunda linha. Sem GPU tudo funciona do mesmo jeito, só que na CPU.

O `rodar_tudo.py` imprime na primeira linha onde a MLP vai rodar e, no fim, quanto
cada script levou (também salvo em `resultados/tempos_execucao.txt`). Ele roda só o
que precisa:

| script | roda? |
|---|---|
| 01 estudo da base, 02 derivadas × Keras | pulados se os resultados já existem (não dependem da validação cruzada) |
| 03 validação cruzada 5-fold | sempre |
| 04 curvas de aprendizado | sempre |
| 05 mapas de calor e saturação | só se a arquitetura da melhor configuração mudar |
| 06 modelo final no teste | sempre |
| 07 Skip-Gram | sempre |

`python rodar_tudo.py --tudo` roda os 7 do zero; `python rodar_tudo.py 06 07` roda
só os escolhidos. Para só testar se tudo roda (PowerShell): `$env:EPOCAS=2; python
rodar_tudo.py` — e depois `Remove-Item Env:EPOCAS` antes da rodada oficial (50 épocas).

**Onde a GPU entra.** No modo padrão (`USAR_GPU=auto`), os modelos com 300 mil
parâmetros ou mais vão para a placa: as redes 256-128, o Skip-Gram e as representações
densas do script 07. As redes 16-8 e 64-32 ficam na CPU, onde são mais rápidas: com
lotes de 64 tweets, o custo fixo de cada operação na GPU domina numa rede pequena.
`$env:USAR_GPU=1` manda tudo para a GPU; `$env:USAR_GPU=0` roda tudo na CPU.

Os números podem variar um pouco de uma máquina para outra, e entre CPU e GPU (até
cerca de 1 ponto percentual, com as mesmas conclusões): a ReLU é sensível a
diferenças de arredondamento (ver relatório, 3.1).

## Estrutura

```
dados/emotion/   os CSVs da base (treino 16.000 / validação 2.000 / teste 2.000)
src/             módulos: mlp (a rede), backend (CPU/GPU), metricas, keras_mlp, dados, config, graficos
figuras/         figuras em tema claro (relatório)
figuras/escuro/  as mesmas figuras em tema escuro (slides)
resultados/      tabelas (csv) e números (json) de cada script
```

| arquivo | o que faz |
|---|---|
| `src/mlp.py` | **a MLP do zero** — a teoria completa está no cabeçalho do arquivo |
| `src/backend.py` | onde as contas rodam: NumPy (CPU) ou PyTorch (GPU), as mesmas equações |
| `src/metricas.py` | precisão, revocação, F1 micro/macro, curva ROC/AUROC, intervalo de confiança, baseline estratificado (com a teoria) |
| `src/keras_mlp.py` | a mesma rede em Keras, cópia de pesos, gradiente por autograd |
| `src/dados.py` | carga da base, TF-IDF ajustado dentro de cada fold |
| `src/config.py` | todos os hiperparâmetros e a grade (e o porquê de cada escolha) |
| `src/graficos.py` | estilo das figuras (claro e escuro) |
| `01_estudo_base.py` | desbalanceamento, tamanho dos tweets, palavras por emoção |
| `02_derivadas_e_keras.py` | diferenças finitas, gradiente do Keras no mesmo lote, treino lado a lado |
| `03_validacao_cruzada.py` | ReLU/Sigmoid × com/sem L2 × 3 tamanhos, 5-fold estratificado, IC 95%, todas as métricas e o baseline estratificado |
| `04_curvas_aprendizado.py` | curvas por época e por tamanho do treino; diagnóstico de viés/variância |
| `05_pesos_e_saturacao.py` | variação acumulada dos pesos (ReLU × Sigmoid), saturação, controle com Adam |
| `06_modelo_final.py` | teste: precisão, revocação, F1 micro/macro, curva ROC/AUROC, matriz de confusão, referências |
| `07_skipgram_extra.py` | Skip-Gram com a própria `MLP` (1 camada oculta linear) no lugar do TF-IDF |
| `rodar_tudo.py` | roda os scripts em ordem e registra os tempos |

## O que o enunciado pede e onde está

| pedido | onde |
|---|---|
| MLP com duas camadas, regra da cadeia (pesos e bias) × Keras | `src/mlp.py`, script 02 |
| ReLU/Sigmoid, L2/sem L2, neurônios por camada | grade do script 03 |
| 5-fold cross-validation + intervalo de confiança dos 5 resultados | script 03 (`03_cv_resultados.csv`, colunas `_ic`) |
| overfitting, underfitting, viés e variância pelas curvas de aprendizado | script 04 |
| mapa de calor da variação acumulada dos pesos, ReLU × Sigmoid; saturação | script 05 |
| curva AUROC, precisão, revocação, micro e macro-F1 | scripts 03 (por configuração) e 06 (teste) |
| Random Stratified Baseline | scripts 03 (`03_baseline_estratificado.csv`) e 06 (`06_metricas_resumo.csv`) |
| Skip-Gram com a própria implementação no lugar do TF-IDF | script 07 |

## Decisões registradas

- **SGD com momentum, η = 0,05.** No SGD o passo é proporcional ao gradiente, então
  o gradiente que some aparece nos mapas de calor; o Adam esconderia isso. Com
  η = 0,3 a ReLU diverge; com 0,1 o custo tem picos.
- **λ = 1e-4.** Com 1e-3 o treino ficou instável.
- **F1 macro como critério**, na última época (sem parada antecipada), porque a
  base é desbalanceada. Acurácia (= F1 micro), precisão, revocação e AUROC são
  reportadas junto.
- **Intervalo de confiança** t de Student com 4 graus de liberdade sobre as 5 notas.
  Como os folds compartilham treino, ele é um pouco otimista (Nadeau e Bengio, 2003).
- **TF-IDF ajustado dentro de cada fold**, para nada da validação vazar.
- **Script 05 sem L2**: com L2, todo peso encolhe a cada passo e a "variação" da
  Sigmoid seria quase só esse encolhimento.
- **Skip-Gram**: "representação concatenada" tem mais de uma leitura; o script testa
  três (concatenação das embeddings em ordem, [média ; máximo] e [TF-IDF ; média]).
- **GPU via PyTorch, sem autograd**: as mesmas equações do NumPy, executadas na placa.
  Pesos iniciais e ordem dos lotes são sorteados na CPU, então a rede começa igual
  nos dois dispositivos.
