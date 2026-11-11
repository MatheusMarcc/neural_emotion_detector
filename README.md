# Projeto 3 — Redes Neurais

MLP com duas camadas ocultas **implementada do zero em NumPy** (forward, custo,
regra da cadeia para pesos e bias, SGD com momentum) e comparada com a mesma rede
em **Keras**, numa base de texto difícil: a **Emotion** (20.000 tweets, 6 emoções,
desbalanceada 9,4 : 1).

O Keras só aparece no script 02, para conferir as derivadas. O scikit-learn entra
apenas no TF-IDF, na divisão em folds, nas métricas e na regressão logística de
referência.

## Como rodar

Num terminal aberto nesta pasta:

```
python -m pip install -r requirements.txt
python rodar_tudo.py
```

O `rodar_tudo.py` executa os 7 scripts em ordem e no fim mostra quanto cada um
levou (também salvo em `resultados/tempos_execucao.txt`). Também dá para rodar um
por um, nesta ordem:

```
python 01_estudo_base.py         # estudo da base                         (segundos)
python 02_derivadas_e_keras.py   # derivadas x diferenças finitas x Keras (~4 min)
python 03_validacao_cruzada.py   # grade de 12 configurações, 3-fold      (~8 min)
python 04_curvas_aprendizado.py  # curvas de aprendizado, viés/variância  (~12 min)
python 05_pesos_e_saturacao.py   # mapas de calor e saturação             (~3 min)
python 06_modelo_final.py        # melhor configuração no teste           (~1 min)
python 07_skipgram_extra.py      # ponto extra: Skip-Gram                 (~15 min)
```

Os tempos foram medidos numa máquina de 2 núcleos (cerca de 45 min no total);
num i7 deve ser menos. Tudo roda na CPU — a rede é pequena demais para a GPU
compensar. Os scripts 04 a 07 dependem do `resultados/melhor_config.json` que o
03 grava. Para só testar se tudo roda, use menos épocas (PowerShell):
`$env:EPOCAS=3; python rodar_tudo.py` — e depois `Remove-Item Env:EPOCAS` antes
da rodada oficial, que usa 50.

Os números podem variar um pouco de uma máquina para outra (na casa da 3ª casa
decimal): a ReLU é sensível a diferenças de arredondamento (ver relatório, 3.1).

## Estrutura

```
dados/emotion/   os CSVs da base (treino 16.000 / validação 2.000 / teste 2.000)
src/             módulos: mlp (a rede), keras_mlp, dados, config, graficos
figuras/         figuras em tema claro (relatório)
figuras/escuro/  as mesmas figuras em tema escuro (slides)
resultados/      tabelas (csv) e números (json) de cada script
```

| arquivo | o que faz |
|---|---|
| `src/mlp.py` | **a MLP do zero** — a teoria completa está no cabeçalho do arquivo |
| `src/keras_mlp.py` | a mesma rede em Keras, cópia de pesos, gradiente por autograd |
| `src/dados.py` | carga da base, TF-IDF ajustado dentro de cada fold |
| `src/config.py` | todos os hiperparâmetros e a grade (e o porquê de cada escolha) |
| `src/graficos.py` | estilo das figuras (claro e escuro) |
| `01_estudo_base.py` | desbalanceamento, tamanho dos tweets, palavras por emoção |
| `02_derivadas_e_keras.py` | diferenças finitas, gradiente do Keras no mesmo lote, treino lado a lado |
| `03_validacao_cruzada.py` | ReLU/Sigmoid × com/sem L2 × 3 tamanhos, 3-fold estratificado |
| `04_curvas_aprendizado.py` | curvas por época e por tamanho do treino; diagnóstico de viés/variância |
| `05_pesos_e_saturacao.py` | variação acumulada dos pesos (ReLU × Sigmoid), saturação, controle com Adam |
| `06_modelo_final.py` | teste, métricas por classe, matriz de confusão, regressão logística |
| `07_skipgram_extra.py` | Skip-Gram com a própria `MLP` (1 camada oculta linear) no lugar do TF-IDF |

## Resultados principais

| | acurácia | F1 macro |
|---|---|---|
| Validação cruzada — ReLU 256-128, L2 = 1e-4 (escolhida) | 0,858 ± 0,006 | 0,821 ± 0,007 |
| Teste — MLP do zero | 0,880 | 0,830 |
| Teste — regressão logística no mesmo TF-IDF | 0,883 | 0,831 |
| Teste — sempre a classe majoritária | 0,348 | 0,086 |

- As derivadas batem com as do Keras até ~1e-7 (limite do float32); com os mesmos
  pesos iniciais e lotes, as curvas das duas implementações se sobrepõem.
- ReLU: aprende em poucas épocas e decora o treino (variância); a L2 corta o custo
  de validação em ~30% sem mudar o F1.
- Sigmoid: underfitting — a segunda camada satura, o gradiente que chega a W1 cai a
  1/65 do da saída e os pesos quase não mudam. Com Adam (controle) ela aprende.
- Skip-Gram (extra), treinado com a própria `MLP`: não melhora a acurácia. Só com
  embeddings ela cai para 0,44–0,49 no teste; [TF-IDF ; média] empata com o TF-IDF
  (0,880 × 0,880).

Relatório (Docs) e apresentação (Slides, 13 slides com roteiro nas notas) estão no
claude.ai, com os links na conversa.

## Decisões registradas

- **SGD com momentum, η = 0,05.** No SGD o passo é proporcional ao gradiente, então
  o gradiente que some aparece nos mapas de calor; o Adam esconderia isso. Com
  η = 0,3 a ReLU diverge; com 0,1 o custo tem picos.
- **λ = 1e-4.** Com 1e-3 o treino ficou instável.
- **F1 macro como critério**, na última época (sem parada antecipada), porque a
  base é desbalanceada.
- **TF-IDF ajustado dentro de cada fold**, para nada da validação vazar.
- **Script 05 sem L2**: com L2, todo peso encolhe a cada passo e a "variação" da
  Sigmoid seria quase só esse encolhimento.
- **Skip-Gram**: "representação concatenada" tem mais de uma leitura; o script testa
  três (concatenação das embeddings em ordem, [média ; máximo] e [TF-IDF ; média]).
