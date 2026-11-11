"""
Roda o pipeline em ordem e mostra quanto tempo cada script levou.

Uso (no terminal, dentro da pasta do projeto):
    python rodar_tudo.py            -> roda o que precisa (ver abaixo)
    python rodar_tudo.py --tudo     -> roda os 7 scripts, do zero
    python rodar_tudo.py 05 06      -> roda só alguns, pelo número

O que "precisa" rodar: 03, 04, 06 e 07 sempre. Os scripts 01 (estudo da base)
e 02 (derivadas × Keras) não dependem da validação cruzada: se os resultados
deles já existem, são pulados. O 05 (mapas de calor) só depende da ARQUITETURA
da melhor configuração: se ela for a mesma da última vez que o 05 rodou, ele é
pulado também.

GPU: a MLP usa a placa de vídeo automaticamente se houver PyTorch com CUDA
(ver README e src/backend.py). A primeira linha impressa diz onde vai rodar.

Se um script der erro, o pipeline para ali (os seguintes dependem dele).
Os tempos ficam salvos em resultados/tempos_execucao.txt.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = ["01_estudo_base", "02_derivadas_e_keras", "03_validacao_cruzada",
           "04_curvas_aprendizado", "05_pesos_e_saturacao", "06_modelo_final",
           "07_skipgram_extra"]


def motivo_para_pular(s, resultados):
    """Devolve o motivo para pular o script s (ou None se ele precisa rodar)."""
    if s == "01_estudo_base" and (resultados / "01_estudo_base.json").exists():
        return "resultados já existem (não dependem da validação cruzada)"
    if s == "02_derivadas_e_keras" and (resultados / "02_derivadas.json").exists():
        return "resultados já existem (não dependem da validação cruzada)"
    if s == "05_pesos_e_saturacao":
        try:
            with open(resultados / "melhor_config.json", encoding="utf-8") as f:
                nova = list(json.load(f)["camadas"])
            with open(resultados / "05_pesos_saturacao.json", encoding="utf-8") as f:
                antiga = list(json.load(f)["camadas"])
        except (OSError, KeyError, ValueError):
            return None
        if nova == antiga:
            return f"a melhor arquitetura continua {nova[0]}-{nova[1]}: resultados atuais valem"
    return None


def main():
    args = sys.argv[1:]
    tudo = "--tudo" in args
    pedidos = [a for a in args if a != "--tudo"]
    escolhidos = [s for s in SCRIPTS if not pedidos or s[:2] in pedidos]
    pasta = Path(__file__).resolve().parent
    resultados = pasta / "resultados"
    resultados.mkdir(exist_ok=True)
    # UTF-8 na saída dos scripts: no Windows, sem isso, letras como η e Σ podem
    # dar erro quando a saída é redirecionada para um arquivo
    ambiente = dict(os.environ, PYTHONIOENCODING="utf-8")

    sys.path.insert(0, str(pasta))
    from src import backend
    print("Onde a MLP vai rodar:", backend.descrever(), flush=True)

    resumo = []
    for s in escolhidos:
        motivo = None if (tudo or pedidos) else motivo_para_pular(s, resultados)
        if motivo:
            print(f"\n===== {s}: pulado — {motivo} =====", flush=True)
            resumo.append((s, "pulado", 0.0))
            continue
        print(f"\n===== {s} =====", flush=True)
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, f"{s}.py"], cwd=pasta, env=ambiente)
        dt = time.perf_counter() - t0
        resumo.append((s, "ok" if r.returncode == 0 else "erro", dt))
        if r.returncode != 0:
            print(f"\n{s} terminou com erro (código {r.returncode}); parando aqui.")
            break

    print("\nResumo:")
    for s, status, dt in resumo:
        print(f"  {s:24s} {status:6s}  {dt / 60:5.1f} min")
    with open(resultados / "tempos_execucao.txt", "w", encoding="utf-8") as f:
        f.write(f"# {backend.descrever()}\n")
        for s, status, dt in resumo:
            f.write(f"{s}\t{status}\t{dt:.0f} s\n")


if __name__ == "__main__":
    main()
