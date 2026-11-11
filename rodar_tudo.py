"""
Roda o pipeline inteiro, em ordem, e mostra quanto tempo cada script levou.

Uso (no terminal, dentro da pasta do projeto):
    python rodar_tudo.py            -> roda os 7 scripts
    python rodar_tudo.py 05 06      -> roda só alguns, pelo número

Se um script der erro, o pipeline para ali (os seguintes dependem dele).
Os tempos ficam salvos em resultados/tempos_execucao.txt.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = ["01_estudo_base", "02_derivadas_e_keras", "03_validacao_cruzada",
           "04_curvas_aprendizado", "05_pesos_e_saturacao", "06_modelo_final",
           "07_skipgram_extra"]


def main():
    pedidos = sys.argv[1:]
    escolhidos = [s for s in SCRIPTS if not pedidos or s[:2] in pedidos]
    pasta = Path(__file__).resolve().parent
    (pasta / "resultados").mkdir(exist_ok=True)
    # UTF-8 na saída dos scripts: no Windows, sem isso, letras como η e Σ podem
    # dar erro quando a saída é redirecionada para um arquivo
    ambiente = dict(os.environ, PYTHONIOENCODING="utf-8")

    resumo = []
    for s in escolhidos:
        print(f"\n===== {s} =====", flush=True)
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, f"{s}.py"], cwd=pasta, env=ambiente)
        dt = time.perf_counter() - t0
        resumo.append((s, r.returncode, dt))
        if r.returncode != 0:
            print(f"\n{s} terminou com erro (código {r.returncode}); parando aqui.")
            break

    print("\nResumo:")
    for s, codigo, dt in resumo:
        print(f"  {s:24s} {'ok  ' if codigo == 0 else 'ERRO'}  {dt / 60:5.1f} min")
    with open(pasta / "resultados" / "tempos_execucao.txt", "w", encoding="utf-8") as f:
        for s, codigo, dt in resumo:
            f.write(f"{s}\t{'ok' if codigo == 0 else 'erro'}\t{dt:.0f} s\n")


if __name__ == "__main__":
    main()
