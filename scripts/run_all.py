"""Паралельний запуск експериментів: кожен YAML ділиться на N шардів (окремі процеси).

python scripts/run_all.py configs/experiments/e1_*.yaml ... --workers 4
Готові прогони (з metrics.json) пропускаються, тож запуск можна перервати й продовжити.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("configs", nargs="+")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    (ROOT / "results" / "_logs").mkdir(parents=True, exist_ok=True)
    for cfg in a.configs:
        name = Path(cfg).stem
        procs = []
        for i in range(a.workers):
            log = open(ROOT / "results" / "_logs" / f"{name}_shard{i}.log", "w", encoding="utf-8")
            procs.append(subprocess.Popen(
                [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), cfg, "--shard", f"{i}/{a.workers}"],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONIOENCODING": "utf-8"}))
        codes = [pr.wait() for pr in procs]
        print(f"{name}: exit codes {codes}", flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "collect_results.py")], cwd=ROOT)
