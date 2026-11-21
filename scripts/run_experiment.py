"""CLI: python scripts/run_experiment.py configs/experiments/e1_architecture.yaml [--force] [--quick]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly_ae.runner import run_experiment  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("configs", nargs="+", type=Path, help="YAML-файли експериментів")
    p.add_argument("--force", action="store_true", help="перезапустити навіть готові прогони")
    p.add_argument("--quick", action="store_true", help="2 епохи — швидка перевірка коду")
    p.add_argument("--shard", default="0/1", help="i/n — частина прогонів для паралельного запуску")
    a = p.parse_args()
    i, n = map(int, a.shard.split("/"))
    for c in a.configs:
        df = run_experiment(c, force=a.force, quick=a.quick, shard=(i, n))
        if df.empty:
            continue
        cols = [col for col in ["run", "seed", "val_roc_auc", "test_roc_auc", "test_pr_auc", "test_f1"] if col in df]
        print(df[cols].to_string(index=False))
