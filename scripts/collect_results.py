"""Збирає metrics.json усіх прогонів у results/experiments.csv (таблиця експериментів)."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    rows = []
    for f in sorted((ROOT / "results").glob("*/*/metrics.json")):
        rows.append(json.loads(f.read_text(encoding="utf-8"))["summary"])
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "experiments.csv", index=False)
    for exp, g in df.groupby("experiment"):
        g.to_csv(ROOT / "results" / exp / "runs.csv", index=False)
    print(f"Зібрано {len(df)} прогонів -> results/experiments.csv")
