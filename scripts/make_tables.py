"""Collect results/runs/**/metrics.json into comparison tables.

Writes one CSV per experiment to results/tables/, with mean and std over seeds.
"""
import json
from pathlib import Path

import pandas as pd

RUNS = Path("results/runs")
TABLES = Path("results/tables")


def main() -> None:
    rows = []
    for path in RUNS.glob("*/*/*/*/seed*/metrics.json"):
        experiment, dataset, method, threshold, seed = path.parts[-6:-1]
        rows.append({"experiment": experiment, "dataset": dataset, "method": method,
                     "threshold": threshold, "seed": seed, **json.loads(path.read_text())})
    if not rows:
        print("No runs found under results/runs/")
        return

    df = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    keys = ["dataset", "method", "threshold"]
    for experiment, part in df.groupby("experiment"):
        metrics = part.drop(columns=["experiment", "seed"]).groupby(keys)
        table = metrics.agg(["mean", "std"]).round(4)
        table.to_csv(TABLES / f"{experiment}.csv")
        print(f"wrote {TABLES / f'{experiment}.csv'}")


if __name__ == "__main__":
    main()
