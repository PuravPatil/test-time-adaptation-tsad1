"""Print our results next to the numbers published in the papers.

    python scripts/compare_paper.py                       # experiment smd_reproduce
    python scripts/compare_paper.py --experiment smd_reproduce

Averages over seeds. Paper numbers come from results/paper_reference.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_runs(root: Path, experiment: str) -> pd.DataFrame:
    rows = []
    for path in (root / experiment).glob("*/*/*/seed*/metrics.json"):
        dataset, method, threshold, seed = path.parts[-5:-1]
        rows.append({"dataset": dataset, "method": method, "threshold": threshold, "seed": seed,
                     **json.loads(path.read_text())})
    return pd.DataFrame(rows)


def fmt(vals) -> str:
    vals = np.asarray(vals, dtype=float)
    return f"{vals.mean():.3f}±{vals.std():.3f}" if len(vals) > 1 else f"{vals.mean():.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="smd_reproduce")
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--reference", default="results/paper_reference.json")
    args = ap.parse_args()

    df = load_runs(Path(args.runs), args.experiment)
    if df.empty:
        raise SystemExit(f"No runs under {args.runs}/{args.experiment}. Run scripts/run.py first.")
    ref = json.loads(Path(args.reference).read_text())
    df = df[df.threshold == "fixed"]

    lines = []
    for (dataset, method), g in df.groupby(["dataset", "method"]):
        for source in ("candi_paper", "m2n2_paper"):
            paper = ref[source].get(dataset, {}).get(method)
            if not paper:
                continue
            for metric, value in paper.items():
                if metric in g:
                    lines.append({"dataset": dataset, "method": method, "metric": metric,
                                  "ours": fmt(g[metric]), "paper": f"{value:.3f}", "source": source,
                                  "diff": f"{g[metric].mean() - value:+.3f}"})
    if not lines:
        raise SystemExit("Nothing to compare (no dataset/method pair in the reference file).")
    out = pd.DataFrame(lines)
    pd.set_option("display.width", 160)
    print(out.to_string(index=False))
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    out.to_csv(f"results/tables/{args.experiment}_vs_paper.csv", index=False)

    print("\nAll our metrics (mean over seeds):")
    cols = ["auroc", "auprc", "f1", "f1_best", "f1_pa"]
    print(df.groupby(["dataset", "method"])[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
