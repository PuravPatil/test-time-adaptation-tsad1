"""Paired comparison of two methods over seeds and machines, with intervals.

    python scripts/compare_methods.py                                   # m2n2 vs pretrained, smd_reproduce
    python scripts/compare_methods.py --experiment smd_reproduce --a m2n2 --b pretrained --threshold fixed

Writes results/tables/<experiment>_<a>_vs_<b>.csv (per machine) and
..._summary.csv (across machines). See src/audit/evaluation/stats.py for what
the intervals and the p-value do and do not cover.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from audit.evaluation.stats import across_datasets, paired_by_seed


def load_runs(root: Path, experiment: str, threshold: str) -> pd.DataFrame:
    rows = []
    for path in (root / experiment).glob(f"*/*/{threshold}/seed*/metrics.json"):
        dataset, method, _, seed = path.parts[-5:-1]
        rows.append({"dataset": dataset, "method": method, "seed": int(seed.removeprefix("seed")),
                     **json.loads(path.read_text())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="smd_reproduce")
    ap.add_argument("--a", default="m2n2", help="method under test")
    ap.add_argument("--b", default="pretrained", help="reference method")
    ap.add_argument("--threshold", default="fixed")
    ap.add_argument("--metrics", nargs="+", default=["auroc", "auprc", "f1", "pa_k_50"])
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--tables", default="results/tables")
    args = ap.parse_args()

    runs = load_runs(Path(args.runs), args.experiment, args.threshold)
    if runs.empty:
        raise SystemExit(f"No runs under {args.runs}/{args.experiment}/*/*/{args.threshold}/")
    paired = paired_by_seed(runs, args.a, args.b, args.metrics, args.level)
    if paired.empty:
        raise SystemExit(f"No dataset has runs of both '{args.a}' and '{args.b}'.")
    summary = across_datasets(paired)

    pd.set_option("display.width", 180)
    print(f"{args.a} - {args.b}, {args.experiment}, threshold={args.threshold}, "
          f"{int(args.level * 100)}% t-interval over seeds\n")
    print(paired.round(4).to_string(index=False))
    print("\nAcross machines (Wilcoxon signed-rank on per-machine mean differences):\n")
    print(summary.round(4).to_string(index=False))

    out = Path(args.tables)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.experiment}_{args.a}_vs_{args.b}"
    paired.to_csv(out / f"{stem}.csv", index=False)
    summary.to_csv(out / f"{stem}_summary.csv", index=False)
    print(f"\nwrote {out / (stem + '.csv')} and {out / (stem + '_summary.csv')}")


if __name__ == "__main__":
    main()
