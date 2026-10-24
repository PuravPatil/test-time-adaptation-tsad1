"""Paired comparison of two methods: per seed within a dataset, and across datasets.

Runs of the two methods are paired by dataset and seed. For M2N2 vs the static
baseline the pair also shares the same seeded weight init and pre-training
shuffle (same adapter, same seed), so a per-seed difference isolates test-time
adaptation.

Read the numbers with two limits in mind:
- the per-dataset interval covers training randomness (seeds) only, not
  sampling uncertainty in the test data;
- the across-dataset Wilcoxon test has little power with few datasets: with 6
  machines the smallest possible two-sided p is 2/64 = 0.031.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mean_ci(x, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and two-sided t-interval. With fewer than 2 values the interval is NaN."""
    x = np.asarray(x, dtype=float)
    m = float(x.mean())
    if len(x) < 2:
        return m, float("nan"), float("nan")
    half = stats.t.ppf((1 + level) / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - half, m + half


def paired_by_seed(runs: pd.DataFrame, a: str, b: str, metrics, level: float = 0.95) -> pd.DataFrame:
    """One row per dataset x metric comparing method ``a`` with ``b`` (diff = a - b).

    ``runs`` has one row per run with columns dataset, method, seed and the metrics.
    Only seeds present for both methods are used.
    """
    rows = []
    for dataset, g in runs.groupby("dataset"):
        A = g[g.method == a].set_index("seed")
        B = g[g.method == b].set_index("seed")
        seeds = sorted(set(A.index) & set(B.index))
        if not seeds:
            continue
        for metric in metrics:
            va = A.loc[seeds, metric].to_numpy(dtype=float)
            vb = B.loc[seeds, metric].to_numpy(dtype=float)
            diff, lo, hi = mean_ci(va - vb, level)
            rows.append({"dataset": dataset, "metric": metric, "n_seeds": len(seeds),
                         f"{a}": va.mean(), f"{b}": vb.mean(), "diff": diff,
                         "ci_low": lo, "ci_high": hi, f"{a}_wins": int((va > vb).sum()),
                         "ci_excludes_0": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def across_datasets(paired: pd.DataFrame) -> pd.DataFrame:
    """Per metric: how many datasets favour ``a``, the mean difference, and a
    two-sided Wilcoxon signed-rank p over the per-dataset mean differences."""
    rows = []
    for metric, g in paired.groupby("metric", sort=False):
        d = g["diff"].to_numpy(dtype=float)
        try:
            p = float(stats.wilcoxon(d).pvalue) if len(d) >= 2 and np.any(d != 0) else float("nan")
        except ValueError:
            p = float("nan")
        rows.append({"metric": metric, "n_datasets": len(d), "mean_diff": float(d.mean()),
                     "datasets_up": int((d > 0).sum()), "datasets_down": int((d < 0).sum()),
                     "datasets_ci_up": int(((g.ci_low > 0)).sum()),
                     "datasets_ci_down": int(((g.ci_high < 0)).sum()),
                     "wilcoxon_p": p})
    return pd.DataFrame(rows)
