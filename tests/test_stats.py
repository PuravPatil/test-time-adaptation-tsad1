import numpy as np
import pandas as pd
import pytest
from scipy import stats

from audit.evaluation.stats import across_datasets, mean_ci, paired_by_seed


def test_mean_ci_matches_t_interval():
    x = np.array([0.1, 0.3, 0.2, 0.4])
    m, lo, hi = mean_ci(x)
    half = stats.t.ppf(0.975, 3) * x.std(ddof=1) / 2
    assert m == pytest.approx(0.25) and lo == pytest.approx(0.25 - half) and hi == pytest.approx(0.25 + half)


def test_mean_ci_single_value_has_no_interval():
    m, lo, hi = mean_ci([0.5])
    assert m == 0.5 and np.isnan(lo) and np.isnan(hi)


def _runs():
    rows = []
    # d1: a beats b by 0.1 on every seed; d2: a loses by 0.05; seed 2 of b missing on d2
    for seed in range(3):
        rows += [{"dataset": "d1", "method": "a", "seed": seed, "auroc": 0.8 + 0.01 * seed},
                 {"dataset": "d1", "method": "b", "seed": seed, "auroc": 0.7 + 0.01 * seed},
                 {"dataset": "d2", "method": "a", "seed": seed, "auroc": 0.6}]
        if seed < 2:
            rows.append({"dataset": "d2", "method": "b", "seed": seed, "auroc": 0.65})
    return pd.DataFrame(rows)


def test_paired_by_seed_pairs_only_shared_seeds():
    p = paired_by_seed(_runs(), "a", "b", ["auroc"]).set_index("dataset")
    assert p.loc["d1", "diff"] == pytest.approx(0.1)
    assert p.loc["d1", "n_seeds"] == 3 and p.loc["d1", "a_wins"] == 3
    assert p.loc["d2", "n_seeds"] == 2 and p.loc["d2", "diff"] == pytest.approx(-0.05)
    assert p.loc["d2", "a_wins"] == 0


def test_constant_difference_gives_zero_width_interval():
    p = paired_by_seed(_runs(), "a", "b", ["auroc"]).set_index("dataset")
    assert p.loc["d1", "ci_low"] == pytest.approx(0.1) and p.loc["d1", "ci_high"] == pytest.approx(0.1)


def test_across_datasets_counts_and_wilcoxon():
    s = across_datasets(paired_by_seed(_runs(), "a", "b", ["auroc"])).iloc[0]
    assert s.n_datasets == 2 and s.datasets_up == 1 and s.datasets_down == 1
    assert s.mean_diff == pytest.approx(0.025)
    assert 0 <= s.wilcoxon_p <= 1


def test_wilcoxon_six_datasets_all_up_is_smallest_p():
    paired = pd.DataFrame({"metric": ["m"] * 6, "diff": [0.1, 0.2, 0.05, 0.3, 0.15, 0.25],
                           "ci_low": [0.0] * 6, "ci_high": [1.0] * 6})
    assert across_datasets(paired).iloc[0].wilcoxon_p == pytest.approx(2 / 64)
