"""Read-only index over results/runs/<experiment>/<dataset>/<arm>/<threshold>/seed<k>/."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from audit.data import load_dataset

from .settings import Settings
from .util import clean, series_payload


class RunNotFound(LookupError):
    pass


class ResultsIndex:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._labels = lru_cache(maxsize=16)(self._load_labels)

    def _run_dir(self, experiment: str, dataset: str, arm: str, threshold: str, seed: int) -> Path:
        d = self.settings.runs_dir / experiment / dataset / arm / threshold / f"seed{seed}"
        if not (d / "metrics.json").exists():
            raise RunNotFound(f"no run {experiment}/{dataset}/{arm}/{threshold}/seed{seed}")
        return d

    def experiments(self) -> list[dict]:
        root = self.settings.runs_dir
        if not root.exists():
            return []
        out = []
        for exp in sorted(p for p in root.iterdir() if p.is_dir()):
            datasets, arms, thresholds, seeds, n = set(), set(), set(), set(), 0
            for m in exp.glob("*/*/*/seed*/metrics.json"):
                ds, arm, thr, sd = m.parts[-5:-1]
                datasets.add(ds), arms.add(arm), thresholds.add(thr)
                if sd.removeprefix("seed").isdigit():
                    seeds.add(int(sd.removeprefix("seed")))
                n += 1
            if n:
                out.append({"name": exp.name, "n_runs": n, "datasets": sorted(datasets), "arms": sorted(arms),
                            "thresholds": sorted(thresholds), "seeds": sorted(seeds)})
        return out

    def run(self, experiment: str, dataset: str, arm: str, threshold: str, seed: int) -> dict:
        d = self._run_dir(experiment, dataset, arm, threshold, seed)
        metrics = json.loads((d / "metrics.json").read_text())
        run = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else {}
        labels = self._labels(dataset)
        return clean({"experiment": experiment, "dataset": dataset, "arm": arm, "threshold": threshold,
                      "seed": seed, "metrics": metrics, "detector_stats": run.get("detector_stats", {}),
                      "overrides": run.get("overrides", {}), "commit": run.get("commit"),
                      "test_anomaly_rate": float(labels.mean()) if labels is not None else None})

    def series(self, experiment: str, dataset: str, arm: str, threshold: str, seed: int, points: int) -> dict:
        d = self._run_dir(experiment, dataset, arm, threshold, seed)
        scores = np.load(d / "scores.npy", allow_pickle=False)
        preds = np.load(d / "preds.npy", allow_pickle=False)
        labels = self._labels(dataset)
        if labels is not None and len(labels) != len(scores):
            labels = None
        return series_payload(scores, preds, labels, points)

    def _load_labels(self, dataset: str):
        """Test labels from data/processed/, or None if that dataset isn't on this machine."""
        try:
            return load_dataset(dataset, str(self.settings.processed_dir)).test_labels
        except (FileNotFoundError, KeyError, ValueError, OSError):
            return None

    def compare(self, experiment: str, a: str, b: str, threshold: str, metrics: list[str]) -> dict:
        """Paired comparison of arm a vs arm b over seeds and datasets (see audit.evaluation.stats)."""
        import pandas as pd
        from audit.evaluation.stats import across_datasets, paired_by_seed

        rows = []
        for m in (self.settings.runs_dir / experiment).glob(f"*/*/{threshold}/seed*/metrics.json"):
            ds, arm, _, sd = m.parts[-5:-1]
            if arm in (a, b) and sd.removeprefix("seed").isdigit():
                rows.append({"dataset": ds, "method": arm, "seed": int(sd.removeprefix("seed")),
                             **json.loads(m.read_text())})
        if not rows:
            raise RunNotFound(f"no runs of '{a}' or '{b}' in {experiment} with threshold {threshold}")
        df = pd.DataFrame(rows)
        missing = [m for m in metrics if m not in df.columns]
        if missing:
            raise RunNotFound(f"unknown metric(s): {', '.join(missing)}")
        paired = paired_by_seed(df, a, b, metrics)
        if paired.empty:
            raise RunNotFound(f"no dataset has runs of both '{a}' and '{b}'")
        return clean({"experiment": experiment, "a": a, "b": b, "threshold": threshold,
                      "per_dataset": paired.to_dict(orient="records"),
                      "summary": across_datasets(paired).to_dict(orient="records")})

    def samples(self) -> list[dict]:
        """Datasets that can be run as a one-click sample (SMD machines on this machine)."""
        root = self.settings.processed_dir / "smd"
        if not root.exists():
            return []
        out = []
        for d in sorted(root.glob("machine-*")):
            if all((d / f).exists() for f in ("train.npy", "test.npy", "test_labels.npy")):
                out.append({"name": "smd_" + d.name.removeprefix("machine-"), "source": "SMD",
                            "description": f"Server Machine Dataset, {d.name}"})
        return out
