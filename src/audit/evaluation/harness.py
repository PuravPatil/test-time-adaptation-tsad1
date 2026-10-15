"""Runs one experiment config: every dataset x method x threshold x seed.

For each run it saves raw scores, predictions, metrics, the resolved config
and the git commit, so metrics can be recomputed later without rerunning.
"""
from __future__ import annotations

import importlib
from itertools import product
from pathlib import Path

import numpy as np

from audit.data import load_dataset
from audit.evaluation.latency import ms_per_step, timer
from audit.evaluation.metrics import evaluate
from audit.thresholds import THRESHOLDS
from audit.utils.io import git_commit, load_yaml, save_json
from audit.utils.seed import set_seed


def _as_list(x):
    return x if isinstance(x, list) else [x]


def _load_class(dotted: str):
    module, _, name = dotted.rpartition(".")
    return getattr(importlib.import_module(module), name)


def run_experiment(exp_path: str, config_dir: str = "configs") -> None:
    base = load_yaml(Path(config_dir) / "base.yaml")
    exp = load_yaml(exp_path)
    runs_root = Path(base["paths"]["runs"]) / exp["name"]
    commit = git_commit()

    for ds_name, method_name, thr_name, seed in product(
        exp["datasets"], exp["methods"], _as_list(exp.get("threshold", "fixed")),
        exp.get("seeds", base["seeds"])
    ):
        method_cfg = load_yaml(Path(config_dir) / "methods" / f"{method_name}.yaml")
        set_seed(seed)
        data = load_dataset(ds_name, base["paths"]["processed"])

        # The adapter picks its per-dataset hyperparameters from method_cfg["hparams"][ds_name].
        run_cfg = {**method_cfg, "dataset": ds_name, **exp.get("method_overrides", {}).get(method_name, {})}
        detector = _load_class(method_cfg["adapter"])(run_cfg)
        detector.fit(data.train)

        # Calibrate the threshold on training scores BEFORE the test stream,
        # so nothing from the test set leaks into it.
        threshold = THRESHOLDS[thr_name](exp.get("threshold_config", {}))
        reference = detector.train_scores()
        if reference is not None:
            threshold.fit(reference)

        timing: dict = {}
        with timer(timing):
            scores = detector.score(data.test)
        preds = threshold.predict(scores)

        metrics = evaluate(data.test_labels, scores, preds, base["metrics"]["pa_k_values"])
        metrics["ms_per_step"] = ms_per_step(timing["seconds"], len(data.test))

        out = runs_root / ds_name / method_name / thr_name / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "scores.npy", scores)
        np.save(out / "preds.npy", preds)
        save_json(out / "metrics.json", metrics)
        save_json(out / "run.json", {
            "experiment": exp, "method": method_cfg, "dataset": ds_name,
            "threshold": thr_name, "seed": seed, "commit": commit,
            "detector_stats": getattr(detector, "stats", {}),
        })
        print(f"{ds_name:>10} {method_name:>6} {thr_name:>9} seed{seed}  "
              f"F1={metrics['f1']:.3f}  AUPRC={metrics['auprc']:.3f}  AUROC={metrics['auroc']:.3f}")
