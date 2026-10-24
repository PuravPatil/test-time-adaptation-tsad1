"""Runs one experiment config: every dataset x method (x override arm) x seed,
evaluated under every threshold.

Each detector is trained and scored ONCE per dataset x method x arm x seed;
all thresholds then read the same scores (a threshold only turns scores into
predictions, it never changes them). For each threshold it saves raw scores,
predictions, metrics, the resolved config and the git commit, so metrics can be
recomputed later without rerunning.

Experiment ``overrides`` set method config keys (e.g. ``normalization``,
``backbone``), per method name or for ``all`` methods:

    overrides:
      all:  {backbone: default}
      m2n2: {normalization: [ema, revin]}   # a list of several values = a sweep

Each sweep value is one arm, saved under ``<method>__<key>-<value>/``. A
method-specific entry wins over ``all``. ``method_overrides`` is the old name
and is still accepted. A key written ``hparams.<name>`` (e.g. ``hparams.adapt``)
overrides that hyperparameter, above the per-machine table in the method config;
its folder label drops the ``hparams.`` prefix.
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


HP = "hparams."


def _arms(overrides: dict, method_name: str) -> list[tuple[str, dict]]:
    """Expand overrides into (output name, settings) arms for one method."""
    merged = {**(overrides.get("all") or {}), **(overrides.get(method_name) or {})}
    keys = list(merged)
    grids = [_as_list(merged[k]) for k in keys]
    arms = []
    for combo in product(*grids):
        label = "__".join(f"{k.removeprefix(HP)}-{v}" for k, v, g in zip(keys, combo, grids) if len(g) > 1)
        arms.append((f"{method_name}__{label}" if label else method_name, dict(zip(keys, combo))))
    return arms


def build_run_config(method_cfg: dict, dataset: str, seed: int, arm_settings: dict | None = None) -> dict:
    """The config one detector sees: the method config plus dataset, seed and arm
    settings. ``hparams.<name>`` keys become ``hparam_overrides``."""
    arm_settings = arm_settings or {}
    # The adapter picks its per-dataset hyperparameters from method_cfg["hparams"][dataset].
    # The seed also drives the pre-training shuffle inside the adapter.
    run_cfg = {**method_cfg, "dataset": dataset, "seed": seed,
               **{k: v for k, v in arm_settings.items() if not k.startswith(HP)}}
    hp_over = {k.removeprefix(HP): v for k, v in arm_settings.items() if k.startswith(HP)}
    if hp_over:
        run_cfg["hparam_overrides"] = {**(run_cfg.get("hparam_overrides") or {}), **hp_over}
    return run_cfg


def fit_and_score(data, run_cfg: dict, seed: int, on_stage=None):
    """Train one detector on data.train and stream data.test through it.

    Returns (test scores, reference train scores, detector stats, ms per test step).
    ``on_stage("training" | "scoring")`` is called before each phase, if given.
    """
    set_seed(seed)
    detector = _load_class(run_cfg["adapter"])(run_cfg)
    if on_stage:
        on_stage("training")
    detector.fit(data.train)
    # Reference scores come from fit(), before the test stream, so nothing
    # from the test set leaks into any threshold calibrated on them.
    reference = detector.train_scores()
    if on_stage:
        on_stage("scoring")
    timing: dict = {}
    with timer(timing):
        scores = detector.score(data.test)
    return scores, reference, getattr(detector, "stats", {}), ms_per_step(timing["seconds"], len(data.test))


def apply_threshold(name: str, config: dict, reference, scores, detector_tau: float | None = None) -> np.ndarray:
    """0/1 predictions from scores, with the threshold calibrated on the reference scores.

    ``quantile: detector`` in the config uses the detector's own tau instead (M2N2
    reports it as stats["tau"]: the one threshold it also adapts with, as in the paper)."""
    config = dict(config or {})
    if config.get("quantile") == "detector":
        if detector_tau is None:
            raise ValueError("threshold 'quantile: detector' needs a detector that reports stats['tau']")
        config.pop("quantile")
        config["tau"] = detector_tau
    threshold = THRESHOLDS[name](config)
    if reference is not None:
        threshold.fit(reference)
    return threshold.predict(scores)


def run_experiment(exp_path: str, config_dir: str = "configs") -> None:
    base = load_yaml(Path(config_dir) / "base.yaml")
    exp = load_yaml(exp_path)
    runs_root = Path(base["paths"]["runs"]) / exp["name"]
    commit = git_commit()
    thresholds = _as_list(exp.get("threshold", "fixed"))
    overrides = exp.get("overrides", exp.get("method_overrides")) or {}

    for ds_name, method_name, seed in product(
        exp["datasets"], exp["methods"], exp.get("seeds", base["seeds"])
    ):
        method_cfg = load_yaml(Path(config_dir) / "methods" / f"{method_name}.yaml")
        data = load_dataset(ds_name, base["paths"]["processed"])

        for arm_name, arm_settings in _arms(overrides, method_name):
            run_cfg = build_run_config(method_cfg, ds_name, seed, arm_settings)
            scores, reference, stats, step_ms = fit_and_score(data, run_cfg, seed)

            for thr_name in thresholds:
                preds = apply_threshold(thr_name, exp.get("threshold_config", {}), reference, scores,
                                        stats.get("tau"))

                metrics = evaluate(data.test_labels, scores, preds, base["metrics"]["pa_k_values"])
                metrics["ms_per_step"] = step_ms

                out = runs_root / ds_name / arm_name / thr_name / f"seed{seed}"
                out.mkdir(parents=True, exist_ok=True)
                np.save(out / "scores.npy", scores)
                np.save(out / "preds.npy", preds)
                save_json(out / "metrics.json", metrics)
                save_json(out / "run.json", {
                    "experiment": exp, "method": method_cfg, "overrides": arm_settings,
                    "dataset": ds_name, "threshold": thr_name, "seed": seed, "commit": commit,
                    "detector_stats": stats,
                })
                print(f"{ds_name:>10} {arm_name:>6} {thr_name:>9} seed{seed}  "
                      f"F1={metrics['f1']:.3f}  AUPRC={metrics['auprc']:.3f}  AUROC={metrics['auroc']:.3f}")
