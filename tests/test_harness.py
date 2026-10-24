"""End-to-end test of the experiment harness with a tiny fake detector (no torch needed)."""
import json

import numpy as np
import yaml

from audit.data import Dataset, register
from audit.evaluation.harness import _arms, run_experiment
from audit.methods import Detector
from audit.methods.base import resolve_hparams
from audit.utils import io

FIT_CALLS: list = []
SEEDS_SEEN: list = []
CONFIGS: list = []


@register("synth")
def _load_synth(processed_dir, entity=None):
    rng = np.random.default_rng(0)
    train = rng.normal(size=(200, 3)).astype(np.float32)
    test = rng.normal(size=(100, 3)).astype(np.float32)
    labels = np.zeros(100, dtype=int)
    test[40:50] += 5.0
    labels[40:50] = 1
    return Dataset("synth", train, test, labels)


class FakeDetector(Detector):
    """Scores each timestep by its mean absolute value; records every fit()."""

    def fit(self, train):
        FIT_CALLS.append(self.config.get("normalization"))
        SEEDS_SEEN.append(self.config.get("seed"))
        CONFIGS.append(self.config)
        self._ref = np.abs(train).mean(axis=1)

    def score(self, test):
        self.stats = {"n_updates": 0}
        return np.abs(test).mean(axis=1)

    def train_scores(self):
        return self._ref


def _write_configs(tmp_path, exp):
    cfg = tmp_path / "configs"
    (cfg / "methods").mkdir(parents=True)
    base = {"seeds": [0], "metrics": {"pa_k_values": [0, 100]},
            "paths": {"processed": str(tmp_path / "data"), "runs": str(tmp_path / "runs")}}
    (cfg / "base.yaml").write_text(yaml.safe_dump(base))
    (cfg / "methods" / "fake.yaml").write_text(
        yaml.safe_dump({"name": "fake", "adapter": f"{__name__}.FakeDetector"}))
    exp_path = tmp_path / "exp.yaml"
    exp_path.write_text(yaml.safe_dump(exp))
    return str(exp_path), str(cfg)


def test_trains_once_per_arm_and_seed_and_writes_every_threshold(tmp_path):
    exp = {"name": "e2e", "datasets": ["synth"], "methods": ["fake"], "seeds": [0, 1],
           "threshold": ["fixed", "streaming"], "threshold_config": {"quantile": 0.95},
           "overrides": {"fake": {"normalization": ["ema", "revin"]}}}
    FIT_CALLS.clear()
    SEEDS_SEEN.clear()
    run_experiment(*_write_configs(tmp_path, exp))

    # 2 arms x 2 seeds fits, NOT x 2 thresholds; each fit sees its run seed
    assert sorted(FIT_CALLS) == ["ema", "ema", "revin", "revin"]
    assert sorted(SEEDS_SEEN) == [0, 0, 1, 1]
    root = tmp_path / "runs" / "e2e" / "synth"
    for norm in ("ema", "revin"):
        for thr in ("fixed", "streaming"):
            for seed in (0, 1):
                out = root / f"fake__normalization-{norm}" / thr / f"seed{seed}"
                for name in ("scores.npy", "preds.npy", "metrics.json", "run.json"):
                    assert (out / name).exists(), out / name
                run = json.loads((out / "run.json").read_text())
                assert run["overrides"] == {"normalization": norm}
                assert run["detector_stats"] == {"n_updates": 0}
                assert run["threshold"] == thr and run["seed"] == seed
    arm = root / "fake__normalization-ema"
    fixed, streaming = (np.load(arm / t / "seed0" / "scores.npy") for t in ("fixed", "streaming"))
    assert np.array_equal(fixed, streaming)


def test_old_method_overrides_key_still_works(tmp_path):
    exp = {"name": "old", "datasets": ["synth"], "methods": ["fake"], "threshold": "fixed",
           "method_overrides": {"fake": {"normalization": "revin"}}}
    FIT_CALLS.clear()
    run_experiment(*_write_configs(tmp_path, exp))
    assert FIT_CALLS == ["revin"]
    assert (tmp_path / "runs" / "old" / "synth" / "fake" / "fixed" / "seed0" / "metrics.json").exists()


def test_arms_expand_sweeps_and_merge_all():
    arms = _arms({"all": {"backbone": ["default"]}, "m2n2": {"normalization": ["ema", "revin"]}}, "m2n2")
    assert arms == [("m2n2__normalization-ema", {"backbone": "default", "normalization": "ema"}),
                    ("m2n2__normalization-revin", {"backbone": "default", "normalization": "revin"})]
    assert _arms({}, "m2n2") == [("m2n2", {})]
    assert _arms({"other": {"normalization": ["ema", "revin"]}}, "m2n2") == [("m2n2", {})]
    assert _arms({"all": {"backbone": "mlp"}, "m2n2": {"backbone": "default"}}, "m2n2") == \
        [("m2n2", {"backbone": "default"})]


def test_hparam_overrides_reach_the_detector_and_name_the_arm(tmp_path):
    exp = {"name": "hp", "datasets": ["synth"], "methods": ["fake"], "threshold": "fixed",
           "overrides": {"fake": {"normalization": "none", "hparams.adapt": [False, True]}}}
    CONFIGS.clear()
    run_experiment(*_write_configs(tmp_path, exp))
    got = sorted((c["normalization"], c["hparam_overrides"]["adapt"]) for c in CONFIGS)
    assert got == [("none", False), ("none", True)]
    assert all("hparams.adapt" not in c for c in CONFIGS)
    root = tmp_path / "runs" / "hp" / "synth"
    assert sorted(p.name for p in root.iterdir()) == ["fake__adapt-False", "fake__adapt-True"]


def test_resolve_hparams_precedence():
    cfg = {"dataset": "smd_1-4", "hparams": {"default": {"gamma": 0.99, "lr": 0.05},
                                               "smd_1-4": {"gamma": 0.4}},
           "hparam_overrides": {"gamma": 0.9}}
    assert resolve_hparams(cfg, {"gamma": 0.5, "epochs": 30}) == {"gamma": 0.9, "lr": 0.05, "epochs": 30}
    del cfg["hparam_overrides"]
    assert resolve_hparams(cfg, {})["gamma"] == 0.4


def test_ablation_config_expands_to_the_four_table3_arms():
    exp = yaml.safe_load(open("configs/experiments/smd_ablation.yaml"))
    names = [name for name, _ in _arms(exp["overrides"], "m2n2")]
    assert names == ["m2n2__normalization-none__adapt-False", "m2n2__normalization-none__adapt-True",
                     "m2n2__normalization-ema__adapt-False", "m2n2__normalization-ema__adapt-True"]
    ref = json.load(open("results/paper_reference.json"))["m2n2_paper"]["smd_2-1"]
    assert set(names) <= set(ref)


def test_threshold_can_use_the_detectors_tau():
    import pytest
    from audit.evaluation.harness import apply_threshold
    s = np.array([0.1, 0.5, 0.9])
    assert apply_threshold("fixed", {"quantile": "detector"}, s, s, detector_tau=0.5).tolist() == [0, 1, 1]
    assert apply_threshold("fixed", {"quantile": 0.5}, s, s).tolist() == [0, 1, 1]
    with pytest.raises(ValueError):
        apply_threshold("fixed", {"quantile": "detector"}, s, s)


def test_git_commit_falls_back_to_env(monkeypatch):
    def no_git(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(io.subprocess, "check_output", no_git)
    monkeypatch.setenv("GIT_COMMIT", "abc123-dirty")
    assert io.git_commit() == "abc123-dirty"
    monkeypatch.delenv("GIT_COMMIT")
    assert io.git_commit() == "unknown"
