"""Background jobs: train a detector on a sample or uploaded dataset, then stream its test part.

Jobs run in a small thread pool (one worker by default: training is CPU-bound and
test-time adaptation is sequential). Job status lives in memory; finished results
are written to results/uploads/jobs/<id>/ so they survive until the TTL cleanup.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from audit.data import load_dataset
from audit.data.upload import UploadError, prepare, read_table
from audit.evaluation.harness import apply_threshold, build_run_config, fit_and_score
from audit.evaluation.metrics import evaluate
from audit.utils.io import load_yaml

from .settings import Settings
from .uploads import UploadStore
from .util import clean, safe_id, series_payload

log = logging.getLogger("tsad_api.jobs")

STAGES = {"queued": 0.0, "loading": 0.1, "training": 0.3, "scoring": 0.7, "evaluating": 0.9, "done": 1.0}
PUBLIC = ("job_id", "state", "stage", "progress", "error", "created", "started", "finished",
          "request", "dataset", "metrics", "detector_stats", "ms_per_step", "tau")


class JobError(ValueError):
    pass


class JobRunner:
    def __init__(self, settings: Settings, store: UploadStore, samples):
        self.settings, self.store, self.samples = settings, store, samples
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=max(1, settings.max_workers), thread_name_prefix="tsad-job")

    # -- public ------------------------------------------------------------
    def submit(self, req: dict) -> dict:
        if (req.get("upload_id") is None) == (req.get("sample") is None):
            raise JobError("give exactly one of 'upload_id' or 'sample'")
        if req["method"] not in self.settings.methods:
            raise JobError(f"unknown method '{req['method']}'. Known: {', '.join(self.settings.methods)}")
        if req.get("upload_id") is not None and self.store.get(req["upload_id"]) is None:
            raise JobError("unknown or expired upload_id")
        if req.get("sample") is not None and req["sample"] not in {s["name"] for s in self.samples()}:
            raise JobError(f"unknown sample '{req['sample']}'")
        job_id = uuid.uuid4().hex
        job = {"job_id": job_id, "state": "queued", "stage": "queued", "progress": 0.0, "error": None,
               "created": time.time(), "started": None, "finished": None, "request": req,
               "dataset": None, "metrics": None, "detector_stats": None, "ms_per_step": None, "tau": None}
        with self.lock:
            self.jobs[job_id] = job
        self.pool.submit(self._run, job_id)
        return self.get(job_id)

    def get(self, job_id: str) -> dict | None:
        if not safe_id(job_id):
            return None
        with self.lock:
            job = self.jobs.get(job_id)
            return clean({k: job[k] for k in PUBLIC}) if job else None

    def series(self, job_id: str, points: int) -> dict | None:
        job = self.get(job_id)
        if not job or job["state"] != "done":
            return None
        d = self.store.jobs_dir / job_id
        labels = np.load(d / "labels.npy") if (d / "labels.npy").exists() else None
        return series_payload(np.load(d / "scores.npy"), np.load(d / "preds.npy"), labels, points)

    def csv(self, job_id: str) -> str | None:
        """Full-resolution result: one row per test timestep (t, score, alarm[, label])."""
        job = self.get(job_id)
        if not job or job["state"] != "done":
            return None
        d = self.store.jobs_dir / job_id
        scores, preds = np.load(d / "scores.npy"), np.load(d / "preds.npy")
        cols = [np.arange(len(scores)), scores, preds]
        header = "t,score,alarm"
        if (d / "labels.npy").exists():
            cols.append(np.load(d / "labels.npy"))
            header += ",label"
        lines = [header] + [",".join(f"{v:.8g}" if isinstance(v, float) else str(v) for v in row)
                            for row in zip(*(c.tolist() for c in cols))]
        return "\n".join(lines) + "\n"

    # -- worker ------------------------------------------------------------
    def _set(self, job_id: str, **kw):
        with self.lock:
            self.jobs[job_id].update(kw)

    def _stage(self, job_id: str, stage: str):
        self._set(job_id, stage=stage, progress=STAGES[stage])

    def _load(self, req: dict):
        if req.get("sample"):
            data = load_dataset(req["sample"], str(self.settings.processed_dir))
            report = {"name": data.name, "source": "sample", "n_train": len(data.train), "n_test": len(data.test),
                      "n_features": int(data.train.shape[1]), "has_labels": True,
                      "test_anomaly_rate": float(data.test_labels.mean()), "warnings": []}
            return data, report
        df = read_table(self.store.data_path(req["upload_id"]))
        data, report = prepare(df, time_col=req.get("time_col"), label_col=req.get("label_col"),
                               feature_cols=req.get("feature_cols"), train_frac=req.get("train_frac", 0.5))
        report["source"] = "upload"
        return data, report

    def _run(self, job_id: str):
        self._set(job_id, state="running", started=time.time())
        req = self.jobs[job_id]["request"]
        try:
            self._stage(job_id, "loading")
            data, report = self._load(req)
            self._set(job_id, dataset=report)

            config_name, extra = self.settings.methods[req["method"]]
            method_cfg = load_yaml(self.settings.config_dir / "methods" / f"{config_name}.yaml")
            run_cfg = build_run_config(method_cfg, data.name, req["seed"])
            for k, v in extra.items():
                run_cfg[k] = {**(run_cfg.get(k) or {}), **v} if isinstance(v, dict) else v
            if req.get("window"):
                run_cfg["hparam_overrides"] = {**(run_cfg.get("hparam_overrides") or {}),
                                               "window": req["window"], "stride": req["window"]}
            if req.get("quantile") is not None:   # one tau for alarms and adaptation, as in the paper
                run_cfg["hparam_overrides"] = {**(run_cfg.get("hparam_overrides") or {}),
                                               "tau_quantile": req["quantile"]}

            scores, reference, stats, step_ms = fit_and_score(
                data, run_cfg, req["seed"], on_stage=lambda s: self._stage(job_id, s))
            self._stage(job_id, "evaluating")
            # Fixed tau = the detector's own tau (its tau_quantile of training scores, the same tau it
            # adapted with); a detector that reports none falls back to the requested or 0.95 quantile.
            det_tau = stats.get("tau")
            q = req.get("quantile")
            thr_cfg = {"quantile": "detector"} if q is None and det_tau is not None else {"quantile": q or 0.95}
            preds = apply_threshold(req["threshold"], thr_cfg, reference, scores, det_tau)
            tau = None                                     # for drawing; a streaming tau moves, so none
            if req["threshold"] == "fixed":
                if thr_cfg["quantile"] == "detector":
                    tau = float(det_tau)
                elif reference is not None:
                    tau = float(np.quantile(reference, thr_cfg["quantile"]))
            metrics = None
            if report["has_labels"]:
                metrics = evaluate(data.test_labels, scores, preds, (0, 20, 50, 80, 100))
                metrics["ms_per_step"] = step_ms

            out = self.store.jobs_dir / job_id
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "scores.npy", scores)
            np.save(out / "preds.npy", preds)
            if report["has_labels"]:
                np.save(out / "labels.npy", data.test_labels)
            (out / "result.json").write_text(json.dumps(clean(
                {"request": req, "dataset": report, "metrics": metrics, "detector_stats": stats,
                 "ms_per_step": step_ms, "tau": tau}), indent=1))
            self._set(job_id, state="done", stage="done", progress=1.0, finished=time.time(),
                      metrics=metrics, detector_stats=stats, ms_per_step=step_ms, tau=tau)
        except UploadError as e:
            self._set(job_id, state="failed", error=str(e), finished=time.time())
        except Exception as e:   # report, never kill the worker
            log.error("job %s failed:\n%s", job_id, traceback.format_exc())
            self._set(job_id, state="failed", error=f"{type(e).__name__}: {e}", finished=time.time())
