"""The demo API end to end, with a fake detector (no torch) and a temporary project folder."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")
from fastapi.testclient import TestClient  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo" / "api"))
from tsad_api.main import create_app  # noqa: E402
from tsad_api.settings import Settings  # noqa: E402

from audit.methods import Detector  # noqa: E402


class FakeDetector(Detector):
    """Scores each timestep by its mean absolute value."""

    def fit(self, train):
        self._ref = np.abs(train).mean(axis=1)

    def score(self, test):
        self.stats = {"n_updates": 0, "normalization": self.config.get("normalization")}
        return np.abs(test).mean(axis=1)

    def train_scores(self):
        return self._ref


def _project(tmp_path):
    rng = np.random.default_rng(0)
    (tmp_path / "configs" / "methods").mkdir(parents=True)
    (tmp_path / "configs" / "methods" / "fake.yaml").write_text(
        yaml.safe_dump({"name": "fake", "adapter": f"{__name__}.FakeDetector"}))
    # saved runs: experiment e, dataset smd_9-9, arms a and b, seeds 0-2
    for arm, shift in (("a", 0.1), ("b", 0.0)):
        for seed in range(3):
            d = tmp_path / "results" / "runs" / "e" / "smd_9-9" / arm / "fixed" / f"seed{seed}"
            d.mkdir(parents=True)
            np.save(d / "scores.npy", rng.random(1000))
            np.save(d / "preds.npy", (rng.random(1000) > 0.9).astype(int))
            (d / "metrics.json").write_text(json.dumps(
                {"auroc": 0.7 + shift + 0.01 * seed, "auprc": float("nan"), "f1": 0.3 + shift, "pa_k_50": 0.4}))
            (d / "run.json").write_text(json.dumps({"detector_stats": {"n_updates": 5}, "commit": "abc"}))
    # one SMD-style sample on disk, loadable by the real loader
    m = tmp_path / "data" / "processed" / "smd" / "machine-9-9"
    m.mkdir(parents=True)
    np.save(m / "train.npy", rng.normal(size=(400, 3)).astype(np.float32))
    test = rng.normal(size=(1000, 3)).astype(np.float32)
    test[500:520] += 6
    lab = np.zeros(1000, dtype=int)
    lab[500:520] = 1
    np.save(m / "test.npy", test)
    np.save(m / "test_labels.npy", lab)
    return Settings(root=tmp_path, max_upload_mb=1.0, methods={"fake": ("fake", {}),
                                                              "fake_none": ("fake", {"normalization": "none"})})


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(_project(tmp_path))) as c:
        yield c


def _csv(n=300, label=True):
    rng = np.random.default_rng(1)
    rows = ["timestamp,cpu,mem" + (",label" if label else "")]
    for i in range(n):
        spike = 5.0 if 250 <= i < 260 else 0.0
        row = f"2026-01-01 00:{i // 60:02d}:{i % 60:02d},{rng.normal() + spike:.4f},{rng.normal():.4f}"
        rows.append(row + (f",{int(250 <= i < 260)}" if label else ""))
    return "\n".join(rows).encode()


def _wait(client, job_id, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/jobs/{job_id}").json()
        if job["state"] in ("done", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_health_and_experiments(client):
    assert client.get("/health").json()["status"] == "ok"
    exps = client.get("/experiments").json()
    assert exps == [{"name": "e", "n_runs": 6, "datasets": ["smd_9-9"], "arms": ["a", "b"],
                     "thresholds": ["fixed"], "seeds": [0, 1, 2]}]


def test_run_detail_series_and_nan_become_null(client):
    r = client.get("/runs/e/smd_9-9/a/fixed/0").json()
    assert r["metrics"]["auprc"] is None and r["detector_stats"] == {"n_updates": 5}
    assert r["test_anomaly_rate"] == pytest.approx(0.02)
    s = client.get("/runs/e/smd_9-9/a/fixed/0/series?points=100").json()
    assert s["n"] == 1000 and len(s["scores"]) == 100 and len(s["labels"]) == 100
    assert max(s["labels"]) == 1.0                   # max-pooling keeps the anomaly segment


def test_compare(client):
    c = client.get("/compare", params={"experiment": "e", "a": "a", "b": "b", "metrics": ["auroc", "f1"]}).json()
    row = {r["metric"]: r for r in c["per_dataset"]}
    assert row["auroc"]["diff"] == pytest.approx(0.1) and row["auroc"]["a_wins"] == 3
    assert {s["metric"] for s in c["summary"]} == {"auroc", "f1"}


def test_bad_names_and_missing_runs(client):
    assert client.get("/runs/e/smd_9-9/a/fixed/7").status_code == 404
    assert client.get("/runs/e/bad$name/a/fixed/0").status_code == 400
    assert client.get("/runs/e/..smd/a/fixed/0").status_code == 400
    assert client.get("/uploads/not-an-id").status_code == 404
    assert client.get("/jobs/not-an-id").status_code == 404


def test_samples(client):
    assert [s["name"] for s in client.get("/samples").json()] == ["smd_9-9"]


def test_upload_inspects_the_file(client):
    r = client.post("/uploads", files={"file": ("readings.csv", _csv(), "text/csv")})
    assert r.status_code == 201
    up = r.json()
    assert up["n_rows"] == 300 and up["time_col"] == "timestamp" and up["label_col"] == "label"
    assert client.get(f"/uploads/{up['upload_id']}").json()["filename"] == "readings.csv"


def test_upload_rejects_bad_type_size_and_content(client):
    assert client.post("/uploads", files={"file": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 415
    big = b"1.0\n" * 400_000                         # ~1.6 MB > the 1 MB test limit
    assert client.post("/uploads", files={"file": ("big.csv", big, "text/csv")}).status_code == 413
    assert client.post("/uploads", files={"file": ("bad.json", b"{oops", "application/json")}).status_code == 422


def test_job_on_upload_with_labels_and_stream(client):
    up = client.post("/uploads", files={"file": ("r.csv", _csv(), "text/csv")}).json()
    r = client.post("/jobs", json={"upload_id": up["upload_id"], "method": "fake", "label_col": "label",
                                   "train_frac": 0.5, "threshold": "streaming"})
    assert r.status_code == 202
    job = _wait(client, r.json()["job_id"])
    assert job["state"] == "done", job["error"]
    assert job["dataset"]["has_labels"] and job["dataset"]["n_test"] == 150
    assert job["metrics"]["auroc"] > 0.9
    s = client.get(f"/jobs/{job['job_id']}/series").json()
    assert s["n"] == 150 and sum(s["labels"]) == 10
    with client.websocket_connect(f"/jobs/{job['job_id']}/stream?chunk=50&delay=0") as ws:
        msgs = []
        while True:
            m = ws.receive_json()
            msgs.append(m)
            if m["type"] == "end":
                break
    types = [m["type"] for m in msgs]
    assert types[0] == "status" and "metrics" in types and types[-1] == "end"
    assert sum(len(m["scores"]) for m in msgs if m["type"] == "series") == 150


def test_job_without_labels_has_no_metrics(client):
    up = client.post("/uploads", files={"file": ("r.csv", _csv(label=False), "text/csv")}).json()
    job = _wait(client, client.post("/jobs", json={"upload_id": up["upload_id"], "method": "fake_none"}).json()["job_id"])
    assert job["state"] == "done" and job["metrics"] is None and not job["dataset"]["has_labels"]
    assert job["detector_stats"]["normalization"] == "none"     # method extras reach the detector


def test_job_on_sample(client):
    job = _wait(client, client.post("/jobs", json={"sample": "smd_9-9", "method": "fake"}).json()["job_id"])
    assert job["state"] == "done" and job["metrics"]["auroc"] > 0.9
    assert job["tau"] is not None and job["tau"] > 0                   # fixed threshold -> drawable tau
    r = client.get(f"/jobs/{job['job_id']}/download")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    lines = r.text.strip().splitlines()
    assert lines[0] == "t,score,alarm,label" and len(lines) == 1001
    assert lines[501].split(",")[3] == "1"                             # t = 500 is labelled


def test_job_validation_and_failures(client):
    assert client.post("/jobs", json={"method": "fake"}).status_code == 422                         # no source
    assert client.post("/jobs", json={"sample": "smd_9-9", "upload_id": "0" * 32,
                                      "method": "fake"}).status_code == 422                         # two sources
    assert client.post("/jobs", json={"sample": "smd_9-9", "method": "nope"}).status_code == 422
    assert client.post("/jobs", json={"sample": "smd_0-0", "method": "fake"}).status_code == 422
    assert client.post("/jobs", json={"sample": "smd_9-9", "method": "fake", "train_frac": 2}).status_code == 422
    up = client.post("/uploads", files={"file": ("r.csv", _csv(), "text/csv")}).json()
    job = _wait(client, client.post("/jobs", json={"upload_id": up["upload_id"], "method": "fake",
                                                   "label_col": "nope"}).json()["job_id"])
    assert job["state"] == "failed" and "nope" in job["error"]
    assert client.get(f"/jobs/{job['job_id']}/series").status_code == 409
    assert client.get(f"/jobs/{job['job_id']}/download").status_code == 409
