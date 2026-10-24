"""HTTP and WebSocket routes.

    GET  /health
    GET  /experiments                                         saved experiments and what they contain
    GET  /runs/{experiment}/{dataset}/{arm}/{threshold}/{seed}          metrics + detector stats
    GET  /runs/{experiment}/{dataset}/{arm}/{threshold}/{seed}/series   scores, alarms, labels (downsampled)
    GET  /compare?experiment=&a=&b=                           paired comparison, intervals, Wilcoxon
    GET  /samples                                             datasets that run with one click
    POST /uploads                                             upload a file -> columns, guesses, preview
    GET  /uploads/{upload_id}
    POST /jobs                                                train + stream on an upload or a sample
    GET  /jobs/{job_id}                                       status, stage, metrics
    GET  /jobs/{job_id}/series
    GET  /jobs/{job_id}/download                              full-resolution CSV (t, score, alarm, label)
    WS   /jobs/{job_id}/stream                               status updates, then the series in chunks
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from audit.data.upload import UploadError

from .jobs import JobError, JobRunner
from .results import ResultsIndex, RunNotFound
from .settings import Settings
from .uploads import UploadStore, UploadTooLarge
from .util import safe_name

DEFAULT_METRICS = ["auroc", "auprc", "f1", "pa_k_50"]


class JobRequest(BaseModel):
    upload_id: str | None = None
    sample: str | None = None
    method: str = "m2n2"
    time_col: str | None = "auto"
    label_col: str | None = None
    feature_cols: list[str] | None = None
    train_frac: float = Field(0.5, gt=0.05, lt=0.95)
    window: int | None = Field(None, ge=2, le=200)
    seed: int = Field(0, ge=0, le=10_000)
    threshold: Literal["fixed", "streaming"] = "fixed"
    # None: the method's own tau_quantile (the paper's per-machine setting for SMD samples)
    quantile: float | None = Field(None, gt=0.5, lt=1.0)


def _check_names(*names: str) -> None:
    for n in names:
        if not safe_name(n):
            raise HTTPException(400, f"invalid name '{n}'")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    results = ResultsIndex(settings)
    store = UploadStore(settings)
    jobs = JobRunner(settings, store, results.samples)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        store.cleanup()                     # drop expired uploads and job results
        yield
        jobs.pool.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="TSAD audit API", version="0.1.0", lifespan=lifespan,
                  description="Saved M2N2 audit runs, dataset upload, and detector jobs.")
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                       allow_methods=["GET", "POST"], allow_headers=["*"])
    app.state.settings, app.state.jobs, app.state.store = settings, jobs, store

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "methods": list(settings.methods), "max_upload_mb": settings.max_upload_mb}

    # -- saved runs ------------------------------------------------------
    @app.get("/experiments")
    def experiments() -> list[dict]:
        return results.experiments()

    @app.get("/runs/{experiment}/{dataset}/{arm}/{threshold}/{seed}")
    def run(experiment: str, dataset: str, arm: str, threshold: str, seed: int) -> dict:
        _check_names(experiment, dataset, arm, threshold)
        try:
            return results.run(experiment, dataset, arm, threshold, seed)
        except RunNotFound as e:
            raise HTTPException(404, str(e))

    @app.get("/runs/{experiment}/{dataset}/{arm}/{threshold}/{seed}/series")
    def run_series(experiment: str, dataset: str, arm: str, threshold: str, seed: int,
                   points: int = Query(600, ge=10, le=20_000)) -> dict:
        _check_names(experiment, dataset, arm, threshold)
        try:
            return results.series(experiment, dataset, arm, threshold, seed, points)
        except RunNotFound as e:
            raise HTTPException(404, str(e))

    @app.get("/compare")
    def compare(experiment: str, a: str, b: str, threshold: str = "fixed",
                metrics: list[str] = Query(DEFAULT_METRICS)) -> dict:
        _check_names(experiment, a, b, threshold)
        try:
            return results.compare(experiment, a, b, threshold, metrics)
        except RunNotFound as e:
            raise HTTPException(404, str(e))

    @app.get("/samples")
    def samples() -> list[dict]:
        return results.samples()

    # -- uploads and jobs ------------------------------------------------
    @app.post("/uploads", status_code=201)
    async def upload(file: UploadFile = File(...)) -> dict:
        try:
            return await store.save(file)
        except UploadTooLarge as e:
            raise HTTPException(413, str(e))
        except UploadError as e:
            code = 415 if str(e).startswith("unsupported file type") else 422
            raise HTTPException(code, str(e))

    @app.get("/uploads/{upload_id}")
    def get_upload(upload_id: str) -> dict:
        meta = store.get(upload_id)
        if meta is None:
            raise HTTPException(404, "unknown or expired upload")
        return meta

    @app.post("/jobs", status_code=202)
    def create_job(req: JobRequest) -> dict:
        try:
            return jobs.submit(req.model_dump())
        except JobError as e:
            raise HTTPException(422, str(e))

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        return job

    @app.get("/jobs/{job_id}/series")
    def job_series(job_id: str, points: int = Query(600, ge=10, le=20_000)) -> dict:
        if jobs.get(job_id) is None:
            raise HTTPException(404, "unknown job")
        s = jobs.series(job_id, points)
        if s is None:
            raise HTTPException(409, "job is not finished")
        return s

    @app.get("/jobs/{job_id}/download")
    def job_download(job_id: str) -> Response:
        if jobs.get(job_id) is None:
            raise HTTPException(404, "unknown job")
        body = jobs.csv(job_id)
        if body is None:
            raise HTTPException(409, "job is not finished")
        return Response(body, media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="tsad_job_{job_id[:8]}.csv"'})

    @app.websocket("/jobs/{job_id}/stream")
    async def job_stream(ws: WebSocket, job_id: str, points: int = 600, chunk: int = 60,
                         delay: float = 0.05) -> None:
        """Status messages while the job runs, then the series in chunks (a replay), then 'end'."""
        await ws.accept()
        points, chunk, delay = min(max(points, 10), 20_000), min(max(chunk, 1), 5_000), min(max(delay, 0.0), 2.0)
        try:
            if jobs.get(job_id) is None:
                await ws.send_json({"type": "error", "detail": "unknown job"})
                await ws.close(code=4404)
                return
            last = None
            while True:
                job = jobs.get(job_id)
                status = {k: job[k] for k in ("state", "stage", "progress", "error")}
                if status != last:
                    await ws.send_json({"type": "status", "job_id": job_id, **status})
                    last = status
                if job["state"] in ("done", "failed"):
                    break
                await asyncio.sleep(0.25)
            if job["state"] == "done":
                await ws.send_json({"type": "metrics", "metrics": job["metrics"], "tau": job["tau"],
                                    "detector_stats": job["detector_stats"], "dataset": job["dataset"],
                                    "ms_per_step": job["ms_per_step"]})
                s = jobs.series(job_id, points)
                for i in range(0, len(s["scores"]), chunk):
                    await ws.send_json({"type": "series", "start": i, "n": s["n"], "bucket": s["bucket"],
                                        "scores": s["scores"][i:i + chunk], "preds": s["preds"][i:i + chunk],
                                        "labels": s["labels"][i:i + chunk] if s["labels"] is not None else None})
                    await asyncio.sleep(delay)
            await ws.send_json({"type": "end"})
            await ws.close()
        except WebSocketDisconnect:
            return

    return app
