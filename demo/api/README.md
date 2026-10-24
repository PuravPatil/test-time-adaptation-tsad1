# TSAD demo API (FastAPI)

Backend for the demo frontend (`demo/web/`, next). It does three things:

1. **Serves saved runs** from `results/runs/` — metrics, detector stats, downsampled score series, paired comparisons.
2. **Accepts uploads** of any numeric time series — `.csv .tsv .txt .xlsx .parquet .json .npy` — and reports its columns, a guessed time column and label column, and a preview.
3. **Runs jobs**: trains a detector on an upload (or a one-click SMD sample), streams the test part through it, and returns scores, alarms and — if there is a label column — metrics.

Runs on uploaded data are written to `results/uploads/`, never to `results/runs/`: they are exploratory, not benchmark evidence.

## Run it

```bash
# Docker (from the project root)
docker compose build api
docker compose up api                      # http://localhost:8000/docs (interactive API docs)

# or locally
pip install -r requirements.txt -r demo/api/requirements.txt && pip install -e .
uvicorn tsad_api.main:create_app --factory --app-dir demo/api --reload --port 8000
```

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Status, available methods, upload limit |
| GET | `/experiments` | Saved experiments with their datasets, arms, thresholds, seeds |
| GET | `/runs/{experiment}/{dataset}/{arm}/{threshold}/{seed}` | Metrics, `detector_stats`, overrides, commit |
| GET | `/runs/.../series?points=600` | Scores, alarms and labels, max-pooled to `points` so spikes stay visible |
| GET | `/compare?experiment=&a=&b=&threshold=fixed&metrics=auroc` | Paired per-seed differences with 95% intervals, Wilcoxon across datasets |
| GET | `/samples` | SMD machines available as one-click datasets |
| POST | `/uploads` (multipart `file`) | Save, read and inspect a file → `upload_id`, columns, guesses, preview |
| GET | `/uploads/{upload_id}` | The same inspection again |
| POST | `/jobs` | Start a job (body below) → `job_id` |
| GET | `/jobs/{job_id}` | State (`queued/running/done/failed`), stage, progress, metrics |
| GET | `/jobs/{job_id}/series?points=600` | The finished job's series |
| WS | `/jobs/{job_id}/stream?points=600&chunk=60&delay=0.05` | Status messages, then `metrics`, then the series in chunks (a replay), then `end` |

Job body (`POST /jobs`), exactly one of `upload_id` / `sample`:

```json
{"upload_id": "…", "method": "m2n2", "time_col": "auto", "label_col": "label",
 "feature_cols": null, "train_frac": 0.5, "window": 5, "seed": 0,
 "threshold": "fixed", "quantile": null}
```

`quantile: null` (the default) uses the method's own τ: its `tau_quantile` of the training scores — for the SMD
samples 1-4 and 2-1 the paper's per-machine setting. As in the paper, that one τ decides both the alarms and
which timesteps the model adapts on; a number (e.g. `0.95`) sets it for this job.

`method`: `m2n2` (detrend + adapt), `detrend_only` (the `pretrained` baseline), `static` (plain MLP).
Without `label_col` a job returns scores and alarms but `metrics: null`.

## Limits and assumptions

- 50 MB per file, 2,000,000 rows, uploads and job results deleted after 24 h, one job at a time
  (env: `TSAD_MAX_UPLOAD_MB`, `TSAD_MAX_ROWS`, `TSAD_UPLOAD_TTL_HOURS`, `TSAD_MAX_WORKERS`, `TSAD_CORS_ORIGINS`).
- The first `train_frac` of the rows (in time order) is assumed normal and used for training.
- Hyperparameters are the SMD defaults (`configs/methods/m2n2.yaml`); on other data, results are exploratory.
- Job status is kept in memory: restarting the API forgets running jobs (finished results stay on disk until the TTL).
- Progress is reported per stage (loading, training, scoring, evaluating), not per window.
