# Drift Studio — demo frontend (React + Vite)

Design D: dark theme, kinetic headline, and an **Upload → Map columns → Results** flow that
talks to the FastAPI backend in `demo/api/`.

- **Upload**: drag and drop any numeric time series (`.csv .tsv .txt .xlsx .parquet .json .npy`), or run an SMD sample.
- **Map columns**: time column, optional label column, features, training split, model (M2N2 / detrend only / plain MLP), window, threshold. Shows the first rows and checks before training.
- **Results**: live stages (loading → training → streaming → evaluating), then the scores replayed over a WebSocket into the chart; metrics when there is a label column; full-resolution CSV download.
- **What the audit found**: headline numbers computed live from `results/runs/smd_ablation` through the API (hidden if that experiment is missing).

## Run it

With Docker (from the project root), both services together:

```bash
docker compose up api web        # app: http://localhost:8080   API docs: http://localhost:8000/docs
```

For development, with hot reload (needs Node 20+):

```bash
docker compose up api            # or run uvicorn locally, see demo/api/README.md
cd demo/web
npm install
npm run dev                      # http://localhost:5173 ; /api is forwarded to localhost:8000
```

The app always calls `/api/...`; Vite (dev) and nginx (Docker) forward that to the API, WebSockets
included, so no CORS setup is needed. `API_TARGET=http://host:port npm run dev` points it elsewhere.

## Files

| File | What it is |
|---|---|
| `src/App.jsx` | Page layout, step state, API health |
| `src/api.js` | Fetch + WebSocket client for every endpoint |
| `src/findings.js` | Loads the audit's headline numbers from the API |
| `src/components/` | `Nav`, `Hero` (kinetic type), `Ticker`, `UploadStep`, `MapStep`, `ResultsStep`, `ScoreChart` (log-scale SVG), `AuditSection` |
| `src/styles.css` | Design tokens and all styles; animations stop under "reduce motion" |
| `Dockerfile`, `nginx.conf` | Production build served by nginx, `/api` proxied to the `api` service |
