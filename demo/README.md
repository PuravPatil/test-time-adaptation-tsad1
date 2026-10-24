# Live demo (weeks 7–8)

Design direction: "D · Modern, kinetic type + upload any dataset" (dark theme, bold
animated headline, Upload → Map columns → Results flow).

- `api/` — FastAPI backend: saved runs, uploads of any numeric time series
  (.csv .tsv .txt .xlsx .parquet .json .npy), background jobs, WebSocket replay.
  See `api/README.md`. Run: `docker compose up api` → http://localhost:8000/docs
- `web/` — React frontend (Vite), design D: upload → map columns → live results, and the
  audit's headline numbers read from the API. See `web/README.md`.
  Run both: `docker compose up api web` → http://localhost:8080
