# Test-Time Adaptation for TSAD: an M2N2 → CANDI audit

A shared, honest evaluation harness for two test-time adaptation methods for unsupervised time-series anomaly detection:

- **M2N2** — Kim, Park, Choo, *When Model Meets New Normals*, AAAI 2024 ([code](https://github.com/carrtesy/M2N2))
- **CANDI** — AAAI 2026 ([paper](https://arxiv.org/pdf/2604.01845), [code](https://github.com/kimanki/CANDI))

Both methods run on the same data, with the same metrics (plain F1, AUPRC, PA%K; F1-PA for reference only), on CPU with fixed seeds. Three diagnostic extensions plug into the harness:

| Extension | Owner | Where |
| --- | --- | --- |
| RevIN in place of EMA detrending | Tanishka | `src/audit/normalization/` |
| Streaming threshold in place of fixed τ | Purav | `src/audit/thresholds/` |
| Backbone generality + inference latency | Unnati | `src/audit/backbones/`, `src/audit/evaluation/latency.py` |

## Setup

```bash
git clone --recurse-submodules https://github.com/unnatipatro12/test-time-adaptation-tsad
cd test-time-adaptation-tsad
python -m venv .venv && source .venv/bin/activate
make setup          # installs requirements and this package in editable mode
make submodules     # first time only: adds M2N2 and CANDI under third_party/
make test
```

## Running an experiment

```bash
python scripts/run.py --config configs/experiments/reproduce.yaml
python scripts/make_tables.py
```

Each run writes raw scores, predictions, metrics, the resolved config and the git commit hash to `results/runs/<experiment>/<dataset>/<method>/seed<k>/`.

## Layout

```
configs/        YAML for datasets, methods, experiments (base.yaml: seeds, device=cpu)
data/           gitignored; raw/ downloads, processed/ common format
src/audit/      data loaders, method adapters, normalization, backbones,
                thresholds, evaluation (metrics, latency, harness), utils
scripts/        download, preprocess, run one experiment, build tables
results/        runs/ (gitignored) and tables/ (committed)
third_party/    upstream M2N2 and CANDI as pinned git submodules — never edited
demo/           live streaming dashboard (weeks 7–8)
tests/          metric, loader and threshold tests
docs/           project plan and design
```

## Rules

- `main` is protected. Branch as `<name>/<topic>` (e.g. `tanny/revin`) and merge by pull request with one review.
- Never edit `third_party/`. Changes to upstream behaviour go in the adapters.
- No hard-coded paths or hyperparameters in code; they belong in `configs/`.
- Any change to `src/audit/evaluation/metrics.py` needs a test.

## Timeline

Weeks 1–3 shared reproduction · weeks 4–6 extensions · weeks 7–8 demo and deployment.
