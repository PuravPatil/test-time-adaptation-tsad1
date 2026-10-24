# Test-Time Adaptation for TSAD: an M2N2 audit

A shared, honest evaluation harness for test-time adaptation in unsupervised time-series anomaly detection. We reproduce M2N2 and compare it with a static (no-adaptation) baseline:

- **M2N2** — Kim, Park, Choo, *When Model Meets New Normals*, AAAI 2024 ([code](https://github.com/carrtesy/M2N2))

Both models run on the same data, with the same metrics (plain F1, AUPRC, PA%K; F1-PA for reference only), on CPU with fixed seeds. Three diagnostic extensions plug into the harness:

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
make submodules     # first time only: adds M2N2 under third_party/
make test
```

## Running an experiment

```bash
python scripts/run.py --config configs/experiments/reproduce.yaml
python scripts/make_tables.py
```

Each run writes raw scores, predictions, metrics, the resolved config and the git commit hash to `results/runs/<experiment>/<dataset>/<method>/<threshold>/seed<k>/`. A detector is trained once per dataset, method and seed; every threshold in the experiment is applied to the same scores.

`overrides` in an experiment config set method keys for one method or for `all`; a list is a sweep, one arm per value, saved as `<method>__<key>-<value>/`:

```yaml
overrides:
  m2n2: {normalization: [ema, revin]}   # -> m2n2__normalization-ema/, m2n2__normalization-revin/
```

`run.json` also stores `detector_stats` (for M2N2: windows seen, test-time updates taken, update rate).

## Reproducing the SMD results (M2N2 and the static MLP baseline)

```bash
pip install -r requirements.txt && pip install -e .
python scripts/download_data.py          # SMD machines 1-4, 1-7, 1-8, 2-1, 2-4, 3-2 (needs internet)
python scripts/preprocess.py
pytest -q                                # includes torch smoke tests of the detectors
python scripts/run.py --config configs/experiments/smd_quick.yaml      # ~minutes: one machine, one seed
python scripts/run.py --config configs/experiments/smd_reproduce.yaml  # 6 machines x 2 models x 5 seeds
python scripts/compare_paper.py          # ours vs the numbers in the papers
python scripts/compare_methods.py        # M2N2 vs static, paired over seeds, 95% intervals + Wilcoxon
python scripts/run.py --config configs/experiments/smd_ablation.yaml   # paper Table 3: detrend x adaptation
python scripts/compare_paper.py --experiment smd_ablation              # vs Table 3 (M-2-1)
python scripts/compare_methods.py --experiment smd_ablation --a m2n2__normalization-ema__adapt-True --b m2n2__normalization-ema__adapt-False
```

What is implemented, and what the papers leave open, is in the docstrings of
`src/audit/methods/m2n2_adapter.py` and the comments in
`configs/methods/*.yaml`. Paper numbers: `results/paper_reference.json`.

## Layout

```
configs/        YAML for datasets, methods, experiments (base.yaml: seeds, device=cpu)
data/           gitignored; raw/ downloads, processed/ common format
src/audit/      data loaders, method adapters, normalization, backbones,
                thresholds, evaluation (metrics, latency, harness), utils
scripts/        download, preprocess, run one experiment, build tables
results/        runs/ (gitignored) and tables/ (committed)
third_party/    upstream M2N2 as a pinned git submodule — never edited
demo/           demo: api/ (FastAPI backend, uploads + jobs), web/ (React frontend); `docker compose up api web`
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

## Running in Docker (CPU)

```bash
export GIT_COMMIT=$(git describe --always --dirty --abbrev=40)   # PowerShell: $env:GIT_COMMIT = git describe --always --dirty --abbrev=40
docker compose build
docker compose run --rm data      # download + preprocess SMD into ./data
docker compose run --rm test
docker compose run --rm quick     # one machine, one seed
docker compose run --rm smd       # full SMD reproduction (6 machines x 2 models x 5 seeds)
docker compose run --rm compare   # ours vs the papers
```

CPU-only on purpose: the MLPs are tiny and test-time adaptation updates one window at a time,
so a GPU would not help. Raise `cpus` / `mem_limit` in `docker-compose.yml` if your machine has spare capacity.
Give Docker Desktop at least 4 GB of RAM. Results land in `./results` on your disk.
