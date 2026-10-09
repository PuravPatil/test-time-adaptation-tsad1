# Design

The full design (high-level design, architecture diagram, folder structure,
ownership) lives in the shared doc:
https://claude.ai/code/artifact/a20cd416-4311-4157-89f0-131f8145cc8e

Summary:

- Every method sits behind `audit.methods.Detector` (`fit`, `score`, `train_scores`).
- Thresholds are separate from methods (`audit.thresholds`), so the streaming
  threshold works for both M2N2 and CANDI.
- Metrics live in one module (`audit.evaluation.metrics`): plain F1, AUPRC,
  PA%K, and F1-PA for reference only.
- Every run saves raw scores, so metrics can be recomputed without rerunning.
- Upstream code stays untouched in `third_party/` as pinned submodules.

Add PROJECT_PLAN.md to this folder.
