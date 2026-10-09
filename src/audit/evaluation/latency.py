"""Per-step inference latency on CPU. Owner: Uno.

Neither paper reports latency. The harness wraps a detector's score() call
and records wall-clock time per test timestep.

TODO (weeks 4-6): add per-step timing inside the adapters' streaming loop
(warm-up excluded) so the number reflects one timestep, not the whole batch.
"""
from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timer(result: dict, key: str = "seconds"):
    start = time.perf_counter()
    yield
    result[key] = time.perf_counter() - start


def ms_per_step(total_seconds: float, n_steps: int) -> float:
    return 1000.0 * total_seconds / max(n_steps, 1)
