"""Where the API reads and writes, and its limits. Every value can be set by environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Methods a job may run: name shown to the client -> (method config file, extra run-config keys).
#   m2n2          EMA detrending + test-time adaptation (the paper's method)
#   detrend_only  EMA detrending, frozen after training  (= the `pretrained` config)
#   static        plain MLP: no detrending, no adaptation (= the paper's MLP baseline, `mlp` config)
DEFAULT_METHODS: dict[str, tuple[str, dict]] = {
    "m2n2": ("m2n2", {}),
    "detrend_only": ("pretrained", {}),
    "static": ("mlp", {}),
}


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    return [s.strip() for s in raw.split(",") if s.strip()] if raw else default


@dataclass
class Settings:
    root: Path = Path(".")                     # project root (holds configs/, data/, results/)
    max_upload_mb: float = 50.0
    max_rows: int = 2_000_000
    upload_ttl_hours: float = 24.0
    max_workers: int = 1                       # jobs run one at a time: CPU-bound
    app_url: str = "http://localhost:8080"     # where a browser opening the API's own address is sent
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"])
    methods: dict[str, tuple[str, dict]] = field(default_factory=lambda: dict(DEFAULT_METHODS))

    @property
    def config_dir(self) -> Path:
        return self.root / "configs"

    @property
    def runs_dir(self) -> Path:
        return self.root / "results" / "runs"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def uploads_dir(self) -> Path:
        """Uploaded files and job results. Kept apart from results/runs/ on purpose:
        runs on uploaded data are not benchmark evidence."""
        return self.root / "results" / "uploads"

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.root = Path(os.environ.get("TSAD_ROOT", ".")).resolve()
        s.max_upload_mb = float(os.environ.get("TSAD_MAX_UPLOAD_MB", s.max_upload_mb))
        s.max_rows = int(os.environ.get("TSAD_MAX_ROWS", s.max_rows))
        s.upload_ttl_hours = float(os.environ.get("TSAD_UPLOAD_TTL_HOURS", s.upload_ttl_hours))
        s.max_workers = int(os.environ.get("TSAD_MAX_WORKERS", s.max_workers))
        s.cors_origins = _env_list("TSAD_CORS_ORIGINS", s.cors_origins)
        s.app_url = os.environ.get("TSAD_APP_URL", s.app_url)
        return s
