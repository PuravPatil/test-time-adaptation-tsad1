from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml


def load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def save_json(path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def git_commit() -> str:
    """Current commit hash, marked -dirty if there are uncommitted changes."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
