from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml


def load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def save_json(path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def git_commit() -> str:
    """Current commit hash, marked -dirty if tracked files have staged or
    unstaged changes. Where git is unavailable (e.g. inside Docker, which has
    no git and no .git), falls back to the GIT_COMMIT environment variable."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        text=True, stderr=subprocess.DEVNULL).strip()
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, OSError):
        return os.environ.get("GIT_COMMIT", "unknown")
