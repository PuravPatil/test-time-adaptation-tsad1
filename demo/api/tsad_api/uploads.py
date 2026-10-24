"""Uploaded files: saved under results/uploads/files/<id>/, inspected once, expired after a TTL."""
from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

from audit.data.upload import FORMATS, UploadError, inspect, read_table

from .settings import Settings
from .util import safe_id

CHUNK = 1 << 20


class UploadTooLarge(UploadError):
    pass


class UploadStore:
    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def files_dir(self) -> Path:
        return self.settings.uploads_dir / "files"

    @property
    def jobs_dir(self) -> Path:
        return self.settings.uploads_dir / "jobs"

    async def save(self, upload) -> dict:
        """Stream a FastAPI UploadFile to disk with a size cap, then read and inspect it."""
        original = Path(upload.filename or "upload").name
        ext = Path(original).suffix.lower()
        if ext not in FORMATS:
            raise UploadError(f"unsupported file type '{ext or '(none)'}'. Supported: {', '.join(FORMATS)}")
        self.cleanup()
        upload_id = uuid.uuid4().hex
        folder = self.files_dir / upload_id
        folder.mkdir(parents=True)
        dest = folder / f"data{ext}"
        limit = int(self.settings.max_upload_mb * 1024 * 1024)
        size = 0
        try:
            with open(dest, "wb") as f:
                while chunk := await upload.read(CHUNK):
                    size += len(chunk)
                    if size > limit:
                        raise UploadTooLarge(f"the file is over the {self.settings.max_upload_mb:g} MB limit")
                    f.write(chunk)
            info = inspect(read_table(dest), max_rows=self.settings.max_rows)
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        meta = {"upload_id": upload_id, "filename": original, "format": ext, "size_bytes": size,
                "created": time.time(), **info}
        (folder / "meta.json").write_text(json.dumps(meta, default=str))
        return meta

    def get(self, upload_id: str) -> dict | None:
        if not safe_id(upload_id):
            return None
        meta = self.files_dir / upload_id / "meta.json"
        return json.loads(meta.read_text()) if meta.exists() else None

    def data_path(self, upload_id: str) -> Path | None:
        meta = self.get(upload_id)
        return self.files_dir / upload_id / f"data{meta['format']}" if meta else None

    def cleanup(self) -> int:
        """Delete uploads and job results older than the TTL. Returns how many were removed."""
        cutoff = time.time() - self.settings.upload_ttl_hours * 3600
        removed = 0
        for parent in (self.files_dir, self.jobs_dir):
            if not parent.exists():
                continue
            for d in parent.iterdir():
                if d.is_dir() and safe_id(d.name) and d.stat().st_mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
                    removed += 1
        return removed
