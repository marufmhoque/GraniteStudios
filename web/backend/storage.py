"""Session-scoped temporary file storage for the web service.

Uploads and results live in a per-session directory under a single root and
are swept on a TTL, so a public deployment never accumulates other people's
media. Nothing here is a durable store — the browser is expected to download
anything it wants to keep.
"""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# How long a session's files survive after its last write.
SESSION_TTL_SECONDS = 60 * 60  # 1 hour

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp",
    ".heic", ".heif",
    ".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".orf", ".rw2",
}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


@dataclass(frozen=True)
class StoredFile:
    file_id: str
    session_id: str
    path: Path
    original_name: str
    kind: str  # "image" | "video"


class SessionStorage:
    """Flat per-session directories under one root, swept by TTL."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, StoredFile] = {}

    # -- session helpers ------------------------------------------------- #

    def new_session_id(self) -> str:
        return uuid.uuid4().hex

    def session_dir(self, session_id: str) -> Path:
        # uuid4().hex is hex-only; reject anything else so a caller-supplied
        # session id can never escape the storage root via path traversal.
        if not session_id.isalnum():
            raise ValueError("Invalid session id")
        path = self.root / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    # -- file helpers ---------------------------------------------------- #

    @staticmethod
    def classify(filename: str) -> str:
        ext = Path(filename).suffix.lower()
        if ext in VIDEO_EXTENSIONS:
            return "video"
        if ext in IMAGE_EXTENSIONS:
            return "image"
        raise ValueError(f"Unsupported file type: {ext or filename}")

    def save_upload(self, session_id: str, filename: str, data: bytes) -> StoredFile:
        kind = self.classify(filename)
        ext = Path(filename).suffix.lower()
        file_id = uuid.uuid4().hex
        # The stored name is derived entirely from generated ids, so a hostile
        # upload filename can't influence where bytes land on disk.
        dest = self.session_dir(session_id) / f"{file_id}{ext}"
        dest.write_bytes(data)

        stored = StoredFile(
            file_id=file_id,
            session_id=session_id,
            path=dest,
            original_name=Path(filename).name,
            kind=kind,
        )
        self._index[file_id] = stored
        return stored

    def register(self, session_id: str, path: Path, original_name: str, kind: str) -> StoredFile:
        """Register a file this service produced (an edit result) so it can be
        fetched by id like an upload."""
        file_id = uuid.uuid4().hex
        stored = StoredFile(
            file_id=file_id,
            session_id=session_id,
            path=Path(path),
            original_name=original_name,
            kind=kind,
        )
        self._index[file_id] = stored
        return stored

    def get(self, file_id: str) -> StoredFile | None:
        stored = self._index.get(file_id)
        if stored is None:
            return None
        if not stored.path.exists():
            self._index.pop(file_id, None)
            return None
        return stored

    def output_path(self, session_id: str, stem: str, suffix: str, ext: str) -> Path:
        return self.session_dir(session_id) / f"{stem}_{suffix}{ext}"

    # -- cleanup --------------------------------------------------------- #

    def sweep(self, ttl_seconds: int = SESSION_TTL_SECONDS) -> int:
        """Delete session directories untouched for longer than the TTL."""
        cutoff = time.time() - ttl_seconds
        removed = 0
        for entry in self.root.iterdir():
            if not entry.is_dir():
                continue
            try:
                newest = max(
                    (p.stat().st_mtime for p in entry.rglob("*") if p.is_file()),
                    default=entry.stat().st_mtime,
                )
                if newest < cutoff:
                    shutil.rmtree(entry, ignore_errors=True)
                    removed += 1
            except OSError:
                logger.warning("Could not sweep session dir %s", entry, exc_info=True)

        if removed:
            # Drop index entries whose files went away with the directory.
            for file_id in [fid for fid, f in self._index.items() if not f.path.exists()]:
                self._index.pop(file_id, None)
            logger.info("Swept %d expired session(s)", removed)
        return removed
