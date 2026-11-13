"""Fetching and verifying the ONNX models the AI features depend on.

The models are not bundled with the app: together they are tens of megabytes,
and most of them are only needed if the matching feature is used. Each one is
downloaded once, verified, and cached under the user's home directory.

Downloads come from OpenCV's model zoo, which keeps them in git-lfs. A plain
raw URL therefore returns a short pointer file rather than the model, so the
lfs batch API is asked for a signed URL to the real object. The pointer file
is also where the pinned digest and size below come from, which is why a
substituted or truncated download is rejected rather than handed to the DNN
loader.

Every fetch is content-addressed: the digest is the lfs object id, so asking
the batch API for a digest and then checking what came back against the same
digest is self-consistent. A mismatch means the download is wrong, not that
the pin is stale.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_DIR = Path.home() / ".granitestudios" / "models"

_LFS_BATCH_URL = "https://github.com/opencv/opencv_zoo.git/info/lfs/objects/batch"
_MEDIA_BASE = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"

_USER_AGENT = "GraniteStudios/1.0"


@dataclass(frozen=True)
class ModelSpec:
    """One downloadable model, pinned by digest.

    ``zoo_dir`` is the directory inside opencv_zoo's ``models/`` tree, used to
    build the direct media URL. ``label`` is what the user is told about when
    the model is missing.

    ``url`` is for the few models that are not in opencv_zoo. When it is set
    the file is fetched from exactly that address and nowhere else. The
    digest check is identical either way, so a different host changes where
    the bytes come from, not how far they are trusted.
    """

    filename: str
    sha256: str
    size: int
    zoo_dir: str
    label: str
    url: str | None = None

    @property
    def path(self) -> Path:
        return MODEL_DIR / self.filename

    @property
    def media_url(self) -> str:
        return f"{_MEDIA_BASE}/{self.zoo_dir}/{self.filename}"


def _fetch(url: str, timeout: int, data: bytes | None = None, headers: dict | None = None) -> bytes:
    request = urllib.request.Request(
        url, data=data, headers=headers or {"User-Agent": _USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _lfs_download_url(spec: ModelSpec) -> str | None:
    """Ask GitHub's git-lfs batch API where the real object lives."""
    payload = json.dumps(
        {
            "operation": "download",
            "transfers": ["basic"],
            "objects": [{"oid": spec.sha256, "size": spec.size}],
        }
    ).encode()
    try:
        raw = _fetch(
            _LFS_BATCH_URL,
            timeout=30,
            data=payload,
            headers={
                "Content-Type": "application/vnd.git-lfs+json",
                "Accept": "application/vnd.git-lfs+json",
                "User-Agent": "git-lfs/3.0.0",
            },
        )
        return json.loads(raw)["objects"][0]["actions"]["download"]["href"]
    except Exception as exc:
        logger.debug("LFS batch lookup failed for %s: %s", spec.filename, exc)
        return None


def _matches(data: bytes, spec: ModelSpec) -> bool:
    return len(data) == spec.size and hashlib.sha256(data).hexdigest() == spec.sha256


def verify(spec: ModelSpec) -> bool:
    """True when the cached file on disk is the model we pinned."""
    try:
        return _matches(spec.path.read_bytes(), spec)
    except OSError:
        return False


def is_cached(spec: ModelSpec) -> bool:
    return spec.path.exists() and verify(spec)


def ensure_model(spec: ModelSpec, download: bool = True, timeout: int = 300) -> Path | None:
    """Return a verified path for ``spec``, downloading it once if needed.

    Returns None when the model is absent and cannot be fetched. Callers must
    treat the feature as unavailable in that case rather than raising, so the
    app degrades the same way it does when any other optional engine is
    missing.
    """
    dest = spec.path
    if dest.exists():
        if verify(spec):
            return dest
        logger.warning("Cached %s failed verification; refetching.", spec.filename)
        try:
            dest.unlink()
        except OSError:
            return None

    if not download:
        return None

    if spec.url:
        candidates = [spec.url]
    else:
        candidates = [spec.media_url]
        lfs = _lfs_download_url(spec)
        if lfs:
            # Signed LFS URL first: it is the only one that works when the
            # media host is unreachable, which is common behind proxies.
            candidates.insert(0, lfs)

    for url in candidates:
        try:
            data = _fetch(url, timeout=timeout)
        except Exception as exc:
            logger.debug("Download failed from %s: %s", url.split("?")[0], exc)
            continue
        if not _matches(data, spec):
            logger.warning("Rejected %s from %s: checksum mismatch.", spec.filename, url.split("?")[0])
            continue
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        except OSError as exc:
            logger.warning("Could not save %s: %s", spec.filename, exc)
            return None
        logger.info("Downloaded and verified %s.", spec.label)
        return dest

    logger.info("%s unavailable — the features that need it will be skipped.", spec.label)
    return None


def ensure_models(specs, download: bool = True, timeout: int = 300) -> dict[str, Path] | None:
    """Fetch several models, returning None unless *every* one succeeded.

    A pipeline that needs two models is no use with one of them, so this is
    all-or-nothing rather than leaving the caller to check each path.
    """
    paths: dict[str, Path] = {}
    for spec in specs:
        path = ensure_model(spec, download=download, timeout=timeout)
        if path is None:
            return None
        paths[spec.filename] = path
    return paths


def total_download_mb(specs) -> float:
    """Megabytes that would be fetched for the specs not already cached."""
    return sum(s.size for s in specs if not is_cached(s)) / (1024.0 * 1024.0)
