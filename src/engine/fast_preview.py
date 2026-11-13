"""Fast previews for culling and analysis.

Every camera RAW carries a JPEG the camera rendered at capture time. Pulling
that out is two orders of magnitude cheaper than demosaicing the sensor data,
and it is what the camera's own screen shows, so it is the right thing to
cull on.

What the measurements say (eleven CC0 samples, ``tests/fetch_media.py``):

* *Finding* the embedded preview costs 1-10 ms.
* *Decoding* it is the real cost: 30-120 ms at full size for a 15-30 MP
  preview. JPEG can be decoded at 1/2, 1/4 or 1/8 scale inside the DCT,
  which roughly halves that.
* Not every embedded preview is usable. Phone DNGs and some medium-format
  backs carry only a postage stamp (160 px wide in one sample). Those fall
  through to a half-size demosaic, and :class:`Preview` records that it
  happened, so a slow frame is explicable rather than mysterious.

Two decode modes, because speed and measurement pull in different directions
-----------------------------------------------------------------------------

Reduced-size decoding is a low-pass filter. Measured on five cameras, it cut
the Laplacian sharpness score of a *sharp* frame to 0.4-0.7x of the score
from a full decode, while barely moving a blurred frame. Every blur threshold
in this project was calibrated on "decode in full, shrink once with area
averaging", so feeding reduced decodes to those thresholds would flag good
photographs as soft.

* ``exact=False`` (default) -- reduced-size decode. For anything a person
  looks at or that only compares frames loosely: grid thumbnails, perceptual
  hashes, face detection.
* ``exact=True`` -- full decode, one area-averaged shrink. For anything
  compared against an absolute threshold: blur triage and sharpness scoring.

Order of attempts for a RAW file:

1. LibRaw's embedded-thumbnail extractor.
2. A scan of the file for JPEG streams (covers files LibRaw cannot open).
3. A half-size demosaic.
"""

from __future__ import annotations

import io
import logging
import mmap
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import rawpy
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

RAW_EXTENSIONS = {
    ".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef", ".3fr",
}

# Where a preview came from. Reported, so timings can be explained.
EMBEDDED_JPEG = "embedded_jpeg"
EMBEDDED_BITMAP = "embedded_bitmap"
SCANNED_JPEG = "scanned_jpeg"
DEMOSAIC_HALF = "demosaic_half"
DECODED = "decoded"

# An embedded preview narrower than this is too small to judge focus on.
DEFAULT_MIN_SIDE = 900
DEFAULT_MAX_SIDE = 1600

_SOI = b"\xff\xd8\xff"
_MAX_SCAN_CANDIDATES = 24

_EXIF_SUBIFD = 0x8769
_EXIF_DATETIMES = (0x9003, 0x9004)   # DateTimeOriginal, DateTimeDigitized
_EXIF_DATETIME = 0x0132              # DateTime, in the base IFD
_EXIF_ORIENTATION = 0x0112

# LibRaw's ``flip`` field -> PIL transpose that puts the frame upright.
_FLIP_TO_TRANSPOSE = {
    3: Image.Transpose.ROTATE_180,
    5: Image.Transpose.ROTATE_90,
    6: Image.Transpose.ROTATE_270,
}


@dataclass
class Preview:
    """An RGB preview and a record of how it was obtained."""

    image: np.ndarray                 # H x W x 3, uint8, RGB, upright
    source: str
    native_size: tuple[int, int]      # (w, h) of the preview before scaling
    extract_ms: float = 0.0
    decode_ms: float = 0.0
    captured_at: datetime | None = None   # from the preview's own EXIF

    @property
    def total_ms(self) -> float:
        return self.extract_ms + self.decode_ms

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h

    def to_pil(self) -> Image.Image:
        return Image.fromarray(self.image)

    def fitted(self, max_side: int) -> np.ndarray:
        """The image shrunk to ``max_side`` with a single area average."""
        return fit(self.image, max_side)


# --------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------- #

def fit(array: np.ndarray, max_side: int | None) -> np.ndarray:
    """Shrink so the long edge is at most ``max_side``. Never enlarges."""
    if not max_side:
        return array
    h, w = array.shape[:2]
    if max(h, w) <= max_side:
        return array
    scale = max_side / float(max(h, w))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(array, size, interpolation=cv2.INTER_AREA)


def _draft_size(native: tuple[int, int], max_side: int) -> tuple[int, int]:
    """The size to ask ``Image.draft`` for.

    ``draft`` picks the largest power-of-two reduction that keeps *both*
    dimensions at or above the request. Asking for a square therefore lets
    the short side veto the reduction; asking in the image's own aspect
    ratio gets the reduction the long side allows.
    """
    w, h = native
    scale = max_side / float(max(w, h))
    return max(1, int(w * scale)), max(1, int(h * scale))


def read_capture_time(image: Image.Image) -> datetime | None:
    """Capture time from an opened image's EXIF, or None."""
    try:
        exif = image.getexif()
        candidates = []
        try:
            sub = exif.get_ifd(_EXIF_SUBIFD)
            candidates += [sub.get(tag) for tag in _EXIF_DATETIMES]
        except Exception:
            pass
        candidates.append(exif.get(_EXIF_DATETIME))
        for value in candidates:
            if not value:
                continue
            try:
                return datetime.strptime(str(value).strip("\x00 "), "%Y:%m:%d %H:%M:%S")
            except ValueError:
                continue
    except Exception:
        pass
    return None


def _open_upright(
    source, max_side: int | None, exact: bool
) -> tuple[Image.Image, tuple[int, int], datetime | None, bool]:
    """Open an image upright in RGB.

    Returns ``(image, native_size, captured_at, had_orientation_tag)``.
    """
    image = Image.open(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source)
    native = image.size
    captured = read_capture_time(image)
    try:
        tagged = image.getexif().get(_EXIF_ORIENTATION, 1) not in (None, 0, 1)
    except Exception:
        tagged = False
    if not exact and max_side and image.format == "JPEG" and max(native) > max_side:
        image.draft("RGB", _draft_size(native, max_side))
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    turned = (image.size[0] >= image.size[1]) != (native[0] >= native[1])
    return image, (native[::-1] if turned else native), captured, tagged


def _apply_raw_flip(image: Image.Image, flip: int, raw_landscape: bool) -> Image.Image:
    """Turn a preview upright using the RAW's own orientation flag.

    Some cameras store the preview already turned and some do not, and the
    flag alone cannot tell them apart. For quarter turns the shape settles
    it: if the preview is already the same way round as the turned sensor
    frame, it has been turned and is left alone.

    Unverified: every sample this was tested against was shot in landscape
    (``flip == 0``), so this branch has not seen a real portrait RAW.
    """
    transpose = _FLIP_TO_TRANSPOSE.get(flip)
    if transpose is None:
        return image
    if flip in (5, 6):
        preview_landscape = image.size[0] >= image.size[1]
        upright_landscape = not raw_landscape
        if preview_landscape == upright_landscape:
            return image
    return image.transpose(transpose)


# --------------------------------------------------------------------- #
# Finding a JPEG inside a file LibRaw cannot read
# --------------------------------------------------------------------- #

def _jpeg_size_at(view, offset: int) -> tuple[int, int] | None:
    """Read a JPEG's dimensions by walking its header segments.

    Returns None for anything that is not a decodable baseline or
    progressive JPEG. That deliberately excludes lossless JPEG (marker C3),
    which is how many RAW formats store the sensor data itself, and rejects
    the stray start-of-image byte patterns that occur inside compressed data.
    """
    end = len(view)
    pos = offset + 2
    for _ in range(96):
        if pos + 4 > end or view[pos] != 0xFF:
            return None
        marker = view[pos + 1]
        if marker == 0xFF:          # fill byte
            pos += 1
            continue
        if marker in (0xC0, 0xC1, 0xC2):
            if pos + 9 > end:
                return None
            height = int.from_bytes(view[pos + 5:pos + 7], "big")
            width = int.from_bytes(view[pos + 7:pos + 9], "big")
            return (width, height) if width and height else None
        if marker in (0x00, 0x01, 0xC3, 0xD8, 0xD9, 0xDA) or 0xD0 <= marker <= 0xD7:
            return None
        length = int.from_bytes(view[pos + 2:pos + 4], "big")
        if length < 2:
            return None
        pos += 2 + length
    return None


def scan_for_jpeg(path: str | Path, min_side: int = 0) -> bytes | None:
    """Find the largest JPEG stream stored inside a file.

    The fallback for RAW files LibRaw cannot parse. Nearly every RAW
    container stores its preview as an ordinary JPEG stream, so looking for
    start-of-image markers and keeping the biggest real one is enough.
    """
    best: tuple[int, int] | None = None  # (area, offset)
    try:
        with open(path, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as view:
            offset, found = 0, 0
            while found < _MAX_SCAN_CANDIDATES:
                offset = view.find(_SOI, offset)
                if offset < 0:
                    break
                size = _jpeg_size_at(view, offset)
                if size is not None:
                    found += 1
                    area = size[0] * size[1]
                    if max(size) >= min_side and (best is None or area > best[0]):
                        best = (area, offset)
                offset += 3
            if best is None:
                return None
            # Hand over the rest of the file; the decoder stops at the
            # stream's own end marker. Searching for that marker here would
            # stop early at the end of a nested thumbnail.
            return bytes(view[best[1]:])
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------- #
# RAW
# --------------------------------------------------------------------- #

def _elapsed(since: float) -> float:
    return (time.perf_counter() - since) * 1e3


def _scanned_preview(
    path: Path, max_side: int | None, min_side: int, exact: bool, started: float,
    captured: datetime | None = None,
) -> Preview | None:
    data = scan_for_jpeg(path, min_side=min_side)
    extract_ms = _elapsed(started)
    if data is None:
        return None
    decoding = time.perf_counter()
    try:
        image, native, found, _ = _open_upright(data, max_side, exact)
    except Exception:
        return None
    array = fit(np.asarray(image), max_side)
    return Preview(array, SCANNED_JPEG, native, extract_ms, _elapsed(decoding), found or captured)


def _raw_preview(path: Path, max_side: int | None, min_side: int, exact: bool) -> Preview | None:
    started = time.perf_counter()
    try:
        raw = rawpy.imread(str(path))
    except Exception as exc:
        logger.debug("LibRaw could not open %s (%s); scanning for a JPEG.", path.name, exc)
        return _scanned_preview(path, max_side, min_side, exact, started)

    with raw:
        flip = int(getattr(raw.sizes, "flip", 0) or 0)
        raw_landscape = raw.sizes.raw_width >= raw.sizes.raw_height
        thumb = None
        try:
            thumb = raw.extract_thumb()
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            thumb = None
        except Exception as exc:  # a corrupt thumbnail must not sink the frame
            logger.debug("Thumbnail extraction failed for %s: %s", path.name, exc)
        extract_ms = _elapsed(started)
        decoding = time.perf_counter()
        captured = None

        if thumb is not None and thumb.format == rawpy.ThumbFormat.JPEG:
            try:
                image, native, captured, tagged = _open_upright(thumb.data, max_side, exact)
                if max(native) >= min_side:
                    if not tagged:
                        image = _apply_raw_flip(image, flip, raw_landscape)
                    array = fit(np.asarray(image), max_side)
                    return Preview(array, EMBEDDED_JPEG, native, extract_ms, _elapsed(decoding), captured)
            except Exception as exc:
                logger.debug("Embedded JPEG in %s did not decode: %s", path.name, exc)

        if thumb is not None and thumb.format == rawpy.ThumbFormat.BITMAP:
            bitmap = np.asarray(thumb.data)
            native = (bitmap.shape[1], bitmap.shape[0])
            if max(native) >= min_side:
                image = _apply_raw_flip(Image.fromarray(bitmap), flip, raw_landscape)
                array = fit(np.asarray(image), max_side)
                return Preview(array, EMBEDDED_BITMAP, native, extract_ms, _elapsed(decoding))

        # No usable preview: develop the sensor data at half size. LibRaw
        # applies the orientation itself on this path.
        try:
            rgb = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
        except Exception as exc:
            logger.debug("Half-size demosaic failed for %s: %s", path.name, exc)
            return _scanned_preview(path, max_side, 0, exact, started, captured)
        # A preview too small to cull on still knows when it was taken.
        return Preview(
            fit(rgb, max_side), DEMOSAIC_HALF, (rgb.shape[1], rgb.shape[0]),
            extract_ms, _elapsed(decoding), captured,
        )


def _container_capture_time(path: Path) -> datetime | None:
    """Capture time from the RAW file's own tags.

    Phone DNGs and some medium-format files carry a preview with no EXIF of
    its own. Most RAW containers are TIFF underneath, so the tags can be
    read from the file header without decoding anything.
    """
    try:
        with Image.open(path) as container:
            return read_capture_time(container)
    except Exception:
        return None


def _plain_preview(path: Path, max_side: int | None, exact: bool) -> Preview | None:
    started = time.perf_counter()
    image, native, captured, _ = _open_upright(path, max_side, exact)
    array = fit(np.asarray(image), max_side)
    return Preview(array, DECODED, native, 0.0, _elapsed(started), captured)


# --------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------- #

def load_preview(
    path: str | Path,
    max_side: int | None = DEFAULT_MAX_SIDE,
    min_side: int = DEFAULT_MIN_SIDE,
    exact: bool = False,
) -> Preview | None:
    """Return an upright RGB preview no larger than ``max_side``.

    ``max_side=None`` keeps the preview at its native size. ``min_side`` is
    the smallest embedded preview worth accepting; anything smaller is
    passed over for a half-size demosaic. Pass 0 when any preview will do,
    as for a grid thumbnail. ``exact`` selects the measurement-grade decode
    described in the module docstring.

    Returns None when the file cannot be read at all. It never raises.
    """
    path = Path(path)
    try:
        if path.suffix.lower() in RAW_EXTENSIONS:
            floor = min(min_side, max_side) if max_side else min_side
            preview = _raw_preview(path, max_side, floor, exact)
            if preview is not None and preview.captured_at is None:
                preview.captured_at = _container_capture_time(path)
            return preview
        return _plain_preview(path, max_side, exact)
    except Exception as exc:
        logger.debug("No preview for %s: %s", path.name, exc)
        return None


def default_workers() -> int:
    """Threads for a batch.

    Decoding releases the GIL, so threads do scale. One core is left for
    the interface, and the count is capped by free memory because an exact
    decode of a 30 MP preview holds about 90 MB while it is being measured.
    """
    workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    try:
        import psutil

        free_gb = psutil.virtual_memory().available / (1024 ** 3)
        workers = max(1, min(workers, int(free_gb / 0.35)))
    except Exception:
        pass
    return workers


def map_previews(
    paths: Iterable[str | Path],
    fn: Callable[[str, Preview | None], object],
    max_side: int | None = DEFAULT_MAX_SIDE,
    min_side: int = DEFAULT_MIN_SIDE,
    exact: bool = False,
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Decode each file and run ``fn(path, preview)`` inside the same worker.

    Results come back keyed by ``str(path)`` in input order. No more than
    ``workers`` previews are alive at once, which is what lets a
    thousand-frame cull run in bounded memory. ``fn`` runs on a worker
    thread and must not touch the interface.
    """
    ordered = [str(p) for p in paths]
    total = len(ordered)
    results: dict[str, object] = {}

    def work(key: str):
        return fn(key, load_preview(key, max_side, min_side, exact))

    with ThreadPoolExecutor(max_workers=workers or default_workers()) as pool:
        for done, (key, value) in enumerate(zip(ordered, pool.map(work, ordered)), start=1):
            results[key] = value
            if progress:
                progress(done, total)
    return results


def load_previews(
    paths: Iterable[str | Path],
    max_side: int | None = DEFAULT_MAX_SIDE,
    min_side: int = DEFAULT_MIN_SIDE,
    exact: bool = False,
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Preview | None]:
    """Preview many files at once. Holds every result; use
    :func:`map_previews` for a large batch."""
    return map_previews(  # type: ignore[return-value]
        paths, lambda _key, preview: preview, max_side, min_side, exact, workers, progress
    )
