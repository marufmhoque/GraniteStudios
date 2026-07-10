"""Universal image loader for Granite Studios.

Handles every input format a working photographer throws at it:
  * DSLR/mirrorless RAW (.nef, .cr2, .arw, .dng, .raf, .orf, .rw2) via rawpy,
    preferring the camera's embedded JPEG thumbnail for high-speed previews.
  * Apple HEIC/HEIF via pillow-heif.
  * Standard JPEG / PNG / TIFF / WebP / BMP via Pillow.

Previews are cached in RAM (bounded LRU) so grid scrolling stays fluid even
on integrated-GPU laptops.
"""

from __future__ import annotations

import io
import logging
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import rawpy
from PIL import Image

import pillow_heif

# Register HEIF/HEIC support with Pillow once, at import time.
pillow_heif.register_heif_opener()

logger = logging.getLogger(__name__)

RAW_EXTENSIONS = {".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".orf", ".rw2"}
HEIF_EXTENSIONS = {".heic", ".heif"}
STANDARD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | HEIF_EXTENSIONS | STANDARD_EXTENSIONS


class _LRUCache:
    """Thread-safe bounded LRU cache for decoded preview images."""

    def __init__(self, max_items: int = 256):
        self._max_items = max_items
        self._data: OrderedDict[str, Image.Image] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Image.Image | None:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key: str, value: Image.Image) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._max_items:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class UniversalLoader:
    """Load previews and full-resolution images from any supported format."""

    def __init__(self, cache_size: int = 256, preview_max_px: int = 1600):
        self._cache = _LRUCache(max_items=cache_size)
        self.preview_max_px = preview_max_px

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @staticmethod
    def is_supported(path: str | Path) -> bool:
        return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS

    def scan_folder(self, folder: str | Path) -> list[Path]:
        """Return all supported image files in a folder, sorted by name."""
        folder = Path(folder)
        if not folder.is_dir():
            return []
        return sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def load_preview(self, path: str | Path) -> Image.Image | None:
        """Load a fast RGB preview (RAM-cached). Ideal for grid thumbnails."""
        path = Path(path)
        key = str(path.resolve())
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        try:
            image = self._decode_preview(path)
        except Exception:
            logger.exception("Failed to load preview: %s", path)
            return None

        if image is None:
            return None

        image = self._downscale(image, self.preview_max_px)
        self._cache.put(key, image)
        return image

    def load_full(self, path: str | Path) -> Image.Image | None:
        """Load the full-resolution RGB image (full RAW demosaic for RAW)."""
        path = Path(path)
        ext = path.suffix.lower()
        try:
            if ext in RAW_EXTENSIONS:
                with rawpy.imread(str(path)) as raw:
                    rgb = raw.postprocess(
                        use_camera_wb=True,
                        no_auto_bright=False,
                        output_bps=8,
                    )
                return Image.fromarray(rgb)
            image = Image.open(path)
            return self._to_rgb(image)
        except Exception:
            logger.exception("Failed to load full image: %s", path)
            return None

    def load_array(self, path: str | Path) -> np.ndarray | None:
        """Load the full image as an RGB numpy array (for the AI pipelines)."""
        image = self.load_full(path)
        if image is None:
            return None
        return np.asarray(image)

    def clear_cache(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _decode_preview(self, path: Path) -> Image.Image | None:
        ext = path.suffix.lower()

        if ext in RAW_EXTENSIONS:
            return self._decode_raw_preview(path)

        # HEIC is registered with Pillow via pillow-heif, so both HEIF and
        # standard formats go through Image.open.
        image = Image.open(path)
        return self._to_rgb(image)

    def _decode_raw_preview(self, path: Path) -> Image.Image | None:
        """Extract the embedded JPEG thumbnail from a RAW file.

        This is 10-50x faster than a full demosaic, which is what makes
        RAM-cached culling viable on modest hardware. Falls back to a
        half-size demosaic if no usable thumbnail is embedded.
        """
        with rawpy.imread(str(path)) as raw:
            try:
                thumb = raw.extract_thumb()
            except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
                thumb = None

            if thumb is not None:
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    image = Image.open(io.BytesIO(thumb.data))
                    return self._to_rgb(image)
                if thumb.format == rawpy.ThumbFormat.BITMAP:
                    return Image.fromarray(thumb.data)

            # Fallback: fast half-size demosaic.
            rgb = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
            return Image.fromarray(rgb)

    @staticmethod
    def _to_rgb(image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    @staticmethod
    def _downscale(image: Image.Image, max_px: int) -> Image.Image:
        if max(image.size) <= max_px:
            return image
        image = image.copy()
        image.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
        return image
