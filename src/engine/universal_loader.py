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
from dataclasses import dataclass
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

# EXIF numeric tag ids. Using raw ids keeps us independent of the evolving
# names in Pillow's ExifTags enums across versions.
_EXIF_MAKE = 0x010F
_EXIF_MODEL = 0x0110
_EXIF_SUBIFD = 0x8769  # pointer to the Exif sub-IFD
_EXIF_EXPOSURE_TIME = 0x829A
_EXIF_FNUMBER = 0x829D
_EXIF_ISO = 0x8827
_EXIF_LENS_MODEL = 0xA434


@dataclass
class ImageMetadata:
    """Camera/lens/exposure info extracted from an image's EXIF."""

    camera_make: str | None = None
    camera_model: str | None = None
    lens_model: str | None = None
    iso: int | None = None
    aperture: float | None = None
    shutter_speed: str | None = None

    @property
    def camera_label(self) -> str:
        """Clean camera name, e.g. "Nikon D3400" (make prefix de-duplicated)."""
        make = (self.camera_make or "").strip()
        model = (self.camera_model or "").strip()
        make_short = make.split()[0].title() if make else ""
        if model:
            trimmed = model
            if make_short and trimmed.upper().startswith(make_short.upper()):
                trimmed = trimmed[len(make_short):].strip()
            return f"{make_short} {trimmed}".strip() if make_short else trimmed
        return make_short

    @property
    def lens_label(self) -> str:
        """Lens description, e.g. "18-55mm" (empty when unknown)."""
        return (self.lens_model or "").strip()

    @property
    def aperture_label(self) -> str:
        if self.aperture is None:
            return ""
        value = ("%.1f" % self.aperture).rstrip("0").rstrip(".")
        return f"f/{value}"

    @property
    def settings_label(self) -> str:
        """Compact exposure triangle, e.g. "ISO 200 · f/5.6 · 1/250s"."""
        parts: list[str] = []
        if self.iso is not None:
            parts.append(f"ISO {self.iso}")
        if self.aperture is not None:
            parts.append(self.aperture_label)
        if self.shutter_speed:
            parts.append(self.shutter_speed)
        return " · ".join(parts)

    @property
    def gear_label(self) -> str:
        """Camera + lens on one line, e.g. "Nikon D3400 | 18-55mm"."""
        pieces = [p for p in (self.camera_label, self.lens_label) if p]
        return " | ".join(pieces)

    def as_dict(self) -> dict:
        return {
            "camera": self.camera_label,
            "lens": self.lens_label,
            "settings": self.settings_label,
            "gear": self.gear_label,
        }


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
    # Metadata / EXIF
    # ------------------------------------------------------------------ #

    def read_metadata(self, path: str | Path) -> ImageMetadata:
        """Extract camera, lens, ISO, aperture and shutter speed from EXIF.

        Standard and HEIC images are read directly. RAW files carry EXIF in
        their embedded JPEG thumbnail, so we read the metadata from there.
        Always returns an ``ImageMetadata`` — empty if nothing is available.
        """
        path = Path(path)
        try:
            if path.suffix.lower() in RAW_EXTENSIONS:
                return self._metadata_from_raw(path)
            with Image.open(path) as img:
                return self._extract_exif(img)
        except Exception:
            logger.debug("Metadata read failed for %s", path, exc_info=True)
            return ImageMetadata()

    def _metadata_from_raw(self, path: Path) -> ImageMetadata:
        with rawpy.imread(str(path)) as raw:
            try:
                thumb = raw.extract_thumb()
            except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
                thumb = None
            data = thumb.data if (thumb and thumb.format == rawpy.ThumbFormat.JPEG) else None
        if data is None:
            return ImageMetadata()
        with Image.open(io.BytesIO(data)) as img:
            return self._extract_exif(img)

    def _extract_exif(self, img: Image.Image) -> ImageMetadata:
        exif = img.getexif()
        if not exif:
            return ImageMetadata()

        make = self._clean_str(exif.get(_EXIF_MAKE))
        model = self._clean_str(exif.get(_EXIF_MODEL))

        try:
            sub = exif.get_ifd(_EXIF_SUBIFD)
        except Exception:
            sub = {}

        lens = self._clean_str(sub.get(_EXIF_LENS_MODEL))
        iso = self._clean_iso(sub.get(_EXIF_ISO))
        aperture = self._clean_float(sub.get(_EXIF_FNUMBER))
        shutter = self._format_shutter(sub.get(_EXIF_EXPOSURE_TIME))

        return ImageMetadata(
            camera_make=make,
            camera_model=model,
            lens_model=lens,
            iso=iso,
            aperture=aperture,
            shutter_speed=shutter,
        )

    @staticmethod
    def _clean_str(value) -> str | None:
        if value is None:
            return None
        text = str(value).strip().rstrip("\x00").strip()
        return text or None

    @staticmethod
    def _clean_iso(value) -> int | None:
        if isinstance(value, (tuple, list)):
            value = value[0] if value else None
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _clean_float(value) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_shutter(value) -> str | None:
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            return None
        if seconds <= 0:
            return None
        if seconds >= 1:
            return f"{seconds:g}s"
        return f"1/{round(1 / seconds)}s"

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
