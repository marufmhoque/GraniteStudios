"""Universal image loader for Granite Studios.

Handles every input format a working photographer throws at it:
  * DSLR/mirrorless RAW (.nef, .cr2, .cr3, .arw, .dng, .raf, .orf, .rw2,
    .pef, .3fr) via rawpy. Previews come from src/engine/fast_preview.py,
    which prefers the camera's embedded JPEG over developing the sensor data.
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
from datetime import datetime
from pathlib import Path

import numpy as np
import rawpy
from PIL import Image, ImageOps

import pillow_heif

# Register HEIF/HEIC support with Pillow once, at import time.
pillow_heif.register_heif_opener()

from src.engine import fast_preview  # noqa: E402

logger = logging.getLogger(__name__)

RAW_EXTENSIONS = fast_preview.RAW_EXTENSIONS
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
_EXIF_BODY_SERIAL = 0xA431
_EXIF_LENS_SERIAL = 0xA435
_EXIF_FOCAL_LENGTH = 0x920A
_EXIF_FOCAL_35MM = 0xA405
_EXIF_OFFSET_TIME = 0x9011      # UTC offset of DateTimeOriginal


@dataclass
class ImageMetadata:
    """Camera/lens/exposure info extracted from an image's EXIF."""

    camera_make: str | None = None
    camera_model: str | None = None
    lens_model: str | None = None
    iso: int | None = None
    aperture: float | None = None
    shutter_speed: str | None = None
    # Added in 2.0. All optional: older callers construct this positionally.
    focal_length: float | None = None
    focal_length_35mm: int | None = None
    body_serial: str | None = None
    lens_serial: str | None = None
    captured_at: datetime | None = None
    utc_offset: str | None = None
    color_profile: str | None = None

    @property
    def focal_label(self) -> str:
        if self.focal_length is None:
            return ""
        value = ("%.1f" % self.focal_length).rstrip("0").rstrip(".")
        return f"{value}mm"

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
            "focal_length": self.focal_label,
            "captured_at": self.captured_at.isoformat() if self.captured_at else "",
            "color_profile": self.color_profile or "",
            "body_serial": self.body_serial or "",
            "lens_serial": self.lens_serial or "",
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

    def scan_videos(self, folder: str | Path) -> list[Path]:
        """Return every video file in a folder, sorted by name."""
        from src.video.ffmpeg_utils import VIDEO_EXTENSIONS

        folder = Path(folder)
        if not folder.is_dir():
            return []
        return sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )

    def live_photo_pairs(self, folder: str | Path) -> dict[Path, Path]:
        """Stills in a folder mapped to the clip that shares their name."""
        from src.engine import phone_media

        return phone_media.find_live_photo_pairs(folder)

    def media_traits(self, path: str | Path, live_pairs: dict | None = None):
        """Phone-specific structure of one file (gain map, motion clip, ...)."""
        from src.engine import phone_media

        return phone_media.inspect(path, live_pairs)

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
            # Honour the EXIF orientation flag, so a phone photo taken in
            # portrait is exported upright rather than on its side.
            image = ImageOps.exif_transpose(Image.open(path))
            return self._to_rgb(image)
        except Exception:
            logger.exception("Failed to load full image: %s", path)
            return None

    def load_thumbnail(self, path: str | Path, size: int = 320) -> Image.Image | None:
        """A small preview for the grid, as fast as the file allows.

        Accepts whatever preview the file carries, however small, so a
        folder of phone DNGs fills the grid without developing each frame.
        Not cached: the analysis-size preview is a different image.
        """
        preview = fast_preview.load_preview(path, max_side=size, min_side=0)
        return preview.to_pil() if preview is not None else None

    def load_analysis(self, path: str | Path, max_side: int = 1600) -> np.ndarray | None:
        """An upright RGB array sized for analysis rather than display."""
        preview = fast_preview.load_preview(path, max_side=max_side)
        return preview.image if preview is not None else None

    def load_array(self, path: str | Path) -> np.ndarray | None:
        """Load the full image as an RGB numpy array (for the AI pipelines)."""
        image = self.load_full(path)
        if image is None:
            return None
        return np.asarray(image)

    def clear_cache(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------ #
    # Edit subfolder — keeps originals (especially RAW) untouched
    # ------------------------------------------------------------------ #

    def get_edit_folder(self, folder: str | Path, create: bool = True) -> Path:
        """Return the ``<FolderName>-edit`` subfolder for a loaded folder.

        Every edited JPEG, resized export, and "keeper" (see
        :meth:`export_keeper`) is written here instead of next to the
        originals — RAW files and everything else in the source folder are
        never modified or moved.
        """
        folder = Path(folder)
        edit_folder = folder / f"{folder.name}-edit"
        if create:
            edit_folder.mkdir(parents=True, exist_ok=True)
        return edit_folder

    def export_keeper(
        self,
        source: str | Path,
        folder: str | Path,
        quality: int = 95,
    ) -> Path:
        """Flatten ``source`` into a full-resolution JPEG in the edit folder.

        Idempotent — if a keeper JPEG for this source already exists, it is
        reused rather than re-decoded (cheap to call repeatedly, e.g. once
        per "Publish to Web" toggle). This is what makes a RAW file usable
        by every downstream export (color grading, IG formatting, web
        publishing) without ever writing next to the original.
        """
        source = Path(source)
        edit_folder = self.get_edit_folder(folder)
        dest = edit_folder / f"{source.stem}.jpg"
        if not dest.exists():
            image = self.load_full(source)
            if image is None:
                raise IOError(f"Could not load image: {source}")
            image.save(dest, "JPEG", quality=quality)
        return dest

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

        focal_35 = sub.get(_EXIF_FOCAL_35MM)
        return ImageMetadata(
            camera_make=make,
            camera_model=model,
            lens_model=lens,
            iso=iso,
            aperture=aperture,
            shutter_speed=shutter,
            focal_length=self._clean_float(sub.get(_EXIF_FOCAL_LENGTH)),
            focal_length_35mm=int(focal_35) if isinstance(focal_35, (int, float)) and focal_35 else None,
            body_serial=self._clean_str(sub.get(_EXIF_BODY_SERIAL)),
            lens_serial=self._clean_str(sub.get(_EXIF_LENS_SERIAL)),
            captured_at=fast_preview.read_capture_time(img),
            utc_offset=self._clean_str(sub.get(_EXIF_OFFSET_TIME)),
            color_profile=self._profile_name(img),
        )

    @staticmethod
    def _profile_name(img: Image.Image) -> str | None:
        """Name of the embedded ICC profile, e.g. "Display P3"."""
        icc = img.info.get("icc_profile")
        if not icc:
            return None
        try:
            from PIL import ImageCms

            profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            return (ImageCms.getProfileDescription(profile) or "").strip() or None
        except Exception:
            return None

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
        preview = fast_preview.load_preview(path, max_side=self.preview_max_px)
        return preview.to_pil() if preview is not None else None

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
