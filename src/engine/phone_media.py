"""Phone-camera formats: paired motion clips, HDR gain maps, deep HEIC.

Phones do not write one file per photograph the way a camera does. This
module reads the extra structure so the rest of the app can treat a phone
frame like any other.

What is here, and how far each part has been checked
----------------------------------------------------

* **Live Photos (Apple).** A still plus a ``.mov`` with the same name.
  Paired by filename. The two files also share a content identifier in
  their metadata; that is *not* checked, so two unrelated files that happen
  to share a stem would be paired.
* **Motion Photos (Google, Samsung).** An MP4 appended to the still. Located
  from the XMP the phone writes, with a structural search of the file tail
  as a fallback. Tested on files built here to the published layout.
* **Ultra HDR gain maps (Android 14+).** A second JPEG holding a per-pixel
  gain, with its parameters in XMP (the ``hdrgm`` namespace). Parsed and
  applied with the published formula. The newer binary ISO 21496-1 metadata
  block is **not** parsed; files that carry only that are reported as having
  no gain map.
* **10/12-bit HEIC.** Decoded to float without first collapsing to 8 bits.

**Not built: Apple ProRAW gain tables.** A ProRAW DNG carries a
``ProfileGainTableMap`` that Apple's own renderer uses for local tone
mapping. LibRaw does not apply it and neither does this module, so a ProRAW
file develops flatter here than in Photos. :func:`has_profile_gain_table`
detects the tag so the interface can say so. Implementing it needs a real
ProRAW file to check the result against, and none was available.
"""

from __future__ import annotations

import io
import logging
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

STILL_EXTENSIONS = {".heic", ".heif", ".jpg", ".jpeg"}
LIVE_VIDEO_EXTENSIONS = {".mov", ".mp4"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".mxf", ".avi"}

_XMP_PACKET = re.compile(rb"<x:xmpmeta.*?</x:xmpmeta>", re.DOTALL)
_PROFILE_GAIN_TABLE_MAP = 0xCD2D


# --------------------------------------------------------------------- #
# XMP helpers
# --------------------------------------------------------------------- #

def _xmp_packets(data: bytes) -> list[str]:
    return [m.group(0).decode("utf-8", "replace") for m in _XMP_PACKET.finditer(data)]


def _xmp_values(packet: str, name: str) -> list[float] | None:
    """Read a numeric XMP property written as an attribute or an rdf:Seq."""
    attr = re.search(rf'{re.escape(name)}\s*=\s*"([^"]*)"', packet)
    if attr:
        try:
            return [float(attr.group(1))]
        except ValueError:
            return None
    element = re.search(rf"<{re.escape(name)}>(.*?)</{re.escape(name)}>", packet, re.DOTALL)
    if element:
        items = re.findall(r"<rdf:li>\s*([^<\s]+)\s*</rdf:li>", element.group(1))
        try:
            return [float(v) for v in (items or [element.group(1).strip()])]
        except ValueError:
            return None
    return None


def _xmp_text(packet: str, name: str) -> str | None:
    attr = re.search(rf'{re.escape(name)}\s*=\s*"([^"]*)"', packet)
    return attr.group(1) if attr else None


def _container_items(packet: str) -> list[dict[str, str]]:
    """The ``Container:Directory`` listing: one dict per appended item."""
    items = []
    for block in re.findall(r"<Container:Item\b(.*?)(?:/>|</Container:Item>)", packet, re.DOTALL):
        items.append(dict(re.findall(r'Item:(\w+)\s*=\s*"([^"]*)"', block)))
    return items


# --------------------------------------------------------------------- #
# Live Photos
# --------------------------------------------------------------------- #

def find_live_photo_pairs(folder: str | Path) -> dict[Path, Path]:
    """Map each still to the clip that shares its filename.

    Matching is by stem, case-insensitively. See the module note: the
    shared content identifier is not verified.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return {}
    stills: dict[str, Path] = {}
    clips: dict[str, Path] = {}
    for path in folder.iterdir():
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in STILL_EXTENSIONS:
            stills.setdefault(path.stem.lower(), path)
        elif suffix in LIVE_VIDEO_EXTENSIONS:
            clips.setdefault(path.stem.lower(), path)
    return {stills[stem]: clips[stem] for stem in sorted(stills) if stem in clips}


# --------------------------------------------------------------------- #
# Motion Photos
# --------------------------------------------------------------------- #

def _looks_like_mp4(data: bytes, offset: int) -> bool:
    """True when an ISO media file plausibly starts at ``offset``."""
    if offset < 0 or offset + 12 > len(data):
        return False
    size = struct.unpack(">I", data[offset:offset + 4])[0]
    return data[offset + 4:offset + 8] == b"ftyp" and 8 <= size <= 4096


def motion_video_range(path: str | Path) -> tuple[int, int] | None:
    """Byte range ``(offset, length)`` of the clip inside a Motion Photo.

    Returns None when the file is an ordinary still.
    """
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    total = len(data)

    for packet in _xmp_packets(data[: min(total, 1 << 20)]):
        if "MotionPhoto" not in packet and "MicroVideo" not in packet:
            continue
        # Current layout: the clip is the last item in the container list.
        for item in _container_items(packet):
            if item.get("Semantic") == "MotionPhoto" and item.get("Length", "").isdigit():
                length = int(item["Length"])
                offset = total - length
                if length and _looks_like_mp4(data, offset):
                    return offset, length
        # Older layout: an offset measured back from the end of the file.
        legacy = _xmp_values(packet, "GCamera:MicroVideoOffset")
        if legacy:
            length = int(legacy[0])
            offset = total - length
            if length and _looks_like_mp4(data, offset):
                return offset, length

    # No usable XMP: look for a media file sitting after the still.
    position = data.rfind(b"ftyp")
    while position > 4:
        if _looks_like_mp4(data, position - 4) and data[: position - 4].rstrip(b"\x00").endswith(b"\xff\xd9"):
            return position - 4, total - (position - 4)
        position = data.rfind(b"ftyp", 0, position)
    return None


def extract_motion_video(path: str | Path, destination: str | Path) -> Path | None:
    """Write a Motion Photo's clip to ``destination``. The still is only read."""
    found = motion_video_range(path)
    if found is None:
        return None
    offset, length = found
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "rb") as source:
        source.seek(offset)
        destination.write_bytes(source.read(length))
    return destination


# --------------------------------------------------------------------- #
# Ultra HDR gain maps
# --------------------------------------------------------------------- #

@dataclass
class GainMap:
    """A decoded gain map and the parameters needed to apply it.

    The numeric fields are log2 values, one per colour channel (a single
    value is repeated three times), exactly as the format stores them.
    """

    image: np.ndarray                      # H x W or H x W x 3, float32, 0..1
    gain_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gain_max: tuple[float, float, float] = (1.0, 1.0, 1.0)
    gamma: tuple[float, float, float] = (1.0, 1.0, 1.0)
    offset_sdr: tuple[float, float, float] = (1 / 64, 1 / 64, 1 / 64)
    offset_hdr: tuple[float, float, float] = (1 / 64, 1 / 64, 1 / 64)
    capacity_min: float = 0.0
    capacity_max: float = 1.0
    base_is_hdr: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def max_boost(self) -> float:
        """Largest linear brightening the map can ask for."""
        return float(2.0 ** max(self.gain_max))


def _triple(values: list[float] | None, default: float) -> tuple[float, float, float]:
    if not values:
        return (default, default, default)
    if len(values) >= 3:
        return (values[0], values[1], values[2])
    return (values[0], values[0], values[0])


def read_gain_map(path: str | Path) -> GainMap | None:
    """Read the gain map from an Ultra HDR JPEG, or None if it has none."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None

    packets = _xmp_packets(data)
    params = next((p for p in packets if "hdrgm:GainMapMax" in p), None)
    if params is None:
        return None

    # The gain map is an appended JPEG; the primary image's container list
    # gives its length, counted back from the end of the file.
    map_bytes: bytes | None = None
    for packet in packets:
        for item in _container_items(packet):
            if item.get("Semantic") == "GainMap" and item.get("Length", "").isdigit():
                length = int(item["Length"])
                if 0 < length < len(data):
                    map_bytes = data[len(data) - length:]
    try:
        if map_bytes is not None:
            decoded = Image.open(io.BytesIO(map_bytes))
            decoded.load()
        else:
            # No container entry: fall back to the multi-picture index.
            decoded = Image.open(io.BytesIO(data))
            if getattr(decoded, "n_frames", 1) < 2:
                return None
            decoded.seek(1)
            decoded.load()
    except Exception as exc:
        logger.debug("Gain map in %s did not decode: %s", path, exc)
        return None

    if decoded.mode not in ("L", "RGB"):
        decoded = decoded.convert("RGB")
    image = np.asarray(decoded, dtype=np.float32) / 255.0

    capacity_max = _xmp_values(params, "hdrgm:HDRCapacityMax")
    gain_max = _xmp_values(params, "hdrgm:GainMapMax")
    result = GainMap(
        image=image,
        gain_min=_triple(_xmp_values(params, "hdrgm:GainMapMin"), 0.0),
        gain_max=_triple(gain_max, 1.0),
        gamma=_triple(_xmp_values(params, "hdrgm:Gamma"), 1.0),
        offset_sdr=_triple(_xmp_values(params, "hdrgm:OffsetSDR"), 1 / 64),
        offset_hdr=_triple(_xmp_values(params, "hdrgm:OffsetHDR"), 1 / 64),
        capacity_min=(_xmp_values(params, "hdrgm:HDRCapacityMin") or [0.0])[0],
        capacity_max=(capacity_max or [max(gain_max or [1.0])])[0],
        base_is_hdr=(_xmp_text(params, "hdrgm:BaseRenditionIsHDR") or "False").lower() == "true",
    )
    if result.base_is_hdr:
        result.notes.append("base rendition is HDR; only SDR-base files are applied")
    return result


def srgb_to_linear(encoded: np.ndarray) -> np.ndarray:
    encoded = np.clip(encoded, 0.0, 1.0)
    return np.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4).astype(np.float32)


def apply_gain_map(sdr_linear: np.ndarray, gain_map: GainMap, display_boost: float | None = None) -> np.ndarray:
    """Reconstruct the HDR rendition from the SDR image and its gain map.

    ``sdr_linear`` is linear-light RGB, 0..1. ``display_boost`` is how many
    times brighter than SDR white the target can show; ``None`` means the
    full range the file was authored for. The result is linear and may
    exceed 1.0 -- that headroom is the point.

    Follows the published Ultra HDR formula: the stored value is undone
    through gamma, mapped between the minimum and maximum gain in log2
    space, scaled by how much of the file's range the display can use, and
    applied with the two offsets that keep the division stable near black.
    """
    sdr = np.asarray(sdr_linear, dtype=np.float32)
    h, w = sdr.shape[:2]

    recovery = gain_map.image
    if recovery.shape[:2] != (h, w):
        import cv2

        recovery = cv2.resize(recovery, (w, h), interpolation=cv2.INTER_LINEAR)
    if recovery.ndim == 2:
        recovery = recovery[:, :, None]

    gamma = np.asarray(gain_map.gamma, np.float32)
    low = np.asarray(gain_map.gain_min, np.float32)
    high = np.asarray(gain_map.gain_max, np.float32)

    log_recovery = np.power(np.clip(recovery, 0.0, 1.0), 1.0 / gamma)
    log_boost = low * (1.0 - log_recovery) + high * log_recovery

    span = gain_map.capacity_max - gain_map.capacity_min
    if display_boost is None or span <= 0:
        weight = 1.0
    else:
        weight = float(np.clip((np.log2(max(display_boost, 1.0)) - gain_map.capacity_min) / span, 0.0, 1.0))

    offset_sdr = np.asarray(gain_map.offset_sdr, np.float32)
    offset_hdr = np.asarray(gain_map.offset_hdr, np.float32)
    hdr = (sdr + offset_sdr) * np.exp2(log_boost * weight) - offset_hdr
    return np.maximum(hdr, 0.0).astype(np.float32)


def load_hdr(path: str | Path, display_boost: float | None = None) -> np.ndarray | None:
    """Linear float RGB for an Ultra HDR file, with its gain map applied.

    Returns None for a file with no gain map, so the caller can fall back to
    the ordinary loader.
    """
    gain_map = read_gain_map(path)
    if gain_map is None or gain_map.base_is_hdr:
        return None
    with Image.open(path) as image:
        base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return apply_gain_map(srgb_to_linear(base), gain_map, display_boost)


# --------------------------------------------------------------------- #
# Deep HEIC
# --------------------------------------------------------------------- #

def heif_bit_depth(path: str | Path) -> int | None:
    try:
        import pillow_heif

        return int(pillow_heif.open_heif(str(path), convert_hdr_to_8bit=False).info.get("bit_depth", 8))
    except Exception:
        return None


def load_heif_float(path: str | Path) -> tuple[np.ndarray, int] | None:
    """Decode a HEIC to float32 RGB in 0..1 at its own bit depth.

    Returns ``(array, bit_depth)``. The values are still gamma-encoded, as
    stored; nothing here guesses a transfer function.
    """
    try:
        import pillow_heif

        heif = pillow_heif.open_heif(str(path), convert_hdr_to_8bit=False)
        depth = int(heif.info.get("bit_depth", 8))
        array = np.asarray(heif)
    except Exception as exc:
        logger.debug("HEIC decode failed for %s: %s", path, exc)
        return None
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    array = array[:, :, :3]
    scale = 65535.0 if array.dtype == np.uint16 else 255.0
    return (array.astype(np.float32) / scale), depth


# --------------------------------------------------------------------- #
# ProRAW
# --------------------------------------------------------------------- #

def has_profile_gain_table(path: str | Path) -> bool:
    """True when a DNG carries Apple's ProfileGainTableMap.

    Detection only. See the module note: the table is not applied.
    """
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as tif:
            pages = list(tif.pages)
            for page in pages:
                pages.extend(getattr(page, "pages", None) or [])
            return any(_PROFILE_GAIN_TABLE_MAP in page.tags for page in pages)
    except Exception:
        return False


# --------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------- #

@dataclass
class MediaTraits:
    """What is unusual about one file, for the interface to show."""

    motion_video: bool = False
    live_photo_clip: Path | None = None
    gain_map: bool = False
    max_boost: float | None = None
    bit_depth: int | None = None
    profile_gain_table: bool = False

    @property
    def labels(self) -> list[str]:
        out = []
        if self.live_photo_clip is not None:
            out.append("Live Photo")
        if self.motion_video:
            out.append("Motion Photo")
        if self.gain_map:
            out.append(f"HDR gain map (up to {self.max_boost:.1f}x)" if self.max_boost else "HDR gain map")
        if self.bit_depth and self.bit_depth > 8:
            out.append(f"{self.bit_depth}-bit")
        if self.profile_gain_table:
            out.append("ProRAW gain table (not applied)")
        return out


def inspect(path: str | Path, live_pairs: dict[Path, Path] | None = None) -> MediaTraits:
    """Describe a file's phone-specific structure. Never raises."""
    path = Path(path)
    suffix = path.suffix.lower()
    traits = MediaTraits()
    try:
        if live_pairs:
            traits.live_photo_clip = live_pairs.get(path)
        if suffix in (".jpg", ".jpeg"):
            traits.motion_video = motion_video_range(path) is not None
            gain = read_gain_map(path)
            if gain is not None:
                traits.gain_map, traits.max_boost = True, gain.max_boost
        elif suffix in (".heic", ".heif"):
            traits.bit_depth = heif_bit_depth(path)
            traits.motion_video = motion_video_range(path) is not None
        elif suffix == ".dng":
            traits.profile_gain_table = has_profile_gain_table(path)
    except Exception:
        logger.debug("Could not inspect %s", path, exc_info=True)
    return traits
