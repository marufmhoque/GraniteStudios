"""The develop pipeline: a photograph plus a recipe gives a rendered image.

Nothing here writes to the source file, and nothing is written at all until
:func:`export` is called. Rendering is a pure function of the pixels and the
recipe, so the same recipe gives the same result however many times, and in
whatever order, its settings were changed.

How it works
------------
Pixels are processed as 32-bit floats. A RAW file is developed to 16-bit
linear light and stays linear through white balance, exposure and lens
vignetting, where the arithmetic is physically meaningful. It is then
encoded for display and the perceptual adjustments (tone, presence, colour)
are applied there.

Two properties the design is built around:

* **A neutral recipe changes nothing.** A JPEG rendered with no settings
  comes back identical. The highlight roll-off, the only step that could
  bend an untouched image, is the identity when no gain has been applied.
* **The preview is the export.** Every large-radius effect (clarity, the
  shadow and highlight masks, dehaze) is computed on a small copy of the
  frame and scaled up. The preview and the full-size export use the same
  small copy, so they agree; and because no step needs a wide neighbourhood
  at full size, the export can be rendered in horizontal strips that fit in
  memory on a modest machine.

Speed
-----
Most of the work on a pixel depends only on that pixel's own value: decode,
white balance, exposure, highlight roll-off, encode, levels, contrast and
curves. For 8- and 16-bit sources all of that is folded into one lookup
table per channel, computed with the exact formulas for every possible
input value. That is what makes dragging a slider responsive on a modest
laptop, and because the table is exact it is also what the export uses.
Only effects that look at neighbouring pixels run as image operations.

What it is not
--------------
This is this app's own rendering. It is not Lightroom's, and a recipe here
will not match the same numbers typed into another editor. "GPU
accelerated" is not claimed: this is vectorised NumPy and OpenCV on the CPU.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.engine import fast_preview
from src.engine.parametric_recipe import HSL_CENTRES, Geometry, Recipe

logger = logging.getLogger(__name__)

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

CONTEXT_SIDE = 768          # the small copy large-radius effects are built on
STRIP_ROWS = 512
KNEE = 0.6                  # highlight roll-off starts here
CURVE_SAMPLES = 4096
WORK_SAMPLES = 65536


# --------------------------------------------------------------------- #
# Transfer functions
# --------------------------------------------------------------------- #

def srgb_decode(encoded: np.ndarray) -> np.ndarray:
    encoded = np.asarray(encoded, dtype=np.float32)
    low = encoded / 12.92
    high = np.power(np.maximum(encoded + 0.055, 0.0) / 1.055, 2.4)
    return np.where(encoded <= 0.04045, low, high).astype(np.float32)


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(np.asarray(linear, dtype=np.float32), 0.0, 1.0)
    low = linear * 12.92
    high = 1.055 * np.power(linear, 1.0 / 2.4) - 0.055
    return np.where(linear <= 0.0031308, low, high).astype(np.float32)


_DECODE_8 = srgb_decode(np.arange(256, dtype=np.float32) / 255.0)
_DECODE_16: np.ndarray | None = None


def roll_off(linear: np.ndarray, peak: float) -> np.ndarray:
    """Compress values up to ``peak`` into 0..1, leaving shadows alone.

    Below the knee the image is untouched. Above it, the range up to
    ``peak`` is folded smoothly into what is left, so pushing exposure
    brightens the midtones by the full amount while highlights bend rather
    than clip. With ``peak <= 1`` this is exactly the identity, which is
    what keeps an unedited image unchanged.
    """
    if peak <= 1.0 + 1e-6:
        return linear
    span = 1.0 - KNEE
    top = (peak - KNEE) / span
    u = np.maximum(linear - KNEE, 0.0) / span
    folded = u * (1.0 + u / (top * top)) / (1.0 + u)
    return np.where(linear > KNEE, KNEE + span * folded, linear).astype(np.float32)


# --------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------- #

@dataclass
class Source:
    """A photograph's pixels, as loaded, before any editing.

    ``data`` is H x W x 3. ``linear`` says whether the values are
    linear-light or already encoded for display.
    """

    data: np.ndarray
    linear: bool
    path: Path | None = None
    is_raw: bool = False
    note: str | None = None      # set when the load was a fallback

    @property
    def size(self) -> tuple[int, int]:
        return self.data.shape[1], self.data.shape[0]

    def to_linear(self, rows: slice | None = None) -> np.ndarray:
        """Linear float32 for the whole frame or a band of rows."""
        global _DECODE_16
        block = self.data if rows is None else self.data[rows]
        if block.dtype == np.uint8:
            return _DECODE_8[block] if not self.linear else block.astype(np.float32) / 255.0
        if block.dtype == np.uint16:
            if self.linear:
                return block.astype(np.float32) / 65535.0
            if _DECODE_16 is None:
                _DECODE_16 = srgb_decode(np.arange(65536, dtype=np.float32) / 65535.0)
            return _DECODE_16[block]
        block = block.astype(np.float32, copy=False)
        return block if self.linear else srgb_decode(block)

    def proxy(self, max_side: int) -> "Source":
        """A smaller copy, for interactive preview."""
        return Source(fast_preview.fit(self.data, max_side), self.linear, self.path, self.is_raw, self.note)


def from_array(array: np.ndarray, linear: bool = False) -> Source:
    """Wrap pixels already in memory (RGB, uint8/uint16/float)."""
    array = np.asarray(array)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    return Source(np.ascontiguousarray(array[:, :, :3]), linear)


def open_source(path: str | Path, half_size: bool = False, hdr: bool = False) -> Source | None:
    """Load a photograph for developing. Returns None if it cannot be read.

    RAW files are demosaiced to 16-bit linear. If the decoder does not
    support the camera, the embedded JPEG is used instead and ``note`` says
    so, so the frame can still be edited. ``hdr`` applies an Ultra HDR gain
    map when the file has one.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix in fast_preview.RAW_EXTENSIONS:
            import rawpy

            try:
                with rawpy.imread(str(path)) as raw:
                    rgb = raw.postprocess(
                        gamma=(1, 1), no_auto_bright=True, output_bps=16,
                        use_camera_wb=True, half_size=half_size,
                        output_color=rawpy.ColorSpace.sRGB,
                    )
                return Source(rgb, True, path, True)
            except Exception as exc:
                preview = fast_preview.load_preview(path, max_side=None, exact=True)
                if preview is None:
                    return None
                return Source(
                    preview.image, False, path, False,
                    f"the RAW decoder does not support this file ({type(exc).__name__}); "
                    "editing its embedded preview instead",
                )

        if suffix in (".heic", ".heif"):
            from src.engine import phone_media

            deep = phone_media.load_heif_float(path)
            if deep is not None and deep[1] > 8:
                return Source(deep[0], False, path)

        if hdr and suffix in (".jpg", ".jpeg"):
            from src.engine import phone_media

            boosted = phone_media.load_hdr(path)
            if boosted is not None:
                return Source(boosted, True, path, note="HDR gain map applied")

        if suffix in (".png", ".tif", ".tiff"):
            # Pillow reduces 16-bit colour to 8; OpenCV keeps it. Read via
            # a buffer so paths with non-ASCII characters work on Windows.
            deep = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if deep is not None and deep.dtype == np.uint16:
                if deep.ndim == 2:
                    deep = np.stack([deep] * 3, axis=-1)
                else:
                    deep = cv2.cvtColor(deep[:, :, :3], cv2.COLOR_BGR2RGB)
                return Source(np.ascontiguousarray(deep), False, path)

        from PIL import Image, ImageOps

        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            return Source(np.asarray(image.convert("RGB")), False, path)
    except Exception as exc:
        logger.debug("Could not open %s for developing: %s", path, exc)
        return None


# --------------------------------------------------------------------- #
# Curves
# --------------------------------------------------------------------- #

def curve_lut(points: list, samples: int = CURVE_SAMPLES) -> np.ndarray:
    """A monotone curve through the control points, sampled on 0..1.

    Uses a shape-preserving cubic, so the curve never overshoots between
    points the way an ordinary spline does. Two points give a straight line.
    """
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    grid = np.linspace(0.0, 1.0, samples)
    if len(xs) < 3:
        return np.interp(grid, xs, ys).astype(np.float32)
    from scipy.interpolate import PchipInterpolator

    inner = PchipInterpolator(xs, ys, extrapolate=False)(np.clip(grid, xs[0], xs[-1]))
    inner = np.where(grid < xs[0], ys[0], np.where(grid > xs[-1], ys[-1], inner))
    return np.clip(inner, 0.0, 1.0).astype(np.float32)


def _is_identity(points: list) -> bool:
    return all(abs(p[0] - p[1]) < 1e-6 for p in points)


def tone_lut(recipe: Recipe, samples: int = CURVE_SAMPLES) -> np.ndarray | None:
    """Levels, contrast and the master curve folded into one table.

    Returns None when all three are neutral, so the step can be skipped.
    """
    b = recipe.basic
    if not (b.whites or b.blacks or b.contrast) and _is_identity(recipe.curve.master):
        return None
    x = np.linspace(0.0, 1.0, samples, dtype=np.float64)

    # Levels: blacks moves the black point, whites the white point.
    black = 0.12 * (-b.blacks / 100.0)
    white = 1.0 - 0.20 * (b.whites / 100.0)
    y = np.clip((x - black) / max(white - black, 1e-3), 0.0, 1.0)

    # Contrast: a symmetric power curve about middle grey.
    if b.contrast:
        amount = 0.6 * b.contrast / 100.0
        power = 1.0 / (1.0 + amount) if amount > 0 else 1.0 - amount * 2.5
        centred = 2.0 * y - 1.0
        y = 0.5 + 0.5 * np.sign(centred) * np.power(np.abs(centred), power)

    if not _is_identity(recipe.curve.master):
        y = np.interp(y, x, curve_lut(recipe.curve.master, samples))
    return y.astype(np.float32)


def _apply_lut(values: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Look ``values`` (0..1) up in a table sampled evenly on 0..1.

    The working tables have 65,536 entries, so the nearest entry is within
    a hundredth of an 8-bit step and no interpolation is needed.
    """
    index = np.clip(values, 0.0, 1.0) * (len(lut) - 1) + 0.5
    return lut[index.astype(np.int32)]


def _fine(lut: np.ndarray | None) -> np.ndarray | None:
    """Resample a curve to the 65,536-entry working size."""
    if lut is None or len(lut) == WORK_SAMPLES:
        return lut
    grid = np.linspace(0.0, 1.0, WORK_SAMPLES)
    return np.interp(grid, np.linspace(0.0, 1.0, len(lut)), lut).astype(np.float32)


# --------------------------------------------------------------------- #
# 3D LUTs
# --------------------------------------------------------------------- #

@dataclass
class CubeLut:
    """A 3D colour lookup table in the ``.cube`` format."""

    table: np.ndarray                      # N x N x N x 3, indexed [b, g, r]
    domain_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    domain_max: tuple[float, float, float] = (1.0, 1.0, 1.0)
    title: str = ""

    @property
    def size(self) -> int:
        return self.table.shape[0]

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """Trilinear lookup. ``rgb`` is float, H x W x 3."""
        low = np.asarray(self.domain_min, np.float32)
        span = np.maximum(np.asarray(self.domain_max, np.float32) - low, 1e-6)
        position = np.clip((rgb - low) / span, 0.0, 1.0) * (self.size - 1)
        base = np.minimum(position.astype(np.int32), self.size - 2)
        frac = position - base
        r0, g0, b0 = base[..., 0], base[..., 1], base[..., 2]
        fr, fg, fb = frac[..., 0:1], frac[..., 1:2], frac[..., 2:3]
        t = self.table
        c00 = t[b0, g0, r0] * (1 - fr) + t[b0, g0, r0 + 1] * fr
        c01 = t[b0, g0 + 1, r0] * (1 - fr) + t[b0, g0 + 1, r0 + 1] * fr
        c10 = t[b0 + 1, g0, r0] * (1 - fr) + t[b0 + 1, g0, r0 + 1] * fr
        c11 = t[b0 + 1, g0 + 1, r0] * (1 - fr) + t[b0 + 1, g0 + 1, r0 + 1] * fr
        return ((c00 * (1 - fg) + c01 * fg) * (1 - fb) + (c10 * (1 - fg) + c11 * fg) * fb).astype(np.float32)


def parse_cube(text: str) -> CubeLut:
    """Parse the text of a ``.cube`` file. Raises ``ValueError`` if invalid."""
    size = None
    low, high, title = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), ""
    rows: list[tuple[float, float, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key = line.split()[0].upper()
        if key == "TITLE":
            title = line[5:].strip().strip('"')
        elif key == "LUT_3D_SIZE":
            size = int(line.split()[1])
        elif key == "LUT_1D_SIZE":
            raise ValueError("1D LUTs are not supported; this needs a 3D .cube file")
        elif key == "DOMAIN_MIN":
            low = tuple(float(v) for v in line.split()[1:4])
        elif key == "DOMAIN_MAX":
            high = tuple(float(v) for v in line.split()[1:4])
        elif key[0].isdigit() or key[0] in "-+.":
            parts = line.split()
            if len(parts) >= 3:
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if not size or size < 2:
        raise ValueError("missing or invalid LUT_3D_SIZE")
    if len(rows) != size ** 3:
        raise ValueError(f"expected {size ** 3} entries for size {size}, found {len(rows)}")
    # The file lists red changing fastest, then green, then blue.
    table = np.asarray(rows, dtype=np.float32).reshape(size, size, size, 3)
    return CubeLut(table, low, high, title)


_cube_cache: dict[tuple[str, float], CubeLut | None] = {}


def load_cube(path: str | Path) -> CubeLut | None:
    """Read a ``.cube`` file, or None if it is missing or malformed."""
    path = Path(path)
    try:
        key = (str(path), path.stat().st_mtime)
    except OSError:
        return None
    if key not in _cube_cache:
        try:
            _cube_cache[key] = parse_cube(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError) as exc:
            logger.warning("Could not use LUT %s: %s", path.name, exc)
            _cube_cache[key] = None
    return _cube_cache[key]


# --------------------------------------------------------------------- #
# Colour helpers
# --------------------------------------------------------------------- #

def white_balance_gains(temperature: float, tint: float) -> np.ndarray:
    """Per-channel gains for a relative temperature/tint shift.

    Warmer raises red and lowers blue; a positive tint lowers green
    (magenta). The gains are scaled so a neutral grey keeps its luminance.
    """
    t, k = temperature / 100.0, tint / 100.0
    gains = np.array([2.0 ** (0.45 * t), 2.0 ** (-0.30 * k), 2.0 ** (-0.45 * t)], dtype=np.float32)
    return gains / float(gains @ LUMA)


def _hue_table(values: list, scale: float) -> np.ndarray:
    """A per-degree table interpolating eight band values round the circle."""
    centres = np.array(HSL_CENTRES + (360.0,), dtype=np.float64)
    amounts = np.array(list(values) + [values[0]], dtype=np.float64) * scale
    return np.interp(np.arange(361, dtype=np.float64), centres, amounts).astype(np.float32)


def _hue_to_rgb(hue_deg: float) -> np.ndarray:
    pixel = np.array([[[hue_deg, 1.0, 1.0]]], dtype=np.float32)
    return cv2.cvtColor(pixel, cv2.COLOR_HSV2RGB)[0, 0]


# --------------------------------------------------------------------- #
# Large-radius effects, built once on a small copy
# --------------------------------------------------------------------- #

def _box_guided(guide: np.ndarray, source: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Edge-preserving smoothing of ``source`` following ``guide``."""
    size = (2 * radius + 1, 2 * radius + 1)
    mean = lambda a: cv2.boxFilter(a, cv2.CV_32F, size)  # noqa: E731
    mean_g, mean_s = mean(guide), mean(source)
    cov = mean(guide * source) - mean_g * mean_s
    var = mean(guide * guide) - mean_g * mean_g
    a = cov / (var + eps)
    b = mean_s - a * mean_g
    return mean(a) * guide + mean(b)


@dataclass
class Context:
    """Everything the full-size pass needs that was measured on the small copy."""

    full_size: tuple[int, int]
    peak: float = 1.0
    light_blur: np.ndarray | None = None      # pre-tone luminance, blurred
    tone_blur: np.ndarray | None = None       # post-tone luminance, blurred
    transmission: np.ndarray | None = None
    airlight: np.ndarray | None = None
    vignette_gain: np.ndarray | None = None   # from a lens profile
    tone: np.ndarray | None = None
    channel_curves: list = field(default_factory=list)
    hue_shift: np.ndarray | None = None
    hue_saturation: np.ndarray | None = None
    hue_luminance: np.ndarray | None = None
    cube: CubeLut | None = None
    # Per-channel tables from source code value straight to display value.
    point_luts: list | None = None
    tone_folded: bool = False     # True when point_luts already include tone

    def upsample(self, small: np.ndarray, rows: slice, width: int) -> np.ndarray:
        """A band of a small map, scaled up to the full frame's pixels."""
        full_w, full_h = self.full_size
        h, w = small.shape[:2]
        y0, y1 = rows.start, rows.stop
        ys = (np.arange(y0, y1, dtype=np.float32) + 0.5) * (h / full_h) - 0.5
        xs = (np.arange(width, dtype=np.float32) + 0.5) * (w / full_w) - 0.5
        map_x, map_y = np.meshgrid(xs, ys)
        return cv2.remap(small, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _linear_stage(linear: np.ndarray, recipe: Recipe, ctx: Context, rows: slice) -> np.ndarray:
    """White balance, exposure and vignetting, in linear light."""
    b = recipe.basic
    height, width = linear.shape[:2]

    if ctx.vignette_gain is not None:
        linear = linear * ctx.upsample(ctx.vignette_gain, rows, width)[:, :, None]
    if recipe.lens.vignette:
        full_w, full_h = ctx.full_size
        ys = (np.arange(rows.start, rows.stop, dtype=np.float32) - full_h / 2.0) / (full_h / 2.0)
        xs = (np.arange(width, dtype=np.float32) - full_w / 2.0) / (full_w / 2.0)
        r2 = (xs[None, :] ** 2 + ys[:, None] ** 2) / 2.0
        linear = linear * (1.0 + (recipe.lens.vignette / 100.0) * 0.9 * r2)[:, :, None]

    gain = np.float32(2.0 ** b.exposure)
    if b.temperature or b.tint:
        linear = linear * (white_balance_gains(b.temperature, b.tint) * gain)
    elif b.exposure:
        linear = linear * gain
    return linear


def _dark_channel(encoded: np.ndarray, radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
    return cv2.erode(encoded.min(axis=2), kernel)


def _code_values(source: Source) -> np.ndarray | None:
    """Every value an integer source can hold, as linear light; else None."""
    global _DECODE_16
    if source.data.dtype == np.uint8:
        return _DECODE_8.copy() if not source.linear else np.arange(256, dtype=np.float32) / 255.0
    if source.data.dtype == np.uint16:
        ramp = np.arange(65536, dtype=np.float32) / 65535.0
        if source.linear:
            return ramp
        if _DECODE_16 is None:
            _DECODE_16 = srgb_decode(ramp)
        return _DECODE_16.copy()
    return None


def _uses_position(recipe: Recipe, ctx: Context) -> bool:
    """True when the linear stage depends on where a pixel is."""
    return ctx.vignette_gain is not None or bool(recipe.lens.vignette)


def _spatial_before_tone(recipe: Recipe) -> bool:
    b = recipe.basic
    return bool(b.dehaze or b.highlights or b.shadows or recipe.detail.noise_reduction)


def _build_point_luts(source: Source, recipe: Recipe, ctx: Context, fold_tone: bool) -> list | None:
    """Tables taking a source code value to its display value, per channel."""
    linear = _code_values(source)
    if linear is None or _uses_position(recipe, ctx):
        return None
    b = recipe.basic
    gains = white_balance_gains(b.temperature, b.tint) if (b.temperature or b.tint) else np.ones(3, np.float32)
    gains = gains * np.float32(2.0 ** b.exposure)
    tables = []
    for channel in range(3):
        encoded = srgb_encode(roll_off(linear * gains[channel], ctx.peak))
        if fold_tone:
            if ctx.tone is not None:
                encoded = _apply_lut(encoded, ctx.tone)
            if ctx.channel_curves[channel] is not None:
                encoded = _apply_lut(encoded, ctx.channel_curves[channel])
        tables.append(encoded.astype(np.float32))
    return tables


def _encode_rows(source: Source, recipe: Recipe, ctx: Context, rows: slice) -> np.ndarray:
    """Source rows to display-encoded float, before any neighbourhood effect."""
    if ctx.point_luts is not None:
        block = source.data[rows]
        return np.stack([ctx.point_luts[c][block[:, :, c]] for c in range(3)], axis=-1)
    linear = _linear_stage(source.to_linear(rows), recipe, ctx, rows)
    return srgb_encode(roll_off(linear, ctx.peak))


def prepare(source: Source, recipe: Recipe) -> Context:
    """Measure the frame on a small copy and build every lookup table."""
    ctx = Context(full_size=source.size)
    b = recipe.basic
    small = source.proxy(CONTEXT_SIDE)
    rows = slice(0, small.data.shape[0])
    small_ctx = Context(full_size=small.size)

    if recipe.lens.profile and source.path is not None:
        ctx.vignette_gain = lens_vignette_gain(source.path, small.size)
        small_ctx.vignette_gain = ctx.vignette_gain

    # How bright the frame gets after gain decides the highlight roll-off.
    # It is measured on the small copy so preview and export agree.
    gained = _linear_stage(small.to_linear(), recipe, small_ctx, rows)
    ctx.peak = small_ctx.peak = max(1.0, float(gained.max()))

    ctx.tone = _fine(tone_lut(recipe))
    ctx.channel_curves = [
        None if _is_identity(points) else _fine(curve_lut(points))
        for points in (recipe.curve.red, recipe.curve.green, recipe.curve.blue)
    ]

    fold = not _spatial_before_tone(recipe)
    ctx.point_luts = _build_point_luts(source, recipe, ctx, fold)
    ctx.tone_folded = fold and ctx.point_luts is not None

    if not (b.dehaze or b.highlights or b.shadows or b.clarity):
        _prepare_colour(recipe, ctx)
        return ctx          # nothing below needs the small copy rendered

    encoded = srgb_encode(roll_off(gained, ctx.peak))
    long_side = max(encoded.shape[:2])

    if b.dehaze > 0:
        radius = max(2, long_side // 60)
        dark = _dark_channel(encoded, radius)
        count = max(1, dark.size // 1000)
        brightest = np.argpartition(dark.reshape(-1), -count)[-count:]
        airlight = np.maximum(encoded.reshape(-1, 3)[brightest].mean(axis=0), 0.2)
        omega = 0.95 * b.dehaze / 100.0
        raw_t = 1.0 - omega * _dark_channel(encoded / airlight, radius)
        guide = encoded @ LUMA
        ctx.transmission = np.clip(_box_guided(guide, raw_t.astype(np.float32), radius * 4, 1e-3), 0.1, 1.0)
        ctx.airlight = airlight.astype(np.float32)
        encoded = _dehaze(encoded, ctx.transmission, ctx.airlight)
    elif b.dehaze < 0:
        encoded = _add_haze(encoded, b.dehaze)

    if b.highlights or b.shadows:
        sigma = max(1.0, long_side / 25.0)
        ctx.light_blur = cv2.GaussianBlur(encoded @ LUMA, (0, 0), sigma)

    if b.clarity:
        small_ctx.light_blur, small_ctx.tone = ctx.light_blur, ctx.tone
        small_ctx.channel_curves = ctx.channel_curves
        toned = _tone_stage(encoded, recipe, small_ctx, rows)
        ctx.tone_blur = cv2.GaussianBlur(toned @ LUMA, (0, 0), max(1.0, long_side / 40.0))

    _prepare_colour(recipe, ctx)
    return ctx


def _prepare_colour(recipe: Recipe, ctx: Context) -> None:
    hsl = recipe.hsl
    if any(hsl.hue):
        ctx.hue_shift = _hue_table(hsl.hue, 0.30)
    if any(hsl.saturation):
        ctx.hue_saturation = _hue_table(hsl.saturation, 0.01)
    if any(hsl.luminance):
        ctx.hue_luminance = _hue_table(hsl.luminance, 0.01)
    if recipe.lut.path and recipe.lut.strength > 0:
        ctx.cube = load_cube(recipe.lut.path)


def _only_point_operations(recipe: Recipe, ctx: Context) -> bool:
    """True when every active setting is a per-value curve.

    Then the whole render is one table lookup per channel, with no float
    image ever built.
    """
    b, grade, detail = recipe.basic, recipe.grade, recipe.detail
    return (
        ctx.tone_folded
        and not (b.clarity or b.texture or b.vibrance or b.saturation)
        and ctx.hue_shift is None and ctx.hue_saturation is None and ctx.hue_luminance is None
        and not any(w.saturation or w.luminance for w in (grade.shadows, grade.midtones, grade.highlights))
        and not (recipe.look.matrix is not None and recipe.look.strength > 0)
        and ctx.cube is None
        and not (detail.sharpen_amount or detail.noise_reduction)
    )


def _dehaze(encoded: np.ndarray, transmission: np.ndarray, airlight: np.ndarray) -> np.ndarray:
    return (encoded - airlight) / transmission[:, :, None] + airlight


def _add_haze(encoded: np.ndarray, amount: float) -> np.ndarray:
    veil = 0.5 * (-amount / 100.0)
    return encoded * (1.0 - veil) + 0.85 * veil


# --------------------------------------------------------------------- #
# Per-pixel stages
# --------------------------------------------------------------------- #

def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _tone_stage(encoded: np.ndarray, recipe: Recipe, ctx: Context, rows: slice) -> np.ndarray:
    """Shadows and highlights, then levels, contrast and curves."""
    b = recipe.basic
    width = encoded.shape[1]
    if ctx.light_blur is not None and (b.highlights or b.shadows):
        light = ctx.upsample(ctx.light_blur, rows, width)[:, :, None]
        e = np.clip(encoded, 0.0, 1.0)
        bend = e * (1.0 - e)        # zero at black and white, so neither clips
        if b.shadows:
            encoded = encoded + 0.9 * (b.shadows / 100.0) * _smoothstep(1.0 - light / 0.5) * bend
        if b.highlights:
            encoded = encoded + 0.9 * (b.highlights / 100.0) * _smoothstep((light - 0.5) / 0.5) * bend
    if ctx.tone is not None:
        encoded = _apply_lut(encoded, ctx.tone)
    for channel, lut in enumerate(ctx.channel_curves):
        if lut is not None:
            encoded[:, :, channel] = _apply_lut(encoded[:, :, channel], lut)
    return encoded


def _presence_stage(encoded: np.ndarray, recipe: Recipe, ctx: Context, rows: slice, long_side: int) -> np.ndarray:
    """Clarity and texture: local contrast at two scales, on luminance."""
    b = recipe.basic
    if not (b.clarity or b.texture):
        return encoded
    luma = encoded @ LUMA
    midtone = 1.0 - np.square(2.0 * np.clip(luma, 0.0, 1.0) - 1.0)
    delta = np.zeros_like(luma)
    if b.clarity and ctx.tone_blur is not None:
        base = ctx.upsample(ctx.tone_blur, rows, encoded.shape[1])
        delta += 0.6 * (b.clarity / 100.0) * (luma - base) * midtone
    if b.texture:
        fine = cv2.GaussianBlur(luma, (0, 0), max(0.6, long_side / 1500.0))
        coarse = cv2.GaussianBlur(luma, (0, 0), max(1.5, long_side / 300.0))
        delta += 1.2 * (b.texture / 100.0) * (fine - coarse) * midtone
    return encoded + delta[:, :, None]


def _colour_stage(encoded: np.ndarray, recipe: Recipe, ctx: Context) -> np.ndarray:
    """HSL bands, vibrance, saturation, colour grading, look and LUT."""
    b = recipe.basic
    needs_hsv = (
        ctx.hue_shift is not None or ctx.hue_saturation is not None
        or ctx.hue_luminance is not None or b.vibrance or b.saturation
    )
    if needs_hsv:
        hsv = cv2.cvtColor(np.clip(encoded, 0.0, 1.0), cv2.COLOR_RGB2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        degree = np.clip(hue, 0.0, 359.999)
        index = degree.astype(np.int32)
        frac = degree - index
        look = lambda table: table[index] * (1.0 - frac) + table[index + 1] * frac  # noqa: E731
        # Greys have no meaningful hue; fade the band controls out for them.
        colourful = _smoothstep(sat / 0.12)
        if ctx.hue_luminance is not None:
            val = val * (1.0 + 0.6 * look(ctx.hue_luminance) * colourful * sat)
        if ctx.hue_saturation is not None:
            sat = sat * (1.0 + look(ctx.hue_saturation) * colourful)
        if ctx.hue_shift is not None:
            hue = (hue + look(ctx.hue_shift) * colourful) % 360.0
        if b.vibrance:
            sat = sat * (1.0 + (b.vibrance / 100.0) * (1.0 - sat))
        if b.saturation:
            sat = sat * (1.0 + b.saturation / 100.0)
        hsv = np.stack([hue, np.clip(sat, 0.0, 1.0), np.clip(val, 0.0, 1.0)], axis=-1).astype(np.float32)
        encoded = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    grade = recipe.grade
    wheels = (grade.shadows, grade.midtones, grade.highlights)
    if any(w.saturation or w.luminance for w in wheels):
        luma = np.clip(encoded @ LUMA, 0.0, 1.0)
        pivot = 0.5 + 0.3 * grade.balance / 100.0
        width = 0.15 + 0.35 * grade.blending / 100.0
        shadow = _smoothstep((pivot - luma) / width)
        highlight = _smoothstep((luma - pivot) / width)
        middle = np.clip(1.0 - shadow - highlight, 0.0, 1.0)
        for wheel, weight in zip(wheels, (shadow, middle, highlight)):
            if wheel.saturation:
                tint = _hue_to_rgb(wheel.hue)
                tint = tint - float(tint @ LUMA)        # colour only, no brightness
                encoded = encoded + (0.35 * wheel.saturation / 100.0) * weight[:, :, None] * tint
            if wheel.luminance:
                encoded = encoded + (0.25 * wheel.luminance / 100.0) * weight[:, :, None]

    look_ = recipe.look
    if look_.matrix is not None and look_.strength > 0:
        matrix = np.asarray(look_.matrix, dtype=np.float32)
        graded = np.clip(np.clip(encoded, 0.0, 1.0) * 255.0 @ matrix.T + look_.lift, 0.0, 255.0) / 255.0
        graded = np.power(graded, 1.0 / look_.gamma)
        amount = look_.strength / 100.0
        encoded = encoded * (1.0 - amount) + graded * amount

    if ctx.cube is not None:
        amount = recipe.lut.strength / 100.0
        encoded = encoded * (1.0 - amount) + ctx.cube.apply(np.clip(encoded, 0.0, 1.0)) * amount
    return encoded


def _detail_stage(encoded: np.ndarray, recipe: Recipe, long_side: int) -> np.ndarray:
    detail = recipe.detail
    if detail.noise_reduction:
        strength = detail.noise_reduction / 100.0
        smooth = cv2.bilateralFilter(np.clip(encoded, 0.0, 1.0), 5, 0.02 + 0.10 * strength, 2.0 + 3.0 * strength)
        encoded = encoded * (1.0 - strength) + smooth * strength
    if detail.sharpen_amount:
        luma = encoded @ LUMA
        radius = detail.sharpen_radius * max(1.0, long_side / 3000.0)
        edge = luma - cv2.GaussianBlur(luma, (0, 0), radius)
        encoded = encoded + (detail.sharpen_amount / 100.0) * edge[:, :, None]
    return encoded


def _padding(recipe: Recipe, long_side: int) -> int:
    """Rows of overlap a strip needs for its small-radius filters."""
    sigma = 0.0
    if recipe.basic.texture:
        sigma = max(sigma, long_side / 300.0, 1.5)
    if recipe.detail.sharpen_amount:
        sigma = max(sigma, recipe.detail.sharpen_radius * max(1.0, long_side / 3000.0))
    if recipe.detail.noise_reduction:
        sigma = max(sigma, 3.0)
    return int(math.ceil(3.0 * sigma)) + 2 if sigma else 0


def _render_rows(source: Source, recipe: Recipe, ctx: Context, rows: slice) -> np.ndarray:
    """Render a band of rows to display-encoded float, 0..1."""
    height = source.data.shape[0]
    long_side = max(source.size)
    pad = _padding(recipe, long_side)
    y0, y1 = max(0, rows.start - pad), min(height, rows.stop + pad)
    band = slice(y0, y1)

    encoded = _encode_rows(source, recipe, ctx, band)
    if ctx.transmission is not None:
        encoded = _dehaze(encoded, ctx.upsample(ctx.transmission, band, encoded.shape[1]), ctx.airlight)
    elif recipe.basic.dehaze < 0:
        encoded = _add_haze(encoded, recipe.basic.dehaze)
    if recipe.detail.noise_reduction:
        encoded = _detail_stage(encoded, _only_noise(recipe), long_side)
    if not ctx.tone_folded:
        encoded = _tone_stage(encoded, recipe, ctx, band)
    encoded = _presence_stage(encoded, recipe, ctx, band, long_side)
    encoded = _colour_stage(encoded, recipe, ctx)
    if recipe.detail.sharpen_amount:
        encoded = _detail_stage(encoded, _only_sharpen(recipe), long_side)
    return np.clip(encoded[rows.start - y0: rows.stop - y0], 0.0, 1.0)


def _only_noise(recipe: Recipe) -> Recipe:
    clone = Recipe()
    clone.detail.noise_reduction = recipe.detail.noise_reduction
    return clone


def _only_sharpen(recipe: Recipe) -> Recipe:
    clone = Recipe()
    clone.detail.sharpen_amount = recipe.detail.sharpen_amount
    clone.detail.sharpen_radius = recipe.detail.sharpen_radius
    return clone


# --------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------- #

def set_rotation(geometry: Geometry, degrees_ccw: float) -> None:
    """Store a total rotation as quarter turns plus a small straightening."""
    quarters = int(round(degrees_ccw / 90.0))
    geometry.quarter_turns = (-quarters) % 4
    geometry.rotation = float(np.clip(degrees_ccw - 90.0 * quarters, -45.0, 45.0))


def _homography(size: tuple[int, int], geometry: Geometry) -> np.ndarray:
    """Rotation and keystone about the frame centre, as one 3 x 3 matrix."""
    w, h = size
    half = np.array([[w / 2.0, 0, w / 2.0], [0, h / 2.0, h / 2.0], [0, 0, 1]], dtype=np.float64)
    theta = math.radians(geometry.rotation)
    aspect = w / float(h)
    # Rotation in normalised coordinates must undo the aspect ratio first.
    rotate = np.array([
        [math.cos(theta), math.sin(theta) / aspect, 0],
        [-math.sin(theta) * aspect, math.cos(theta), 0],
        [0, 0, 1],
    ])
    tilt = np.array([
        [1, 0, 0], [0, 1, 0],
        [0.35 * geometry.keystone_horizontal / 100.0, 0.35 * geometry.keystone_vertical / 100.0, 1],
    ])
    return half @ rotate @ tilt @ np.linalg.inv(half)


def _largest_rect_inside(quad: np.ndarray, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Largest centred rectangle of the frame's shape inside a warped quad."""
    w, h = size
    centre = quad.mean(axis=0)
    contour = quad.reshape(-1, 1, 2).astype(np.float32)
    low, high = 0.0, 1.0
    for _ in range(24):
        mid = (low + high) / 2.0
        half_w, half_h = mid * w / 2.0, mid * h / 2.0
        corners = [(centre[0] + sx * half_w, centre[1] + sy * half_h) for sx in (-1, 1) for sy in (-1, 1)]
        inside = all(
            0 <= x <= w and 0 <= y <= h and cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0
            for x, y in corners
        )
        low, high = (mid, high) if inside else (low, mid)
    # Two pixels in from the edge: cubic resampling blends the outermost
    # row of the warped frame with the blank beyond it.
    half_w, half_h = low * w / 2.0 - 2.0, low * h / 2.0 - 2.0
    x0, y0 = int(math.ceil(centre[0] - half_w)), int(math.ceil(centre[1] - half_h))
    return x0, y0, max(1, int(math.floor(2 * half_w))), max(1, int(math.floor(2 * half_h)))


def apply_geometry(image: np.ndarray, geometry: Geometry, with_coverage: bool = False):
    """Turn, flip, straighten, correct keystone and crop.

    With ``with_coverage`` the result is ``(image, coverage)``, where
    coverage is 255 wherever the output shows the photograph and falls to 0
    in the blank wedges a rotation leaves. It is computed from the
    transform itself, so it is exact where judging blankness from pixel
    values would miss the blended edge.
    """
    coverage = None
    if geometry.quarter_turns % 4:
        image = np.rot90(image, k=-(geometry.quarter_turns % 4))
    if geometry.flip_horizontal:
        image = image[:, ::-1]
    if geometry.flip_vertical:
        image = image[::-1]
    image = np.ascontiguousarray(image)

    if geometry.rotation or geometry.keystone_vertical or geometry.keystone_horizontal:
        h, w = image.shape[:2]
        matrix = _homography((w, h), geometry)
        image = cv2.warpPerspective(image, matrix, (w, h), flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if with_coverage:
            coverage = cv2.warpPerspective(np.full((h, w), 255, np.uint8), matrix, (w, h),
                                           flags=cv2.INTER_LINEAR, borderValue=0)
        if geometry.constrain_to_image:
            corners = np.array([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=np.float64)
            quad = cv2.perspectiveTransform(corners, matrix)[0]
            x, y, cw, ch = _largest_rect_inside(quad, (w, h))
            image = image[y:y + ch, x:x + cw]
            coverage = coverage[y:y + ch, x:x + cw] if coverage is not None else None

    if geometry.crop is not None:
        h, w = image.shape[:2]
        x, y, cw, ch = geometry.crop
        x0, y0 = int(round(x * w)), int(round(y * h))
        x1, y1 = int(round((x + cw) * w)), int(round((y + ch) * h))
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            image = image[y0:y1, x0:x1]
            coverage = coverage[y0:y1, x0:x1] if coverage is not None else None
    image = np.ascontiguousarray(image)
    if not with_coverage:
        return image
    if coverage is None:
        coverage = np.full(image.shape[:2], 255, np.uint8)
    return image, np.ascontiguousarray(coverage)


# --------------------------------------------------------------------- #
# Lens profiles
# --------------------------------------------------------------------- #

_lens_db = None


def _lookup_lens(path: Path):
    """``(camera, lens, focal, aperture)`` from the lens database, or None."""
    global _lens_db
    try:
        import lensfunpy

        from src.engine.universal_loader import UniversalLoader

        meta = UniversalLoader().read_metadata(path)
        if not (meta.camera_make and meta.camera_model and meta.lens_model and meta.focal_length):
            return None
        if _lens_db is None:
            _lens_db = lensfunpy.Database()
        cameras = _lens_db.find_cameras(meta.camera_make, meta.camera_model)
        if not cameras:
            return None
        lenses = _lens_db.find_lenses(cameras[0], None, meta.lens_model)
        if not lenses:
            return None
        return cameras[0], lenses[0], float(meta.focal_length), float(meta.aperture or 8.0)
    except Exception as exc:
        logger.debug("Lens lookup failed for %s: %s", path, exc)
        return None


def lens_profile_available(path: str | Path) -> bool:
    return _lookup_lens(Path(path)) is not None


def lens_vignette_gain(path: str | Path, size: tuple[int, int]) -> np.ndarray | None:
    """Per-pixel brightening that undoes the lens's vignetting, at ``size``."""
    found = _lookup_lens(Path(path))
    if found is None:
        return None
    try:
        import lensfunpy

        camera, lens, focal, aperture = found
        w, h = size
        modifier = lensfunpy.Modifier(lens, camera.crop_factor, w, h)
        modifier.initialize(focal, aperture, 10.0, pixel_format=np.float32,
                            flags=lensfunpy.ModifyFlags.VIGNETTING)
        flat = np.ones((h, w, 3), dtype=np.float32)
        if not modifier.apply_color_modification(flat):
            return None
        return flat[:, :, 1].copy()
    except Exception as exc:
        logger.debug("Vignetting profile failed: %s", exc)
        return None


def correct_lens_geometry(source: Source) -> Source:
    """Undo distortion and lateral colour fringing using the lens profile.

    Returns the source unchanged when no profile matches, with ``note`` set.
    """
    found = _lookup_lens(source.path) if source.path is not None else None
    if found is None:
        source.note = source.note or "no lens profile found for this camera and lens"
        return source
    try:
        import lensfunpy

        camera, lens, focal, aperture = found
        w, h = source.size
        modifier = lensfunpy.Modifier(lens, camera.crop_factor, w, h)
        modifier.initialize(
            focal, aperture, 10.0, pixel_format=np.float32,
            flags=lensfunpy.ModifyFlags.DISTORTION | lensfunpy.ModifyFlags.TCA,
        )
        out = np.empty_like(source.data)
        for y0 in range(0, h, STRIP_ROWS):
            rows = min(STRIP_ROWS, h - y0)
            coords = modifier.apply_subpixel_geometry_distortion(0, y0, w, rows)
            if coords is None:
                source.note = source.note or "the lens profile has no distortion data"
                return source
            for channel in range(3):
                out[y0:y0 + rows, :, channel] = cv2.remap(
                    source.data[:, :, channel], coords[:, :, channel, 0], coords[:, :, channel, 1],
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
                )
        return Source(out, source.linear, source.path, source.is_raw, source.note)
    except Exception as exc:
        logger.debug("Lens geometry correction failed: %s", exc)
        return source


def manual_distortion(source: Source, amount: float) -> Source:
    """A one-term radial correction for when no profile exists."""
    if not amount:
        return source
    w, h = source.size
    k = -0.12 * amount / 100.0
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    nx, ny = (xs - w / 2.0) / (w / 2.0), (ys - h / 2.0) / (h / 2.0)
    scale = 1.0 + k * (nx * nx + ny * ny) / 2.0
    map_x = (nx * scale) * (w / 2.0) + w / 2.0
    map_y = (ny * scale) * (h / 2.0) + h / 2.0
    out = cv2.remap(source.data, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Source(out, source.linear, source.path, source.is_raw, source.note)


# --------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------- #

def _pixels_untouched(recipe: Recipe) -> bool:
    """True when the recipe changes framing at most, not pixel values."""
    from src.engine.parametric_recipe import to_dict

    mine, neutral = to_dict(recipe), to_dict(Recipe())
    return all(mine[key] == neutral[key] for key in mine if key not in ("geometry", "version", "vectors", "layers"))


def _quantise(encoded: np.ndarray, bits: int) -> np.ndarray:
    if bits == 16:
        return np.round(encoded * 65535.0).astype(np.uint16)
    return np.round(encoded * 255.0).astype(np.uint8)


def render(
    source: Source,
    recipe: Recipe,
    bits: int = 8,
    geometry: bool = True,
    layers=None,
    inpainter=None,
) -> np.ndarray:
    """Render ``source`` through ``recipe`` to RGB of the given bit depth.

    The frame is processed in strips, so peak memory is one strip of float
    work on top of the source and the result, whatever the frame size.

    ``layers`` is a ``layer_stack.RenderContext``; pass one to have the
    recipe's layer stack applied (it supplies the models and files layers
    may need). Without it the stack is skipped. ``inpainter`` is used to
    fill the corners when the recipe asks for that.
    """
    out = _render_pixels(source, recipe, bits)
    if layers is not None and recipe.layers:
        from src.engine import layer_stack

        out = layer_stack.composite(out, layer_stack.layers_from_recipe(recipe), layers)
    if not geometry:
        return out
    g = recipe.geometry
    if g.fill_blank and not g.constrain_to_image and (g.rotation or g.keystone_vertical or g.keystone_horizontal):
        out, coverage = apply_geometry(out, g, with_coverage=True)
        return _fill_blank(out, coverage, inpainter)
    return apply_geometry(out, g)


def _fill_blank(image: np.ndarray, coverage: np.ndarray, inpainter) -> np.ndarray:
    """Paint in the wedges straightening left, at the image's own bit depth."""
    from src.ai_tools import retouch

    # Anything not fully covered by the photograph, plus a pixel of margin
    # for the cubic resampling that blended the edge.
    hole = cv2.dilate((coverage < 254).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)
    if not hole.any():
        return image
    engine = inpainter or retouch._FastOnly()
    if image.dtype == np.uint8:
        return engine.fill(image, hole, engine="auto")[0]
    # The fill works in 8 bits; only the wedges take its result, so the
    # rest of a 16-bit frame keeps its precision.
    filled = engine.fill((image >> 8).astype(np.uint8), hole, engine="auto")[0].astype(np.uint16) * 257
    return np.where(hole[:, :, None] > 0, filled, image)


def _render_pixels(source: Source, recipe: Recipe, bits: int) -> np.ndarray:
    """The develop settings only: no layers, no framing."""
    if recipe.lens.profile:
        source = correct_lens_geometry(source)
    if recipe.lens.distortion:
        source = manual_distortion(source, recipe.lens.distortion)

    if _pixels_untouched(recipe) and not source.linear and source.data.dtype == (
        np.uint16 if bits == 16 else np.uint8
    ):
        return source.data.copy()       # nothing to compute

    ctx = prepare(source, recipe)
    width, height = source.size
    if _only_point_operations(recipe, ctx):
        tables = [_quantise(np.clip(table, 0.0, 1.0), bits) for table in ctx.point_luts]
        if source.data.dtype == np.uint8 and bits == 8:
            return cv2.LUT(source.data, np.stack(tables, axis=-1).reshape(256, 1, 3))
        return np.stack([tables[c][source.data[:, :, c]] for c in range(3)], axis=-1)

    out = np.empty((height, width, 3), dtype=np.uint16 if bits == 16 else np.uint8)
    for y0 in range(0, height, STRIP_ROWS):
        rows = slice(y0, min(height, y0 + STRIP_ROWS))
        out[rows] = _quantise(_render_rows(source, recipe, ctx, rows), bits)
    return out


def render_preview(source: Source, recipe: Recipe, max_side: int = 1600, layers=None) -> np.ndarray:
    """An 8-bit preview no larger than ``max_side``, for the interface.

    Corners are filled with the fast engine here even when the export will
    use the neural one, so a preview is never held up for seconds.
    """
    return render(source.proxy(max_side), recipe, bits=8, layers=layers)


def auto_tone(source: Source) -> float:
    """Exposure, in stops, that brings the frame's midtones to middle grey.

    A starting point for a RAW file, which develops dark without one. It
    looks at the median luminance and never suggests more than three stops.
    """
    small = source.proxy(256).to_linear()
    median = float(np.median(small @ LUMA))
    if median <= 1e-5:
        return 0.0
    return float(np.clip(math.log2(0.18 / median), -3.0, 3.0))


# --------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------- #

EXPORT_FORMATS = ("jpeg", "png", "tiff", "tiff16", "png16")


def export(
    path: str | Path,
    recipe: Recipe,
    destination: str | Path,
    fmt: str = "jpeg",
    quality: int = 92,
    max_side: int | None = None,
    dpi: int | None = None,
    layers=None,
    inpainter=None,
) -> Path:
    """Render a photograph through its recipe and write the result.

    The source is only read. ``destination`` must not be the source.
    """
    path, destination = Path(path), Path(destination)
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"Unknown format {fmt!r}. Choose from: {', '.join(EXPORT_FORMATS)}")
    if destination.resolve() == path.resolve():
        raise ValueError("refusing to export over the original file")

    source = open_source(path)
    if source is None:
        raise IOError(f"Could not read {path}")
    bits = 16 if fmt.endswith("16") else 8
    image = render(source, recipe, bits=bits, layers=layers, inpainter=inpainter)
    if max_side:
        image = fast_preview.fit(image, max_side)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if bits == 16:
        if fmt == "tiff16":
            import tifffile

            extra = {"resolution": (dpi, dpi), "resolutionunit": "INCH"} if dpi else {}
            tifffile.imwrite(str(destination), image, photometric="rgb", **extra)
        else:
            ok, data = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if not ok:
                raise IOError("PNG encoding failed")
            destination.write_bytes(data.tobytes())
        return destination

    from PIL import Image

    picture = Image.fromarray(image)
    extra = {"dpi": (dpi, dpi)} if dpi else {}
    if fmt == "jpeg":
        picture.save(destination, "JPEG", quality=int(quality), subsampling=0, **extra)
    elif fmt == "png":
        picture.save(destination, "PNG", **extra)
    else:
        picture.save(destination, "TIFF", compression="tiff_lzw", **extra)
    return destination
