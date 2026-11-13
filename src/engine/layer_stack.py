"""The layer stack: edits that apply to part of a photograph.

A layer sits over the developed image and changes it through a mask. The
stack is stored in the recipe as plain data, so it is as non-destructive as
the sliders: nothing is flattened until export, and any layer can be
hidden, reordered, re-masked or deleted later.

Layer kinds
-----------
``adjustment``  A set of develop settings applied through the mask
                ("brighten the subject", "cool the sky").
``retouch``     A portrait operation computed from the face itself: skin
                smoothing, teeth whitening, eye clarity.
``pixel``       A patch of pixels, such as a healed spot. The patch is a
                PNG kept in the edit folder; the layer records where it goes.
``mask``        Limits everything beneath it in the stack to its mask.
``text``        Vector text, drawn by the vector canvas.
``shape``       A vector shape, drawn by the vector canvas.

Positions and sizes are stored as fractions of the frame, never pixels, so
a stack built on a small preview lands in the same place on the full-size
export.

Blending
--------
Seven modes, with the formulas from the W3C compositing specification,
which are the ones Photoshop documents: normal, multiply, screen, overlay,
soft light, colour dodge and luminosity. Blending is done on
display-encoded values, as those applications do, so a mode behaves the way
a retoucher expects from them.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BLEND_MODES = ("normal", "multiply", "screen", "overlay", "soft_light", "color_dodge", "luminosity")
LAYER_KINDS = ("adjustment", "retouch", "pixel", "mask", "text", "shape")
MASK_KINDS = ("none", "ai", "gradient", "radial", "luminance", "brush", "raster")
RETOUCH_OPS = ("skin_smooth", "teeth_whiten", "eye_clarity")

ASSET_SUFFIX = ".granite.assets"

_LUM = np.array([0.30, 0.59, 0.11], dtype=np.float32)     # the W3C weights


# --------------------------------------------------------------------- #
# Blend modes
# --------------------------------------------------------------------- #

def _lum(colour: np.ndarray) -> np.ndarray:
    return colour @ _LUM


def _set_lum(colour: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Give ``colour`` the luminance ``target``, staying inside the gamut."""
    shifted = colour + (target - _lum(colour))[..., None]
    lum = _lum(shifted)[..., None]
    low = shifted.min(axis=-1, keepdims=True)
    high = shifted.max(axis=-1, keepdims=True)
    under = low < 0
    over = high > 1
    with np.errstate(divide="ignore", invalid="ignore"):
        shifted = np.where(under, lum + (shifted - lum) * lum / np.maximum(lum - low, 1e-6), shifted)
        shifted = np.where(over, lum + (shifted - lum) * (1 - lum) / np.maximum(high - lum, 1e-6), shifted)
    return shifted


def blend(base: np.ndarray, top: np.ndarray, mode: str) -> np.ndarray:
    """Blend ``top`` over ``base``; both float RGB in 0..1, same shape.

    Returns the fully-blended colour. Opacity and masking are applied by
    the caller as a mix between ``base`` and this result.
    """
    if mode == "normal":
        return top
    if mode == "multiply":
        return base * top
    if mode == "screen":
        return base + top - base * top
    if mode == "overlay":
        return np.where(base <= 0.5, 2.0 * base * top, 1.0 - 2.0 * (1.0 - base) * (1.0 - top))
    if mode == "soft_light":
        d = np.where(base <= 0.25, ((16.0 * base - 12.0) * base + 4.0) * base, np.sqrt(np.maximum(base, 0.0)))
        return np.where(
            top <= 0.5,
            base - (1.0 - 2.0 * top) * base * (1.0 - base),
            base + (2.0 * top - 1.0) * (d - base),
        )
    if mode == "color_dodge":
        with np.errstate(divide="ignore", invalid="ignore"):
            dodged = np.minimum(1.0, base / np.maximum(1.0 - top, 1e-6))
        return np.where(base <= 0.0, 0.0, np.where(top >= 1.0, 1.0, dodged))
    if mode == "luminosity":
        return _set_lum(base, _lum(top))
    raise ValueError(f"Unknown blend mode {mode!r}. Choose from: {', '.join(BLEND_MODES)}")


# --------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------- #

@dataclass
class Mask:
    """Where a layer applies. ``kind == "none"`` means everywhere."""

    kind: str = "none"
    params: dict = field(default_factory=dict)
    invert: bool = False
    feather: float = 0.0        # as a fraction of the frame's long edge
    density: float = 100.0      # 0..100; how strongly the mask restricts


@dataclass
class Layer:
    kind: str = "adjustment"
    name: str = ""
    visible: bool = True
    opacity: float = 100.0
    blend: str = "normal"
    mask: Mask = field(default_factory=Mask)
    recipe: dict = field(default_factory=dict)       # adjustment
    op: str = ""                                     # retouch
    amount: float = 50.0                             # retouch, 0..100
    asset: str = ""                                  # pixel: a file in the assets folder
    rect: list = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])   # pixel
    vector: dict = field(default_factory=dict)       # text, shape


def _number(raw: Any, default: float, low: float, high: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return min(high, max(low, value)) if math.isfinite(value) else default


def mask_from_dict(data: Any) -> Mask:
    if not isinstance(data, dict):
        return Mask()
    kind = data.get("kind")
    return Mask(
        kind=kind if kind in MASK_KINDS else "none",
        params=data.get("params") if isinstance(data.get("params"), dict) else {},
        invert=data.get("invert") is True,
        feather=_number(data.get("feather"), 0.0, 0.0, 0.5),
        density=_number(data.get("density"), 100.0, 0.0, 100.0),
    )


def layer_from_dict(data: Any) -> Layer | None:
    """Read one layer, tolerating anything. Returns None if it is unusable."""
    if not isinstance(data, dict) or data.get("kind") not in LAYER_KINDS:
        return None
    rect = data.get("rect")
    try:
        rect = [float(v) for v in rect][:4] if rect is not None else [0.0, 0.0, 1.0, 1.0]
        if len(rect) != 4 or not all(math.isfinite(v) for v in rect):
            raise ValueError
    except (TypeError, ValueError):
        rect = [0.0, 0.0, 1.0, 1.0]
    blend_mode = data.get("blend")
    op = data.get("op")
    return Layer(
        kind=data["kind"],
        name=str(data.get("name", ""))[:120],
        visible=data.get("visible", True) is not False,
        opacity=_number(data.get("opacity"), 100.0, 0.0, 100.0),
        blend=blend_mode if blend_mode in BLEND_MODES else "normal",
        mask=mask_from_dict(data.get("mask")),
        recipe=data.get("recipe") if isinstance(data.get("recipe"), dict) else {},
        op=op if op in RETOUCH_OPS else "",
        amount=_number(data.get("amount"), 50.0, 0.0, 100.0),
        # A bare filename only: an asset can never point outside its folder.
        asset=Path(str(data.get("asset", ""))).name,
        rect=rect,
        vector=data.get("vector") if isinstance(data.get("vector"), dict) else {},
    )


def layer_to_dict(layer: Layer) -> dict:
    import dataclasses

    return dataclasses.asdict(layer)


def layers_from_recipe(recipe) -> list[Layer]:
    """The usable layers of a recipe, bottom first."""
    found = (layer_from_dict(item) for item in getattr(recipe, "layers", []) or [])
    return [layer for layer in found if layer is not None]


def assets_folder(source: str | Path, edit_folder: str | Path) -> Path:
    """Where a photograph's layer patches are kept. Inside the edit folder."""
    return Path(edit_folder) / f"{Path(source).name}{ASSET_SUFFIX}"


def save_asset(source: str | Path, edit_folder: str | Path, name: str, rgba: np.ndarray) -> str:
    """Store a patch for a pixel layer; returns the name to put in the layer."""
    folder = assets_folder(source, edit_folder)
    folder.mkdir(parents=True, exist_ok=True)
    filename = Path(name).with_suffix(".png").name
    channels = cv2.COLOR_RGBA2BGRA if rgba.shape[2] == 4 else cv2.COLOR_RGB2BGR
    ok, data = cv2.imencode(".png", cv2.cvtColor(rgba, channels))
    if not ok:
        raise IOError("could not encode the layer patch")
    (folder / filename).write_bytes(data.tobytes())
    return filename


# --------------------------------------------------------------------- #
# Rendering context
# --------------------------------------------------------------------- #

@dataclass
class RenderContext:
    """What rendering a stack may need beyond the pixels themselves.

    Everything is optional. A missing piece disables the layers that need
    it and leaves the rest working.
    """

    segmenter: Any = None                       # segmentation.Segmenter
    assets: Path | None = None                  # folder holding pixel-layer patches
    vector_renderer: Callable | None = None     # (vector dict, (w, h)) -> RGBA float
    notes: list = field(default_factory=list)   # why a layer was skipped
    _masks: dict = field(default_factory=dict)
    _faces: Any = None

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)

    def ai_mask(self, rgb8: np.ndarray, kind: str) -> np.ndarray | None:
        if self.segmenter is None:
            self.note(f"the {kind} mask needs the segmentation models")
            return None
        key = (kind, rgb8.shape)
        if key not in self._masks:
            result = self.segmenter.mask(rgb8, kind)
            if result.note and result.is_empty:
                self.note(f"{kind} mask: {result.note}")
            self._masks[key] = result.mask
        return self._masks[key]

    def faces(self, rgb8: np.ndarray):
        if self.segmenter is None:
            return None
        if self._faces is None:
            self._faces = self.segmenter.faces(rgb8) or []
        return self._faces


# --------------------------------------------------------------------- #
# Masks
# --------------------------------------------------------------------- #

def _point(raw, default) -> tuple[float, float]:
    try:
        return float(raw[0]), float(raw[1])
    except (TypeError, ValueError, IndexError):
        return default


def _brush(shape: tuple[int, int], strokes) -> np.ndarray:
    """Rasterise brush strokes stored as fractions of the frame."""
    h, w = shape
    long_side = max(h, w)
    canvas = np.zeros((h, w), np.float32)
    for stroke in strokes if isinstance(strokes, list) else []:
        if not isinstance(stroke, dict):
            continue
        points = [
            (int(round(x * w)), int(round(y * h)))
            for x, y in (_point(p, (None, None)) for p in stroke.get("points", []) if p is not None)
            if x is not None
        ]
        if not points:
            continue
        radius = max(1, int(round(_number(stroke.get("radius"), 0.02, 0.0005, 0.5) * long_side)))
        hardness = _number(stroke.get("hardness"), 0.5, 0.0, 1.0)
        layer = np.zeros((h, w), np.float32)
        if len(points) == 1:
            cv2.circle(layer, points[0], radius, 1.0, -1, lineType=cv2.LINE_AA)
        else:
            cv2.polylines(layer, [np.array(points, np.int32)], False, 1.0, 2 * radius, lineType=cv2.LINE_AA)
        if hardness < 0.98:
            layer = cv2.GaussianBlur(layer, (0, 0), max(0.5, radius * (1.0 - hardness) * 0.6))
        canvas = np.minimum(canvas, 1.0 - layer) if stroke.get("erase") is True else np.maximum(canvas, layer)
    return canvas


def render_mask(mask: Mask, image: np.ndarray, ctx: RenderContext) -> np.ndarray | None:
    """A layer's mask at the image's size: float, 0..1.

    ``image`` is float RGB in 0..1. Returns None when the mask cannot be
    built (a missing model, a missing file); the caller then skips the
    layer rather than applying it everywhere.
    """
    h, w = image.shape[:2]
    p = mask.params
    if mask.kind == "none":
        result = np.ones((h, w), np.float32)
    elif mask.kind == "ai":
        kind = p.get("kind")
        rgb8 = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
        if kind == "click":
            if ctx.segmenter is None:
                ctx.note("click selection needs the segmentation models")
                return None
            points = [(x * w, y * h) for x, y in (_point(q, (0.5, 0.5)) for q in p.get("points", []))]
            exclude = [(x * w, y * h) for x, y in (_point(q, (0.5, 0.5)) for q in p.get("exclude", []))]
            result = ctx.segmenter.click(rgb8, points, exclude).mask
        else:
            found = ctx.ai_mask(rgb8, str(kind))
            if found is None:
                return None
            result = found
    elif mask.kind == "gradient":
        (x0, y0), (x1, y1) = _point(p.get("start"), (0.5, 0.0)), _point(p.get("end"), (0.5, 1.0))
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        dx, dy = (x1 - x0) * w, (y1 - y0) * h
        length2 = max(dx * dx + dy * dy, 1e-6)
        result = np.clip(((xs - x0 * w) * dx + (ys - y0 * h) * dy) / length2, 0.0, 1.0)
    elif mask.kind == "radial":
        cx, cy = _point(p.get("centre"), (0.5, 0.5))
        rx = max(1e-3, _number(p.get("radius_x"), 0.25, 0.0, 2.0)) * w
        ry = max(1e-3, _number(p.get("radius_y"), 0.25, 0.0, 2.0)) * h
        soft = _number(p.get("softness"), 0.5, 0.0, 1.0)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        distance = np.sqrt(((xs - cx * w) / rx) ** 2 + ((ys - cy * h) / ry) ** 2)
        result = np.clip((1.0 - distance) / max(soft, 1e-3), 0.0, 1.0)
    elif mask.kind == "luminance":
        low = _number(p.get("low"), 0.0, 0.0, 1.0)
        high = _number(p.get("high"), 1.0, 0.0, 1.0)
        soft = max(1e-3, _number(p.get("softness"), 0.1, 0.0, 1.0))
        lum = image @ _LUM
        result = np.clip((lum - low) / soft + 1.0, 0.0, 1.0) * np.clip((high - lum) / soft + 1.0, 0.0, 1.0)
    elif mask.kind == "brush":
        result = _brush((h, w), p.get("strokes"))
    elif mask.kind == "raster":
        name = Path(str(p.get("asset", ""))).name
        if ctx.assets is None or not name or not (ctx.assets / name).exists():
            ctx.note("a painted mask is missing its file")
            return None
        stored = cv2.imdecode(np.fromfile(str(ctx.assets / name), np.uint8), cv2.IMREAD_GRAYSCALE)
        if stored is None:
            ctx.note("a painted mask could not be read")
            return None
        result = cv2.resize(stored, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    else:
        return None

    result = result.astype(np.float32)
    if mask.feather > 0:
        result = cv2.GaussianBlur(result, (0, 0), max(0.5, mask.feather * max(h, w)))
    if mask.invert:
        result = 1.0 - result
    density = mask.density / 100.0
    if density < 1.0:
        result = 1.0 - density * (1.0 - result)     # lower density lets the layer through
    return np.clip(result, 0.0, 1.0)


# --------------------------------------------------------------------- #
# Layer content
# --------------------------------------------------------------------- #

def _adjust(image: np.ndarray, layer: Layer) -> np.ndarray:
    from src.engine import develop_pipeline as dp
    from src.engine.parametric_recipe import from_dict

    recipe = from_dict(layer.recipe)
    recipe.layers, recipe.vectors = [], []          # a layer does not nest a stack
    deep = np.round(np.clip(image, 0.0, 1.0) * 65535.0).astype(np.uint16)
    return dp.render(dp.from_array(deep), recipe, bits=16, geometry=False).astype(np.float32) / 65535.0


def _retouch(image: np.ndarray, layer: Layer, ctx: RenderContext) -> np.ndarray | None:
    from src.ai_tools import retouch, segmentation

    rgb8 = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
    faces = ctx.faces(rgb8)
    if faces is None:
        ctx.note(f"{layer.op.replace('_', ' ')} needs the face models")
        return None
    if not faces:
        ctx.note(f"{layer.op.replace('_', ' ')}: no face was found")
        return None
    amount = layer.amount / 100.0
    out = image
    for face in faces:
        if layer.op == "skin_smooth":
            out = retouch.smooth_skin(out, segmentation.skin_mask(rgb8, face), amount, face.eye_span)
        elif layer.op == "teeth_whiten":
            out = retouch.whiten_teeth(out, segmentation.teeth_mask(rgb8, face), amount)
        elif layer.op == "eye_clarity":
            eyes = ctx.ai_mask(rgb8, "eyes")
            iris = ctx.ai_mask(rgb8, "iris")
            return retouch.clarify_eyes(out, iris, eyes, amount) if iris is not None else None
    return out


def _pixel(layer: Layer, size: tuple[int, int], ctx: RenderContext):
    """A pixel layer's ``(rgb, alpha)`` at the frame's size, or None."""
    if ctx.assets is None or not layer.asset:
        ctx.note("a pixel layer has no patch folder to read from")
        return None
    path = ctx.assets / layer.asset
    if not path.exists():
        ctx.note(f"a pixel layer's patch is missing ({layer.asset})")
        return None
    patch = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if patch is None:
        ctx.note(f"a pixel layer's patch could not be read ({layer.asset})")
        return None
    if patch.ndim == 2:
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGRA)
    elif patch.shape[2] == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2BGRA)
    scale = 65535.0 if patch.dtype == np.uint16 else 255.0
    patch = cv2.cvtColor(patch, cv2.COLOR_BGRA2RGBA).astype(np.float32) / scale

    w, h = size
    x, y, rw, rh = layer.rect
    x0, y0 = int(round(x * w)), int(round(y * h))
    x1, y1 = int(round((x + rw) * w)), int(round((y + rh) * h))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    patch = cv2.resize(patch, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    rgb = np.zeros((h, w, 3), np.float32)
    alpha = np.zeros((h, w), np.float32)
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    part = patch[cy0 - y0: cy1 - y0, cx0 - x0: cx1 - x0]
    rgb[cy0:cy1, cx0:cx1] = part[:, :, :3]
    alpha[cy0:cy1, cx0:cx1] = part[:, :, 3]
    return rgb, alpha


# --------------------------------------------------------------------- #
# Compositing
# --------------------------------------------------------------------- #

def composite(image: np.ndarray, layers: list[Layer], ctx: RenderContext | None = None) -> np.ndarray:
    """Apply a stack, bottom first, to an RGB image of any supported dtype.

    A layer that cannot be rendered is skipped and the reason recorded in
    ``ctx.notes``; the rest of the stack still applies. With no visible
    layers the image is returned unchanged.
    """
    active = [layer for layer in layers if layer.visible and layer.opacity > 0]
    if not active:
        return image
    ctx = ctx or RenderContext()

    scale = {np.dtype(np.uint8): 255.0, np.dtype(np.uint16): 65535.0}.get(image.dtype, 1.0)
    base = image.astype(np.float32) / scale
    current = base
    size = (image.shape[1], image.shape[0])

    for layer in active:
        try:
            alpha = render_mask(layer.mask, current, ctx)
            if alpha is None:
                continue
            alpha = alpha * (layer.opacity / 100.0)

            if layer.kind == "mask":
                # Everything so far shows only where this mask allows.
                current = base + (current - base) * alpha[:, :, None]
                continue

            if layer.kind == "adjustment":
                top = _adjust(current, layer)
            elif layer.kind == "retouch":
                top = _retouch(current, layer, ctx)
                if top is None:
                    continue
            elif layer.kind == "pixel":
                found = _pixel(layer, size, ctx)
                if found is None:
                    continue
                top, own_alpha = found
                alpha = alpha * own_alpha
            else:   # text, shape
                if ctx.vector_renderer is None:
                    ctx.note("text and shape layers need the vector canvas")
                    continue
                drawn = ctx.vector_renderer(layer.vector, size)
                if drawn is None:
                    continue
                top, alpha = drawn[:, :, :3], alpha * drawn[:, :, 3]

            blended = blend(current, np.clip(top, 0.0, 1.0), layer.blend)
            current = current + (blended - current) * alpha[:, :, None]
        except Exception as exc:    # one bad layer must not lose the picture
            logger.warning("Layer %r failed: %s", layer.name or layer.kind, exc)
            ctx.note(f"a layer could not be rendered ({type(exc).__name__})")

    current = np.clip(current, 0.0, 1.0)
    if scale == 1.0:
        return current
    return np.round(current * scale).astype(image.dtype)
