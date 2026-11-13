"""Vector text, shapes and watermarks over a photograph.

A design is a list of items -- text, rectangles, ellipses, stars, bezier
paths, groups -- stored as plain data. One painter draws that list to
whatever is asked for: pixels at any size for the screen or a 300 dpi
print, an SVG, or a PDF. Because all three go through the same drawing
code, a design looks the same in each.

Units
-----
Nothing in an item is in pixels. Positions and sizes are fractions of the
canvas; a text size is a fraction of the canvas height; a stroke width is a
fraction of the canvas's long edge. A watermark placed on a 1,600 pixel
preview is therefore in the same place, at the same relative size, on a
9,000 pixel export.

Text
----
Text is laid out once at a fixed reference size with hinting off, then
scaled. That makes glyph positions exact at every output size, where
asking the font engine for each size separately would round differently at
each and move letters around between preview and print.

Limits
------
* A drop shadow is blurred in raster output. SVG and PDF get the same
  shadow unblurred, because neither is written with a blur filter here.
* SVG text is written as real ``<text>`` by default so it stays selectable;
  it then depends on the viewer having the font. ``outline_text=True``
  writes glyph outlines instead, which always look right and cannot be
  edited as text.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ITEM_KINDS = ("text", "rect", "ellipse", "star", "path", "group")
ANCHORS = (
    "top-left", "top", "top-right", "left", "center", "right",
    "bottom-left", "bottom", "bottom-right",
)
ALIGNMENTS = ("left", "center", "right")

_REFERENCE_PX = 200.0       # text is laid out at this size, then scaled
_TOKEN = re.compile(r"\{(\w+)\}")


# --------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------- #

@dataclass
class Shadow:
    dx: float = 0.004           # fractions of the canvas long edge
    dy: float = 0.004
    blur: float = 0.004
    color: str = "#00000099"


@dataclass
class Item:
    """One thing on the canvas. Which fields matter depends on ``kind``."""

    kind: str = "rect"
    x: float = 0.0              # the item's box, as fractions of the canvas;
    y: float = 0.0              # for text, the anchor point
    w: float = 0.2
    h: float = 0.2
    rotation: float = 0.0       # degrees, clockwise on screen
    opacity: float = 100.0
    fill: str | None = "#ffffffff"
    stroke: str | None = None
    stroke_width: float = 0.002
    shadow: Shadow | None = None
    # text
    text: str = ""
    font_family: str = "Segoe UI"
    font_size: float = 0.04     # fraction of the canvas height
    bold: bool = False
    italic: bool = False
    kerning: bool = True
    letter_spacing: float = 100.0   # percent; 100 is the font's own spacing
    line_spacing: float = 1.0       # multiple of the font's line height
    align: str = "left"
    anchor: str = "top-left"
    # rect
    corner_radius: float = 0.0  # fraction of the box's shorter side, 0..0.5
    # star
    points: int = 5
    inner_ratio: float = 0.45
    # path: [["M", x, y], ["L", x, y], ["C", x1, y1, x2, y2, x, y], ["Q", ...], ["Z"]]
    commands: list = field(default_factory=list)
    # group
    children: list = field(default_factory=list)
    name: str = ""


def _float(raw: Any, default: float, low: float = -1e6, high: float = 1e6) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return min(high, max(low, value)) if math.isfinite(value) else default


def _colour(raw: Any, default: str | None) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str) and re.fullmatch(r"#[0-9a-fA-F]{6}([0-9a-fA-F]{2})?", raw):
        return raw.lower()
    return default


def item_from_dict(data: Any) -> Item | None:
    """Read one item, tolerating anything. None if it cannot be drawn."""
    if not isinstance(data, dict) or data.get("kind") not in ITEM_KINDS:
        return None
    base = Item()
    shadow = data.get("shadow")
    commands = []
    for command in data.get("commands", []) if isinstance(data.get("commands"), list) else []:
        if not isinstance(command, (list, tuple)) or not command:
            continue
        op = str(command[0]).upper()
        need = {"M": 2, "L": 2, "Q": 4, "C": 6, "Z": 0}.get(op)
        if need is None or len(command) - 1 < need:
            continue
        try:
            commands.append([op, *[float(v) for v in command[1:1 + need]]])
        except (TypeError, ValueError):
            continue
    children = [item_from_dict(child) for child in data.get("children", [])] if isinstance(
        data.get("children"), list) else []
    return Item(
        kind=data["kind"],
        x=_float(data.get("x"), 0.0, -4, 4), y=_float(data.get("y"), 0.0, -4, 4),
        w=_float(data.get("w"), base.w, 0, 8), h=_float(data.get("h"), base.h, 0, 8),
        rotation=_float(data.get("rotation"), 0.0, -3600, 3600),
        opacity=_float(data.get("opacity"), 100.0, 0, 100),
        fill=_colour(data.get("fill", base.fill), base.fill),
        stroke=_colour(data.get("stroke"), None),
        stroke_width=_float(data.get("stroke_width"), base.stroke_width, 0, 0.5),
        shadow=Shadow(
            _float(shadow.get("dx"), 0.004, -1, 1), _float(shadow.get("dy"), 0.004, -1, 1),
            _float(shadow.get("blur"), 0.004, 0, 0.2), _colour(shadow.get("color"), "#00000099") or "#00000099",
        ) if isinstance(shadow, dict) else None,
        text=str(data.get("text", ""))[:4000],
        font_family=str(data.get("font_family", base.font_family))[:100],
        font_size=_float(data.get("font_size"), base.font_size, 0.002, 2.0),
        bold=data.get("bold") is True, italic=data.get("italic") is True,
        kerning=data.get("kerning", True) is not False,
        letter_spacing=_float(data.get("letter_spacing"), 100.0, 30, 400),
        line_spacing=_float(data.get("line_spacing"), 1.0, 0.3, 5.0),
        align=data.get("align") if data.get("align") in ALIGNMENTS else "left",
        anchor=data.get("anchor") if data.get("anchor") in ANCHORS else "top-left",
        corner_radius=_float(data.get("corner_radius"), 0.0, 0, 0.5),
        points=int(_float(data.get("points"), 5, 3, 64)),
        inner_ratio=_float(data.get("inner_ratio"), 0.45, 0.05, 0.95),
        commands=commands,
        children=[child for child in children if child is not None],
        name=str(data.get("name", ""))[:120],
    )


def item_to_dict(item: Item) -> dict:
    import dataclasses

    return dataclasses.asdict(item)


def items_from(data: Any) -> list[Item]:
    found = (item_from_dict(entry) for entry in (data if isinstance(data, list) else []))
    return [item for item in found if item is not None]


# --------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------- #

def expand_tokens(text: str, metadata=None, extra: dict | None = None) -> str:
    """Fill ``{camera}``-style tokens from a photograph's metadata.

    Known tokens: camera, lens, settings, gear, focal, iso, aperture,
    shutter, date, year. An unknown token, or one with no value for this
    photograph, becomes empty rather than being left as braces.
    """
    values: dict[str, str] = {}
    if metadata is not None:
        captured = getattr(metadata, "captured_at", None)
        values = {
            "camera": getattr(metadata, "camera_label", ""),
            "lens": getattr(metadata, "lens_label", ""),
            "settings": getattr(metadata, "settings_label", ""),
            "gear": getattr(metadata, "gear_label", ""),
            "focal": getattr(metadata, "focal_label", ""),
            "iso": f"ISO {metadata.iso}" if getattr(metadata, "iso", None) else "",
            "aperture": getattr(metadata, "aperture_label", ""),
            "shutter": getattr(metadata, "shutter_speed", None) or "",
            "date": captured.strftime("%d %b %Y") if captured else "",
            "year": str(captured.year) if captured else "",
        }
    values.update({k: str(v) for k, v in (extra or {}).items()})
    expanded = _TOKEN.sub(lambda m: values.get(m.group(1), ""), text)
    # Tidy the separators an empty token leaves behind.
    expanded = re.sub(r"(\s*[·|•,/-]\s*){2,}", " · ", expanded)
    return re.sub(r"^\s*[·|•,/-]\s*|\s*[·|•,/-]\s*$", "", expanded).strip()


# --------------------------------------------------------------------- #
# Snapping and guides: pure functions for the design workspace
# --------------------------------------------------------------------- #

def snap_to_grid(value: float, grid: float, threshold: float | None = None) -> float:
    """Move ``value`` to the nearest grid line if it is within ``threshold``."""
    if grid <= 0:
        return value
    nearest = round(value / grid) * grid
    limit = grid / 2.0 if threshold is None else threshold
    return nearest if abs(nearest - value) <= limit else value


def alignment_guides(moving, others, threshold: float = 0.01):
    """Where a dragged box lines up with others or with the canvas.

    Boxes are ``(x, y, w, h)`` in canvas fractions. Returns
    ``(dx, dy, guides)``: the nudge that snaps the moving box, and the guide
    lines to draw as ``("v", x)`` or ``("h", y)``. Edges and centres are
    compared, and the canvas's own edges and centre count as targets.
    """
    x, y, w, h = moving
    targets_x, targets_y = [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]
    for ox, oy, ow, oh in others:
        targets_x += [ox, ox + ow / 2.0, ox + ow]
        targets_y += [oy, oy + oh / 2.0, oy + oh]

    def best(own: list[float], targets: list[float]):
        found = None
        for value in own:
            for target in targets:
                gap = target - value
                if abs(gap) <= threshold and (found is None or abs(gap) < abs(found[0])):
                    found = (gap, target)
        return found

    guides = []
    snap_x = best([x, x + w / 2.0, x + w], targets_x)
    snap_y = best([y, y + h / 2.0, y + h], targets_y)
    if snap_x:
        guides.append(("v", snap_x[1]))
    if snap_y:
        guides.append(("h", snap_y[1]))
    return (snap_x[0] if snap_x else 0.0), (snap_y[0] if snap_y else 0.0), guides


# --------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------- #

def system_font_folder() -> str | None:
    """Where this operating system keeps its fonts, if that can be told."""
    import sys

    candidates = {
        "win32": [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")],
        "darwin": ["/System/Library/Fonts", "/Library/Fonts"],
    }.get(sys.platform, ["/usr/share/fonts"])
    return next((folder for folder in candidates if os.path.isdir(folder)), None)


def prepare_headless_fonts() -> None:
    """Point Qt's windowless mode at the system fonts.

    With a real window Qt asks the operating system for fonts. In its
    windowless mode it looks only in a folder of its own, finds nothing, and
    draws every letter as an empty box. Naming the system folder before the
    application object exists fixes that. Harmless when a window is used.
    """
    folder = system_font_folder()
    if folder:
        os.environ.setdefault("QT_QPA_FONTDIR", folder)


def _ensure_app():
    """Fonts need a Qt application object; make a headless one if needed."""
    from PyQt6.QtCore import QCoreApplication
    from PyQt6.QtGui import QGuiApplication

    if QCoreApplication.instance() is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        prepare_headless_fonts()
        _ensure_app.keep = QGuiApplication([])     # type: ignore[attr-defined]


def available_fonts() -> list[str]:
    """Font families installed on this computer."""
    _ensure_app()
    from PyQt6.QtGui import QFontDatabase

    return sorted(QFontDatabase.families())


def _qcolor(value: str | None, opacity: float = 100.0):
    from PyQt6.QtGui import QColor

    if not value:
        return None
    colour = QColor(int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16),
                    int(value[7:9], 16) if len(value) == 9 else 255)
    colour.setAlphaF(colour.alphaF() * max(0.0, min(1.0, opacity / 100.0)))
    return colour


def _font(item: Item):
    from PyQt6.QtGui import QFont

    font = QFont(item.font_family)
    font.setPixelSize(int(_REFERENCE_PX))
    font.setBold(item.bold)
    font.setItalic(item.italic)
    font.setKerning(item.kerning)
    font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, item.letter_spacing)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    return font


_ANCHOR_FACTORS = {
    "top-left": (0.0, 0.0), "top": (0.5, 0.0), "top-right": (1.0, 0.0),
    "left": (0.0, 0.5), "center": (0.5, 0.5), "right": (1.0, 0.5),
    "bottom-left": (0.0, 1.0), "bottom": (0.5, 1.0), "bottom-right": (1.0, 1.0),
}


@dataclass
class TextLayout:
    """Where each line of a text item goes, in output pixels, before the
    item's own rotation. ``origin`` is the anchor point."""

    lines: list            # (text, x, baseline_y, width)
    width: float
    height: float
    scale: float           # output pixels per reference pixel
    origin: tuple[float, float]


def layout_text(item: Item, size: tuple[int, int], metadata=None) -> TextLayout:
    """Lay a text item out for a canvas of ``size`` pixels."""
    _ensure_app()
    from PyQt6.QtGui import QFontMetricsF

    canvas_w, canvas_h = size
    metrics = QFontMetricsF(_font(item))
    scale = item.font_size * canvas_h / _REFERENCE_PX
    rows = expand_tokens(item.text, metadata).split("\n") if metadata is not None or "{" in item.text \
        else item.text.split("\n")
    widths = [metrics.horizontalAdvance(row) * scale for row in rows]
    block_w = max(widths) if widths else 0.0
    step = metrics.height() * item.line_spacing * scale
    block_h = metrics.height() * scale + step * (len(rows) - 1)

    fx, fy = _ANCHOR_FACTORS[item.anchor]
    left, top = -fx * block_w, -fy * block_h
    lines = []
    for index, (row, width) in enumerate(zip(rows, widths)):
        offset = {"left": 0.0, "center": (block_w - width) / 2.0, "right": block_w - width}[item.align]
        lines.append((row, left + offset, top + metrics.ascent() * scale + index * step, width))
    return TextLayout(lines, block_w, block_h, scale, (item.x * canvas_w, item.y * canvas_h))


def _text_path(item: Item, layout: TextLayout):
    """The text as outlines, in output pixels relative to the anchor."""
    from PyQt6.QtGui import QPainterPath, QTransform

    font = _font(item)
    path = QPainterPath()
    for row, x, baseline, _ in layout.lines:
        # Built at the reference size and scaled, so glyph positions are
        # identical at every output size.
        line = QPainterPath()
        line.addText(0.0, 0.0, font, row)
        path.addPath(QTransform().translate(x, baseline).scale(layout.scale, layout.scale).map(line))
    return path


def _shape_path(item: Item, size: tuple[int, int]):
    """A shape's outline in output pixels, centred on its own box centre."""
    from PyQt6.QtCore import QPointF, QRectF
    from PyQt6.QtGui import QPainterPath

    w, h = item.w * size[0], item.h * size[1]
    path = QPainterPath()
    if item.kind == "rect":
        radius = item.corner_radius * min(w, h)
        box = QRectF(-w / 2.0, -h / 2.0, w, h)
        path.addRoundedRect(box, radius, radius) if radius > 0 else path.addRect(box)
    elif item.kind == "ellipse":
        path.addEllipse(QRectF(-w / 2.0, -h / 2.0, w, h))
    elif item.kind == "star":
        count = max(3, item.points)
        for index in range(count * 2):
            reach = 1.0 if index % 2 == 0 else item.inner_ratio
            angle = -math.pi / 2.0 + index * math.pi / count
            point = QPointF(math.cos(angle) * reach * w / 2.0, math.sin(angle) * reach * h / 2.0)
            path.lineTo(point) if index else path.moveTo(point)
        path.closeSubpath()
    return path


def _free_path(item: Item, size: tuple[int, int]):
    from PyQt6.QtGui import QPainterPath

    sw, sh = size
    path = QPainterPath()
    for command in item.commands:
        op, v = command[0], command[1:]
        if op == "M":
            path.moveTo(v[0] * sw, v[1] * sh)
        elif op == "L":
            path.lineTo(v[0] * sw, v[1] * sh)
        elif op == "Q":
            path.quadTo(v[0] * sw, v[1] * sh, v[2] * sw, v[3] * sh)
        elif op == "C":
            path.cubicTo(v[0] * sw, v[1] * sh, v[2] * sw, v[3] * sh, v[4] * sw, v[5] * sh)
        elif op == "Z":
            path.closeSubpath()
    return path


def _stroke_and_fill(painter, path, item: Item, long_edge: float, opacity: float) -> None:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QBrush, QPen

    fill, stroke = _qcolor(item.fill, opacity), _qcolor(item.stroke, opacity)
    painter.setBrush(QBrush(fill) if fill is not None else Qt.BrushStyle.NoBrush)
    if stroke is not None and item.stroke_width > 0:
        pen = QPen(stroke, item.stroke_width * long_edge)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
    else:
        painter.setPen(Qt.PenStyle.NoPen)
    painter.drawPath(path)


def _paint_item(painter, item: Item, size: tuple[int, int], metadata, opacity: float,
                live_text: bool, shadows: list | None) -> None:
    """Draw one item. ``shadows`` collects raster shadow jobs when given."""
    from PyQt6.QtGui import QColor

    opacity = opacity * item.opacity / 100.0
    if opacity <= 0:
        return
    long_edge = float(max(size))

    if item.kind == "group":
        painter.save()
        painter.translate((item.x + item.w / 2.0) * size[0], (item.y + item.h / 2.0) * size[1])
        painter.rotate(item.rotation)
        painter.translate(-(item.x + item.w / 2.0) * size[0], -(item.y + item.h / 2.0) * size[1])
        for child in item.children:
            _paint_item(painter, child, size, metadata, opacity, live_text, shadows)
        painter.restore()
        return

    painter.save()
    layout = None
    if item.kind == "text":
        layout = layout_text(item, size, metadata)
        painter.translate(*layout.origin)
        painter.rotate(item.rotation)
        path = _text_path(item, layout)
    elif item.kind == "path":
        path = _free_path(item, size)
        centre = path.boundingRect().center()
        painter.translate(centre)
        painter.rotate(item.rotation)
        painter.translate(-centre)
    else:
        painter.translate((item.x + item.w / 2.0) * size[0], (item.y + item.h / 2.0) * size[1])
        painter.rotate(item.rotation)
        path = _shape_path(item, size)

    if item.shadow is not None:
        shadow = item.shadow
        if shadows is not None:
            # Raster output: remembered and blurred by the caller.
            shadows.append((painter.transform(), path, shadow, opacity, long_edge))
        else:
            painter.save()
            painter.translate(shadow.dx * long_edge, shadow.dy * long_edge)
            ghost = Item(kind=item.kind, fill=shadow.color, stroke=None)
            _stroke_and_fill(painter, path, ghost, long_edge, opacity)
            painter.restore()

    if item.kind == "text" and live_text and layout is not None:
        # Real text, so an SVG stays selectable and searchable. An outline,
        # if there is one, goes underneath as a path.
        if item.stroke and item.stroke_width > 0:
            _stroke_and_fill(painter, path, Item(kind="text", fill=None, stroke=item.stroke,
                                                 stroke_width=item.stroke_width), long_edge, opacity)
        font = _font(item)
        font.setPixelSize(max(1, int(round(_REFERENCE_PX * layout.scale))))
        painter.setFont(font)
        painter.setPen(_qcolor(item.fill, opacity) or QColor(0, 0, 0, 0))
        for row, x, baseline, _ in layout.lines:
            painter.drawText(int(round(x)), int(round(baseline)), row)
    else:
        _stroke_and_fill(painter, path, item, long_edge, opacity)
    painter.restore()


def paint(painter, items: list[Item], size: tuple[int, int], metadata=None,
          live_text: bool = False, shadows: list | None = None) -> None:
    """Draw a design onto any Qt painter, for a canvas of ``size`` pixels."""
    from PyQt6.QtGui import QPainter

    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    for item in items:
        _paint_item(painter, item, size, metadata, 100.0, live_text, shadows)


# --------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------- #

def _image_to_rgba(image) -> np.ndarray:
    from PyQt6.QtGui import QImage

    image = image.convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = image.width(), image.height()
    buffer = image.constBits()
    buffer.setsize(image.sizeInBytes())
    array = np.frombuffer(buffer, np.uint8).reshape(h, image.bytesPerLine() // 4, 4)[:, :w]
    return array.astype(np.float32) / 255.0


def render_rgba(items: list[Item], size: tuple[int, int], metadata=None) -> np.ndarray:
    """Rasterise a design to float RGBA (straight alpha), H x W x 4.

    ``size`` is ``(width, height)`` in pixels; ask for whatever the output
    needs, since nothing in a design is tied to a resolution.
    """
    _ensure_app()
    from PyQt6.QtGui import QImage, QPainter

    w, h = max(1, int(size[0])), max(1, int(size[1]))
    canvas = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    canvas.fill(0)
    shadows: list = []
    painter = QPainter(canvas)
    paint(painter, items, (w, h), metadata, shadows=shadows)
    painter.end()
    top = _image_to_rgba(canvas)
    if not shadows:
        return top

    import cv2

    # Shadows go underneath everything, each blurred by its own radius.
    under = np.zeros((h, w, 4), np.float32)
    for transform, path, shadow, opacity, long_edge in shadows:
        layer = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        layer.fill(0)
        brush = QPainter(layer)
        brush.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        brush.translate(shadow.dx * long_edge, shadow.dy * long_edge)
        brush.setTransform(transform, True)
        _stroke_and_fill(brush, path, Item(fill=shadow.color, stroke=None), long_edge, opacity)
        brush.end()
        drawn = _image_to_rgba(layer)
        sigma = shadow.blur * long_edge
        if sigma > 0.3:
            premultiplied = np.dstack([drawn[:, :, :3] * drawn[:, :, 3:4], drawn[:, :, 3]])
            premultiplied = cv2.GaussianBlur(premultiplied, (0, 0), sigma)
            alpha = premultiplied[:, :, 3:4]
            drawn = np.dstack([premultiplied[:, :, :3] / np.maximum(alpha, 1e-6), alpha[:, :, 0]])
        under = over(drawn, under)
    return over(top, under)


def over(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Composite one straight-alpha RGBA image over another."""
    ta, ba = top[:, :, 3:4], bottom[:, :, 3:4]
    alpha = ta + ba * (1.0 - ta)
    colour = (top[:, :, :3] * ta + bottom[:, :, :3] * ba * (1.0 - ta)) / np.maximum(alpha, 1e-6)
    return np.dstack([colour, alpha[:, :, 0]]).astype(np.float32)


def composite_onto(image: np.ndarray, items: list[Item], metadata=None) -> np.ndarray:
    """Draw a design over an RGB image of any supported dtype."""
    if not items:
        return image
    h, w = image.shape[:2]
    overlay = render_rgba(items, (w, h), metadata)
    scale = {np.dtype(np.uint8): 255.0, np.dtype(np.uint16): 65535.0}.get(image.dtype, 1.0)
    base = image.astype(np.float32) / scale
    alpha = overlay[:, :, 3:4]
    mixed = np.clip(base * (1.0 - alpha) + overlay[:, :, :3] * alpha, 0.0, 1.0)
    return mixed if scale == 1.0 else np.round(mixed * scale).astype(image.dtype)


def render_layer(vector: dict, size: tuple[int, int]) -> np.ndarray | None:
    """Adapter for the layer stack: one item's dict to float RGBA."""
    item = item_from_dict(vector)
    return render_rgba([item], size) if item is not None else None


def pixels_for_print(width_inches: float, height_inches: float, dpi: int = 300) -> tuple[int, int]:
    return int(round(width_inches * dpi)), int(round(height_inches * dpi))


def export_svg(items: list[Item], path: str | Path, size: tuple[int, int], metadata=None,
               outline_text: bool = False, title: str = "Granite Studios design") -> Path:
    """Write a design as SVG, ``size`` user units wide and high.

    The document's width, height and viewBox are all ``size``, so one user
    unit is one pixel of the canvas the design was made for.
    """
    _ensure_app()
    from PyQt6.QtCore import QRect, QSize
    from PyQt6.QtGui import QPainter
    from PyQt6.QtSvg import QSvgGenerator

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = int(size[0]), int(size[1])
    generator = QSvgGenerator()
    generator.setFileName(str(path))
    generator.setSize(QSize(w, h))
    generator.setViewBox(QRect(0, 0, w, h))
    generator.setResolution(96)
    generator.setTitle(title)
    painter = QPainter(generator)
    paint(painter, items, (w, h), metadata, live_text=not outline_text)
    painter.end()
    return path


def export_pdf(items: list[Item], path: str | Path, size_inches: tuple[float, float],
               metadata=None, dpi: int = 300, background: np.ndarray | None = None) -> Path:
    """Write a design as a one-page PDF of the given physical size.

    The design stays vector. ``background`` is an optional RGB uint8 image
    drawn under it, filling the page.
    """
    _ensure_app()
    from PyQt6.QtCore import QMarginsF, QRectF, QSizeF
    from PyQt6.QtGui import QImage, QPageLayout, QPageSize, QPainter, QPdfWriter

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = QPdfWriter(str(path))
    writer.setResolution(int(dpi))
    page = QPageSize(QSizeF(size_inches[0], size_inches[1]), QPageSize.Unit.Inch)
    writer.setPageLayout(QPageLayout(page, QPageLayout.Orientation.Portrait, QMarginsF(0, 0, 0, 0)))
    w, h = pixels_for_print(size_inches[0], size_inches[1], dpi)
    painter = QPainter(writer)
    if background is not None:
        picture = np.ascontiguousarray(background[:, :, :3])
        frame = QImage(picture.data, picture.shape[1], picture.shape[0], picture.shape[1] * 3,
                       QImage.Format.Format_RGB888)
        painter.drawImage(QRectF(0, 0, w, h), frame)
    paint(painter, items, (w, h), metadata)
    painter.end()
    return path


# --------------------------------------------------------------------- #
# Watermark templates
# --------------------------------------------------------------------- #

_CORNER = {
    "bottom-right": (0.97, 0.96), "bottom-left": (0.03, 0.96), "bottom": (0.5, 0.96),
    "top-right": (0.97, 0.04), "top-left": (0.03, 0.04), "top": (0.5, 0.04), "center": (0.5, 0.5),
}
_CORNER_ALIGN = {"left": "left", "right": "right"}


def watermark_signature(name: str, position: str = "bottom-right", size: float = 0.032,
                        color: str = "#ffffffd9", font_family: str = "Segoe Script") -> list[Item]:
    """A signature in a corner, with a soft shadow so it reads on any tone."""
    x, y = _CORNER.get(position, _CORNER["bottom-right"])
    side = position.split("-")[-1]
    return [Item(
        kind="text", name="Signature", text=name, x=x, y=y, anchor=position if position in ANCHORS else "bottom-right",
        align=_CORNER_ALIGN.get(side, "center"), font_family=font_family, font_size=size, fill=color,
        shadow=Shadow(0.0015, 0.0015, 0.003, "#00000080"),
    )]


def watermark_seal(name: str, tagline: str = "PHOTOGRAPHY", position: str = "bottom-right",
                   diameter: float = 0.16, color: str = "#ffffffcc") -> list[Item]:
    """A round photographer's seal: two rings, a star, a name and a tagline.

    ``diameter`` is a fraction of the canvas height. The seal is built for
    a 3:2 canvas; on other shapes it stays round because its width is
    derived from the height where it is drawn.
    """
    cx, cy = _CORNER.get(position, _CORNER["bottom-right"])
    half = diameter / 2.0
    cx = min(1.0 - half * 0.75, max(half * 0.75, cx + (-half * 0.7 if cx > 0.5 else half * 0.7 if cx < 0.5 else 0)))
    cy = min(1.0 - half * 1.05, max(half * 1.05, cy + (-half if cy > 0.5 else half if cy < 0.5 else 0)))
    aspect = 2.0 / 3.0      # height over width of the assumed canvas
    ring_w, ring_h = diameter * aspect, diameter
    # Shrink a long name to fit inside the ring. A bold capital is roughly
    # 0.7 of the font size wide, and the inner ring is 0.88 of the diameter.
    name_size = min(diameter * 0.135, 0.70 * diameter / (0.7 * max(len(name), 1)))
    return [Item(kind="group", name="Seal", x=cx - ring_w / 2, y=cy - ring_h / 2, w=ring_w, h=ring_h, children=[
        Item(kind="ellipse", x=cx - ring_w / 2, y=cy - ring_h / 2, w=ring_w, h=ring_h,
             fill=None, stroke=color, stroke_width=0.0022),
        Item(kind="ellipse", x=cx - ring_w * 0.44, y=cy - ring_h * 0.44, w=ring_w * 0.88, h=ring_h * 0.88,
             fill=None, stroke=color, stroke_width=0.0009),
        Item(kind="star", x=cx - ring_w * 0.07, y=cy - ring_h * 0.36, w=ring_w * 0.14, h=ring_h * 0.14,
             fill=color, points=5),
        Item(kind="text", text=name, x=cx, y=cy, anchor="center", align="center",
             font_size=name_size, bold=True, letter_spacing=104, fill=color),
        Item(kind="text", text=tagline, x=cx, y=cy + ring_h * 0.2, anchor="center", align="center",
             font_size=diameter * 0.052, letter_spacing=150, fill=color),
    ])]


def watermark_exif_strip(template: str = "{camera}  ·  {lens}  ·  {settings}",
                         height: float = 0.055, color: str = "#ffffffff",
                         band: str = "#00000073") -> list[Item]:
    """A band along the bottom carrying the photograph's own shooting data.

    The text holds tokens, filled in from each photograph when it is drawn,
    so one template serves a whole batch.
    """
    return [
        Item(kind="rect", name="EXIF band", x=0.0, y=1.0 - height, w=1.0, h=height, fill=band),
        Item(kind="text", name="EXIF text", text=template, x=0.5, y=1.0 - height / 2.0, anchor="center",
             align="center", font_size=height * 0.42, fill=color, letter_spacing=103),
    ]


WATERMARK_TEMPLATES = {
    "signature": watermark_signature,
    "seal": watermark_seal,
    "exif_strip": watermark_exif_strip,
}
