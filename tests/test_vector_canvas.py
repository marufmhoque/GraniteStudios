"""The vector canvas: one design, drawn to pixels, SVG and PDF."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pytest

from src.engine import develop_pipeline as dp
from src.engine import layer_stack as ls
from src.engine.parametric_recipe import Recipe
from src.engine.universal_loader import ImageMetadata
from src.graphics import vector_canvas as vc

SVG = "{http://www.w3.org/2000/svg}"


def _alpha(items, size=(600, 400)) -> np.ndarray:
    return vc.render_rgba(items, size)[:, :, 3]


def _box(mask) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0.5)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


@pytest.fixture(scope="module", autouse=True)
def _fonts():
    if not vc.available_fonts():
        pytest.skip("no fonts are available to headless Qt on this machine")


# --------------------------------------------------------------------- #
# Reading designs
# --------------------------------------------------------------------- #

def test_items_round_trip():
    item = vc.Item(kind="text", text="Hi", x=0.3, y=0.4, rotation=15, bold=True, letter_spacing=120,
                   shadow=vc.Shadow(), stroke="#102030ff", anchor="center", align="center")
    assert vc.item_from_dict(vc.item_to_dict(item)) == item
    group = vc.Item(kind="group", children=[item, vc.Item(kind="star", points=7)])
    assert vc.item_from_dict(vc.item_to_dict(group)) == group


@pytest.mark.parametrize("bad", [None, 4, "text", {"kind": "hologram"}, {}])
def test_unusable_items_are_dropped(bad):
    assert vc.item_from_dict(bad) is None
    assert vc.items_from([bad, {"kind": "rect"}]) == [vc.item_from_dict({"kind": "rect"})]


def test_bad_fields_fall_back():
    item = vc.item_from_dict({
        "kind": "path", "x": "left", "opacity": 900, "fill": "red", "stroke": "#12345",
        "anchor": "middle-ish", "align": "justify", "points": "many", "font_size": float("nan"),
        "commands": [["M", 0.1, 0.1], ["L", 0.5], ["X", 1, 2], "junk", ["C", 0, 0, 0, 0, 1, 1], ["Z"]],
    })
    assert item.x == 0.0 and item.opacity == 100.0 and item.fill == "#ffffffff" and item.stroke is None
    assert item.anchor == "top-left" and item.align == "left" and item.points == 5
    assert [c[0] for c in item.commands] == ["M", "C", "Z"], "malformed path commands should be skipped"
    assert vc.items_from("not a list") == []


# --------------------------------------------------------------------- #
# Shapes land where they are put
# --------------------------------------------------------------------- #

def test_a_rectangle_fills_its_box():
    mask = _alpha([vc.Item(kind="rect", x=0.25, y=0.25, w=0.5, h=0.5)])
    assert _box(mask) == pytest.approx((150, 100, 449, 299), abs=1)
    assert mask[200, 300] == pytest.approx(1.0) and mask[10, 10] == 0.0


def test_rounded_corners_and_ellipses_cut_the_corners():
    square = _alpha([vc.Item(kind="rect", x=0.25, y=0.25, w=0.5, h=0.5)])
    rounded = _alpha([vc.Item(kind="rect", x=0.25, y=0.25, w=0.5, h=0.5, corner_radius=0.4)])
    ellipse = _alpha([vc.Item(kind="ellipse", x=0.25, y=0.25, w=0.5, h=0.5)])
    assert square[102, 152] > 0.9 and rounded[102, 152] < 0.1 and ellipse[102, 152] < 0.1
    assert ellipse.sum() == pytest.approx(math.pi / 4.0 * square.sum(), rel=0.02)
    assert ellipse.sum() < rounded.sum() < square.sum()


def test_a_star_has_its_points():
    def points_of(count: int) -> int:
        mask = _alpha([vc.Item(kind="star", x=0.3, y=0.2, w=0.4, h=0.6, points=count, inner_ratio=0.4)])
        # Walk a ring between the inner and outer radius and count crossings.
        angles = np.linspace(0, 2 * math.pi, 1440, endpoint=False)
        xs = (300 + np.cos(angles) * 0.4 * 600 * 0.35).astype(int)
        ys = (200 + np.sin(angles) * 0.6 * 400 * 0.35).astype(int)
        ring = mask[ys, xs] > 0.5
        return int(np.sum(ring & ~np.roll(ring, 1)))

    assert points_of(5) == 5 and points_of(8) == 8


def test_a_bezier_path_is_drawn_and_closed():
    path = vc.Item(kind="path", fill="#ffffffff", commands=[
        ["M", 0.2, 0.8], ["C", 0.2, 0.2, 0.8, 0.2, 0.8, 0.8], ["Z"],
    ])
    mask = _alpha([path])
    assert mask[220, 300] > 0.9, "the inside of the closed curve is not filled"
    assert mask[20, 300] == 0.0 and mask[390, 300] == 0.0
    x0, y0, x1, y1 = _box(mask)
    assert (x0, x1) == pytest.approx((120, 479), abs=2)
    assert y0 == pytest.approx(0.35 * 400, abs=3), "a cubic with these handles peaks at three quarters of their height"


def test_stroke_only_and_opacity():
    ring = vc.render_rgba([vc.Item(kind="rect", x=0.25, y=0.25, w=0.5, h=0.5, fill=None,
                                   stroke="#ff0000ff", stroke_width=0.01)], (600, 400))
    assert ring[200, 300, 3] == 0.0, "an unfilled shape was filled"
    assert ring[100, 300, 3] > 0.9 and ring[100, 300, 0] > 0.9 and ring[100, 300, 1] < 0.1
    half = vc.render_rgba([vc.Item(kind="rect", x=0, y=0, w=1, h=1, opacity=50)], (40, 40))
    assert half[20, 20, 3] == pytest.approx(0.5, abs=0.01)


def test_rotation_turns_a_shape_about_its_centre():
    flat = _alpha([vc.Item(kind="rect", x=0.3, y=0.45, w=0.4, h=0.1)])
    turned = _alpha([vc.Item(kind="rect", x=0.3, y=0.45, w=0.4, h=0.1, rotation=90)])
    fx0, fy0, fx1, fy1 = _box(flat)
    tx0, ty0, tx1, ty1 = _box(turned)
    assert (fx1 - fx0) == pytest.approx(ty1 - ty0, abs=2) and (fy1 - fy0) == pytest.approx(tx1 - tx0, abs=2)
    assert ((tx0 + tx1) / 2, (ty0 + ty1) / 2) == pytest.approx((300, 200), abs=1.5)


def test_a_group_turns_its_children_together():
    child = vc.Item(kind="rect", x=0.6, y=0.45, w=0.1, h=0.1)
    group = vc.Item(kind="group", x=0.25, y=0.25, w=0.5, h=0.5, rotation=180, children=[child])
    x0, _, x1, _ = _box(_alpha([group]))
    assert (x0 + x1) / 2 == pytest.approx(0.35 * 600, abs=2), "the child was not carried round the group's centre"


# --------------------------------------------------------------------- #
# Text
# --------------------------------------------------------------------- #

def _text(**overrides) -> vc.Item:
    base = dict(kind="text", text="Granite", x=0.5, y=0.5, anchor="center", font_size=0.2, fill="#ffffffff")
    return vc.Item(**{**base, **overrides})


def test_text_is_drawn_and_centred_on_its_anchor():
    x0, y0, x1, y1 = _box(_alpha([_text()]))
    assert (x0 + x1) / 2 == pytest.approx(300, abs=6) and (y0 + y1) / 2 == pytest.approx(200, abs=25)
    assert x1 - x0 > 100


@pytest.mark.parametrize("anchor,edge", [("top-left", "left"), ("bottom-right", "right")])
def test_anchors_pin_the_named_corner(anchor, edge):
    x0, y0, x1, y1 = _box(_alpha([_text(anchor=anchor, x=0.5, y=0.5)]))
    if edge == "left":
        assert x0 >= 298 and y0 >= 198
    else:
        assert x1 <= 302 and y1 <= 202


def test_text_size_is_a_fraction_of_the_canvas_height():
    small = vc.layout_text(_text(font_size=0.1), (600, 400))
    large = vc.layout_text(_text(font_size=0.2), (600, 400))
    assert large.width == pytest.approx(2 * small.width, rel=0.01)
    assert large.height == pytest.approx(2 * small.height, rel=0.01)


def test_text_is_the_same_design_at_every_resolution():
    """Laid out once and scaled, so nothing shifts between preview and print."""
    small = vc.layout_text(_text(text="Wavy AV To"), (600, 400))
    large = vc.layout_text(_text(text="Wavy AV To"), (6000, 4000))
    assert large.width == pytest.approx(10 * small.width, rel=1e-6)
    a = _alpha([_text(text="Wavy AV To")], (600, 400)).sum() / (600 * 400)
    b = _alpha([_text(text="Wavy AV To")], (1800, 1200)).sum() / (1800 * 1200)
    assert a == pytest.approx(b, rel=0.03)


def test_letter_spacing_bold_and_kerning_change_the_width():
    def width(**kw):
        return vc.layout_text(_text(text="AVATAR Wave", **kw), (600, 400)).width

    assert width(letter_spacing=150) > width() * 1.3
    assert width(bold=True) > width()
    assert width(kerning=False) > width(), "kerning should pull AV and Wa together"


def test_lines_and_line_spacing():
    one = vc.layout_text(_text(text="Line"), (600, 400))
    two = vc.layout_text(_text(text="Line\nLine"), (600, 400))
    loose = vc.layout_text(_text(text="Line\nLine", line_spacing=2.0), (600, 400))
    assert len(two.lines) == 2 and two.height > one.height * 1.8
    assert loose.height > two.height * 1.4
    assert two.lines[1][2] > two.lines[0][2]


@pytest.mark.parametrize("align", ["left", "center", "right"])
def test_alignment_of_a_short_line_under_a_long_one(align):
    layout = vc.layout_text(_text(text="A much longer line\nshort", align=align, anchor="top-left"), (600, 400))
    (_, long_x, _, long_w), (_, short_x, _, short_w) = layout.lines
    expected = {"left": long_x, "center": long_x + (long_w - short_w) / 2, "right": long_x + long_w - short_w}[align]
    assert short_x == pytest.approx(expected, abs=1e-6)


def test_outline_and_shadow():
    plain = vc.render_rgba([_text()], (600, 400))
    outlined = vc.render_rgba([_text(stroke="#ff0000ff", stroke_width=0.01)], (600, 400))
    assert outlined[:, :, 3].sum() > plain[:, :, 3].sum() * 1.05
    shadowed = vc.render_rgba([_text(shadow=vc.Shadow(0.02, 0.02, 0.01, "#000000ff"))], (600, 400))
    extra = (shadowed[:, :, 3] > 0.3) & (plain[:, :, 3] < 0.05)
    assert extra.sum() > 200, "no shadow was drawn"
    assert shadowed[extra][:, :3].max() < 0.3, "the shadow is not dark"
    ys, xs = np.where(extra)
    px0, py0, px1, py1 = _box(plain[:, :, 3])
    assert xs.mean() > (px0 + px1) / 2 and ys.mean() > (py0 + py1) / 2, "the shadow is not offset down and right"
    soft = np.unique(np.round(shadowed[:, :, 3][extra], 2))
    assert len(soft) > 10, "the shadow edge is not blurred"


# --------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------- #

def _meta() -> ImageMetadata:
    return ImageMetadata(camera_make="Canon", camera_model="EOS R6", lens_model="RF 50mm F1.8",
                         iso=400, aperture=2.8, shutter_speed="1/250s", focal_length=50.0,
                         captured_at=datetime(2025, 6, 1, 9, 30))


def test_tokens_are_filled_from_the_photograph():
    text = vc.expand_tokens("{camera} · {lens} · {iso} · {aperture} · {shutter} · {year}", _meta())
    assert text == "Canon EOS R6 · RF 50mm F1.8 · ISO 400 · f/2.8 · 1/250s · 2025"
    assert vc.expand_tokens("{date}", _meta()) == "01 Jun 2025"


def test_missing_values_leave_no_stray_separators():
    sparse = ImageMetadata(camera_make="Canon", camera_model="EOS R6")
    assert vc.expand_tokens("{camera} · {lens} · {settings}", sparse) == "Canon EOS R6"
    assert vc.expand_tokens("{lens} · {camera}", sparse) == "Canon EOS R6"
    assert vc.expand_tokens("{nonsense} plain", None) == "plain"
    assert vc.expand_tokens("By {who}", None, {"who": "Maruf"}) == "By Maruf"


# --------------------------------------------------------------------- #
# Snapping
# --------------------------------------------------------------------- #

def test_snap_to_grid():
    assert vc.snap_to_grid(0.26, 0.05) == pytest.approx(0.25)
    assert vc.snap_to_grid(0.262, 0.05, threshold=0.005) == 0.262, "outside the threshold it should stay put"
    assert vc.snap_to_grid(0.3, 0) == 0.3


def test_alignment_guides_find_edges_and_centres():
    other = (0.6, 0.2, 0.2, 0.2)
    dx, dy, guides = vc.alignment_guides((0.105, 0.205, 0.2, 0.1), [other])
    # Its top lines up with the other box's top and its bottom with that
    # box's centre line; both are 0.005 away, and either is a correct snap.
    assert dy == pytest.approx(-0.005)
    assert [g for g in guides if g[0] == "h" and min(abs(g[1] - 0.2), abs(g[1] - 0.3)) < 1e-9]
    dx, dy, guides = vc.alignment_guides((0.395, 0.7, 0.2, 0.1), [])
    assert dx == pytest.approx(0.005), "the box's centre should snap to the canvas centre"
    assert ("v", 0.5) in guides
    assert vc.alignment_guides((0.13, 0.63, 0.2, 0.1), [other])[2] == []


# --------------------------------------------------------------------- #
# SVG
# --------------------------------------------------------------------- #

def _svg(tmp_path, items, size=(1200, 800), **kw):
    path = vc.export_svg(items, tmp_path / "design.svg", size, **kw)
    return ET.parse(path).getroot(), path.read_text(encoding="utf-8")


def _mm(value: str) -> float:
    return float(value.replace("mm", ""))


def test_svg_matches_the_pixel_dimensions(tmp_path):
    root, _ = _svg(tmp_path, [vc.Item(kind="rect", x=0.25, y=0.25, w=0.5, h=0.5)], (1200, 800))
    assert [float(v) for v in root.get("viewBox").split()] == [0, 0, 1200, 800]
    # The physical size is the same pixels at the SVG standard of 96 per inch.
    assert _mm(root.get("width")) * 96 / 25.4 == pytest.approx(1200, abs=0.5)
    assert _mm(root.get("height")) * 96 / 25.4 == pytest.approx(800, abs=0.5)


def _numbers(text: str) -> list[float]:
    return [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", text)]


def test_svg_shapes_are_vector_and_in_the_right_place(tmp_path):
    root, text = _svg(tmp_path, [vc.Item(kind="rect", x=0.25, y=0.25, w=0.5, h=0.5, fill="#336699ff")])
    assert "<image" not in text, "the design was embedded as a picture instead of drawn"
    paths = root.iter(f"{SVG}path")
    assert any(p.get("d") for p in paths)
    assert "#336699" in text.lower()
    group = next(g for g in root.iter(f"{SVG}g") if list(g.iter(f"{SVG}path")) and g.get("transform"))
    a, b, c, d, e, f = _numbers(group.get("transform"))
    assert (e, f) == pytest.approx((600, 400), abs=0.5), "the shape is not centred where it was placed"


@pytest.mark.parametrize("angle", [0.0, 30.0, -45.0, 90.0])
def test_svg_text_carries_its_content_position_and_rotation(tmp_path, angle):
    item = vc.Item(kind="text", text="Granite Studios", x=0.25, y=0.5, rotation=angle, font_size=0.05,
                   font_family="Segoe UI", fill="#ffffffff")
    root, _ = _svg(tmp_path, [item])
    found = [(g, t) for g in root.iter(f"{SVG}g") for t in g.findall(f"{SVG}text")]
    assert found, "the text was not written as text"
    group, element = found[0]
    assert "".join(element.itertext()).strip() == "Granite Studios"
    a, b, c, d, e, f = _numbers(group.get("transform"))
    assert math.degrees(math.atan2(b, a)) == pytest.approx(angle, abs=0.1), "the rotation did not survive"
    assert math.hypot(a, b) == pytest.approx(1.0, abs=1e-3), "the text was scaled as well as turned"
    assert (e, f) == pytest.approx((0.25 * 1200, 0.5 * 800), abs=0.5), "the anchor point moved"
    assert float(group.get("font-size")) == pytest.approx(0.05 * 800, abs=1.0)
    assert "Segoe UI" in group.get("font-family")


def test_svg_text_can_be_written_as_outlines(tmp_path):
    item = vc.Item(kind="text", text="Outline", x=0.5, y=0.5, anchor="center", font_size=0.1)
    root, _ = _svg(tmp_path, [item], outline_text=True)
    assert not list(root.iter(f"{SVG}text"))
    assert any(len(p.get("d", "")) > 200 for p in root.iter(f"{SVG}path")), "no glyph outlines were written"


def test_svg_tokens_are_expanded(tmp_path):
    _, text = _svg(tmp_path, vc.watermark_exif_strip("{camera} · {iso}"), metadata=_meta())
    assert "Canon EOS R6" in text and "ISO 400" in text and "{camera}" not in text


# --------------------------------------------------------------------- #
# Raster at print size, and PDF
# --------------------------------------------------------------------- #

def test_print_size_and_resolution_independence():
    assert vc.pixels_for_print(6, 4, 300) == (1800, 1200)
    item = vc.Item(kind="ellipse", x=0.2, y=0.3, w=0.3, h=0.4)
    small = _alpha([item], (300, 200))
    large = _alpha([item], vc.pixels_for_print(6, 4, 300))
    assert large.shape == (1200, 1800)
    assert large.mean() == pytest.approx(small.mean(), rel=0.01), "the design changed size with the resolution"
    assert vc.render_rgba([], (20, 10)).shape == (10, 20, 4)


def test_pdf_is_written_at_the_requested_size(tmp_path):
    path = vc.export_pdf(vc.watermark_signature("Maruf Hoque"), tmp_path / "out.pdf", (6, 4),
                         background=np.full((40, 60, 3), 90, np.uint8))
    data = path.read_bytes()
    assert data.startswith(b"%PDF-")
    box = re.search(rb"/MediaBox\s*\[\s*([\d.\s]+)\]", data)
    assert box, "no page size in the PDF"
    _, _, width, height = (float(v) for v in box.group(1).split())
    assert (width, height) == pytest.approx((6 * 72, 4 * 72), abs=1.0)        # PDF units are 72 per inch


# --------------------------------------------------------------------- #
# Watermarks, and through the pipeline
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("name,args", [
    ("signature", ("Maruf Hoque",)), ("seal", ("M. HOQUE",)), ("exif_strip", ()),
])
def test_watermark_templates_draw_inside_the_frame(name, args):
    items = vc.WATERMARK_TEMPLATES[name](*args)
    assert vc.items_from([vc.item_to_dict(i) for i in items]) == items, "a template is not plain data"
    rgba = vc.render_rgba(items, (900, 600), _meta())
    assert rgba[:, :, 3].max() > 0.5, "the watermark drew nothing"
    edge = np.concatenate([rgba[0, :, 3], rgba[:, 0, 3], rgba[:, -1, 3]])
    if name != "exif_strip":
        assert edge.max() < 0.2 and rgba[-1, :, 3].max() < 0.2, "the watermark runs off the frame"


def test_a_signature_goes_in_the_corner_it_is_asked_for():
    for position, right, bottom in (("bottom-right", True, True), ("top-left", False, False)):
        x0, y0, x1, y1 = _box(_alpha(vc.watermark_signature("Maruf", position=position), (900, 600)))
        assert ((x0 + x1) / 2 > 450) is right and ((y0 + y1) / 2 > 300) is bottom


def test_compositing_keeps_the_dtype_and_leaves_the_rest_alone():
    for dtype, grey in ((np.uint8, 100), (np.uint16, 25000)):
        image = np.full((200, 300, 3), grey, dtype)
        out = vc.composite_onto(image, [vc.Item(kind="rect", x=0.5, y=0.5, w=0.4, h=0.4, fill="#ffffffff")])
        assert out.dtype == dtype and out[20, 20, 0] == grey and out[140, 210, 0] == np.iinfo(dtype).max
    untouched = np.zeros((10, 10, 3), np.uint8)
    assert vc.composite_onto(untouched, []) is untouched


def test_recipe_vectors_are_drawn_over_the_final_framing():
    image = np.full((400, 600, 3), 60, np.uint8)
    recipe = Recipe()
    recipe.vectors = [vc.item_to_dict(vc.Item(kind="rect", x=0.9, y=0.9, w=0.1, h=0.1, fill="#ffffffff"))]
    assert dp.render(dp.from_array(image), recipe)[-5, -5, 0] == 255
    recipe.geometry.crop = [0.0, 0.0, 0.5, 0.5]
    cropped = dp.render(dp.from_array(image), recipe)
    assert cropped.shape[:2] == (200, 300)
    assert cropped[-5, -5, 0] == 255, "the watermark should sit in the corner of the cropped frame"


def test_text_and_shape_layers_render_through_the_stack():
    image = np.full((200, 300, 3), 40, np.uint8)
    layers = [ls.layer_from_dict(d) for d in (
        {"kind": "shape", "vector": {"kind": "ellipse", "x": 0.1, "y": 0.1, "w": 0.3, "h": 0.4, "fill": "#ff0000ff"}},
        {"kind": "text", "opacity": 50,
         "vector": {"text": "Hi", "x": 0.75, "y": 0.5, "anchor": "center", "font_size": 0.4, "fill": "#ffffffff"}},
    )]
    ctx = ls.RenderContext()
    out = ls.composite(image, layers, ctx)
    assert ctx.notes == []
    assert tuple(out[60, 75]) == (255, 0, 0)
    right = out[:, 150:]
    assert 100 < right.max() < 200, "a half-opacity white text layer should land mid-grey"
