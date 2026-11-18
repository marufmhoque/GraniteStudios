"""The edge-aware Color Splash Brush.

SB-01  a stroke stops at a hard edge; erasing takes it back
SB-02  sensitivity, snapping off, drags without gaps, the live preview
SB-03  the painted mask is a layer: it renders and exports, originals untouched
SB-04  the Retouch tool: painting, size keys, invert, reset, the lock
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.ai_tools.smart_brush import SmartBrushEngine, reveal_preview, tolerances
from src.engine import export_presets as ep
from src.engine import layer_stack
from src.engine import parametric_recipe as pr
from tests.conftest import assert_originals_untouched, pump, snapshot

RED, BLUE = (220, 30, 30), (30, 40, 220)


def _halves(h=200, w=300, left=RED, right=BLUE, noise=8, seed=0) -> np.ndarray:
    image = np.zeros((h, w, 3), np.uint8)
    image[:, : w // 2] = left
    image[:, w // 2:] = right
    rng = np.random.default_rng(seed)
    return np.clip(image.astype(int) + rng.integers(-noise, noise + 1, image.shape), 0, 255).astype(np.uint8)


def _circle(shape, centre, radius) -> np.ndarray:
    ys, xs = np.mgrid[: shape[0], : shape[1]]
    return (xs - centre[0]) ** 2 + (ys - centre[1]) ** 2 <= radius ** 2


# --------------------------------------------------------------------- #
# SB-01
# --------------------------------------------------------------------- #

def test_the_stroke_stops_at_the_edge():
    image = _halves()
    mask = np.zeros(image.shape[:2], np.uint8)
    mask, rect = SmartBrushEngine().paint_stroke(image, mask, (140, 100), 40, edge_sensitivity=0.6)
    circle = _circle(mask.shape, (140, 100), 40)
    xs = np.arange(mask.shape[1])[None, :].repeat(mask.shape[0], 0)
    assert (mask[circle & (xs < 150)] > 0).all(), "the red side under the brush is painted, to the edge"
    assert mask[:, 150:].max() == 0, "nothing crosses onto the blue side"
    assert (rect.x, rect.y, rect.w, rect.h) == (100, 60, 81, 81)


def test_erase_takes_the_paint_back():
    image = _halves()
    engine = SmartBrushEngine()
    mask = np.zeros(image.shape[:2], np.uint8)
    engine.paint_stroke(image, mask, (140, 100), 40)
    assert mask.max() == 255
    engine.paint_stroke(image, mask, (140, 100), 40, is_erase=True)
    assert mask[_circle(mask.shape, (140, 100), 40)].max() == 0
    assert mask.max() == 0


def test_erase_stops_at_the_edge_too():
    image = _halves()
    engine = SmartBrushEngine()
    mask = np.full(image.shape[:2], 255, np.uint8)
    engine.paint_stroke(image, mask, (140, 100), 40, is_erase=True)
    assert mask[:, 150:].min() == 255, "erasing the red side leaves the blue side revealed"


def test_the_mask_must_match_the_image():
    with pytest.raises(ValueError):
        SmartBrushEngine().paint_stroke(_halves(), np.zeros((10, 10), np.uint8), (5, 5), 4)


# --------------------------------------------------------------------- #
# SB-02
# --------------------------------------------------------------------- #

def test_sensitivity_decides_whether_a_faint_edge_stops_the_brush():
    image = _halves(left=(200, 40, 40), right=(200, 90, 40))       # red into orange-red
    engine = SmartBrushEngine()
    loose, strict = np.zeros(image.shape[:2], np.uint8), np.zeros(image.shape[:2], np.uint8)
    engine.paint_stroke(image, loose, (140, 100), 40, edge_sensitivity=0.1)
    engine.paint_stroke(image, strict, (140, 100), 40, edge_sensitivity=0.9)
    assert loose[:, 150:].max() == 255, "a loose brush paints through a faint edge"
    assert strict[:, 150:].max() == 0, "a strict one stops at it"
    assert tolerances(0.1)[0] > tolerances(0.9)[0] and tolerances(0.1)[1] > tolerances(0.9)[1]


def test_snapping_off_is_a_plain_round_brush():
    image = _halves()
    mask = np.zeros(image.shape[:2], np.uint8)
    SmartBrushEngine().paint_stroke(image, mask, (140, 100), 40, snapping=False)
    assert mask[100, 170] == 255 and mask[100, 185] == 0


def test_a_brush_centred_on_the_edge_still_paints_one_side():
    image = _halves(noise=0)
    mask = np.zeros(image.shape[:2], np.uint8)
    SmartBrushEngine().paint_stroke(image, mask, (150, 100), 30)
    left, right = (mask[:, :150] > 0).sum(), (mask[:, 150:] > 0).sum()
    assert left + right > 0 and min(left, right) == 0


def test_a_fast_drag_leaves_no_gaps():
    image = _halves(w=600, right=RED)                    # all red
    mask = np.zeros(image.shape[:2], np.uint8)
    _, rect = SmartBrushEngine().paint_line(image, mask, (20, 100), (560, 100), 15)
    assert (mask[100, 20:561] == 255).all()
    assert rect.x <= 5 and rect.x + rect.w >= 575


def test_a_dab_is_fast_once_the_image_is_prepared():
    import time

    image = _halves(1000, 1500)
    engine = SmartBrushEngine()
    mask = np.zeros(image.shape[:2], np.uint8)
    engine.paint_stroke(image, mask, (300, 300), 100)
    started = time.perf_counter()
    for x in range(200, 700, 25):
        engine.paint_stroke(image, mask, (x, 300), 100)
    assert (time.perf_counter() - started) / 20 < 0.05, "well inside a 60 fps frame budget per dab"


def test_the_live_preview_is_colour_where_revealed():
    image = _halves()
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[:, :150] = 255
    shown = reveal_preview(image, mask).astype(int)
    assert np.abs(shown[:, :150] - image[:, :150].astype(int)).max() == 0
    assert np.abs(shown[:, 150:, 0] - shown[:, 150:, 2]).max() == 0, "black and white elsewhere"


# --------------------------------------------------------------------- #
# SB-03: the layer
# --------------------------------------------------------------------- #

def _splash_photo(tmp_path) -> tuple[Path, Path]:
    folder = tmp_path / "Shoot"
    folder.mkdir()
    photo = folder / "IMG_0001.png"
    cv2.imwrite(str(photo), cv2.cvtColor(_halves(noise=0), cv2.COLOR_RGB2BGR))
    return folder, photo


def test_the_mask_is_saved_and_read_back(tmp_path):
    mask = np.zeros((50, 80), np.uint8)
    mask[10:20, 10:30] = 255
    name = layer_stack.save_mask_asset("IMG_0001.jpg", tmp_path, "color-splash-brush.png", mask)
    assert (layer_stack.load_mask_asset("IMG_0001.jpg", tmp_path, name) == mask).all()
    assert layer_stack.load_mask_asset("IMG_0001.jpg", tmp_path, name, (160, 100)).shape == (100, 160)
    assert layer_stack.load_mask_asset("IMG_0001.jpg", tmp_path, "nothing.png") is None
    layer = layer_stack.color_splash_brush_layer(name)
    again = layer_stack.layer_from_dict(layer_stack.layer_to_dict(layer))
    assert layer_stack.is_splash_brush(again) and again.mask.kind == "raster" and again.mask.invert


def test_the_painted_layer_exports_colour_only_where_revealed(tmp_path):
    folder, photo = _splash_photo(tmp_path)
    before = snapshot(folder)
    edit = tmp_path / "edit"
    image = _halves(noise=0)
    mask = np.zeros(image.shape[:2], np.uint8)
    SmartBrushEngine().paint_line(image, mask, (40, 100), (140, 100), 45)
    name = layer_stack.save_mask_asset(photo, edit, layer_stack.splash_brush_asset(), mask)
    recipe = pr.Recipe()
    recipe.layers = [layer_stack.layer_to_dict(layer_stack.color_splash_brush_layer(name))]
    context = layer_stack.RenderContext(assets=layer_stack.assets_folder(photo, edit))
    out = ep.render_frame(ep.FrameJob(photo, recipe, context)).astype(int)
    revealed, grey = out[100, 90], out[100, 250]
    assert revealed[0] - revealed[2] > 120, "the painted red stays red"
    assert abs(grey[0] - grey[2]) <= 3, "the unpainted blue is black and white"
    assert abs(out[180, 20, 0] - out[180, 20, 2]) <= 3, "red the brush never reached is black and white"
    assert_originals_untouched(before, folder)


# --------------------------------------------------------------------- #
# SB-04: Retouch
# --------------------------------------------------------------------- #

def _retouch(harness):
    w = harness.window
    key = str(harness.folder / "IMG_0001.jpg")
    w.session.set_current(key)
    w.switch_to(w.retouch)
    retouch = w.retouch
    pump(harness.app, lambda: retouch.key == key and retouch.base is not None)
    button = next(b for b in retouch.tool_group.buttons() if b.property("tool") == "splash_brush")
    button.setChecked(True)
    return w, retouch, key


def test_the_tool_shows_its_bar_and_cursor(harness):
    from src.ui.widgets.canvas import CANVAS_MODE_COLOR_SPLASH_BRUSH, CANVAS_MODE_NORMAL

    w, retouch, _ = _retouch(harness)
    assert not retouch.splash_bar.isHidden()
    assert retouch.canvas.mode == CANVAS_MODE_COLOR_SPLASH_BRUSH
    assert retouch.splash_snap.isChecked() and retouch.splash_sensitivity.currentText() == "Medium"
    size = retouch.splash_size.value()
    retouch._resize_splash(1.2)
    assert retouch.splash_size.value() > size and retouch.canvas.brush_radius == retouch.splash_size.value() / 2
    retouch._resize_splash(1 / 100)
    assert retouch.splash_size.value() == 10
    retouch.tool_group.button(0).setChecked(True)
    assert retouch.splash_bar.isHidden() and retouch.canvas.mode == CANVAS_MODE_NORMAL


def test_painting_makes_a_layer_and_saves_its_mask(harness):
    from PyQt6.QtCore import Qt

    w, retouch, key = _retouch(harness)
    before = snapshot(harness.folder)
    retouch._on_press(0.30, 0.50, Qt.MouseButton.LeftButton)
    retouch._on_drag(0.45, 0.50)
    retouch._on_release(0.45, 0.50)
    layers = layer_stack.layers_from_recipe(w.session.recipe(key))
    brush = [layer for layer in layers if layer_stack.is_splash_brush(layer)]
    assert len(brush) == 1 and brush[0].mask.params["revision"] == 1
    h, w_ = retouch.base.shape[:2]
    mask = layer_stack.load_mask_asset(key, w.edit_folder, brush[0].mask.params["asset"])
    assert mask.shape == (h, w_) and mask.max() == 255 and mask.mean() < 255
    assert w.session.history(key).steps[-1].label == "Color Splash Brush"
    # Invert, then reset.
    painted = int((mask > 0).sum())
    retouch.splash_invert()
    inverted = layer_stack.load_mask_asset(key, w.edit_folder, brush[0].mask.params["asset"])
    assert int((inverted > 0).sum()) == mask.size - painted
    retouch.splash_reset()
    assert layer_stack.load_mask_asset(key, w.edit_folder, brush[0].mask.params["asset"]).max() == 0
    assert_originals_untouched(before, harness.folder)


def test_alt_or_the_erase_button_erases(harness, monkeypatch):
    from PyQt6.QtCore import Qt

    w, retouch, key = _retouch(harness)
    retouch.splash_snap.setChecked(False)
    retouch._on_press(0.5, 0.5, Qt.MouseButton.LeftButton)
    retouch._on_release(0.5, 0.5)
    assert retouch._splash_mask.max() == 255
    retouch.splash_erase_button.setChecked(True)
    assert retouch.canvas.brush_erase
    retouch._on_press(0.5, 0.5, Qt.MouseButton.LeftButton)
    retouch._on_release(0.5, 0.5)
    h, w_ = retouch._splash_mask.shape
    inner = _circle((h, w_), (w_ * 0.5, h * 0.5), retouch._splash_radius() * 0.8)
    assert retouch._splash_mask[inner].max() == 0, "the brush's core is erased (its soft rim fades)"


def test_a_locked_photo_is_not_painted(harness):
    from PyQt6.QtCore import Qt

    w, retouch, key = _retouch(harness)
    harness.reject("IMG_0001.jpg")
    retouch._on_press(0.5, 0.5, Qt.MouseButton.LeftButton)
    retouch._on_release(0.5, 0.5)
    assert not any(layer_stack.is_splash_brush(l) for l in layer_stack.layers_from_recipe(w.session.recipe(key)))
    assert not retouch.splash_bar.isEnabled()


def test_the_canvas_repaints_only_a_region(qapp):
    from src.ui.widgets.canvas import ImageCanvas

    canvas = ImageCanvas()
    canvas.resize(300, 200)
    canvas.set_image(np.zeros((100, 150, 3), np.uint8))
    canvas.update_region(np.full((10, 20, 3), 255, np.uint8), 30, 40)
    image = canvas._image
    assert image.pixelColor(35, 45).red() == 255 and image.pixelColor(10, 10).red() == 0
