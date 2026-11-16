"""Color Splash (Isolate Object): one click keeps an object in colour and
turns the rest black and white, as an ordinary, reversible layer.

The click-to-select model is replaced by a stand-in that "finds" the left
half of the frame, so these tests check the tool, the layer and the render,
not the model (``test_retouch.py`` exercises the model on a real photo).
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import Qt

from src.ai_tools import model_store
from src.ai_tools.segmentation import MaskResult, Segmenter
from src.engine import develop_pipeline as dp
from src.engine import layer_stack as ls
from src.engine import parametric_recipe as pr
from tests.conftest import assert_originals_untouched, snapshot

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


class _LeftHalf:
    """Finds "the object" as the left half of the frame, and counts calls."""

    def __init__(self):
        self.calls = []

    def click(self, rgb, points, exclude=()):
        self.calls.append(list(points))
        mask = np.zeros(rgb.shape[:2], np.float32)
        mask[:, : rgb.shape[1] // 2] = 1.0
        return MaskResult(mask, "click-to-select")


def _colourful(w=160, h=100) -> np.ndarray:
    """Saturated stripes of red, green and blue, float RGB 0..1."""
    image = np.zeros((h, w, 3), np.float32)
    for index in range(w):
        image[:, index, (index // 10) % 3] = 0.9
    return image


def _spread(pixels: np.ndarray) -> float:
    return float((pixels.max(axis=-1) - pixels.min(axis=-1)).max())


def _splash(**changes) -> ls.Layer:
    layer = ls.color_splash_layer(0.25, 0.5, 6000)
    for name, value in changes.items():
        setattr(layer, name, value)
    return layer


# --------------------------------------------------------------------- #
# The layer
# --------------------------------------------------------------------- #

def test_the_layer_is_an_inverted_desaturating_adjustment():
    layer = ls.color_splash_layer(0.25, 0.5, 6000)
    assert layer.kind == "adjustment" and layer.name == "Color Splash"
    assert layer.recipe == {"basic": {"saturation": -100.0}}
    assert layer.mask.kind == "ai" and layer.mask.params == {"kind": "click", "points": [[0.25, 0.5]]}
    assert layer.mask.invert is True
    assert layer.mask.feather == pytest.approx(1.5 / 6000)       # 1.5 px at full size
    assert ls.is_color_splash(layer)
    assert not ls.is_color_splash(ls.Layer(kind="adjustment"))


def test_it_lives_in_the_recipe_and_survives_a_round_trip():
    recipe = pr.Recipe()
    recipe.layers = [ls.layer_to_dict(_splash())]
    again = ls.layers_from_recipe(pr.from_dict(pr.to_dict(recipe)))
    assert len(again) == 1 and ls.is_color_splash(again[0])
    assert again[0].mask.invert and again[0].recipe["basic"]["saturation"] == -100.0


def test_the_object_keeps_its_colour_and_the_rest_turns_grey():
    image = _colourful()
    ctx = ls.RenderContext(segmenter=_LeftHalf())
    out = ls.composite(image, [_splash()], ctx)
    w = image.shape[1]
    # Away from the feathered edge: the object untouched, the rest pure grey.
    assert np.allclose(out[:, : w // 2 - 4], image[:, : w // 2 - 4], atol=1e-3)
    assert _spread(out[:, w // 2 + 4:]) < 1e-3
    # Grey of the same brightness, not washed out (luminance is kept).
    assert np.allclose(out[:, w // 2 + 4:] @ LUMA, image[:, w // 2 + 4:] @ LUMA, atol=2e-3)


def test_it_can_be_turned_off_inverted_and_adjusted():
    image = _colourful()
    w = image.shape[1]
    left, right = slice(None, w // 2 - 4), slice(w // 2 + 4, None)
    ctx = ls.RenderContext(segmenter=_LeftHalf())

    hidden = ls.composite(image, [_splash(visible=False)], ctx)
    assert np.allclose(hidden, image, atol=1e-4)

    swapped = _splash()
    swapped.mask.invert = False                 # the object grey, the rest in colour
    out = ls.composite(image, [swapped], ctx)
    assert _spread(out[:, left]) < 1e-3
    assert np.allclose(out[:, right], image[:, right], atol=1e-3)

    partial = _splash(recipe={"basic": {"saturation": -50.0}})
    out = ls.composite(image, [partial], ctx)
    assert 0.2 < _spread(out[:, right]) < _spread(image[:, right]) - 0.2


def test_the_slow_selection_runs_once_however_often_the_stack_renders():
    segmenter = _LeftHalf()
    ctx = ls.RenderContext(segmenter=segmenter)
    image = _colourful()
    for saturation in (-100.0, -60.0, -20.0):
        ls.composite(image, [_splash(recipe={"basic": {"saturation": saturation}})], ctx)
    assert len(segmenter.calls) == 1
    grown = _splash()
    grown.mask.params = {**grown.mask.params, "points": [[0.25, 0.5], [0.1, 0.1]]}
    ls.composite(image, [grown], ctx)
    assert len(segmenter.calls) == 2, "a new click must select again"


def test_without_the_model_the_layer_is_skipped_not_applied_everywhere():
    image = _colourful()
    ctx = ls.RenderContext(segmenter=None)
    out = ls.composite(image, [_splash()], ctx)
    assert np.allclose(out, image, atol=1e-4)
    assert any("click selection" in note for note in ctx.notes)


def test_segment_object_at_point_is_one_click(monkeypatch):
    seen = []
    monkeypatch.setattr(Segmenter, "click", lambda self, rgb, points, exclude=(): seen.append(points) or "mask")
    assert Segmenter(download=False).segment_object_at_point(np.zeros((8, 8, 3), np.uint8), 3, 4) == "mask"
    assert seen == [[(3, 4)]]


# --------------------------------------------------------------------- #
# Saturation itself: -100 must be a real black and white
# --------------------------------------------------------------------- #

def test_full_desaturation_keeps_brightness():
    red = np.zeros((4, 4, 3), np.float32)
    red[..., 0] = 1.0
    out = dp._saturate(red, 0.0, -100.0)
    assert np.allclose(out, 0.2126, atol=1e-4), "full red must become its own grey, not white"


def test_a_boost_stays_in_gamut_and_keeps_its_hue():
    colour = np.array([[[0.9, 0.3, 0.2]]], np.float32)
    out = dp._saturate(colour, 0.0, 100.0)
    assert out.min() >= -1e-6 and out.max() <= 1.0 + 1e-6
    before, after = colour - colour @ LUMA, out - out @ LUMA
    cosine = float((before * after).sum() / (np.linalg.norm(before) * np.linalg.norm(after)))
    assert cosine > 0.999, "the boost changed the colour's hue"
    assert (out @ LUMA).item() == pytest.approx((colour @ LUMA).item(), abs=1e-4)


# --------------------------------------------------------------------- #
# The Retouch tool
# --------------------------------------------------------------------- #

@pytest.fixture
def splash_ready(monkeypatch):
    stand_in = _LeftHalf()
    monkeypatch.setattr(model_store, "is_cached", lambda spec: True)
    monkeypatch.setattr(Segmenter, "click", lambda self, rgb, points, exclude=(): stand_in.click(rgb, points))
    return stand_in


def _open_in_retouch(harness, name="IMG_0001.jpg"):
    from tests.test_workspaces import _settle

    w = harness.window
    key = str(harness.folder / name)
    w.session.set_current(key)
    w.switch_to(w.retouch)
    _settle(harness, w.retouch)
    return w, w.retouch, key


def test_one_click_makes_a_color_splash_layer(harness, splash_ready, library):
    from tests.test_workspaces import _settle

    before = snapshot(library)
    w, retouch, key = _open_in_retouch(harness)
    retouch.tool_group.button(4).setChecked(True)
    assert retouch._tool() == "splash" and retouch.tool_group.button(4).text() == "Color Splash"

    retouch._on_press(0.25, 0.5, Qt.MouseButton.LeftButton)
    _settle(harness, retouch)
    layers = w.session.recipe(key).layers
    assert len(layers) == 1
    layer = ls.layer_from_dict(layers[0])
    assert ls.is_color_splash(layer) and layer.mask.invert
    assert layer.recipe["basic"]["saturation"] == -100.0
    # 1.5 px of the full-size photo, not of the reduced preview.
    full_long_edge = max(retouch.source.size) / retouch.source.scale
    assert layer.mask.feather == pytest.approx(1.5 / full_long_edge, rel=1e-3)

    # The controls show it as it is: inverted, saturation -100.
    assert retouch.invert.isChecked()
    assert retouch.adjust_rows["saturation"].value() == -100.0

    retouch._on_press(0.1, 0.1, Qt.MouseButton.LeftButton)          # adds to the object
    layer = ls.layer_from_dict(w.session.recipe(key).layers[0])
    assert layer.mask.params["points"] == [[0.25, 0.5], [0.1, 0.1]]
    assert len(w.session.recipe(key).layers) == 1

    retouch.invert.setChecked(False)                                  # inverted back
    assert ls.layer_from_dict(w.session.recipe(key).layers[0]).mask.invert is False
    assert_originals_untouched(before, harness.folder)


def test_the_splash_reaches_the_exported_file(harness, splash_ready, tmp_path):
    w, retouch, key = _open_in_retouch(harness, "IMG_0002.jpg")
    retouch.tool_group.button(4).setChecked(True)
    retouch._on_press(0.25, 0.5, Qt.MouseButton.LeftButton)
    recipe = w.session.recipe(key)
    source = dp.open_source(key)
    ctx = ls.RenderContext(segmenter=Segmenter(download=False))
    out = dp.render(source, recipe, layers=ctx)
    right = out[:, out.shape[1] // 2 + 8:].astype(np.float32)
    left = out[:, : out.shape[1] // 2 - 8].astype(np.float32)
    assert _spread(right / 255.0) < 0.02, "the background kept its colour"
    assert _spread(left / 255.0) > 0.1, "the object lost its colour"
