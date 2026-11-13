"""The layer stack: blend-mode arithmetic against hand-computed values,
masks, tolerance of bad data, and compositing."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.engine import develop_pipeline as dp
from src.engine import layer_stack as ls
from src.engine.parametric_recipe import Recipe
from tests.conftest import assert_originals_untouched, make_scene, snapshot


def _px(*rgb) -> np.ndarray:
    return np.array([[list(rgb)]], dtype=np.float32)


def _grey(value: float) -> np.ndarray:
    return _px(value, value, value)


def _image(size=(120, 80), value=128) -> np.ndarray:
    return np.full((size[1], size[0], 3), value, np.uint8)


def _adjust(exposure=1.0, **extra) -> dict:
    return {"kind": "adjustment", "recipe": {"basic": {"exposure": exposure}}, **extra}


# --------------------------------------------------------------------- #
# Blend modes: hand-computed
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("mode,base,top,expected", [
    ("normal", 0.2, 0.7, 0.7),
    ("multiply", 0.5, 0.4, 0.20),
    ("multiply", 0.8, 1.0, 0.80),               # white is the identity
    ("screen", 0.5, 0.4, 0.70),                 # 1 - 0.5 * 0.6
    ("screen", 0.3, 0.0, 0.30),                 # black is the identity
    ("overlay", 0.25, 0.6, 0.30),               # base <= .5: 2 * .25 * .6
    ("overlay", 0.75, 0.6, 0.80),               # base > .5: 1 - 2 * .25 * .4
    ("overlay", 0.4, 0.5, 0.40),                # mid grey is the identity
    ("soft_light", 0.4, 0.5, 0.40),             # mid grey is the identity
    ("soft_light", 0.5, 0.25, 0.375),           # top <= .5: .5 - .5 * .5 * .5
    ("soft_light", 0.25, 0.75, 0.375),          # d(.25) = .5; .25 + .5 * .25
    ("soft_light", 0.64, 1.0, 0.80),            # top = 1 gives sqrt(base)
    ("color_dodge", 0.3, 0.5, 0.60),
    ("color_dodge", 0.3, 0.0, 0.30),            # black is the identity
    ("color_dodge", 0.6, 0.5, 1.00),            # clips
    ("color_dodge", 0.0, 1.0, 0.00),            # black stays black
])
def test_blend_mode_values(mode, base, top, expected):
    assert float(ls.blend(_grey(base), _grey(top), mode)[0, 0, 0]) == pytest.approx(expected, abs=1e-5)


def test_luminosity_takes_brightness_and_keeps_colour():
    base, top = _px(0.8, 0.2, 0.2), _grey(0.6)
    out = ls.blend(base, top, "luminosity")[0, 0]
    assert float(out @ ls._LUM) == pytest.approx(0.6, abs=1e-4)
    assert out[0] > out[1] and out[1] == pytest.approx(out[2], abs=1e-5), "the hue was not kept"
    same = ls.blend(base, base, "luminosity")[0, 0]
    assert np.allclose(same, base[0, 0], atol=1e-5), "an image's own luminosity should change nothing"


def test_luminosity_stays_in_gamut():
    out = ls.blend(_px(1.0, 0.0, 0.0), _grey(0.95), "luminosity")
    assert out.min() >= -1e-5 and out.max() <= 1.0 + 1e-5


def test_every_mode_stays_in_range_and_unknown_modes_are_refused():
    rng = np.random.default_rng(1)
    base, top = rng.random((20, 20, 3), dtype=np.float32), rng.random((20, 20, 3), dtype=np.float32)
    for mode in ls.BLEND_MODES:
        out = ls.blend(base, top, mode)
        assert np.isfinite(out).all() and out.min() >= -1e-4 and out.max() <= 1.0 + 1e-4, mode
    with pytest.raises(ValueError):
        ls.blend(base, top, "difference")


# --------------------------------------------------------------------- #
# Reading layers
# --------------------------------------------------------------------- #

def test_layers_round_trip_through_a_recipe():
    recipe = Recipe()
    recipe.layers = [
        _adjust(0.5, name="Subject", opacity=80, blend="soft_light",
                mask={"kind": "radial", "params": {"centre": [0.5, 0.4]}, "feather": 0.02, "invert": True}),
        {"kind": "retouch", "op": "skin_smooth", "amount": 40},
    ]
    from src.engine import parametric_recipe as pr

    again = ls.layers_from_recipe(pr.loads(pr.dumps(recipe)))
    assert [l.kind for l in again] == ["adjustment", "retouch"]
    assert again[0].blend == "soft_light" and again[0].opacity == 80 and again[0].mask.invert
    assert again[1].op == "skin_smooth" and again[1].amount == 40
    assert ls.layer_from_dict(ls.layer_to_dict(again[0])) == again[0]


@pytest.mark.parametrize("bad", [None, 3, "layer", [], {"kind": "hologram"}, {"name": "no kind"}])
def test_unusable_layers_are_dropped(bad):
    assert ls.layer_from_dict(bad) is None


def test_bad_fields_fall_back_to_safe_values():
    layer = ls.layer_from_dict({
        "kind": "pixel", "opacity": "lots", "blend": "explode", "rect": [1, "x"],
        "asset": "../../secrets/key.png", "mask": {"kind": "telepathy", "feather": 99, "density": -5},
        "op": "delete_everything",
    })
    assert layer.opacity == 100.0 and layer.blend == "normal" and layer.rect == [0.0, 0.0, 1.0, 1.0]
    assert layer.asset == "key.png", "an asset name must not be able to leave its folder"
    assert layer.mask.kind == "none" and layer.mask.feather == 0.5 and layer.mask.density == 0.0
    assert layer.op == ""


# --------------------------------------------------------------------- #
# Masks
# --------------------------------------------------------------------- #

def _mask(kind, **params) -> np.ndarray:
    extra = {k: params.pop(k) for k in ("invert", "feather", "density") if k in params}
    mask = ls.mask_from_dict({"kind": kind, "params": params, **extra})
    return ls.render_mask(mask, np.full((80, 120, 3), 0.5, np.float32), ls.RenderContext())


def test_no_mask_is_everywhere():
    assert np.all(_mask("none") == 1.0)


def test_gradient_runs_from_start_to_end():
    mask = _mask("gradient", start=[0.0, 0.5], end=[1.0, 0.5])
    assert mask[40, 0] == pytest.approx(0.0, abs=0.02) and mask[40, -1] == pytest.approx(1.0, abs=0.02)
    assert mask[40, 60] == pytest.approx(0.5, abs=0.02)
    assert np.all(np.diff(mask[40]) >= -1e-6)


def test_radial_is_full_at_the_centre_and_empty_outside():
    mask = _mask("radial", centre=[0.5, 0.5], radius_x=0.25, radius_y=0.25, softness=0.5)
    assert mask[40, 60] == pytest.approx(1.0) and mask[0, 0] == 0.0


def test_luminance_range_selects_by_brightness():
    ramp = np.repeat(np.linspace(0, 1, 100, dtype=np.float32)[None, :, None], 3, axis=2).repeat(10, axis=0)
    mask = ls.render_mask(ls.mask_from_dict({"kind": "luminance", "params": {"low": 0.7, "high": 1.0, "softness": 0.05}}),
                          ramp, ls.RenderContext())
    assert mask[5, 95] == pytest.approx(1.0) and mask[5, 30] == 0.0


def test_brush_strokes_paint_and_erase():
    stroke = {"points": [[0.2, 0.5], [0.8, 0.5]], "radius": 0.05, "hardness": 1.0}
    painted = _mask("brush", strokes=[stroke])
    assert painted[40, 60] == pytest.approx(1.0) and painted[5, 60] == 0.0
    erased = _mask("brush", strokes=[stroke, {"points": [[0.5, 0.5]], "radius": 0.08, "hardness": 1.0, "erase": True}])
    assert erased[40, 60] == pytest.approx(0.0, abs=0.02) and erased[40, 30] == pytest.approx(1.0)
    assert _mask("brush", strokes="not a list").max() == 0.0


def test_brush_positions_follow_the_frame_size():
    stroke = {"points": [[0.5, 0.5]], "radius": 0.05, "hardness": 1.0}
    mask = ls.mask_from_dict({"kind": "brush", "params": {"strokes": [stroke]}})
    small = ls.render_mask(mask, np.zeros((100, 200, 3), np.float32), ls.RenderContext())
    large = ls.render_mask(mask, np.zeros((400, 800, 3), np.float32), ls.RenderContext())
    assert small.mean() == pytest.approx(large.mean(), rel=0.08)
    assert large[200, 400] == 1.0


def test_invert_feather_and_density():
    assert _mask("radial", centre=[0.5, 0.5], radius_x=0.2, radius_y=0.2, invert=True)[40, 60] == pytest.approx(0.0)
    hard = _mask("brush", strokes=[{"points": [[0.5, 0.5]], "radius": 0.1, "hardness": 1.0}])
    soft = _mask("brush", strokes=[{"points": [[0.5, 0.5]], "radius": 0.1, "hardness": 1.0}], feather=0.05)
    assert len(np.unique(np.round(soft, 2))) > len(np.unique(np.round(hard, 2))) + 5
    half = _mask("radial", centre=[0.5, 0.5], radius_x=0.2, radius_y=0.2, density=50)
    assert half[0, 0] == pytest.approx(0.5), "half density should let half the layer through outside the mask"


def test_a_mask_that_cannot_be_built_is_reported(tmp_path):
    ctx = ls.RenderContext(assets=tmp_path)
    image = np.zeros((20, 20, 3), np.float32)
    assert ls.render_mask(ls.mask_from_dict({"kind": "raster", "params": {"asset": "gone.png"}}), image, ctx) is None
    assert ls.render_mask(ls.mask_from_dict({"kind": "ai", "params": {"kind": "sky"}}), image, ctx) is None
    assert len(ctx.notes) == 2


# --------------------------------------------------------------------- #
# Compositing
# --------------------------------------------------------------------- #

def _layers(*dicts) -> list:
    return [ls.layer_from_dict(d) for d in dicts]


def test_an_empty_or_hidden_stack_changes_nothing():
    image = _image()
    assert ls.composite(image, []) is image
    assert np.array_equal(ls.composite(image, _layers(_adjust(2.0, visible=False))), image)
    assert np.array_equal(ls.composite(image, _layers(_adjust(2.0, opacity=0))), image)


def test_an_adjustment_layer_applies_inside_its_mask_only():
    image = _image()
    mask = {"kind": "gradient", "params": {"start": [0.45, 0.5], "end": [0.55, 0.5]}}
    out = ls.composite(image, _layers(_adjust(1.0, mask=mask)))
    assert out[40, 5, 0] == 128, "the masked-out side changed"
    assert out[40, -5, 0] > 150, "the masked-in side did not change"
    assert out.dtype == np.uint8


def test_opacity_scales_the_effect():
    image = _image()
    full = int(ls.composite(image, _layers(_adjust(1.0)))[0, 0, 0])
    half = int(ls.composite(image, _layers(_adjust(1.0, opacity=50)))[0, 0, 0])
    assert half == pytest.approx((128 + full) / 2, abs=1.5)


def test_layers_apply_bottom_first():
    image = _image()
    darken = {"kind": "adjustment", "recipe": {"basic": {"exposure": -1.0}}}
    up_then_down = ls.composite(image, _layers(_adjust(1.0), darken))[0, 0, 0]
    assert abs(int(up_then_down) - 128) <= 6, "a stop up then a stop down should come back near the start"


def test_a_mask_layer_limits_everything_beneath_it():
    image = _image()
    limit = {"kind": "mask", "mask": {"kind": "gradient", "params": {"start": [0.45, 0.5], "end": [0.55, 0.5]}}}
    out = ls.composite(image, _layers(_adjust(1.0), limit))
    assert out[40, 5, 0] == 128 and out[40, -5, 0] > 150


def test_sixteen_bit_images_keep_their_depth():
    image = np.full((40, 60, 3), 30000, np.uint16)
    out = ls.composite(image, _layers(_adjust(0.5)))
    assert out.dtype == np.uint16 and out[0, 0, 0] > 30000


def test_pixel_layers_place_a_patch_by_fraction(tmp_path):
    source = tmp_path / "IMG_1.jpg"
    patch = np.zeros((10, 10, 4), np.uint8)
    patch[:, :] = (255, 0, 0, 255)
    name = ls.save_asset(source, tmp_path / "edit", "heal-1", patch)
    assert name == "heal-1.png"
    ctx = ls.RenderContext(assets=ls.assets_folder(source, tmp_path / "edit"))
    layer = {"kind": "pixel", "asset": name, "rect": [0.5, 0.5, 0.25, 0.25]}
    for size in ((120, 80), (480, 320)):
        out = ls.composite(_image(size), _layers(layer), ctx)
        w, h = size
        assert tuple(out[int(0.6 * h), int(0.6 * w)]) == (255, 0, 0)
        assert tuple(out[int(0.2 * h), int(0.2 * w)]) == (128, 128, 128)
    assert ctx.notes == []


def test_a_layer_that_cannot_render_is_skipped_and_explained(tmp_path):
    ctx = ls.RenderContext(assets=tmp_path)
    image = _image()
    out = ls.composite(image, _layers(
        {"kind": "pixel", "asset": "missing.png"},
        {"kind": "retouch", "op": "skin_smooth"},
        {"kind": "text", "vector": {"text": "hello"}},
        _adjust(1.0),
    ), ctx)
    assert out[0, 0, 0] > 150, "the layers that could render were not applied"
    assert len(ctx.notes) == 3


def test_blend_mode_is_used_when_compositing():
    image = _image(value=128)
    grey = {"kind": "adjustment", "recipe": {}, "blend": "multiply"}
    out = ls.composite(image, _layers(grey))
    assert out[0, 0, 0] == pytest.approx(round(255 * (128 / 255) ** 2), abs=1)


# --------------------------------------------------------------------- #
# Through the develop pipeline
# --------------------------------------------------------------------- #

def test_the_pipeline_applies_the_stack_only_when_given_a_context():
    recipe = Recipe()
    recipe.layers = [_adjust(1.0)]
    source = dp.from_array(_image())
    assert np.array_equal(dp.render(source, recipe), _image()), "layers ran without a context"
    assert dp.render(source, recipe, layers=ls.RenderContext())[0, 0, 0] > 150


def test_layers_follow_the_picture_when_it_is_cropped():
    """Layers are placed before framing, so cropping does not slide them."""
    image = _image((200, 100))
    recipe = Recipe()
    recipe.layers = [_adjust(1.0, mask={"kind": "gradient", "params": {"start": [0.49, 0.5], "end": [0.51, 0.5]}})]
    recipe.geometry.crop = [0.5, 0.0, 0.5, 1.0]             # keep the right half
    out = dp.render(dp.from_array(image), recipe, layers=ls.RenderContext())
    assert out.shape[1] == 100 and out[50, 50, 0] > 150, "the brightened half was not the half that was kept"


def test_exporting_with_layers_leaves_originals_alone(library):
    before = snapshot(library)
    recipe = Recipe()
    recipe.layers = [_adjust(0.7, mask={"kind": "radial", "params": {"centre": [0.5, 0.5]}})]
    out = dp.export(library / "IMG_0001.jpg", recipe, library / "Shoot-edit" / "layered.jpg",
                    layers=ls.RenderContext())
    assert cv2.imread(str(out)).mean() > cv2.imread(str(library / "IMG_0001.jpg")).mean()
    assert_originals_untouched(before, library)
