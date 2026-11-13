"""Retouching, inpainting and the masks they rely on."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.ai_tools import face_mesh as fm
from src.ai_tools import model_store, retouch, segmentation
from src.engine import develop_pipeline as dp
from src.engine import layer_stack as ls
from src.engine.parametric_recipe import Recipe
from tests.conftest import make_scene
from tests.fetch_media import FACES, MEDIA_DIR, face


def _rgb(seed=1, size=(320, 240)) -> np.ndarray:
    return cv2.cvtColor(make_scene(seed, size), cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------- #
# Frequency separation
# --------------------------------------------------------------------- #

def test_the_three_bands_add_back_to_the_image():
    image = _rgb().astype(np.float32) / 255.0
    fine, middle, coarse = retouch.split_bands(image, 2.0, 12.0)
    assert np.abs(fine + middle + coarse - image).max() < 1e-5


def _skin_like(size=400, eye_distance=100.0):
    """A flat tone carrying fine grain ("pores") and broad blotches."""
    rng = np.random.default_rng(5)
    pores = rng.normal(0, 0.020, (size, size)).astype(np.float32)
    blotch = cv2.GaussianBlur(rng.normal(0, 1.0, (size, size)).astype(np.float32), (0, 0), 6.0)
    blotch *= 0.05 / blotch.std()
    image = np.clip(0.6 + pores + blotch, 0, 1)[:, :, None].repeat(3, axis=2)
    return image.astype(np.float32), np.ones((size, size), np.float32), eye_distance


def test_smoothing_removes_blotches_and_keeps_pores():
    image, mask, eyes = _skin_like()
    fine_before, middle_before = retouch.band_energy(image, mask, eyes)
    smoothed = retouch.smooth_skin(image, mask, 1.0, eyes)
    fine_after, middle_after = retouch.band_energy(smoothed, mask, eyes)
    assert middle_after < 0.2 * middle_before, "blotchiness was not reduced"
    assert fine_after > 0.9 * fine_before, "fine texture was lost: this is the plastic look"


def test_a_plain_blur_would_have_destroyed_the_pores():
    """Why frequency separation is used at all."""
    image, mask, eyes = _skin_like()
    fine_before, _ = retouch.band_energy(image, mask, eyes)
    blurred = cv2.GaussianBlur(image, (0, 0), 0.08 * eyes)
    assert retouch.band_energy(blurred, mask, eyes)[0] < 0.2 * fine_before


def test_smoothing_stays_inside_the_mask_and_scales_with_amount():
    image, _, eyes = _skin_like()
    mask = np.zeros(image.shape[:2], np.float32)
    mask[:, :200] = 1.0
    out = retouch.smooth_skin(image, mask, 1.0, eyes)
    assert np.abs(out[:, 300:] - image[:, 300:]).max() < 1e-4, "pixels outside the mask changed"
    half = retouch.smooth_skin(image, np.ones_like(mask), 0.5, eyes)
    full = retouch.smooth_skin(image, np.ones_like(mask), 1.0, eyes)
    ones = np.ones_like(mask)
    assert retouch.band_energy(full, ones, eyes)[1] < retouch.band_energy(half, ones, eyes)[1]
    assert np.array_equal(retouch.smooth_skin(image, ones, 0.0, eyes), image)


def test_retouch_functions_keep_the_dtype():
    image8 = _rgb()
    mask = np.ones(image8.shape[:2], np.float32)
    assert retouch.smooth_skin(image8, mask, 0.5, 80.0).dtype == np.uint8
    assert retouch.whiten_teeth(image8, mask, 0.5).dtype == np.uint8
    assert retouch.clarify_eyes(image8, mask, mask, 0.5).dtype == np.uint8
    deep = image8.astype(np.uint16) * 257
    assert retouch.smooth_skin(deep, mask, 0.5, 80.0).dtype == np.uint16


# --------------------------------------------------------------------- #
# Teeth and eyes
# --------------------------------------------------------------------- #

def test_whitening_removes_yellow_without_glowing():
    yellowed = np.full((40, 40, 3), (215, 200, 150), np.uint8)
    mask = np.zeros((40, 40), np.float32)
    mask[10:30, 10:30] = 1.0
    out = retouch.whiten_teeth(yellowed, mask, 1.0)
    inside = out[20, 20].astype(int)
    assert inside.max() - inside.min() < 0.5 * (215 - 150), "the yellow cast was not reduced"
    assert inside.max() < 250, "teeth were pushed to pure white"
    assert tuple(out[2, 2]) == (215, 200, 150), "pixels outside the mask changed"


def test_eye_clarity_adds_contrast_in_the_iris_only():
    rng = np.random.default_rng(2)
    image = np.clip(110 + rng.normal(0, 14, (80, 80, 3)), 0, 255).astype(np.uint8)
    iris = np.zeros((80, 80), np.float32)
    cv2.circle(iris, (40, 40), 12, 1.0, -1)
    eye = np.zeros((80, 80), np.float32)
    cv2.ellipse(eye, (40, 40), (26, 14), 0, 0, 360, 1.0, -1)
    out = retouch.clarify_eyes(image, iris, eye, 1.0)
    inside = iris > 0.5
    assert out[inside].std() > image[inside].std() * 1.15
    assert np.array_equal(out[:5, :5], image[:5, :5])


# --------------------------------------------------------------------- #
# Inpainting
# --------------------------------------------------------------------- #

def test_fast_fill_touches_only_the_hole_and_blends_in():
    gradient = np.repeat(np.linspace(40, 220, 200)[None, :, None], 3, axis=2).repeat(120, axis=0).astype(np.uint8)
    damaged = gradient.copy()
    hole = np.zeros((120, 200), np.uint8)
    cv2.circle(hole, (100, 60), 8, 1, -1)
    damaged[hole > 0] = 0
    healed, engine = retouch._FastOnly.fill(damaged, hole)
    assert engine == "fast"
    assert np.array_equal(healed[hole == 0], damaged[hole == 0]), "pixels outside the hole changed"
    assert np.abs(healed[hole > 0].astype(int) - gradient[hole > 0].astype(int)).mean() < 8


def test_heal_spots_and_empty_requests():
    image = _rgb()
    assert np.array_equal(retouch.heal_spots(image, []), image)
    healed = retouch.heal_spots(image, [(100, 100, 6)])
    assert not np.array_equal(healed, image)
    untouched = np.ones(image.shape[:2], bool)
    cv2.circle(untouched.view(np.uint8), (100, 100), 8, 0, -1)
    assert np.array_equal(healed[untouched], image[untouched])


def test_blank_corners_are_found_and_a_black_object_is_not():
    image = np.full((200, 300, 3), 150, np.uint8)
    image[90:110, 140:160] = 0                     # a black object in the middle
    turned = cv2.warpAffine(image, cv2.getRotationMatrix2D((150, 100), 8, 1.0), (300, 200))
    wedges = retouch.blank_corners(turned)
    assert wedges[0, 0] == 1 and wedges[-1, -1] == 1
    assert wedges[100, 150] == 0, "an object inside the frame was taken for a blank corner"
    assert not retouch.blank_corners(image).any()


def test_filling_corners_leaves_no_blank_and_keeps_the_picture():
    image = _rgb(3, (300, 200))
    turned = cv2.warpAffine(image, cv2.getRotationMatrix2D((150, 100), 6, 1.0), (300, 200))
    filled, engine = retouch.fill_corners(turned)
    assert engine == "fast"
    assert not retouch.blank_corners(filled).any(), "a blank wedge survived"
    kept = retouch.blank_corners(turned) == 0
    assert np.array_equal(filled[kept], turned[kept])


def test_the_recipe_can_fill_corners_instead_of_cropping():
    image = np.full((200, 300, 3), 160, np.uint8)
    recipe = Recipe()
    recipe.geometry.rotation = 6.0
    cropped = dp.render(dp.from_array(image), recipe)
    assert cropped.shape[0] < 200, "the default should still be to crop"
    recipe.geometry.constrain_to_image = False
    assert dp.render(dp.from_array(image), recipe).min() == 0, "without filling, the wedges are blank"
    recipe.geometry.fill_blank = True
    filled = dp.render(dp.from_array(image), recipe)
    assert filled.shape[:2] == (200, 300) and filled.min() > 100, "the full frame was not kept and filled"


def test_without_the_model_inpainting_falls_back_to_the_fast_fill(tmp_path, monkeypatch):
    monkeypatch.setattr(model_store, "MODEL_DIR", tmp_path)
    engine = retouch.Inpainter(download=False)
    assert not engine.available and engine.load_error
    image = _rgb()
    hole = np.zeros(image.shape[:2], np.uint8)
    hole[50:120, 60:160] = 1                        # large enough to want the neural engine
    filled, used = engine.fill(image, hole, engine="neural")
    assert used == "fast"
    assert np.array_equal(filled[hole == 0], image[hole == 0])
    with pytest.raises(TypeError):
        engine.fill(image.astype(np.float32), hole)


@pytest.mark.model
@pytest.mark.slow
def test_neural_inpainting_fills_only_the_hole_with_matching_colour():
    if not retouch.Inpainter.is_cached():
        pytest.skip("inpainting model not cached")
    sample = MEDIA_DIR / "raw" / "samsung_s10.dng"
    if not sample.exists():
        pytest.skip("sample not downloaded")
    from src.engine import fast_preview

    image = fast_preview.load_preview(sample, max_side=640).image        # a church against blue sky
    hole = np.zeros(image.shape[:2], np.uint8)
    hole[20:90, 30:130] = 1                                              # a block of sky
    damaged = image.copy()
    damaged[hole > 0] = 0
    filled, used = retouch.Inpainter().fill(damaged, hole, engine="neural")
    assert used == "neural"
    assert np.array_equal(filled[hole == 0], damaged[hole == 0]), "pixels outside the hole changed"
    inside, truth = filled[hole > 0].astype(float), image[hole > 0].astype(float)
    assert inside.min() > 20, "part of the hole was left blank"
    # Sky is blue: if the model's channel order were wrong it would come back orange.
    assert inside[:, 2].mean() > inside[:, 0].mean(), "the filled sky is not blue"
    assert np.abs(inside.mean(axis=0) - truth.mean(axis=0)).max() < 30


# --------------------------------------------------------------------- #
# Masks
# --------------------------------------------------------------------- #

def test_sky_is_found_at_the_top_and_not_lower_down():
    image = np.zeros((300, 400, 3), np.uint8)
    image[:120] = (110, 160, 230)                                        # sky
    image[120:] = cv2.cvtColor(make_scene(2, (400, 180)), cv2.COLOR_BGR2RGB)
    image[200:240, 150:250] = (110, 160, 230)                            # a sky-blue object, not sky
    mask = segmentation.sky_mask(image)
    assert mask[40, 200] > 0.9
    assert mask[220, 200] < 0.1, "a blue object away from the top edge was taken for sky"
    assert segmentation.sky_mask(cv2.cvtColor(make_scene(1), cv2.COLOR_BGR2RGB)).mean() < 0.05


def test_unknown_mask_kinds_are_refused_and_missing_models_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(model_store, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(segmentation, "U2NET_PATH", tmp_path / "absent.onnx")
    seg = segmentation.Segmenter(download=False)
    image = _rgb()
    with pytest.raises(ValueError):
        seg.mask(image, "aura")
    for kind in ("subject", "person", "skin", "teeth", "hair"):
        result = seg.mask(image, kind)
        assert result.is_empty and result.note, kind
        assert result.mask.shape == image.shape[:2]
    assert seg.click(image, [(10, 10)]).note
    assert segmentation.Segmenter.download_mb("skin") > 0


@pytest.fixture(scope="module")
def portrait():
    if not (all(f.present for f in FACES) and fm.FaceMesh.is_cached()):
        pytest.skip("reference faces or face-mesh model not present")
    seg = segmentation.Segmenter()
    if seg.faces(np.zeros((64, 64, 3), np.uint8)) is None:
        pytest.skip("face models not cached")
    image = cv2.resize(cv2.imread(str(face("pose.png").path)), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return seg, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@pytest.mark.model
@pytest.mark.media
def test_face_masks_land_on_the_face(portrait):
    seg, image = portrait
    regions = seg.faces(image)[0]
    x, y, w, h = regions.box
    skin = seg.mask(image, "skin").mask
    assert skin[y + h // 2, x + w // 4] > 0.5, "the cheek is not in the skin mask"
    assert skin[5, 5] < 0.05, "a white backdrop was taken for skin"
    right_eye = regions.outline(fm.RIGHT_EYE_RING).mean(axis=0).astype(int)
    assert skin[right_eye[1], right_eye[0]] < 0.5, "an eye was left inside the skin mask"

    eyes, iris = seg.mask(image, "eyes").mask, seg.mask(image, "iris").mask
    assert eyes[right_eye[1], right_eye[0]] > 0.5
    assert iris.sum() > 0 and np.all(iris <= eyes + 1e-3), "the iris mask spills outside the eyelids"

    teeth = seg.mask(image, "teeth").mask
    lips = seg.mask(image, "lips").mask
    assert teeth.sum() > 0, "a broad smile showed no teeth"
    assert (teeth * lips).sum() < 0.1 * teeth.sum(), "teeth and lips overlap"
    selected = image[teeth > 0.5]
    assert selected.mean() > image[lips > 0.5].mean(), "the teeth mask is not the bright part of the mouth"


@pytest.mark.model
@pytest.mark.media
def test_a_black_and_white_portrait_gets_a_geometric_skin_mask():
    if not (face("basketball1.png").present and fm.FaceMesh.is_cached()):
        pytest.skip("reference face or model not present")
    image = cv2.resize(cv2.imread(str(face("basketball1.png").path)), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    result = segmentation.Segmenter().mask(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "skin")
    assert 0.002 < result.coverage < 0.05, "with no colour to go on, the mask should be just the face"


@pytest.mark.model
@pytest.mark.media
def test_retouch_layers_run_through_the_stack(portrait):
    seg, image = portrait
    layers = [ls.layer_from_dict(d) for d in (
        {"kind": "retouch", "op": "skin_smooth", "amount": 80},
        {"kind": "retouch", "op": "teeth_whiten", "amount": 80},
        {"kind": "retouch", "op": "eye_clarity", "amount": 80},
    )]
    ctx = ls.RenderContext(segmenter=seg)
    out = ls.composite(image, layers, ctx)
    assert ctx.notes == []
    changed = np.abs(out.astype(int) - image.astype(int)).max(axis=2) > 2
    assert changed.any(), "the retouch layers changed nothing"
    x, y, w, h = seg.faces(image)[0].box
    region = np.zeros_like(changed)
    region[max(0, y - h): y + 3 * h, max(0, x - w): x + 2 * w] = True
    assert not changed[~region].any(), "retouching reached pixels far from the face"


@pytest.mark.model
@pytest.mark.media
@pytest.mark.slow
def test_click_selects_the_object_under_the_point(portrait):
    seg, image = portrait
    if not model_store.is_cached(segmentation.SAM_SPEC):
        pytest.skip("click-to-select model not cached")
    x, y, w, h = seg.faces(image)[0].box
    torso = (x + w / 2.0, y + 2.2 * h)
    result = seg.click(image, [torso])
    assert result.mask[int(torso[1]), int(torso[0])] > 0.5
    assert result.mask[5, 5] < 0.5, "the backdrop was selected along with the sweater"
