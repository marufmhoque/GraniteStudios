"""The develop pipeline, on constructed images with known answers."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.ai_tools import orientation
from src.engine import develop_pipeline as dp
from src.engine import parametric_recipe as pr
from src.engine.parametric_recipe import Recipe
from tests.conftest import assert_originals_untouched, make_architecture, make_scene, snapshot
from tests.fetch_media import SAMPLES


def _scene(size=(480, 320), seed=1) -> np.ndarray:
    return cv2.cvtColor(make_scene(seed, size), cv2.COLOR_BGR2RGB)


def _flat(rgb, size=(64, 64)) -> np.ndarray:
    return np.full((size[1], size[0], 3), rgb, dtype=np.uint8)


def _ramp(width=512, height=32) -> np.ndarray:
    row = np.linspace(0, 255, width).round().astype(np.uint8)
    return np.repeat(np.repeat(row[None, :, None], height, axis=0), 3, axis=2)


def _render(image, **basic) -> np.ndarray:
    recipe = Recipe()
    for key, value in basic.items():
        setattr(recipe.basic, key, value)
    return dp.render(dp.from_array(image), recipe)


def _diff(a, b) -> int:
    return int(np.abs(a.astype(np.int64) - b.astype(np.int64)).max())


# --------------------------------------------------------------------- #
# The identity
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("dtype,scale", [(np.uint8, 255), (np.uint16, 65535)])
def test_a_neutral_recipe_changes_nothing(dtype, scale):
    image = (_scene().astype(np.float64) / 255 * scale).round().astype(dtype)
    bits = 16 if dtype == np.uint16 else 8
    assert np.array_equal(dp.render(dp.from_array(image), Recipe(), bits=bits), image)


def test_a_neutral_recipe_through_the_full_float_path_changes_nothing():
    """The shortcut for untouched pixels must not be what makes this true."""
    image = _scene()
    as_float = dp.from_array(image.astype(np.float32) / 255.0)
    assert _diff(dp.render(as_float, Recipe()), image) == 0


def test_roll_off_is_the_identity_without_gain():
    x = np.linspace(0, 1, 1000, dtype=np.float32)
    assert np.array_equal(dp.roll_off(x, 1.0), x)


def test_roll_off_is_monotone_and_lands_the_peak_on_white():
    x = np.linspace(0, 4.0, 4000, dtype=np.float32)
    y = dp.roll_off(x, 4.0)
    assert np.all(np.diff(y) >= 0)
    assert y[-1] == pytest.approx(1.0, abs=1e-5)
    below = x <= dp.KNEE
    assert np.array_equal(y[below], x[below]), "values below the knee were altered"


def test_transfer_functions_are_inverses():
    x = np.linspace(0, 1, 4096, dtype=np.float32)
    assert np.abs(dp.srgb_encode(dp.srgb_decode(x)) - x).max() < 1e-5


# --------------------------------------------------------------------- #
# Tone
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("stops", [-2.0, -1.0, 1.0])
def test_exposure_scales_linear_light_in_the_midtones(stops):
    grey = 64                                   # linear 0.051: stays below the knee at +1
    out = _render(_flat((grey,) * 3), exposure=stops)
    before = float(dp.srgb_decode(np.float32(grey / 255.0)))
    after = float(dp.srgb_decode(np.float32(out[0, 0, 0] / 255.0)))
    assert after == pytest.approx(before * 2.0 ** stops, rel=0.03)


def test_pushing_exposure_bends_highlights_instead_of_clipping_them():
    out = _render(_ramp(), exposure=1.5)[0, :, 0].astype(int)
    assert np.all(np.diff(out) >= 0)
    assert out[-1] >= 254
    top = out[int(0.75 * len(out)):]
    assert len(np.unique(top)) > 12, "the top quarter of the ramp collapsed to white"


def test_contrast_pivots_on_middle_grey():
    ramp = _ramp()
    more, less = _render(ramp, contrast=60)[0, :, 0], _render(ramp, contrast=-60)[0, :, 0]
    mid = ramp.shape[1] // 2
    assert abs(int(more[mid]) - 128) <= 2 and abs(int(less[mid]) - 128) <= 2
    assert more.std() > ramp[0, :, 0].std() > less.std()
    assert np.all(np.diff(more.astype(int)) >= 0) and np.all(np.diff(less.astype(int)) >= 0)


def test_blacks_and_whites_move_the_ends():
    ramp = _ramp()
    assert _render(ramp, blacks=-80)[0, 20, 0] < ramp[0, 20, 0]
    assert _render(ramp, blacks=80)[0, 0, 0] > 0
    assert _render(ramp, whites=80)[0, -30, 0] > ramp[0, -30, 0]
    assert _render(ramp, whites=-80)[0, -1, 0] < 255


def test_a_point_curve_passes_through_its_points_and_never_overshoots():
    points = [[0.0, 0.0], [0.25, 0.10], [0.5, 0.55], [0.75, 0.92], [1.0, 1.0]]
    lut = dp.curve_lut(points)
    for x, y in points:
        assert lut[int(round(x * (len(lut) - 1)))] == pytest.approx(y, abs=1e-3)
    assert np.all(np.diff(lut) >= -1e-6), "a monotone set of points produced a curve that dips"


def test_channel_curves_touch_only_their_channel():
    recipe = Recipe()
    recipe.curve.red = [[0.0, 0.0], [0.5, 0.7], [1.0, 1.0]]
    out = dp.render(dp.from_array(_flat((128, 128, 128))), recipe)[0, 0]
    assert out[0] > 150 and out[1] == 128 and out[2] == 128


def test_shadows_lift_the_dark_side_and_highlights_pull_the_bright_side():
    image = np.zeros((200, 400, 3), np.uint8)
    image[:, :200], image[:, 200:] = 45, 210
    lifted = _render(image, shadows=80)
    assert lifted[100, 60, 0] > 45 + 8, "shadows did not lift the dark half"
    assert abs(int(lifted[100, 340, 0]) - 210) <= 3, "shadows moved the bright half"
    recovered = _render(image, highlights=-80)
    assert recovered[100, 340, 0] < 210 - 8
    assert abs(int(recovered[100, 60, 0]) - 45) <= 3


def test_white_balance_tints_grey_and_keeps_its_brightness():
    warm = _render(_flat((128, 128, 128)), temperature=60)[0, 0].astype(float)
    cool = _render(_flat((128, 128, 128)), temperature=-60)[0, 0].astype(float)
    assert warm[0] > warm[1] > warm[2] and cool[2] > cool[1] > cool[0]
    gains = dp.white_balance_gains(60, 0)
    assert float(gains @ dp.LUMA) == pytest.approx(1.0, abs=1e-5)
    magenta = _render(_flat((128, 128, 128)), tint=60)[0, 0].astype(int)
    assert magenta[1] < magenta[0] and magenta[1] < magenta[2]


# --------------------------------------------------------------------- #
# Colour
# --------------------------------------------------------------------- #

def test_full_desaturation_is_grey():
    out = _render(_scene(), saturation=-100)
    assert _diff(out[:, :, 0], out[:, :, 1]) <= 1 and _diff(out[:, :, 1], out[:, :, 2]) <= 1


def test_vibrance_moves_muted_colour_more_than_vivid_colour():
    def chroma(pixel):
        return int(pixel.max()) - int(pixel.min())

    muted, vivid = (140, 120, 115), (230, 40, 30)
    gain_muted = chroma(_render(_flat(muted), vibrance=80)[0, 0]) / chroma(np.array(muted))
    gain_vivid = chroma(_render(_flat(vivid), vibrance=80)[0, 0]) / chroma(np.array(vivid))
    assert gain_muted > gain_vivid >= 1.0


def _patches() -> np.ndarray:
    image = np.zeros((60, 180, 3), np.uint8)
    image[:, :60], image[:, 60:120], image[:, 120:] = (200, 40, 40), (40, 60, 200), (128, 128, 128)
    return image


def test_an_hsl_band_reaches_its_own_colour_only():
    recipe = Recipe()
    recipe.hsl.saturation[pr.HSL_BANDS.index("red")] = -100
    out = dp.render(dp.from_array(_patches()), recipe)
    red, blue, grey = out[30, 30].astype(int), out[30, 90].astype(int), out[30, 150].astype(int)
    assert red.max() - red.min() <= 2, "the red patch kept its colour"
    assert abs(blue - [40, 60, 200]).max() <= 2, "the blue patch changed"
    assert abs(grey - 128).max() <= 1, "a neutral grey changed"


def test_hsl_hue_and_luminance():
    recipe = Recipe()
    recipe.hsl.hue[pr.HSL_BANDS.index("red")] = 100           # towards orange
    shifted = dp.render(dp.from_array(_patches()), recipe)[30, 30].astype(int)
    assert shifted[1] > 40 + 15, "red did not move towards orange"

    recipe = Recipe()
    recipe.hsl.luminance[pr.HSL_BANDS.index("blue")] = -100
    out = dp.render(dp.from_array(_patches()), recipe)
    assert out[30, 90].astype(int).max() < 200 - 30, "blue was not darkened"
    assert abs(out[30, 30].astype(int) - [200, 40, 40]).max() <= 2


def test_colour_grading_tints_only_its_tonal_range():
    image = np.zeros((60, 120, 3), np.uint8)
    image[:, :60], image[:, 60:] = 30, 225
    recipe = Recipe()
    recipe.grade.shadows = pr.GradeWheel(hue=240.0, saturation=80.0)
    out = dp.render(dp.from_array(image), recipe)
    dark, bright = out[30, 30].astype(int), out[30, 90].astype(int)
    assert dark[2] > dark[0] + 5, "shadows were not tinted blue"
    assert abs(bright - 225).max() <= 1, "the highlights were tinted too"


# --------------------------------------------------------------------- #
# 3D LUTs
# --------------------------------------------------------------------- #

def _cube_text(size, fn) -> str:
    lines = [f"TITLE \"test\"", f"LUT_3D_SIZE {size}"]
    for b in range(size):
        for g in range(size):
            for r in range(size):
                lines.append(" ".join(f"{v:.6f}" for v in fn(r / (size - 1), g / (size - 1), b / (size - 1))))
    return "\n".join(lines) + "\n"


def test_an_identity_cube_changes_nothing(tmp_path):
    path = tmp_path / "identity.cube"
    path.write_text(_cube_text(9, lambda r, g, b: (r, g, b)))
    recipe = Recipe()
    recipe.lut = pr.Lut(str(path), 100.0)
    image = _scene()
    assert _diff(dp.render(dp.from_array(image), recipe), image) <= 1


def test_a_cube_is_read_red_fastest(tmp_path):
    path = tmp_path / "swap.cube"
    path.write_text(_cube_text(5, lambda r, g, b: (b, g, r)))       # swap red and blue
    recipe = Recipe()
    recipe.lut = pr.Lut(str(path), 100.0)
    out = dp.render(dp.from_array(_flat((200, 100, 30))), recipe)[0, 0].astype(int)
    assert abs(out - [30, 100, 200]).max() <= 1


def test_cube_strength_blends(tmp_path):
    path = tmp_path / "invert.cube"
    path.write_text(_cube_text(5, lambda r, g, b: (1 - r, 1 - g, 1 - b)))
    recipe = Recipe()
    recipe.lut = pr.Lut(str(path), 50.0)
    out = dp.render(dp.from_array(_flat((200, 100, 30))), recipe)[0, 0].astype(int)
    assert abs(out - 128).max() <= 2, "half strength of an inversion should land on mid grey"


@pytest.mark.parametrize("text", [
    "LUT_3D_SIZE 2\n0 0 0\n1 1 1\n",             # too few rows
    "0 0 0\n1 1 1\n",                            # no size
    "LUT_1D_SIZE 4\n0 0 0\n",                    # wrong kind
])
def test_a_malformed_cube_is_rejected_then_ignored(tmp_path, text):
    with pytest.raises(ValueError):
        dp.parse_cube(text)
    path = tmp_path / "bad.cube"
    path.write_text(text)
    recipe = Recipe()
    recipe.lut = pr.Lut(str(path), 100.0)
    image = _scene()
    assert _diff(dp.render(dp.from_array(image), recipe), image) <= 1, "a bad LUT altered the picture"
    recipe.lut = pr.Lut(str(tmp_path / "missing.cube"), 100.0)
    assert _diff(dp.render(dp.from_array(image), recipe), image) <= 1


# --------------------------------------------------------------------- #
# Presence and detail
# --------------------------------------------------------------------- #

def _local_contrast(image) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _soft_scene() -> np.ndarray:
    return cv2.GaussianBlur(_scene((640, 480)), (0, 0), 1.5)


@pytest.mark.parametrize("control", ["clarity", "texture"])
def test_presence_controls_raise_and_lower_local_contrast(control):
    image = _soft_scene()
    base = _local_contrast(image)
    assert _local_contrast(_render(image, **{control: 80})) > base * 1.05
    assert _local_contrast(_render(image, **{control: -80})) < base * 0.98


def test_dehaze_restores_contrast_lost_to_a_veil():
    clear = _scene((480, 320))
    hazy = (clear.astype(np.float32) * 0.55 + 255 * 0.45 * 0.9).astype(np.uint8)
    restored = _render(hazy, dehaze=80)
    assert restored.std() > hazy.std() * 1.25
    assert _render(hazy, dehaze=-80).std() < hazy.std()


def test_sharpening_and_noise_reduction():
    image = _soft_scene()
    recipe = Recipe()
    recipe.detail.sharpen_amount = 100
    assert _local_contrast(dp.render(dp.from_array(image), recipe)) > _local_contrast(image) * 1.3

    rng = np.random.default_rng(4)
    noisy = np.clip(128 + rng.normal(0, 12, (200, 200, 3)), 0, 255).astype(np.uint8)
    recipe = Recipe()
    recipe.detail.noise_reduction = 100
    assert dp.render(dp.from_array(noisy), recipe).std() < noisy.std() * 0.8


# --------------------------------------------------------------------- #
# Strips, previews and bit depth
# --------------------------------------------------------------------- #

def _everything() -> Recipe:
    recipe = Recipe()
    recipe.basic.exposure, recipe.basic.contrast = 0.4, 15
    recipe.basic.shadows, recipe.basic.highlights = 40, -30
    recipe.basic.clarity, recipe.basic.texture, recipe.basic.dehaze = 30, 25, 20
    recipe.basic.vibrance = 20
    recipe.hsl.saturation[5] = -30
    recipe.detail.sharpen_amount, recipe.detail.noise_reduction = 50, 20
    return recipe


def test_strips_leave_no_seams(monkeypatch):
    """Rendering in bands must give the same picture as rendering whole."""
    source = dp.from_array(_scene((640, 600)))
    recipe = _everything()
    monkeypatch.setattr(dp, "STRIP_ROWS", 100_000)
    whole = dp.render(source, recipe)
    monkeypatch.setattr(dp, "STRIP_ROWS", 64)
    banded = dp.render(source, recipe)
    assert _diff(whole, banded) <= 1


def test_the_preview_matches_the_export():
    """Large-radius effects are computed on a small copy precisely so this
    holds: a preview is the export, shrunk."""
    source = dp.from_array(cv2.GaussianBlur(_scene((2400, 1600), seed=3), (0, 0), 4))
    recipe = _everything()
    recipe.basic.texture = 0            # fine detail legitimately differs with size
    recipe.detail.sharpen_amount = recipe.detail.noise_reduction = 0
    preview = dp.render_preview(source, recipe, max_side=600).astype(np.float64)
    full = dp.render(source, recipe)
    shrunk = cv2.resize(full, (preview.shape[1], preview.shape[0]), interpolation=cv2.INTER_AREA)
    assert np.abs(preview - shrunk).mean() < 2.0


def test_sixteen_bit_output_keeps_precision():
    ramp = np.linspace(0, 65535, 4096).round().astype(np.uint16)
    image = np.repeat(np.repeat(ramp[None, :, None], 8, axis=0), 3, axis=2)
    recipe = Recipe()
    recipe.basic.contrast = 20
    deep = dp.render(dp.from_array(image), recipe, bits=16)
    assert deep.dtype == np.uint16
    assert len(np.unique(deep[0, :, 0])) > 3000, "16-bit output was quantised to 8"
    assert len(np.unique(dp.render(dp.from_array(image), recipe, bits=8)[0, :, 0])) <= 256


def test_the_lookup_shortcut_matches_the_full_computation():
    image = _scene()
    recipe = Recipe()
    recipe.basic.exposure, recipe.basic.contrast, recipe.basic.temperature = 0.8, 30, 20
    recipe.basic.blacks, recipe.basic.whites = -20, 15
    recipe.curve.master = [[0, 0], [0.3, 0.24], [0.7, 0.8], [1, 1]]
    recipe.curve.red = [[0, 0], [0.5, 0.6], [1, 1]]
    fast = dp.render(dp.from_array(image), recipe)
    full = dp.render(dp.from_array(image.astype(np.float32) / 255.0), recipe)
    assert _diff(fast, full) <= 1


# --------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------- #

def _marked() -> np.ndarray:
    image = np.full((300, 400, 3), 60, np.uint8)
    image[:60, :80] = (255, 0, 0)               # red in the top-left corner
    return image


def test_quarter_turns_and_flips():
    g = pr.Geometry(quarter_turns=1)
    turned = dp.apply_geometry(_marked(), g)
    assert turned.shape[:2] == (400, 300)
    assert tuple(turned[10, -10]) == (255, 0, 0), "a clockwise turn should put the top-left at the top-right"
    flipped = dp.apply_geometry(_marked(), pr.Geometry(flip_horizontal=True))
    assert tuple(flipped[10, -10]) == (255, 0, 0)
    assert tuple(dp.apply_geometry(_marked(), pr.Geometry(flip_vertical=True))[-10, 10]) == (255, 0, 0)
    assert np.array_equal(dp.apply_geometry(_marked(), pr.Geometry(quarter_turns=4)), _marked())


@pytest.mark.parametrize("angle", [-12.0, -3.0, 5.0, 20.0])
def test_straightening_leaves_no_blank_corners(angle):
    image = np.full((400, 600, 3), 180, np.uint8)
    out = dp.apply_geometry(image, pr.Geometry(rotation=angle))
    assert out.min() > 150, "a blank wedge survived the crop"
    assert out.shape[0] < 400 and out.shape[1] < 600
    assert out.shape[1] / out.shape[0] == pytest.approx(600 / 400, rel=0.03)


def test_straightening_can_keep_the_whole_canvas():
    image = np.full((400, 600, 3), 180, np.uint8)
    out = dp.apply_geometry(image, pr.Geometry(rotation=10.0, constrain_to_image=False))
    assert out.shape[:2] == (400, 600) and out.min() == 0


@pytest.mark.parametrize("tilt", [-6.0, 4.0])
def test_rotation_levels_a_tilted_frame(tilt):
    from tests.conftest import rotate_keep_size

    tilted = rotate_keep_size(make_architecture(), tilt)
    found = orientation.detect(tilted)
    assert found.needs_rotation

    # The recipe stores the same correction auto-level applies directly,
    # with the same sign: positive is counter-clockwise.
    geometry = pr.Geometry()
    dp.set_rotation(geometry, found.total_angle)
    levelled = dp.apply_geometry(tilted, geometry)
    reference = orientation.rotate_image(tilted, found.total_angle, crop=True)

    residual, _ = orientation.line_tilt(levelled)
    reference_residual, _ = orientation.line_tilt(reference)
    assert residual is not None and abs(residual) < 1.0
    assert abs(residual - reference_residual) < 0.6


@pytest.mark.parametrize("total,turns,rest", [(0.0, 0, 0.0), (93.0, 3, 3.0), (-88.0, 1, 2.0), (181.5, 2, 1.5), (4.0, 0, 4.0)])
def test_a_total_rotation_splits_into_turns_and_straightening(total, turns, rest):
    geometry = pr.Geometry()
    dp.set_rotation(geometry, total)
    assert geometry.quarter_turns == turns and geometry.rotation == pytest.approx(rest)


def test_crop_uses_fractions_of_the_frame():
    out = dp.apply_geometry(_marked(), pr.Geometry(crop=[0.25, 0.5, 0.5, 0.25]))
    assert out.shape[:2] == (75, 200)


def test_keystone_corrects_without_leaving_blanks():
    image = np.full((400, 600, 3), 180, np.uint8)
    out = dp.apply_geometry(image, pr.Geometry(keystone_vertical=60.0))
    assert out.min() > 150 and out.shape[0] < 400


def test_geometry_is_part_of_a_render():
    recipe = Recipe()
    recipe.geometry.quarter_turns = 1
    assert dp.render(dp.from_array(_marked()), recipe).shape[:2] == (400, 300)
    assert dp.render(dp.from_array(_marked()), recipe, geometry=False).shape[:2] == (300, 400)


# --------------------------------------------------------------------- #
# Lens
# --------------------------------------------------------------------- #

def test_manual_vignette_brightens_corners_and_spares_the_centre():
    image = _flat((100, 100, 100), (300, 200))
    recipe = Recipe()
    recipe.lens.vignette = 80
    out = dp.render(dp.from_array(image), recipe)
    assert out[2, 2, 0] > 100 + 15
    assert abs(int(out[100, 150, 0]) - 100) <= 1
    recipe.lens.vignette = -80
    assert dp.render(dp.from_array(image), recipe)[2, 2, 0] < 100 - 10


def test_manual_distortion_moves_the_edges_and_not_the_centre():
    image = cv2.cvtColor(make_architecture((400, 300)), cv2.COLOR_BGR2RGB)
    recipe = Recipe()
    recipe.lens.distortion = 80
    out = dp.render(dp.from_array(image), recipe)
    centre = (slice(140, 160), slice(190, 210))
    assert _diff(out[centre], image[centre]) <= 40
    assert np.abs(out.astype(int) - image.astype(int)).mean() > 2, "the frame did not move at all"


def test_no_lens_profile_is_reported_not_guessed(tmp_path):
    path = tmp_path / "plain.jpg"
    cv2.imwrite(str(path), make_scene(1))
    assert dp.lens_profile_available(path) is False
    source = dp.open_source(path)
    recipe = Recipe()
    recipe.lens.profile = True
    out = dp.render(source, recipe)
    assert _diff(out, source.data) <= 1, "a missing profile changed the picture"
    assert "no lens profile" in dp.correct_lens_geometry(source).note


def test_a_real_lens_profile_is_applied(tmp_path):
    """Write a file whose EXIF names a camera and lens the database knows,
    and check the profile actually moves pixels."""
    lensfunpy = pytest.importorskip("lensfunpy")
    from PIL import Image

    db = lensfunpy.Database()
    pick = None
    for lens in db.lenses:
        if not lens.calib_distortion:
            continue
        cameras = [c for c in db.cameras if c.mount in lens.mounts]
        for camera in cameras[:3]:
            if db.find_lenses(camera, None, lens.model):
                pick = (camera, lens)
                break
        if pick:
            break
    if pick is None:
        pytest.skip("no calibrated camera and lens pair in the bundled database")
    camera, lens = pick

    exif = Image.Exif()
    exif[0x010F], exif[0x0110] = camera.maker, camera.model
    sub = exif.get_ifd(0x8769)
    sub[0xA434] = lens.model
    sub[0x920A] = float(lens.min_focal)
    sub[0x829D] = float(max(lens.min_aperture or 4.0, 4.0))
    path = tmp_path / "lens.jpg"
    Image.fromarray(cv2.cvtColor(make_architecture((900, 600)), cv2.COLOR_BGR2RGB)).save(path, "JPEG", exif=exif, quality=95)

    assert dp.lens_profile_available(path)
    source = dp.open_source(path)
    corrected = dp.correct_lens_geometry(source)
    assert corrected.note is None
    assert np.abs(corrected.data.astype(int) - source.data.astype(int)).mean() > 0.5, "the profile moved nothing"


# --------------------------------------------------------------------- #
# Sources and export
# --------------------------------------------------------------------- #

def test_open_source_reads_ordinary_and_deep_files(tmp_path):
    jpeg = tmp_path / "a.jpg"
    cv2.imwrite(str(jpeg), make_scene(1))
    source = dp.open_source(jpeg)
    assert source.data.dtype == np.uint8 and not source.linear and not source.is_raw

    deep = tmp_path / "deep.png"
    cv2.imwrite(str(deep), (make_scene(1).astype(np.uint16) * 257))
    source = dp.open_source(deep)
    assert source.data.dtype == np.uint16 and source.data.shape[2] == 3
    assert tuple(source.data[5, 5]) == tuple(int(v) * 257 for v in make_scene(1)[5, 5][::-1]), "channels are in the wrong order"

    assert dp.open_source(tmp_path / "missing.jpg") is None
    junk = tmp_path / "junk.jpg"
    junk.write_bytes(b"nope")
    assert dp.open_source(junk) is None


def test_auto_tone_suggests_a_sensible_exposure():
    dark = dp.from_array(np.full((50, 50, 3), 2000, np.uint16), linear=True)
    assert 1.0 < dp.auto_tone(dark) <= 3.0
    grey = dp.from_array(np.full((50, 50, 3), int(0.18 * 65535), np.uint16), linear=True)
    assert dp.auto_tone(grey) == pytest.approx(0.0, abs=0.05)
    assert dp.auto_tone(dp.from_array(np.zeros((10, 10, 3), np.uint16), linear=True)) == 0.0


@pytest.mark.parametrize("fmt,suffix,dtype", [
    ("jpeg", ".jpg", np.uint8), ("png", ".png", np.uint8), ("tiff", ".tif", np.uint8),
    ("tiff16", ".tif", np.uint16), ("png16", ".png", np.uint16),
])
def test_export_formats(library, fmt, suffix, dtype):
    before = snapshot(library)
    recipe = Recipe()
    recipe.basic.contrast = 20
    out = dp.export(library / "IMG_0003.jpg", recipe, library / "Shoot-edit" / f"out{suffix}", fmt=fmt)
    back = cv2.imdecode(np.fromfile(str(out), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert back is not None and back.dtype == dtype and back.shape[:2] == (480, 640)
    assert_originals_untouched(before, library)


def test_export_size_and_resolution(library):
    from PIL import Image

    out = dp.export(library / "IMG_0003.jpg", Recipe(), library / "Shoot-edit" / "small.jpg",
                    max_side=200, dpi=300)
    with Image.open(out) as image:
        assert max(image.size) == 200
        assert tuple(round(v) for v in image.info["dpi"]) == (300, 300)
    with pytest.raises(ValueError):
        dp.export(library / "IMG_0003.jpg", Recipe(), library / "Shoot-edit" / "x.bmp", fmt="bmp")


@pytest.mark.media
@pytest.mark.slow
def test_a_raw_file_develops_to_linear_sixteen_bit():
    sample = next((s for s in SAMPLES if s.name == "canon_eos_r6.cr3" and s.present), None)
    if sample is None:
        pytest.skip("sample not downloaded")
    source = dp.open_source(sample.path, half_size=True)
    assert source.is_raw and source.linear and source.data.dtype == np.uint16
    recipe = Recipe()
    recipe.basic.exposure = dp.auto_tone(source)
    out = dp.render_preview(source, recipe, max_side=800)
    assert out.dtype == np.uint8 and 60 < out.mean() < 190, "the developed frame is implausibly dark or bright"


@pytest.mark.media
@pytest.mark.slow
def test_a_raw_the_decoder_cannot_read_falls_back_to_its_preview():
    sample = next((s for s in SAMPLES if s.name == "nikon_z50_2.nef" and s.present), None)
    if sample is None:
        pytest.skip("sample not downloaded")
    source = dp.open_source(sample.path)
    assert source is not None, "an unsupported RAW could not be opened at all"
    if source.is_raw:
        pytest.skip("this LibRaw build now supports the camera; the fallback is not exercised")
    assert "embedded preview" in source.note and not source.linear
