"""De-blurring: deconvolution of motion smear and missed focus.

The maths is checked on synthetic blur, where the truth is known: a sharp
frame is blurred with a known PSF and the restoration is scored against the
original. That proves the engine inverts the blur it is told about. It does
not prove it on real photographs, whose blur is never exactly one global
kernel.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import Qt

from src.ai_tools import blur_analysis
from src.ai_tools import deblur as db
from src.ai_tools.segmentation import MaskResult
from src.engine import develop_pipeline as dp
from src.engine import parametric_recipe as pr
from tests.conftest import assert_originals_untouched, make_architecture, make_scene, pump, snapshot, write_jpeg


def _rgb(seed: int = 7, size=(640, 480)) -> np.ndarray:
    return cv2.cvtColor(make_scene(seed, size), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _blur(image: np.ndarray, kernel: np.ndarray, noise: float = 0.002, seed: int = 0) -> np.ndarray:
    blurred = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    if noise:
        blurred = blurred + np.random.default_rng(seed).normal(0, noise, image.shape).astype(np.float32)
    return np.clip(blurred, 0.0, 1.0).astype(np.float32)


def psnr(estimate: np.ndarray, truth: np.ndarray) -> float:
    mse = float(np.mean((estimate.astype(np.float64) - truth.astype(np.float64)) ** 2))
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


# --------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("length,angle", [(1, 0), (5, 0), (10, 45), (17.5, 120), (30, 90)])
def test_motion_kernel_is_a_normalised_line_at_the_angle(length, angle):
    k = db.synthesize_motion_kernel(length, angle)
    assert k.dtype == np.float32 and k.shape[0] == k.shape[1] and k.shape[0] % 2 == 1
    assert k.sum() == pytest.approx(1.0, abs=1e-5) and k.min() >= 0
    if length < 2:
        return
    ys, xs = np.indices(k.shape)
    c = (k.shape[0] - 1) / 2
    x, y = xs - c, -(ys - c)                     # y up, as the angle is defined
    cov = np.array([[np.sum(k * x * x), np.sum(k * x * y)], [np.sum(k * x * y), np.sum(k * y * y)]])
    values, vectors = np.linalg.eigh(cov)
    major = vectors[:, 1]
    measured = np.degrees(np.arctan2(major[1], major[0])) % 180
    assert min(abs(measured - angle % 180), 180 - abs(measured - angle % 180)) < 2.0
    # A uniform segment of length L has variance L^2 / 12 along its axis.
    assert np.sqrt(12 * values[1]) == pytest.approx(length, rel=0.12)


@pytest.mark.parametrize("radius", [0.5, 1, 3, 7.5, 20])
def test_defocus_kernel_is_a_normalised_disc(radius):
    k = db.synthesize_defocus_kernel(radius)
    assert k.sum() == pytest.approx(1.0, abs=1e-5)
    assert np.allclose(k, k.T, atol=1e-6) and np.allclose(k, k[::-1, ::-1], atol=1e-6)
    if radius >= 3:
        ys, xs = np.indices(k.shape)
        c = (k.shape[0] - 1) / 2
        # A uniform disc has a mean squared radius of R^2 / 2.
        assert np.sqrt(2 * np.sum(k * ((xs - c) ** 2 + (ys - c) ** 2))) == pytest.approx(radius, rel=0.05)


def test_kernel_for_rejects_unknown_types():
    with pytest.raises(ValueError):
        db.kernel_for("wobble")


@pytest.mark.parametrize("angle", [0, 30, 45, 90, 120])
def test_the_angle_means_what_blur_analysis_reports(angle):
    """Auto-detect hands blur_analysis's angle straight to the kernel, so
    the two must share one convention."""
    scene = cv2.cvtColor(make_scene(3, (800, 600)), cv2.COLOR_BGR2GRAY)
    smeared = cv2.filter2D(scene, -1, db.synthesize_motion_kernel(15, angle))
    _, measured = blur_analysis.directional_blur(smeared)
    assert min(abs(measured - angle), 180 - abs(measured - angle)) <= 7


# --------------------------------------------------------------------- #
# Restoration quality
# --------------------------------------------------------------------- #

def test_richardson_lucy_recovers_45_degree_10px_motion_blur():
    """The specification's regression: at least 4.5 dB better than the blur."""
    sharp = _rgb()
    kernel = db.synthesize_motion_kernel(10, 45)
    blurred = _blur(sharp, kernel)
    restored = db.richardson_lucy_damped(blurred, kernel)
    assert restored.dtype == np.float32 and restored.shape == sharp.shape
    gain = psnr(restored, sharp) - psnr(blurred, sharp)
    assert gain >= 4.5, f"only {gain:.2f} dB better"


def test_richardson_lucy_recovers_defocus():
    sharp = _rgb(11)
    kernel = db.synthesize_defocus_kernel(4)
    blurred = _blur(sharp, kernel)
    gain = psnr(db.richardson_lucy_damped(blurred, kernel), sharp) - psnr(blurred, sharp)
    assert gain >= 3.0, f"only {gain:.2f} dB better"


def test_wiener_recovers_motion_blur_quickly_enough_to_follow_a_slider():
    sharp = _rgb()
    kernel = db.synthesize_motion_kernel(10, 45)
    blurred = _blur(sharp, kernel)
    restored = db.wiener_deconvolution(blurred, kernel, snr=dp.wiener_snr(0.05), taper=False)
    assert psnr(restored, sharp) - psnr(blurred, sharp) >= 3.0


def test_greyscale_input_stays_greyscale():
    sharp = _rgb()[:, :, 1]
    kernel = db.synthesize_motion_kernel(8, 10)
    blurred = _blur(sharp, kernel)
    for out in (db.richardson_lucy_damped(blurred, kernel, 5), db.wiener_deconvolution(blurred, kernel)):
        assert out.shape == sharp.shape


def test_damping_holds_back_noise():
    """On a flat noisy area there is nothing to recover: plain RL amplifies
    the noise, damping leaves it near where it was."""
    flat = np.full((200, 200, 3), 0.5, np.float32)
    kernel = db.synthesize_motion_kernel(9, 30)
    noisy = _blur(flat, kernel, noise=0.004, seed=3)
    plain = db.richardson_lucy_damped(noisy, kernel, damping=0.0)
    damped = db.richardson_lucy_damped(noisy, kernel, damping=0.5)
    core = (slice(30, -30), slice(30, -30))
    assert damped[core].std() < plain[core].std() * 0.7
    assert damped[core].std() < noisy[core].std() * 1.5


def test_divergence_is_clamped_to_the_photographs_range():
    sharp = _rgb()
    kernel = db.synthesize_motion_kernel(20, 0)
    blurred = _blur(sharp, kernel, noise=0.02)
    out = db.richardson_lucy_damped(blurred, kernel, iterations=60, damping=0.0)
    assert np.isfinite(out).all()
    assert out.max() <= min(1.0, blurred.max() * 1.04 + 0.01) + 1e-6


# --------------------------------------------------------------------- #
# Boundaries and tiles
# --------------------------------------------------------------------- #

def _border_error(out: np.ndarray, truth: np.ndarray, width: int = 24) -> float:
    edge = np.ones(truth.shape[:2], bool)
    edge[width:-width, width:-width] = False
    return float(np.abs(out[edge] - truth[edge]).mean())


def test_boundary_handling_beats_a_plain_fft():
    """The specification's edge test: deconvolving the frame as a periodic
    signal rings at the borders; extension plus taper does not."""
    sharp = _rgb()
    kernel = db.synthesize_motion_kernel(10, 45)
    blurred = _blur(sharp, kernel)
    plain = db.wiener_deconvolution(blurred, kernel, snr=100, taper=False, extend=False)
    handled = db.wiener_deconvolution(blurred, kernel, snr=100, taper=True, extend=True)
    assert _border_error(handled, sharp) < _border_error(plain, sharp) * 0.5


def test_edge_taper_alone_reduces_border_ringing():
    sharp = _rgb()
    kernel = db.synthesize_motion_kernel(10, 0)
    blurred = _blur(sharp, kernel)
    plain = db.wiener_deconvolution(blurred, kernel, snr=100, taper=False, extend=False)
    tapered = db.wiener_deconvolution(blurred, kernel, snr=100, taper=True, extend=False)
    assert _border_error(tapered, sharp) < _border_error(plain, sharp)


def test_edge_taper_touches_only_the_border():
    image = _rgb()
    kernel = db.synthesize_motion_kernel(9, 0)
    tapered = db.edge_taper(image, kernel)
    k = kernel.shape[0]
    assert np.allclose(tapered[k:-k, k:-k], image[k:-k, k:-k], atol=1e-5)
    assert not np.allclose(tapered[:, :2], image[:, :2], atol=1e-3)
    # At the very edge the frame becomes periodic: left and right columns agree.
    assert np.abs(tapered[:, 0] - tapered[:, -1]).mean() < np.abs(image[:, 0] - image[:, -1]).mean() * 0.5


def test_tiled_deconvolution_matches_the_whole_frame():
    sharp = cv2.resize(_rgb(5), (1100, 760))
    kernel = db.synthesize_motion_kernel(12, 20)
    blurred = _blur(sharp, kernel)
    whole = db.richardson_lucy_damped(blurred, kernel)
    tiles = db.deconvolve(blurred, kernel, "parametric", tile=400)
    assert np.abs(whole - tiles).mean() < 1e-3
    assert np.abs(whole - tiles).max() < 0.05


def test_overlapping_tiles_blend_without_seams():
    image = _rgb(9, (1300, 900))
    assert np.allclose(db.tiled(image, lambda block: block), image, atol=1e-6)
    local = lambda block: cv2.GaussianBlur(block, (0, 0), 2.0, borderType=cv2.BORDER_REFLECT_101)  # noqa: E731
    tiled = db.tiled(image, local, tile=256, overlap=64)
    whole = local(image)
    assert np.abs(tiled - whole).max() < 0.02, "a seam shows where tiles meet"


# --------------------------------------------------------------------- #
# Estimating the blur
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("scene", ["texture", "facade", "large"])
@pytest.mark.parametrize("length", [8, 12, 20, 30])
def test_motion_length_is_estimated(scene, length):
    image = {
        "texture": lambda: make_scene(3, (800, 600)),
        "facade": lambda: make_architecture((800, 600)),
        "large": lambda: make_scene(9, (1200, 800)),      # measured on a reduced copy
    }[scene]()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smeared = cv2.filter2D(gray, -1, db.synthesize_motion_kernel(length, 30))
    estimate = db.estimate_motion_length(smeared, 30)
    assert estimate is not None and estimate == pytest.approx(length, abs=max(1.5, 0.1 * length))


@pytest.mark.parametrize("radius", [5, 8, 12])
def test_defocus_radius_is_estimated(radius):
    for image in (make_scene(3, (800, 600)), make_architecture((800, 600))):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        estimate = db.estimate_defocus_radius(cv2.filter2D(gray, -1, db.synthesize_defocus_kernel(radius)))
        assert estimate is not None and estimate == pytest.approx(radius, abs=1.0)


def test_a_featureless_frame_gives_no_estimate():
    flat = np.full((400, 400), 128, np.uint8)
    assert db.estimate_motion_length(flat, 0) is None
    assert db.estimate_defocus_radius(flat) is None


# --------------------------------------------------------------------- #
# The neural engine
# --------------------------------------------------------------------- #

def test_the_neural_model_is_pinned():
    spec = db.NAFNET_SPEC
    assert len(spec.sha256) == 64 and spec.size == 91_736_251
    assert spec.url.startswith("https://huggingface.co/opencv/deblurring_nafnet/")


def test_without_the_model_neural_deblur_is_off_not_faked(tmp_path, monkeypatch):
    from src.ai_tools import model_store

    monkeypatch.setattr(model_store, "MODEL_DIR", tmp_path)
    engine = db.TiledNeuralDeblur(download=False)
    assert not engine.available
    with pytest.raises(RuntimeError):
        engine.restore(np.zeros((64, 64, 3), np.uint8))
    dp._engines.pop("neural", None)
    source = dp.from_array((_rgb() * 255).astype(np.uint8))
    out = dp.deblur_source(source, pr.Deblur(enabled=True, engine="neural"))
    assert out.data is source.data and "not downloaded" in out.note
    dp._engines.pop("neural", None)


@pytest.mark.model
def test_neural_deblur_improves_a_smeared_photograph():
    """NAFNet was trained on photographs, so it is judged on one: the flat
    synthetic scenes the other tests use are outside what it knows, and it
    distorts them (a sharp scene comes back at 18 dB). Measured on this
    photograph: +1.5 dB for a 15 px smear at 30 degrees."""
    if not db.TiledNeuralDeblur.is_cached():
        pytest.skip("the NAFNet model is not downloaded")
    skimage_data = pytest.importorskip("skimage.data")
    engine = db.TiledNeuralDeblur(download=False)
    sharp = skimage_data.astronaut().astype(np.float32) / 255.0
    blurred = _blur(sharp, db.synthesize_motion_kernel(15, 30), noise=0.0)
    restored = engine.restore(blurred)
    assert restored.shape == sharp.shape and np.isfinite(restored).all()
    assert psnr(restored, sharp) > psnr(blurred, sharp) + 0.5, "the model did not de-blur the photograph"


# --------------------------------------------------------------------- #
# The recipe and the pipeline
# --------------------------------------------------------------------- #

def test_deblur_settings_round_trip_and_tolerate_nonsense():
    recipe = pr.Recipe()
    recipe.deblur = pr.Deblur(True, "wiener_fast", "defocus", 33.0, 12.5, 4.0, 0.3, "face_only")
    assert pr.from_dict(pr.to_dict(recipe)).deblur == recipe.deblur
    odd = pr.from_dict({"deblur": {"enabled": "yes", "engine": "magic", "blur_type": 3, "angle": 999,
                                   "length": -4, "radius": "big", "damping": float("nan"), "mask_mode": None}})
    assert odd.deblur == pr.Deblur(False, "parametric", "motion", 180.0, 1.0, 3.0, 0.05, "full")
    assert pr.Recipe().is_neutral and not pr.from_dict({"deblur": "junk"}).deblur.enabled


def _deblur_recipe(**changes) -> pr.Recipe:
    recipe = pr.Recipe()
    recipe.deblur = pr.Deblur(enabled=True, angle=45, length=10, **changes)
    return recipe


def _blur_like_a_camera(sharp: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Blur in linear light, as a lens or a moving sensor does, then encode."""
    linear = _blur(dp.srgb_decode(sharp), kernel, noise=0.0005)
    return np.round(dp.srgb_encode(linear) * 255).astype(np.uint8)


def test_the_pipeline_deblurs_a_photograph():
    sharp = _rgb()
    blurred8 = _blur_like_a_camera(sharp, db.synthesize_motion_kernel(10, 45))
    source = dp.from_array(blurred8)
    plain = dp.render(source, pr.Recipe()).astype(np.float32) / 255
    fixed = dp.render(source, _deblur_recipe()).astype(np.float32) / 255
    truth = np.round(sharp * 255) / 255
    assert psnr(fixed, truth) > psnr(plain, truth) + 3.0
    fast = dp.render(source, _deblur_recipe(engine="wiener_fast")).astype(np.float32) / 255
    assert psnr(fast, truth) > psnr(plain, truth) + 1.5


def test_lengths_are_in_full_size_pixels_so_the_preview_matches_the_export():
    sharp = _rgb(5, (1280, 960))
    blurred8 = _blur_like_a_camera(sharp, db.synthesize_motion_kernel(12, 30))
    full = dp.from_array(blurred8)
    recipe = _deblur_recipe(engine="wiener_fast")
    recipe.deblur.angle, recipe.deblur.length = 30, 12
    exported = dp.render(full, recipe)
    preview_source = full.proxy(640)
    assert preview_source.scale == pytest.approx(0.5)
    preview = dp.render(preview_source, recipe).astype(np.float32)
    reference = cv2.resize(exported, (640, 480), interpolation=cv2.INTER_AREA).astype(np.float32)
    unscaled = dp.from_array(preview_source.data)          # forgets it is a proxy
    wrong = dp.render(unscaled, recipe).astype(np.float32)
    assert np.abs(preview - reference).mean() < np.abs(wrong - reference).mean() * 0.6


def test_a_disabled_or_neutral_deblur_changes_nothing():
    image = (_rgb() * 255).astype(np.uint8)
    source = dp.from_array(image)
    recipe = pr.Recipe()
    recipe.deblur.length = 25                               # set, but not enabled
    assert np.array_equal(dp.render(source, recipe), image)


class _HalfMask:
    """A segmenter that reports the left half of the frame as the subject."""

    def __init__(self, empty: bool = False) -> None:
        self.empty = empty

    def mask(self, rgb, kind):
        m = np.zeros(rgb.shape[:2], np.float32)
        if not self.empty:
            m[:, : rgb.shape[1] // 2] = 1.0
        return MaskResult(m, "test", "nothing found" if self.empty else None)

    def face(self, rgb):
        return self.mask(rgb, "face")


def test_subject_only_leaves_the_background_alone():
    blurred8 = _blur_like_a_camera(_rgb(), db.synthesize_motion_kernel(10, 45))
    source = dp.from_array(blurred8)
    settings = pr.Deblur(enabled=True, angle=45, length=10, mask_mode="subject_only")
    out = dp.deblur_source(source, settings, segmenter=_HalfMask())
    before = source.to_linear()
    after = out.to_linear()
    w = before.shape[1]
    assert np.abs(after[:, w * 3 // 4:] - before[:, w * 3 // 4:]).max() < 1e-3, "the background was changed"
    assert np.abs(after[:, : w // 4] - before[:, : w // 4]).mean() > 0.01, "the subject was not de-blurred"


def test_a_missing_mask_falls_back_to_the_whole_frame_and_says_so():
    blurred8 = _blur_like_a_camera(_rgb(), db.synthesize_motion_kernel(10, 45))
    settings = pr.Deblur(enabled=True, angle=45, length=10, mask_mode="face_only")
    out = dp.deblur_source(dp.from_array(blurred8), settings, segmenter=_HalfMask(empty=True))
    assert "whole frame" in out.note


def test_results_are_cached_per_settings():
    source = dp.from_array((_rgb() * 255).astype(np.uint8))
    settings = pr.Deblur(enabled=True, angle=10, length=8)
    first = dp.deblur_source(source, settings)
    assert dp.deblur_source(source, settings) is first
    other = dp.deblur_source(source, pr.Deblur(enabled=True, angle=10, length=9))
    assert other is not first


# --------------------------------------------------------------------- #
# Non-destructive, end to end through the Develop workspace
# --------------------------------------------------------------------- #

@pytest.fixture
def blurred_shoot(harness, tmp_path):
    """A folder with one motion-blurred frame larger than the preview, opened
    in the real window."""
    folder = tmp_path / "Blurred"
    folder.mkdir()
    sharp = make_scene(13, (2400, 1600))
    smeared = cv2.filter2D(sharp, -1, db.synthesize_motion_kernel(16, 30))
    write_jpeg(folder / "SMEAR.jpg", smeared, quality=97)
    w = harness.window
    w.cull.open_folder(folder)
    pump(harness.app, lambda: w.cull._thumb_worker is None or not w.cull._thumb_worker.isRunning())
    return folder


def _settle(harness, workspace) -> None:
    for _ in range(3):
        workspace.loader_job.wait_idle(harness.app)
        workspace.renderer.wait_idle(harness.app)


def test_deblur_in_develop_writes_only_the_sidecar(harness, blurred_shoot, monkeypatch):
    before = snapshot(blurred_shoot)
    w = harness.window
    key = str(blurred_shoot / "SMEAR.jpg")
    w.open_in_develop(key)
    develop = w.develop
    _settle(harness, develop)
    assert develop.source.scale == pytest.approx(1600 / 2400)

    engines = []
    real = dp.deblur_source
    monkeypatch.setattr(dp, "deblur_source", lambda s, d, seg=None: (engines.append(d.engine), real(s, d, seg))[1])

    develop._on_deblur_auto()
    _settle(harness, develop)
    d = develop.recipe.deblur
    assert d.enabled and d.blur_type == "motion"
    assert min(abs(d.angle - 30), 180 - abs(d.angle - 30)) <= 8
    assert d.length == pytest.approx(16, abs=3), "the length should be in full-size pixels"

    row = develop.sliders[("deblur", "length")]
    engines.clear()
    row.set_value(14.0, announce=True)                      # dragging
    _settle(harness, develop)
    assert engines and engines[-1] == "wiener_fast"
    row.released.emit()                                     # let go
    _settle(harness, develop)
    assert engines[-1] == "parametric"

    w.switch_to(w.cull)                                     # leaving saves
    edit_files = sorted(p.name for p in w.edit_folder.iterdir() if p.is_file())
    assert "SMEAR.jpg.granite.json" in edit_files
    assert not [n for n in edit_files if n.lower().endswith((".jpg", ".png", ".tif"))]
    saved = json.loads((w.edit_folder / "SMEAR.jpg.granite.json").read_text(encoding="utf-8"))["deblur"]
    assert saved["enabled"] is True and saved["length"] == 14.0
    assert_originals_untouched(before, blurred_shoot)


def test_deblur_stays_with_its_photo_when_settings_are_pasted(harness, blurred_shoot):
    w = harness.window
    develop = w.develop
    w.open_in_develop(str(blurred_shoot / "SMEAR.jpg"))
    _settle(harness, develop)
    develop.deblur_enable.setChecked(True)
    develop._on_copy()
    target = pr.Recipe()
    pasted = develop._settings_from_clipboard(target)
    assert not pasted.deblur.enabled, "a smear's angle and length belong to one photograph"


def test_choosing_ai_neural_asks_before_downloading(harness, blurred_shoot, tmp_path, monkeypatch):
    from src.ai_tools import model_store

    monkeypatch.setattr(model_store, "MODEL_DIR", tmp_path / "empty")
    asked = []
    monkeypatch.setattr(w := harness.window, "with_models",
                        lambda specs, purpose, then, button: asked.append([s.filename for s in specs]))
    develop = w.develop
    w.open_in_develop(str(blurred_shoot / "SMEAR.jpg"))
    _settle(harness, develop)
    develop.deblur_enable.setChecked(True)
    develop.deblur_mode.setCurrentIndex(develop.deblur_mode.findData("neural"))
    assert asked and db.NAFNET_SPEC.filename in asked[-1]
    assert not develop.sliders[("deblur", "length")].isVisibleTo(develop)
