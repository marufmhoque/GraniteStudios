"""Speed: thumbnails, the Develop preview, and the brush.

PF-01  ten RAW thumbnails decode in under 200 ms (the grid's thread pool)
PF-02  thumbnails come from the embedded JPEG: no full RAW development; a RAW
       without one is decoded at half size
PF-03  the thread pool hands files out 20 at a time, streams every file once,
       and can be cancelled between chunks
PF-04  a slider's round trip on the 1,200 px proxy is under 25 ms
PF-05  the faster saturation step gives exactly the old result; parallel
       preview strips match the one-strip render to within one level
PF-06  Develop's two tiers: a draft while the hand moves, then the full
       preview once it rests; in-between renders are shown, not dropped
PF-07  moving the brush repaints only the rectangles under the cursor

tests/profile_performance.py prints the same measurements in full.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from src.engine import develop_pipeline as dp
from src.engine import fast_preview as fp
from src.engine import parametric_recipe as pr
from src.engine.universal_loader import THUMB_CHUNK, UniversalLoader
from tests.conftest import make_scene, pump, write_jpeg


def _raw_samples() -> list[Path]:
    from tests.fetch_media import RAW_DIR

    return sorted(p for p in RAW_DIR.glob("*") if p.is_file()) if RAW_DIR.is_dir() else []


# --------------------------------------------------------------------- #
# PF-01 .. PF-03: thumbnails
# --------------------------------------------------------------------- #

@pytest.mark.media
@pytest.mark.benchmark
def test_ten_raw_thumbnails_in_under_200_ms():
    samples = _raw_samples()[:10]
    if len(samples) < 10:
        pytest.skip("needs ten RAW samples (python tests/fetch_media.py)")
    loader = UniversalLoader()
    list(loader.iter_thumbnails(samples, size=320))            # the files into the OS cache
    runs = []
    for _ in range(3):
        started = time.perf_counter()
        thumbs = list(loader.iter_thumbnails(samples, size=320))
        runs.append((time.perf_counter() - started) * 1000.0)
    assert all(thumb is not None and max(thumb.size) <= 320 for _path, thumb, _meta in thumbs)
    assert min(runs) < 200.0, f"ten RAW thumbnails took {min(runs):.0f} ms"


class _Spy:
    """Wraps a rawpy file to see which calls the thumbnail path makes."""

    calls: list[str] = []

    def __init__(self, raw):
        self._raw = raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self._raw.__exit__(*exc)

    def __getattr__(self, name):
        if name in ("postprocess", "extract_thumb"):
            _Spy.calls.append(name)
        return getattr(self._raw, name)


@pytest.mark.media
def test_raw_thumbnails_never_develop_the_raw_data(monkeypatch):
    samples = [p for p in _raw_samples() if p.suffix.lower() != ".3fr"]  # the Hasselblad keeps a bitmap
    if not samples:
        pytest.skip("no RAW samples downloaded")
    real = fp.rawpy.imread
    monkeypatch.setattr(fp.rawpy, "imread", lambda path: _Spy(real(path)))
    _Spy.calls = []
    for sample in samples:
        preview = fp.load_preview(sample, max_side=320, min_side=0)
        assert preview is not None and preview.source == fp.EMBEDDED_JPEG, sample.name
    assert "postprocess" not in _Spy.calls
    assert _Spy.calls.count("extract_thumb") == len(samples)


def test_the_fallback_decodes_at_half_size():
    import inspect

    source = inspect.getsource(fp._locate_raw)
    assert "postprocess(half_size=True" in source


def test_the_pool_streams_every_file_once_in_chunks_of_20(tmp_path, monkeypatch):
    paths = [write_jpeg(tmp_path / f"F{i:03d}.jpg", make_scene(i % 5, (200, 150))) for i in range(45)]
    loader = UniversalLoader()
    live, peak = [0], [0]
    real = loader.load_thumbnail

    def counting(path, size=320):
        live[0] += 1
        peak[0] = max(peak[0], live[0])
        try:
            time.sleep(0.002)
            return real(path, size)
        finally:
            live[0] -= 1

    monkeypatch.setattr(loader, "load_thumbnail", counting)
    seen = [path for path, thumb, _meta in loader.iter_thumbnails(paths, size=64, workers=4)
            if thumb is not None]
    assert sorted(seen) == sorted(paths) and len(seen) == len(set(seen))
    assert THUMB_CHUNK == 20 and peak[0] <= 4

    # Cancelled after the first chunk: nothing more is decoded.
    stop = {"now": False}
    first = []
    for path, _thumb, _meta in loader.iter_thumbnails(paths, size=64, cancelled=lambda: stop["now"]):
        first.append(path)
        if len(first) == THUMB_CHUNK:
            stop["now"] = True
    assert len(first) == THUMB_CHUNK


def test_the_grid_still_fills_from_the_pool(harness):
    w = harness.window
    assert set(w.cull.cells) == set(w.session.keys())
    assert all(not cell.pixmap().isNull() for cell in w.cull.cells.values())


# --------------------------------------------------------------------- #
# PF-04, PF-05: the develop pipeline
# --------------------------------------------------------------------- #

@pytest.mark.benchmark
def test_a_slider_round_trip_on_the_proxy_is_under_25_ms():
    from src.ui.workspaces import develop

    proxy = dp.from_array(make_scene(3, (4000, 2667))).proxy(develop.DRAFT_MAX_SIDE)
    recipe = pr.Recipe()
    recipe.basic.exposure, recipe.basic.contrast = 0.5, 20
    recipe.curve.master = [[0, 0], [0.25, 0.2], [0.75, 0.8], [1, 1]]
    dp.render(proxy, recipe)
    best = min(_time(lambda: dp.render(proxy, recipe)) for _ in range(5))
    assert max(proxy.data.shape[:2]) == 1200
    assert best < 25.0, f"the 1,200 px proxy took {best:.1f} ms"


def _time(fn) -> float:
    started = time.perf_counter()
    fn()
    return (time.perf_counter() - started) * 1000.0


def _saturate_before_2_09(encoded, vibrance, saturation):
    """The saturation step as it was, to prove the faster one is exact."""
    luma = np.clip(encoded @ dp.LUMA, 0.0, 1.0)[:, :, None]
    chroma = encoded - luma
    factor = np.float32(1.0 + saturation / 100.0)
    if vibrance:
        high, low = encoded.max(axis=2), encoded.min(axis=2)
        muted = 1.0 - np.clip((high - low) / np.maximum(high, 1e-6), 0.0, 1.0)
        factor = factor * (1.0 + (vibrance / 100.0) * muted)[:, :, None]
    factor = np.broadcast_to(np.asarray(factor, np.float32), luma.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        room = np.where(chroma > 1e-6, (1.0 - luma) / chroma,
                        np.where(chroma < -1e-6, luma / -chroma, np.inf)).min(axis=2, keepdims=True)
    factor = np.where(factor > 1.0, np.minimum(factor, np.maximum(room, 1.0)), np.maximum(factor, 0.0))
    return (luma + chroma * factor).astype(np.float32)


@pytest.mark.parametrize("vibrance,saturation", [(0, 30), (25, 0), (40, -30), (-50, 60), (0, -100), (100, 100)])
def test_the_faster_saturation_gives_exactly_the_old_result(vibrance, saturation):
    rng = np.random.default_rng(7)
    pixels = rng.uniform(-0.05, 1.05, (120, 160, 3)).astype(np.float32)
    pixels[:10] = 0.5                       # greys
    pixels[10:20, :, 0] = 1.0               # a clipped channel
    assert np.array_equal(dp._saturate(pixels, vibrance, saturation),
                          _saturate_before_2_09(pixels, vibrance, saturation))


def test_parallel_preview_strips_match_the_serial_render(monkeypatch):
    source = dp.from_array(make_scene(2, (1600, 1067)))
    recipe = pr.Recipe()
    recipe.basic.highlights, recipe.basic.clarity, recipe.basic.texture = -30, 25, 20
    recipe.basic.vibrance, recipe.detail.sharpen_amount = 20, 40
    parallel = dp.render(source, recipe)
    monkeypatch.setattr(dp, "PARALLEL_MAX_PIXELS", 0)
    serial = dp.render(source, recipe)
    difference = np.abs(parallel.astype(int) - serial.astype(int))
    assert difference.max() <= 1 and (difference > 0).mean() < 0.001


# --------------------------------------------------------------------- #
# PF-06: Develop's two tiers
# --------------------------------------------------------------------- #

@pytest.fixture
def big_develop(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    from src.ui.main_window import MainWindow

    folder = tmp_path / "Shoot"
    folder.mkdir()
    for i in range(2):
        write_jpeg(folder / f"DSC_{i + 1:04d}.jpg", make_scene(i, (3200, 2134)))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(folder)))
    window = MainWindow()
    window._on_load_folder()
    pump(qapp, lambda: window._thumb_worker is None or not window._thumb_worker.isRunning())
    key = window.session.keys()[0]
    assert window.open_in_develop(key)
    develop = window.develop
    pump(qapp, lambda: develop.key == key and develop.source is not None)
    develop.renderer.wait_idle(qapp)
    yield develop
    window.close()


def test_a_draft_follows_the_hand_then_the_full_preview(qapp, big_develop):
    from src.ui.workspaces import develop as dv

    d = big_develop
    assert d._last_render.shape[1] == dv.PREVIEW_SIDE
    d._render_ms = 400.0                    # as if the full preview were slow
    d._draft_side = 640
    shown = []
    d.renderer.ready.connect(lambda result: shown.append(result))
    d._on_value(("basic", "exposure"), 0.4)
    assert d._settle.isActive()
    pump(qapp, lambda: any(isinstance(r[0], str) for r in shown))
    draft = next(r for r in shown if isinstance(r[0], str))[1]
    assert max(draft.shape[:2]) == 640      # the quick tier
    d.renderer.wait_idle(qapp)              # waits for the settle timer and the full render
    assert max(d._last_render.shape[:2]) == dv.PREVIEW_SIDE
    assert not isinstance(shown[-1][0], str)
    assert d._render_ms is not None and d._render_ms < 400.0


def test_fast_previews_skip_the_draft(qapp, big_develop):
    d = big_develop
    d._render_ms = 10.0
    d._on_value(("basic", "exposure"), 0.3)
    assert not d._settle.isActive()
    d.renderer.wait_idle(qapp)
    assert max(d._last_render.shape[:2]) == 1600


def test_overtaken_renders_are_shown_only_when_asked(qapp):
    import threading

    from src.ui.widgets.renderer import LatestOnlyRenderer

    for progress in (False, True):
        renderer = LatestOnlyRenderer()
        renderer.show_progress = progress
        got = []
        renderer.ready.connect(got.append)
        gate = threading.Event()
        renderer.request(lambda: (gate.wait(5), "first")[1])
        renderer.request(lambda: "second")          # queued behind the first
        renderer.request(lambda: "third")           # replaces the second
        gate.set()
        renderer.wait_idle(qapp)
        assert got == (["first", "third"] if progress else ["third"])

    # A different photo: nothing from before cancel() is ever shown.
    renderer = LatestOnlyRenderer()
    renderer.show_progress = True
    got = []
    renderer.ready.connect(got.append)
    gate = threading.Event()
    renderer.request(lambda: (gate.wait(5), "old photo")[1])
    renderer.cancel()
    renderer.request(lambda: "new photo")
    gate.set()
    renderer.wait_idle(qapp)
    assert got == ["new photo"]


# --------------------------------------------------------------------- #
# PF-07: the brush cursor
# --------------------------------------------------------------------- #

def test_moving_the_brush_repaints_only_under_the_cursor(qapp, monkeypatch):
    from PyQt6.QtCore import QEvent, QPointF, QRect, Qt
    from PyQt6.QtGui import QMouseEvent

    from src.ui.widgets.canvas import CANVAS_MODE_COLOR_SPLASH_BRUSH, ImageCanvas

    canvas = ImageCanvas()
    canvas.resize(1200, 800)
    canvas.set_image(np.zeros((1067, 1600, 3), np.uint8))
    canvas.set_mode(CANVAS_MODE_COLOR_SPLASH_BRUSH)
    canvas.set_brush(radius=30)
    repaints = []
    monkeypatch.setattr(canvas, "update", lambda *area: repaints.append(area))
    for x in (300, 340, 380):
        point = QPointF(x, 400)
        canvas.mouseMoveEvent(QMouseEvent(QEvent.Type.MouseMove, point, canvas.mapToGlobal(point),
                                          Qt.MouseButton.NoButton, Qt.MouseButton.NoButton,
                                          Qt.KeyboardModifier.NoModifier))
    assert repaints and all(len(area) == 1 and isinstance(area[0], QRect) for area in repaints)
    whole = canvas.width() * canvas.height()
    assert all(area[0].width() * area[0].height() < whole / 20 for area in repaints)
