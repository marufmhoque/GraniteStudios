"""Profile Granite Studios' interactive speed on this machine.

    python tests/profile_performance.py            # everything
    python tests/profile_performance.py thumbs develop

Not a test (pytest does not collect it); ``tests/test_performance.py`` holds
the assertions. It prints timings for:

    thumbs    loading 100 RAW thumbnails: one at a time, and the grid's way
              (a thread pool, 20 at a time). Uses the downloaded RAW samples
              (tests/fetch_media.py) repeated to 100, or synthetic JPEGs
    develop   a Develop slider's round trip: the full 1,600 px preview and
              the drafts shown while dragging, for point edits (exposure,
              contrast, curves) and for local and colour ones
    brush     the Color Splash brush: one dab, and the frames per second a
              stroke can keep up
    ai        DirectML (or another accelerator) against the CPU for the
              cached ONNX models

Originals are only read. Synthetic files go to a temporary folder.
"""

from __future__ import annotations

import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from tests.conftest import make_scene, write_jpeg  # noqa: E402


def timed(fn, repeat: int = 5, warmup: int = 1) -> tuple[float, float]:
    """Median and best of ``repeat`` runs, in milliseconds."""
    for _ in range(warmup):
        fn()
    runs = []
    for _ in range(repeat):
        started = time.perf_counter()
        fn()
        runs.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(runs), min(runs)


def row(label: str, ms: float, extra: str = "") -> None:
    print(f"  {label:<46} {ms:9.1f} ms  {extra}")


# --------------------------------------------------------------------- #

def raw_samples() -> list[Path]:
    from tests.fetch_media import RAW_DIR

    return sorted(p for p in RAW_DIR.glob("*") if p.is_file()) if RAW_DIR.is_dir() else []


def profile_thumbs() -> None:
    from src.engine.universal_loader import UniversalLoader
    from src.ui.theme import THUMB_SIZE

    print("\n1. Loading 100 thumbnails")
    samples = raw_samples()
    if samples:
        paths = (samples * (100 // len(samples) + 1))[:100]
        kind = f"RAW ({len(samples)} cameras, repeated)"
    else:
        folder = Path(tempfile.mkdtemp())
        paths = [write_jpeg(folder / f"F{i:03d}.jpg", make_scene(i, (3000, 2000))) for i in range(100)]
        kind = "synthetic 6 MP JPEG (no RAW samples downloaded)"
    print(f"  files: {kind}")
    loader = UniversalLoader()
    size = THUMB_SIZE * 2
    for path in paths[:10]:
        loader.load_thumbnail(path, size)          # the files into the OS cache
    serial, _ = timed(lambda: [loader.load_thumbnail(p, size) for p in paths], repeat=3)
    pooled, _ = timed(lambda: list(loader.iter_thumbnails(paths, size)), repeat=3)
    with_meta, _ = timed(lambda: list(loader.iter_thumbnails(paths, size, metadata=True)), repeat=3)
    row("one at a time", serial, f"{serial / len(paths):.1f} ms each")
    row("thread pool, chunks of 20 (the grid)", pooled, f"{serial / max(pooled, 1e-6):.1f}x faster")
    row("  ... with EXIF for Tech Info", with_meta)
    started = time.perf_counter()
    first = next(iter(loader.iter_thumbnails(paths, size)))
    row("first tile on screen", (time.perf_counter() - started) * 1000.0, Path(first[0]).name)


def profile_develop() -> None:
    from src.engine import develop_pipeline as dp
    from src.engine import parametric_recipe as pr
    from src.ui.workspaces import develop

    print("\n2. Develop slider round trip (24 MP frame)")
    source = dp.from_array(make_scene(3, (6000, 4000)))
    preview = source.proxy(develop.PREVIEW_SIDE)

    point = pr.Recipe()
    point.basic.exposure, point.basic.contrast = 0.5, 20
    point.curve.master = [[0, 0], [0.25, 0.2], [0.75, 0.8], [1, 1]]
    local = point.copy()
    local.basic.highlights, local.basic.shadows, local.basic.clarity = -30, 25, 20
    colour = local.copy()
    colour.basic.vibrance, colour.basic.saturation = 20, 10
    colour.hsl.saturation[2] = 30

    for name, recipe in (("exposure, contrast, curve", point), ("+ highlights, shadows, clarity", local),
                         ("+ vibrance, saturation, HSL", colour)):
        print(f"  {name}:")
        full, _ = timed(lambda: dp.render(preview, recipe))
        row(f"  full preview, {develop.PREVIEW_SIDE} px", full, f"{1000 / full:.0f} fps")
        for side in (develop.DRAFT_MAX_SIDE, 800, develop.DRAFT_MIN_SIDE):
            draft = preview.proxy(side)
            ms, _ = timed(lambda: dp.render(draft, recipe))
            row(f"  draft while dragging, {side} px", ms, f"{1000 / ms:.0f} fps")


def profile_brush() -> None:
    from src.ai_tools.smart_brush import SmartBrushEngine

    print("\n3. Color Splash brush")
    image = make_scene(5, (1600, 1067))
    engine = SmartBrushEngine()
    engine.prepare(image)
    mask = np.zeros(image.shape[:2], np.uint8)
    for radius in (40, 120):
        centres = [(200 + 9 * i, 500) for i in range(60)]
        index = iter(range(10 ** 9))

        def dab():
            x, y = centres[next(index) % len(centres)]
            engine.paint_stroke(image, mask, (x, y), radius, 0.6, False, True)

        ms, best = timed(dab, repeat=40, warmup=3)
        row(f"one dab, radius {radius} px (edge snapping)", ms, f"{1000 / ms:.0f} dabs/s, best {best:.1f} ms")
        ms, _ = timed(lambda: engine.paint_stroke(image, mask, (800, 500), radius, 0.6, False, False), repeat=40)
        row(f"one dab, radius {radius} px (snapping off)", ms, f"{1000 / ms:.0f} dabs/s")
    print("  The canvas repaints only the brushed rectangle (ImageCanvas.update_region).")


def profile_ai() -> None:
    import onnxruntime as ort

    ort.set_default_logger_severity(3)      # quiet: some models warn about their initialisers

    from src.ai_tools import model_store
    from src.engine import hardware

    print("\n4. AI inference: accelerator against CPU")
    available = ort.get_available_providers()
    accelerated = [p for p in hardware.get_execution_providers() if p != "CPUExecutionProvider"]
    print(f"  providers: {', '.join(available)}")
    if not accelerated:
        print("  no accelerator in this build of ONNX Runtime; CPU only")

    from src.ai_tools import face_analysis, person_clustering, segmentation

    # Background removal (U2Net, through rembg) is the one model the app runs
    # on ONNX Runtime, so the one DirectML serves. The others run on the CPU
    # through OpenCV's engine; they are timed here on ONNX Runtime only to
    # show whether an accelerator would help them.
    u2net = Path(os.environ.get("U2NET_HOME", Path.home() / ".u2net")) / "u2net.onnx"
    models = [("U2Net background removal (the app: DirectML)", u2net if u2net.is_file() else None)]
    for label, spec in (("YuNet face detection (the app: CPU)", face_analysis.MODEL_SPEC),
                        ("SFace recognition (the app: CPU)", person_clustering.SFACE_SPEC),
                        ("PP-HumanSeg (the app: CPU)", segmentation.HUMANSEG_SPEC)):
        models.append((label, model_store.ensure_model(spec, download=False)
                       if model_store.is_cached(spec) else None))
    for label, path in models:
        if path is None:
            print(f"  {label}: not downloaded, skipped")
            continue
        results = []
        for providers in ([*accelerated, "CPUExecutionProvider"], ["CPUExecutionProvider"]):
            if providers == ["CPUExecutionProvider"] or accelerated:
                try:
                    session = hardware.create_session(path, providers)
                    feeds = _feeds(session)
                    ms, _ = timed(lambda: session.run(None, feeds), repeat=10, warmup=2)
                    results.append((session.get_providers()[0], ms))
                except Exception as exc:  # a model a provider cannot run
                    results.append((providers[0], float("nan")))
                    print(f"  {label} on {providers[0]}: {exc}")
        print(f"  {label}:")
        for provider, ms in results:
            row(f"  {provider.replace('ExecutionProvider', '')}", ms)


def _feeds(session) -> dict:
    feeds = {}
    for item in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else (1 if i == 0 else 320) for i, d in enumerate(item.shape)]
        dtype = np.float16 if "float16" in item.type else (np.int64 if "int64" in item.type else np.float32)
        feeds[item.name] = np.random.default_rng(0).random(shape).astype(dtype)
    return feeds


SECTIONS = {"thumbs": profile_thumbs, "develop": profile_develop, "brush": profile_brush, "ai": profile_ai}


def main(argv: list[str]) -> None:
    from src import __version__

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    print(f"Granite Studios {__version__} performance profile, {os.cpu_count()} logical CPUs")
    for name in argv or list(SECTIONS):
        SECTIONS[name]()


if __name__ == "__main__":
    main(sys.argv[1:])
