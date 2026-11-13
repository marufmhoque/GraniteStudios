"""Fast previews: correctness on synthetic files, and behaviour and timing
on real camera files when they are present."""

from __future__ import annotations

import io
from datetime import datetime

import cv2
import numpy as np
import pytest
from PIL import Image

from src.ai_tools import blur_analysis
from src.ai_tools.culling import SmartCull
from src.engine import fast_preview as fp
from tests.conftest import make_scene
from tests.fetch_media import SAMPLES


def _big_sharp_jpeg(path, size=(4000, 3000), exif=None):
    image = Image.fromarray(cv2.cvtColor(make_scene(7, size), cv2.COLOR_BGR2RGB))
    image.save(path, "JPEG", quality=93, **({"exif": exif} if exif is not None else {}))
    return path


def _jpeg_bytes(size, seed=0) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(cv2.cvtColor(make_scene(seed, size), cv2.COLOR_BGR2RGB)).save(buffer, "JPEG")
    return buffer.getvalue()


# --------------------------------------------------------------------- #
# Synthetic: always run
# --------------------------------------------------------------------- #

def test_preview_respects_max_side(tmp_path):
    preview = fp.load_preview(_big_sharp_jpeg(tmp_path / "a.jpg"), max_side=800)
    assert preview.source == fp.DECODED
    assert max(preview.size) == 800
    assert preview.native_size == (4000, 3000)
    assert preview.image.dtype == np.uint8 and preview.image.shape[2] == 3


def test_native_size_when_no_limit(tmp_path):
    preview = fp.load_preview(_big_sharp_jpeg(tmp_path / "a.jpg"), max_side=None, exact=True)
    assert preview.size == (4000, 3000)


def test_orientation_tag_is_applied(tmp_path):
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 clockwise to view
    preview = fp.load_preview(_big_sharp_jpeg(tmp_path / "a.jpg", exif=exif), max_side=600)
    w, h = preview.size
    assert h > w, "a frame tagged as portrait came back landscape"
    assert preview.native_size == (3000, 4000)


def test_capture_time_is_read_from_exif(tmp_path):
    exif = Image.Exif()
    exif[0x0132] = "2024:03:09 14:05:33"
    preview = fp.load_preview(_big_sharp_jpeg(tmp_path / "a.jpg", exif=exif))
    assert preview.captured_at == datetime(2024, 3, 9, 14, 5, 33)


def test_missing_and_corrupt_files_return_none(tmp_path):
    assert fp.load_preview(tmp_path / "nope.jpg") is None
    junk = tmp_path / "junk.jpg"
    junk.write_bytes(b"not an image at all")
    assert fp.load_preview(junk) is None
    fake_raw = tmp_path / "junk.cr2"
    fake_raw.write_bytes(b"\x00" * 4096)
    assert fp.load_preview(fake_raw) is None


def test_exact_decode_matches_the_calibrated_chain(tmp_path):
    """Blur thresholds were tuned on: decode in full, shrink once."""
    path = _big_sharp_jpeg(tmp_path / "a.jpg")
    reference = blur_analysis.sharpness_score(cv2.imread(str(path)))
    via_preview = blur_analysis.analyze(path).sharpness
    assert via_preview == pytest.approx(reference, rel=0.01)


def test_reduced_decode_would_understate_sharpness(tmp_path):
    """The reason the exact mode exists. If this ever stops holding the two
    modes can be merged."""
    path = _big_sharp_jpeg(tmp_path / "a.jpg")
    exact = fp.load_preview(path, max_side=None, exact=True)
    reduced = fp.load_preview(path, max_side=1600)
    to_bgr = lambda p: cv2.cvtColor(p.image, cv2.COLOR_RGB2BGR)  # noqa: E731
    assert blur_analysis.sharpness_score(to_bgr(reduced)) < 0.9 * blur_analysis.sharpness_score(to_bgr(exact))


def test_draft_request_keeps_the_aspect_ratio():
    assert fp._draft_size((6720, 4480), 1600) == (1600, 1066)
    assert fp._draft_size((3000, 4000), 1000) == (750, 1000)


def test_fit_never_enlarges():
    small = np.zeros((100, 200, 3), np.uint8)
    assert fp.fit(small, 1600) is small
    assert fp.fit(np.zeros((1000, 2000, 3), np.uint8), 500).shape[:2] == (250, 500)


@pytest.mark.parametrize("flip,expect_portrait", [(0, False), (3, False), (5, True), (6, True)])
def test_raw_flip_turns_a_landscape_preview(flip, expect_portrait):
    preview = Image.new("RGB", (300, 200))
    turned = fp._apply_raw_flip(preview, flip, raw_landscape=True)
    assert (turned.size[1] > turned.size[0]) == expect_portrait


def test_raw_flip_leaves_an_already_turned_preview_alone():
    preview = Image.new("RGB", (200, 300))
    assert fp._apply_raw_flip(preview, 6, raw_landscape=True).size == (200, 300)


def test_scan_finds_the_largest_jpeg_and_ignores_decoys(tmp_path):
    rng = np.random.default_rng(3)
    small, large = _jpeg_bytes((160, 120), 1), _jpeg_bytes((1200, 800), 2)
    decoy = b"\xff\xd8\xff" + b"\x12\x34\x56" * 10          # looks like a start, is not
    lossless = b"\xff\xd8\xff\xc3\x00\x0b\x08\x10\x00\x20\x00\x02" + b"\x00" * 16
    container = (
        rng.bytes(2048) + decoy + small + rng.bytes(999) + lossless
        + large + rng.bytes(4096)
    )
    path = tmp_path / "container.bin"
    path.write_bytes(container)
    found = fp.scan_for_jpeg(path)
    assert found is not None
    assert Image.open(io.BytesIO(found)).size == (1200, 800)
    assert fp.scan_for_jpeg(path, min_side=5000) is None


def test_scan_of_a_file_with_no_jpeg(tmp_path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"\x00" * 5000)
    assert fp.scan_for_jpeg(path) is None


def test_map_previews_keeps_order_and_reports_progress(library):
    paths = sorted(library.iterdir())
    seen: list[tuple[int, int]] = []
    result = fp.map_previews(
        paths, lambda key, preview: preview.size if preview else None,
        max_side=200, progress=lambda done, total: seen.append((done, total)),
    )
    assert list(result) == [str(p) for p in paths]
    assert all(size and max(size) <= 200 for size in result.values())
    assert seen[-1] == (len(paths), len(paths))


def test_score_frames_measures_everything_in_one_pass(library):
    paths = sorted(p for p in library.iterdir() if p.is_file())
    frames = SmartCull().score_frames(paths)
    assert [f.path for f in frames] == paths
    assert all(f.sharpness > 0 and f.phash is not None and f.timestamp for f in frames)
    by_name = {f.path.name: f for f in frames}
    assert by_name["IMG_0004.jpg"].sharpness < by_name["IMG_0003.jpg"].sharpness
    assert SmartCull.hamming_distance(by_name["IMG_0001.jpg"].phash, by_name["IMG_0002.jpg"].phash) <= 6


def test_eye_check_with_a_supplied_analyzer_runs_under_threads(library):
    from src.ai_tools.face_analysis import Face, FaceReport

    class Stub:
        available = True
        calls = 0

        def analyze(self, bgr):
            Stub.calls += 1
            return FaceReport(faces=[Face(0, 0, 10, 10, 0.9, (2.0, 3.0), (7.0, 3.0))])

    paths = sorted(p for p in library.iterdir() if p.is_file())
    frames = SmartCull().score_frames(paths, check_eyes=True, analyzer=Stub())
    assert Stub.calls == len(paths)
    assert all(f.eyes_checked and f.face_count == 1 for f in frames)


# --------------------------------------------------------------------- #
# Real camera files
# --------------------------------------------------------------------- #

_media = pytest.mark.parametrize(
    "sample", SAMPLES, ids=[s.name for s in SAMPLES]
)


def _need(sample):
    if not sample.present:
        pytest.skip("sample not downloaded; run tests/fetch_media.py")


@pytest.mark.media
@_media
def test_every_format_yields_a_usable_preview(sample):
    _need(sample)
    preview = fp.load_preview(sample.path, max_side=1600)
    assert preview is not None, "no preview at all"
    assert max(preview.size) >= 900, f"preview too small to cull on: {preview.size}"
    assert max(preview.size) <= 1600
    assert preview.image.std() > 5, "preview is blank"
    assert preview.source in (fp.EMBEDDED_JPEG, fp.SCANNED_JPEG, fp.DEMOSAIC_HALF)


@pytest.mark.media
@pytest.mark.benchmark
@_media
def test_locating_the_embedded_preview_is_fast(sample):
    """The spec's figure is about *finding* the preview, and holds. Decoding
    it is separate and slower; tests/bench_cull.py measures the whole thing."""
    _need(sample)
    fp.load_preview(sample.path, max_side=400, min_side=0)  # warm the file cache
    best = min(fp.load_preview(sample.path, max_side=400, min_side=0).extract_ms for _ in range(5))
    assert best < 25.0, f"extraction took {best:.1f} ms"


@pytest.mark.media
@_media
def test_any_size_preview_is_available_for_the_grid(sample):
    _need(sample)
    preview = fp.load_preview(sample.path, max_side=320, min_side=0)
    assert preview is not None and max(preview.size) <= 320


@pytest.mark.media
@_media
def test_embedded_previews_carry_the_capture_time(sample):
    _need(sample)
    preview = fp.load_preview(sample.path, max_side=400)
    if preview.source == fp.DEMOSAIC_HALF and preview.captured_at is None:
        pytest.skip("this file carries no EXIF in its embedded thumbnail")
    assert preview.captured_at is not None and preview.captured_at.year >= 2000


@pytest.mark.media
@_media
def test_blur_triage_reads_raw_without_developing_it(sample):
    _need(sample)
    report = blur_analysis.analyze(sample.path)
    assert report.error is None and report.sharpness > 0
