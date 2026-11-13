"""INV-1: originals are never modified, moved or deleted.

Every action that reads the library is driven through the real window and
followed by a byte-for-byte comparison of the source folder. A second group
checks the other half of the rule: a rejected frame is never exported.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import assert_originals_untouched, snapshot

U2NET = Path.home() / ".u2net" / "u2net.onnx"


@pytest.fixture
def before(library):
    return snapshot(library)


def test_the_check_itself_catches_a_modified_original(library, before):
    target = library / "IMG_0001.jpg"
    target.write_bytes(target.read_bytes() + b"\0")
    with pytest.raises(AssertionError, match="modified"):
        assert_originals_untouched(before, library)


def test_the_check_itself_catches_a_removed_original(library, before):
    (library / "IMG_0001.jpg").unlink()
    with pytest.raises(AssertionError, match="removed"):
        assert_originals_untouched(before, library)


def test_library_fixture_is_what_we_think(harness, before):
    assert len(harness.window.image_paths) == 7
    assert harness.window.edit_folder.is_dir()
    assert_originals_untouched(before, harness.folder)


def test_smart_cull(harness, before):
    harness.run(harness.window._on_run_cull)
    assert_originals_untouched(before, harness.folder)


def test_cull_to_target_then_reject_non_picks(harness, before):
    w = harness.window
    w.cull_target_input.setValue(3)
    w.cull_eyes_check.setChecked(False)
    harness.run(w._on_cull_to_target)
    assert w.last_cull_frames, "the cull produced no frames, so the rest proves nothing"
    w._on_reject_nonpicks()
    assert w.rejected, "reject non-picks rejected nothing"
    assert_originals_untouched(before, harness.folder)


def test_reject_and_restore(harness, before):
    w = harness.window
    harness.select("IMG_0001.jpg", "IMG_0003.jpg")
    w._on_reject_selected()
    assert len(w.rejected) == 2
    w._on_restore_selected()
    assert not w.rejected
    assert_originals_untouched(before, harness.folder)


def test_find_blurry_and_sharpen(harness, before):
    w = harness.window
    harness.run(w._on_find_blurry)
    assert w.blur_reports, "blur scan returned nothing"
    assert any(r.is_blurry for r in w.blur_reports.values()), (
        "the deliberately blurred frame was not flagged"
    )
    harness.run(w._on_sharpen_recoverable)
    assert_originals_untouched(before, harness.folder)


@pytest.mark.parametrize("mode_index", [0, 1])
def test_grade(harness, before, mode_index):
    w = harness.window
    harness.select("IMG_0001.jpg", "IMG_0003.jpg")
    w.color_mode.setCurrentIndex(mode_index)
    harness.run(w._on_apply_color)
    assert {"IMG_0001_graded.jpg", "IMG_0003_graded.jpg"} <= harness.edit_files()
    assert_originals_untouched(before, harness.folder)


def test_auto_level(harness, before):
    w = harness.window
    w.level_body_check.setChecked(False)
    harness.select("IMG_0005.jpg")
    harness.run(w._on_auto_level)
    assert "IMG_0005_level.jpg" in harness.edit_files(), (
        "the tilted facade was not straightened, so nothing was exercised"
    )
    assert_originals_untouched(before, harness.folder)


def test_write_xmp_adds_sidecars_without_touching_images(harness, before):
    w = harness.window
    w.cull_target_input.setValue(3)
    w.cull_eyes_check.setChecked(False)
    harness.run(w._on_cull_to_target)
    assert w.last_cull_frames, "no cull result to write"
    harness.run(w._on_write_xmp)
    sidecars = [p for p in harness.folder.iterdir() if p.suffix == ".xmp"]
    assert sidecars, "no sidecar was written"
    assert_originals_untouched(before, harness.folder)


@pytest.mark.slow
@pytest.mark.model
def test_background_removal(harness, before):
    if not U2NET.exists():
        pytest.skip("U2Net is not cached; not downloading 176 MB for a test")
    harness.select("IMG_0001.jpg")
    harness.run(harness.window._on_remove_background)
    assert "IMG_0001_nobg.png" in harness.edit_files()
    assert_originals_untouched(before, harness.folder)


def test_instagram_export(harness, before):
    harness.select("IMG_0001.jpg", "IMG_0003.jpg")
    harness.run(harness.window._on_export_ig)
    assert len(list(harness.out_dir.iterdir())) == 2
    assert_originals_untouched(before, harness.folder)


@pytest.mark.slow
def test_reel(harness, before):
    harness.select("IMG_0001.jpg", "IMG_0003.jpg", "IMG_0006.jpg")
    harness.run(harness.window._on_create_reel)
    assert (harness.out_dir / "reel.mp4").stat().st_size > 0
    assert_originals_untouched(before, harness.folder)


def test_gallery(harness, before):
    w = harness.window
    w.web_selected = {str(p) for p in w.image_paths if p.name in {"IMG_0001.jpg", "IMG_0007.png"}}
    harness.run(w._on_generate_gallery)
    assert (harness.out_dir / "index.html").exists()
    assert_originals_untouched(before, harness.folder)


# --------------------------------------------------------------------- #
# A rejected frame never reaches an export
# --------------------------------------------------------------------- #

def test_rejected_frame_is_not_graded(harness, before):
    harness.reject("IMG_0003.jpg")
    harness.select("IMG_0001.jpg", "IMG_0003.jpg")
    harness.run(harness.window._on_apply_color)
    files = harness.edit_files()
    assert "IMG_0001_graded.jpg" in files
    assert "IMG_0003_graded.jpg" not in files
    assert (harness.folder / "IMG_0003.jpg").exists()
    assert_originals_untouched(before, harness.folder)


def test_rejected_frame_is_not_levelled(harness, before):
    w = harness.window
    w.level_body_check.setChecked(False)
    harness.reject("IMG_0005.jpg")
    harness.select("IMG_0005.jpg")
    harness.run(w._on_auto_level)
    assert "IMG_0005_level.jpg" not in harness.edit_files()
    assert_originals_untouched(before, harness.folder)


def test_rejected_frame_is_not_exported_to_instagram(harness, before):
    harness.reject("IMG_0003.jpg")
    harness.select("IMG_0001.jpg", "IMG_0003.jpg")
    harness.run(harness.window._on_export_ig)
    names = [p.name for p in harness.out_dir.iterdir()]
    assert len(names) == 1 and "IMG_0001" in names[0]
    assert_originals_untouched(before, harness.folder)


def test_rejected_frame_is_not_published(harness, before):
    w = harness.window
    w.web_selected = {str(p) for p in w.image_paths if p.name in {"IMG_0001.jpg", "IMG_0003.jpg"}}
    harness.reject("IMG_0003.jpg")
    harness.run(w._on_generate_gallery)
    published = " ".join(p.name for p in harness.out_dir.rglob("*") if p.is_file())
    assert "IMG_0001" in published
    assert "IMG_0003" not in published
    assert_originals_untouched(before, harness.folder)


@pytest.mark.slow
def test_reel_refuses_when_rejection_leaves_too_few(harness, before):
    harness.reject("IMG_0006.jpg")
    harness.select("IMG_0001.jpg", "IMG_0003.jpg", "IMG_0006.jpg")
    harness.run(harness.window._on_create_reel)
    assert not (harness.out_dir / "reel.mp4").exists()
    assert_originals_untouched(before, harness.folder)


# --------------------------------------------------------------------- #
# Video follows the same rule
# --------------------------------------------------------------------- #

@pytest.mark.slow
def test_video_operation_leaves_the_source_alone(tmp_path):
    from src.video import timing
    from src.video.ffmpeg_utils import run_ffmpeg

    source = tmp_path / "clip.mp4"
    run_ffmpeg([
        "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=15",
        "-pix_fmt", "yuv420p", str(source),
    ])
    before = snapshot(tmp_path)
    out = tmp_path / "clip-edit"
    out.mkdir()
    result = timing.trim(source, out / "clip_trim.mp4", 0.2, 1.2)
    assert Path(result).stat().st_size > 0
    assert_originals_untouched(before, tmp_path)
