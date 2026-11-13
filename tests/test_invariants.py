"""INV-1 and INV-7: originals are never modified, moved or deleted, and
editing writes settings, not images, until something is exported.

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
    # A grade is now a recipe setting: a sidecar, and no rendered image.
    assert {"IMG_0001.jpg.granite.json", "IMG_0003.jpg.granite.json"} <= harness.edit_files()
    assert not any(name.endswith(".jpg") for name in harness.edit_files())
    graded = w.session.recipe(harness.folder / "IMG_0001.jpg")
    assert graded.look.matrix is not None or min(graded.hsl.saturation) == -100.0
    assert_originals_untouched(before, harness.folder)


def test_auto_level(harness, before):
    w = harness.window
    w.level_body_check.setChecked(False)
    harness.select("IMG_0005.jpg")
    harness.run(w._on_auto_level)
    recipe = w.session.recipe(harness.folder / "IMG_0005.jpg")
    assert abs(recipe.geometry.rotation) > 3.0, (
        "the tilted facade was not straightened, so nothing was exercised"
    )
    assert harness.edit_files() == {"IMG_0005.jpg.granite.json"}, "levelling rendered an image"
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
    assert "IMG_0001.jpg.granite.json" in files
    assert "IMG_0003.jpg.granite.json" not in files
    assert (harness.folder / "IMG_0003.jpg").exists()
    assert_originals_untouched(before, harness.folder)


def test_rejected_frame_is_not_levelled(harness, before):
    w = harness.window
    w.level_body_check.setChecked(False)
    harness.reject("IMG_0005.jpg")
    harness.select("IMG_0005.jpg")
    harness.run(w._on_auto_level)
    assert w.session.recipe(harness.folder / "IMG_0005.jpg").is_neutral
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


# --------------------------------------------------------------------- #
# Export renders the edits; still nothing touches the originals
# --------------------------------------------------------------------- #

def _export(harness, preset="full"):
    w = harness.window
    combo = w.export.preset_combo
    combo.setCurrentIndex(combo.findData(preset))
    harness.run(w.export._on_export_files)
    return sorted(harness.out_dir.iterdir())


def test_export_renders_recipes_and_leaves_out_rejects(harness, before):
    import cv2

    w = harness.window
    harness.select("IMG_0001.jpg", "IMG_0003.jpg", "IMG_0006.jpg")
    w.color_mode.setCurrentIndex(1)                     # selective colour
    harness.run(w._on_apply_color)
    harness.reject("IMG_0006.jpg")
    written = _export(harness)
    assert [p.name for p in written] == ["IMG_0001.jpg", "IMG_0003.jpg"]
    exported = cv2.imread(str(written[0]))
    original = cv2.imread(str(harness.folder / "IMG_0001.jpg"))
    spread = lambda img: float((img.max(axis=2).astype(int) - img.min(axis=2)).mean())  # noqa: E731
    assert spread(exported) < spread(original) * 0.8, "the export does not show the edit"
    assert_originals_untouched(before, harness.folder)


def test_export_of_a_straightened_frame(harness, before):
    import cv2

    w = harness.window
    w.level_body_check.setChecked(False)
    harness.select("IMG_0005.jpg")
    harness.run(w._on_auto_level)
    written = _export(harness, "web")
    image = cv2.imread(str(written[0]))
    assert image.shape[1] < 800, "the straightened export was not cropped to clean corners"
    assert_originals_untouched(before, harness.folder)


def test_edits_and_marks_survive_reopening_the_folder(harness, before):
    w = harness.window
    harness.select("IMG_0001.jpg")
    w.color_mode.setCurrentIndex(0)
    harness.run(w._on_apply_color)
    w.session.set_rating([str(harness.folder / "IMG_0003.jpg")], 4)
    harness.reject("IMG_0006.jpg")
    w.session.save_all()
    w.cull.open_folder(harness.folder)
    harness.wait()
    key = str(harness.folder / "IMG_0001.jpg")
    assert w.session.has_edits(key) and w.session.recipe(key).look.matrix is not None
    assert w.session.ratings[str(harness.folder / "IMG_0003.jpg")] == 4
    assert str(harness.folder / "IMG_0006.jpg") in w.rejected
    assert_originals_untouched(before, harness.folder)
