"""The workspaces, driven headlessly the way a person would use them."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from src.engine import parametric_recipe as pr
from src.engine.session import STATE_FILENAME
from tests.conftest import assert_originals_untouched, snapshot, write_jpeg
from tests.fetch_media import face


def _settle(harness, workspace) -> None:
    """Wait for a workspace's background decode and preview render."""
    for job in ("loader_job", "renderer"):
        runner = getattr(workspace, job, None)
        if runner is not None:
            runner.wait_idle(harness.app)
    for job in ("loader_job", "renderer"):
        runner = getattr(workspace, job, None)
        if runner is not None:
            runner.wait_idle(harness.app)


def _key(harness, name: str) -> str:
    return str(harness.folder / name)


@pytest.fixture
def before(library):
    return snapshot(library)


# --------------------------------------------------------------------- #
# The bar and the shared session
# --------------------------------------------------------------------- #

def test_six_workspaces_share_one_selection(harness):
    w = harness.window
    assert [ws.title for ws in w.workspaces] == ["Cull", "Develop", "Retouch", "Design", "Export", "Video"]
    harness.select("IMG_0001.jpg", "IMG_0003.jpg")
    harness.reject("IMG_0003.jpg")
    w.switch_to(w.export)
    assert w.current_workspace is w.export and w.nav_buttons[4].isChecked()
    assert "1 selected frame(s) will be exported" in w.export.files_count.text()
    assert "1 rejected" in w.export.files_count.text()
    w.switch_to(0)
    assert w.cull.cells[_key(harness, "IMG_0003.jpg")].rejected


def test_keyboard_rating_labels_picks_and_rejects(harness):
    w = harness.window
    cull = w.cull
    w.session.set_current(_key(harness, "IMG_0001.jpg"))
    QTest.keyClick(cull, Qt.Key.Key_4)
    QTest.keyClick(cull, Qt.Key.Key_6)
    QTest.keyClick(cull, Qt.Key.Key_P)
    first = _key(harness, "IMG_0001.jpg")
    assert w.session.ratings[first] == 4
    assert w.session.color_labels[first] == "Red"
    assert first in w.session.picks
    cell = cull.cells[first]
    assert cell.rating == 4 and cell.colour_label == "Red" and cell.picked

    QTest.keyClick(cull, Qt.Key.Key_Right)
    second = _key(harness, "IMG_0002.jpg")
    assert w.session.current == second
    QTest.keyClick(cull, Qt.Key.Key_X)
    assert second in w.rejected and cull.cells[second].rejected
    QTest.keyClick(cull, Qt.Key.Key_U)
    assert second not in w.rejected

    QTest.keyClick(cull, Qt.Key.Key_6)                  # pressing a label again clears it
    assert second in w.session.color_labels
    QTest.keyClick(cull, Qt.Key.Key_6)
    assert second not in w.session.color_labels
    QTest.keyClick(cull, Qt.Key.Key_0)
    assert second not in w.session.ratings

    w.session.save_state()
    state = json.loads((w.edit_folder / STATE_FILENAME).read_text(encoding="utf-8"))
    assert state["ratings"] == {"IMG_0001.jpg": 4} and state["picks"] == ["IMG_0001.jpg"]


def test_marks_apply_to_the_whole_selection(harness):
    w = harness.window
    harness.select("IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg")
    QTest.keyClick(w.cull, Qt.Key.Key_5)
    assert {k for k, v in w.session.ratings.items() if v == 5} == {
        _key(harness, n) for n in ("IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg")}


def test_a_damaged_state_file_opens_unmarked(harness):
    w = harness.window
    (w.edit_folder / STATE_FILENAME).write_bytes(b"\xff\x00 not json")
    w.cull.open_folder(harness.folder)
    harness.wait()
    assert not w.rejected and not w.session.ratings


# --------------------------------------------------------------------- #
# Survey and face loupe
# --------------------------------------------------------------------- #

def test_survey_locks_zoom_and_pan_together(harness):
    w = harness.window
    harness.select("IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg")
    QTest.keyClick(w.cull, Qt.Key.Key_S)
    assert w.cull.view_stack.currentIndex() == 1
    canvases = w.cull.survey_canvases
    assert len(canvases) == 3 and all(c.has_image for c in canvases)
    canvases[0].set_zoom(2.0, (0.3, 0.6))
    for other in canvases[1:]:
        zoom, cx, cy = other.view()
        assert (zoom, cx, cy) == pytest.approx((2.0, 0.3, 0.6))
    QTest.keyClick(w.cull, Qt.Key.Key_Escape)
    assert w.cull.view_stack.currentIndex() == 0


def test_survey_needs_two_frames(harness):
    harness.select("IMG_0001.jpg")
    harness.window.cull._toggle_survey()
    assert harness.window.cull.view_stack.currentIndex() == 0
    assert any("Survey" in title for _kind, title in harness.messages)


@pytest.mark.model
@pytest.mark.media
def test_face_loupe_shows_every_face_at_full_size(harness, tmp_path):
    from src.ai_tools.face_analysis import FaceAnalyzer

    if not (face("pose.png").present and FaceAnalyzer(download=False).available):
        pytest.skip("reference face or face model not present")
    w = harness.window
    folder = tmp_path / "Portraits"
    folder.mkdir()
    portrait = cv2.resize(cv2.imread(str(face("pose.png").path)), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    write_jpeg(folder / "a.jpg", portrait)
    write_jpeg(folder / "b.jpg", cv2.flip(portrait, 1))
    w.cull.open_folder(folder)
    harness.wait()
    w.selected = {str(folder / "a.jpg"), str(folder / "b.jpg")}
    harness.run(w.cull._on_face_loupe)
    tiles = [w.cull.loupe_layout.itemAt(i).widget() for i in range(w.cull.loupe_layout.count())]
    tiles = [t for t in tiles if t is not None]
    assert len(tiles) == 2 and w.cull.loupe_scroll.isVisibleTo(w.cull)
    assert "100%" in tiles[0].toolTip()


# --------------------------------------------------------------------- #
# Develop
# --------------------------------------------------------------------- #

@pytest.fixture
def develop(harness):
    w = harness.window
    w.open_in_develop(_key(harness, "IMG_0003.jpg"))
    _settle(harness, w.develop)
    assert w.current_workspace is w.develop and w.develop.canvas.has_image
    return w.develop


def test_double_clicking_a_thumbnail_opens_it_in_develop(harness):
    w = harness.window
    cell = w.cull.cells[_key(harness, "IMG_0006.jpg")]
    QTest.mouseDClick(cell, Qt.MouseButton.LeftButton)
    _settle(harness, w.develop)
    assert w.current_workspace is w.develop and w.develop.key == _key(harness, "IMG_0006.jpg")


def test_a_slider_changes_the_preview_and_nothing_is_written_until_saved(harness, develop, before):
    w = harness.window
    first = develop._last_render.astype(float).mean()
    row = develop.sliders[("basic", "exposure")]
    row.set_value(1.0, announce=True)
    _settle(harness, develop)
    assert develop._last_render.astype(float).mean() > first + 10, "the preview did not brighten"
    assert not w.session.has_edits(develop.key), "the edit was committed before the slider was released"
    row.released.emit()
    assert w.session.has_edits(develop.key)
    sidecar = pr.sidecar_path(develop.key, w.edit_folder)
    assert not sidecar.exists(), "a sidecar was written on every change"
    w.switch_to(w.cull)                                 # leaving saves
    assert sidecar.exists() and pr.load(develop.key, w.edit_folder).basic.exposure == 1.0
    assert_originals_untouched(before, harness.folder)


def test_undo_and_redo(harness, develop):
    row = develop.sliders[("basic", "contrast")]
    row.set_value(40, announce=True)
    row.released.emit()
    row.set_value(-20, announce=True)
    row.released.emit()
    develop.undo()
    assert develop.recipe.basic.contrast == 40 and row.value() == 40
    develop.undo()
    assert develop.recipe.basic.contrast == 0
    develop.redo()
    assert develop.recipe.basic.contrast == 40


def test_before_after_and_clipping(harness, develop):
    QTest.keyClick(develop, Qt.Key.Key_Backslash)
    _settle(harness, develop)
    assert develop.canvas.comparing and develop._before_key is not None
    develop.sliders[("basic", "exposure")].set_value(4.0, announce=True)
    _settle(harness, develop)
    QTest.keyClick(develop, Qt.Key.Key_J)
    assert develop._show_clipping and develop.histogram.highlights_clipped > 0.01
    assert "highlights" in develop.clip_readout.text()


def test_arrow_keys_move_through_frames_and_sync_copies_settings(harness, develop):
    w = harness.window
    develop.sliders[("basic", "vibrance")].set_value(30, announce=True)
    develop.sliders[("basic", "vibrance")].released.emit()
    develop.recipe.geometry.crop = [0.1, 0.1, 0.5, 0.5]
    develop._commit()
    harness.select("IMG_0004.jpg", "IMG_0006.jpg")
    develop._on_sync()
    for name in ("IMG_0004.jpg", "IMG_0006.jpg"):
        synced = w.session.recipe(_key(harness, name))
        assert synced.basic.vibrance == 30 and synced.geometry.crop is None, "a crop travelled with sync"
    QTest.keyClick(develop, Qt.Key.Key_Right)
    _settle(harness, develop)
    assert develop.key == _key(harness, "IMG_0004.jpg")
    assert develop.sliders[("basic", "vibrance")].value() == 30


def test_aspect_crop_and_looks(harness, develop):
    develop.aspect_combo.setCurrentText("1:1")
    _settle(harness, develop)
    h, w = develop._last_render.shape[:2]
    assert abs(h - w) <= 2
    develop.look_combo.setCurrentIndex(develop.look_combo.findData("noir"))
    _settle(harness, develop)
    image = develop._last_render.astype(int)
    assert np.abs(image[:, :, 0] - image[:, :, 1]).max() <= 2, "the noir look is not monochrome"


# --------------------------------------------------------------------- #
# Retouch
# --------------------------------------------------------------------- #

def test_healing_is_stored_as_positions_and_rendered(harness, before):
    w = harness.window
    w.session.set_current(_key(harness, "IMG_0001.jpg"))
    w.switch_to(w.retouch)
    retouch = w.retouch
    _settle(harness, retouch)
    first = retouch.canvas._image.copy() if retouch.canvas._image is not None else None
    assert first is not None
    retouch.tool_group.button(0).setChecked(True)          # heal
    retouch._on_press(0.4, 0.4, Qt.MouseButton.LeftButton)
    _settle(harness, retouch)
    layers = w.session.recipe(_key(harness, "IMG_0001.jpg")).layers
    assert layers and layers[0]["op"] == "heal" and len(layers[0]["params"]["spots"]) == 1
    assert retouch.canvas._image != first, "the healed spot did not change the picture"
    retouch._delete()
    assert not w.session.recipe(_key(harness, "IMG_0001.jpg")).layers
    assert_originals_untouched(before, harness.folder)


def test_brushed_adjustment_layer(harness):
    w = harness.window
    w.session.set_current(_key(harness, "IMG_0003.jpg"))
    w.switch_to(w.retouch)
    retouch = w.retouch
    _settle(harness, retouch)
    retouch._add_brush_layer()
    retouch.adjust_rows["exposure"].set_value(1.5, announce=True)
    retouch._on_press(0.2, 0.5, Qt.MouseButton.LeftButton)
    retouch._on_drag(0.8, 0.5)
    retouch._on_release(0.8, 0.5)
    _settle(harness, retouch)
    layer = w.session.recipe(_key(harness, "IMG_0003.jpg")).layers[-1]
    assert layer["mask"]["kind"] == "brush" and len(layer["mask"]["params"]["strokes"]) == 1
    assert layer["recipe"]["basic"]["exposure"] == 1.5


# --------------------------------------------------------------------- #
# Design
# --------------------------------------------------------------------- #

def test_design_items_snap_and_are_stored_in_the_recipe(harness):
    w = harness.window
    w.session.set_current(_key(harness, "IMG_0001.jpg"))
    w.switch_to(w.design)
    design = w.design
    _settle(harness, design)
    design._add_item("rect")
    item = design.items[-1]
    assert (item.x, item.y) == (0.4, 0.4)
    design._on_press(0.5, 0.5, Qt.MouseButton.LeftButton)
    design._on_drag(0.5 + 0.093, 0.5 + 0.051)
    design._on_release(0.6, 0.55)
    assert item.x == pytest.approx(0.5) and item.y == pytest.approx(0.45), "the item did not snap to the 5% grid"
    stored = w.session.recipe(_key(harness, "IMG_0001.jpg")).vectors
    assert stored and stored[-1]["x"] == pytest.approx(0.5)


def test_design_templates_and_svg_export(harness, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    w = harness.window
    w.session.set_current(_key(harness, "IMG_0001.jpg"))
    w.switch_to(w.design)
    design = w.design
    _settle(harness, design)
    design.signature_name.setText("Maruf Hoque")
    design._add_template("signature")
    design._add_template("exif_strip")
    assert len(design.items) == 3
    target = harness.out_dir / "design.svg"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(target), "")))
    design._on_export_svg()
    assert target.exists() and "Maruf Hoque" in target.read_text(encoding="utf-8")


def test_design_overlays_appear_in_the_export(harness, before):
    w = harness.window
    w.session.set_current(_key(harness, "IMG_0004.jpg"))
    w.switch_to(w.design)
    _settle(harness, w.design)
    w.design._add_item("rect")
    item = w.design.items[-1]
    item.x, item.y, item.w, item.h, item.fill = 0.0, 0.0, 0.25, 0.25, "#ff0000ff"
    w.design._commit()
    w.switch_to(w.export)
    harness.select("IMG_0004.jpg")
    w.export.preset_combo.setCurrentIndex(w.export.preset_combo.findData("full"))
    harness.run(w.export._on_export_files)
    exported = cv2.imread(str(harness.out_dir / "IMG_0004.jpg"))
    assert exported is not None, (w.status_label.text(), harness.messages, sorted(harness.out_dir.iterdir()))
    assert tuple(int(v) for v in exported[20, 20]) == pytest.approx((0, 0, 255), abs=3)
    assert_originals_untouched(before, harness.folder)


# --------------------------------------------------------------------- #
# Models are fetched only with consent; people are grouped locally
# --------------------------------------------------------------------- #

def test_declining_a_model_download_fetches_nothing(harness, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from src.ai_tools import model_store
    from src.ai_tools.segmentation import SAM_SPEC

    monkeypatch.setattr(model_store, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
    fetched = []
    monkeypatch.setattr(model_store, "_fetch", lambda *a, **k: fetched.append(a) or b"")
    ran = []
    harness.window.with_models([SAM_SPEC], "Testing", lambda: ran.append(True), harness.window.load_button)
    harness.wait()
    assert not ran and not fetched
    assert "not downloaded" in harness.window.status_label.text()


def test_models_already_present_need_no_question(harness, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    asked = []
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: asked.append(a)))
    ran = []
    harness.window.with_models([], "Testing", lambda: ran.append(True), harness.window.load_button)
    assert ran == [True] and not asked


@pytest.mark.model
@pytest.mark.media
@pytest.mark.slow
def test_group_by_person_end_to_end(harness, tmp_path, before):
    from src.ai_tools import person_clustering

    if not (person_clustering.FaceEmbedder.is_cached() and face("pose.png").present
            and face("grace_hopper_227.png").present):
        pytest.skip("recognition model or reference faces not present")
    w = harness.window
    folder = tmp_path / "Event"
    folder.mkdir()
    for name, sample, change in (("a.jpg", "pose.png", None), ("b.jpg", "pose.png", "bright"),
                                 ("c.jpg", "grace_hopper_227.png", None), ("d.jpg", "grace_hopper_227.png", "bright")):
        image = cv2.resize(cv2.imread(str(face(sample).path)), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        if change:
            image = cv2.convertScaleAbs(image, alpha=1.1, beta=10)
        write_jpeg(folder / name, image)
    folder_before = snapshot(folder)
    w.cull.open_folder(folder)
    harness.wait()
    harness.run(w.cull._on_group_people)
    people = w.session.people
    assert people is not None and len(people.people) == 2
    groups = sorted(sorted(Path(p).name for p in people.paths_of(pid)) for pid in people.people)
    assert groups == [["a.jpg", "b.jpg"], ["c.jpg", "d.jpg"]]

    w.cull.person_filter.setCurrentIndex(1)
    shown = [k for k, cell in w.cull.cells.items() if not cell.isHidden()]
    assert len(shown) == 2
    people.rename(w.cull.person_filter.currentData(), "Bride")
    people.save(w.edit_folder)
    assert person_clustering.PeopleIndex.load(w.edit_folder).label(w.cull.person_filter.currentData()) == "Bride"

    w.cull._on_forget_people()
    assert not (w.edit_folder / person_clustering.INDEX_FILENAME).exists()
    assert w.session.people is None and w.cull.person_filter.count() == 1
    assert_originals_untouched(folder_before, folder)
