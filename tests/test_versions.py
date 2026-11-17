"""Versions ("virtual copies"): several independent edits of one original.

VRS-01  a copy duplicates the edit settings only, never the pixels
VRS-02  versions are edited independently
VRS-03  version pills on the canvas, a badge on the grid, side-by-side compare
VRS-04  three style versions in one action

Plus what has to hold around them: originals untouched, versions survive
reopening the folder (and a lost session file), and every version exports
as its own file.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from src.engine import develop_pipeline as dp
from src.engine import export_presets
from src.engine import parametric_recipe as pr
from src.engine.session import MASTER, STATE_FILENAME, LibrarySession
from src.engine.universal_loader import UniversalLoader
from tests.conftest import assert_originals_untouched, snapshot


def _open(folder: Path) -> LibrarySession:
    session = LibrarySession()
    session.open(folder, UniversalLoader())
    return session


def _first(session: LibrarySession) -> str:
    return session.keys()[0]


# --------------------------------------------------------------------- #
# Where versions are stored
# --------------------------------------------------------------------- #

def test_copy_sidecars_sit_beside_the_master_and_never_collide(tmp_path):
    edit = tmp_path / "Shoot-edit"
    assert pr.sidecar_path("IMG_1.CR2", edit).name == "IMG_1.CR2.granite.json"
    assert pr.sidecar_path("IMG_1.CR2", edit, 2).name == "IMG_1.CR2.granite-v2.json"
    # A file whose own name looks like a copy's sidecar still gets a distinct master sidecar.
    assert pr.sidecar_path("IMG_1.CR2.granite-v2.json", edit).name != pr.sidecar_path("IMG_1.CR2", edit, 2).name
    with pytest.raises(ValueError):
        pr.sidecar_path("IMG_1.CR2", edit, -1)


def test_copies_are_found_from_their_file_names(tmp_path):
    for name in ("A.jpg.granite-v1.json", "A.jpg.granite-v3.json", "B.CR2.granite-v2.json",
                 "A.jpg.granite.json", "C.jpg.granite-vX.json", "D.jpg.granite-v0.json", "notes.txt"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    assert pr.version_files(tmp_path) == {"A.jpg": [1, 3], "B.CR2": [2]}
    assert pr.version_files(tmp_path / "missing") == {}


def test_the_master_recipe_cannot_be_discarded_as_a_copy(tmp_path):
    with pytest.raises(ValueError):
        pr.discard("A.jpg", tmp_path, MASTER)


# --------------------------------------------------------------------- #
# VRS-01 and VRS-02
# --------------------------------------------------------------------- #

def test_a_version_copies_settings_not_pixels(library):
    before = snapshot(library)
    session = _open(library)
    key = _first(session)
    session.recipe(key).basic.exposure = 0.7
    session.touch(key)
    copy = session.create_version(key)
    session.save_all()

    assert session.versions(key)[1].name == "Copy 1"
    assert session.active_version(key) == copy
    assert session.recipe(key, copy).basic.exposure == 0.7, "the copy did not start from the master's settings"
    files = sorted(p.name for p in session.edit_folder.iterdir())
    assert f"{Path(key).name}.granite-v{copy}.json" in files
    # Nothing heavier than settings was written: no image files at all.
    assert not [f for f in files if Path(f).suffix.lower() in (".jpg", ".png", ".tif", ".cr2")]
    assert (session.edit_folder / f"{Path(key).name}.granite-v{copy}.json").stat().st_size < 10_000
    assert_originals_untouched(before, library)


def test_versions_are_edited_independently(library):
    session = _open(library)
    key = _first(session)
    copy = session.create_version(key, "Moody Edit")
    mono = session.recipe(key, copy).copy()
    mono.basic.saturation = -100
    session.set_recipe(key, mono)                         # the active version: the copy
    warm = session.recipe(key, MASTER).copy()
    warm.basic.temperature = 40
    session.set_recipe(key, warm, MASTER)
    session.save_all()

    assert pr.load(key, session.edit_folder).basic.temperature == 40
    assert pr.load(key, session.edit_folder).basic.saturation == 0
    assert pr.load(key, session.edit_folder, copy).basic.saturation == -100
    assert pr.load(key, session.edit_folder, copy).basic.temperature == 0


def test_marks_belong_to_the_photograph_not_the_version(library):
    session = _open(library)
    key = _first(session)
    session.set_rating([key], 4)
    session.create_version(key)
    assert session.ratings[key] == 4
    session.set_rejected([key], True)
    assert session.exportable([key]) == []


# --------------------------------------------------------------------- #
# Names, switching, removing
# --------------------------------------------------------------------- #

def test_names_default_to_copy_n_and_stay_unique(library):
    session = _open(library)
    key = _first(session)
    first, second = session.create_version(key), session.create_version(key)
    assert [v.name for v in session.versions(key)] == ["Master", "Copy 1", "Copy 2"]
    third = session.create_version(key, "  Bright   Edit ")
    assert session.version_name(key, third) == "Bright Edit"
    assert session.version_name(key, session.create_version(key, "bright edit")) == "bright edit 2"
    assert session.display_name(key, first) == f"{Path(key).name} (Copy 1)"
    assert session.display_name(key, MASTER) == Path(key).name

    assert session.rename_version(key, second, "Moody") == "Moody"
    for bad in ("", "   ", "Bright Edit", "master"):
        with pytest.raises(ValueError):
            session.rename_version(key, second, bad)
    with pytest.raises(ValueError):
        session.rename_version(key, MASTER, "Main")
    with pytest.raises(KeyError):
        session.rename_version(key, 99, "Nope")


def test_switching_and_stepping_wraps_round(library):
    session = _open(library)
    key = _first(session)
    heard = []
    session.subscribe(heard.append)
    a = session.create_version(key, activate=False)
    b = session.create_version(key, activate=False)
    assert session.active_version(key) == MASTER
    session.set_active_version(key, b)
    assert "versions" in heard and session.active_version(key) == b
    assert session.step_version(key, 1) == MASTER
    assert session.step_version(key, -1) == b
    assert session.step_version(key, -1) == a
    with pytest.raises(KeyError):
        session.set_active_version(key, 42)


def test_removing_a_copy_deletes_only_its_settings(library):
    before = snapshot(library)
    session = _open(library)
    key = _first(session)
    keep = session.create_version(key, "Keep")
    session.recipe(key, keep).basic.contrast = 30
    session.touch(key, keep)
    drop = session.create_version(key, "Drop")
    session.recipe(key, drop).basic.contrast = -30
    session.touch(key, drop)
    session.save_all()
    drop_file = pr.sidecar_path(key, session.edit_folder, drop)
    assert drop_file.exists()

    session.delete_version(key, drop)
    assert not drop_file.exists()
    assert [v.name for v in session.versions(key)] == ["Master", "Keep"]
    assert session.active_version(key) == MASTER, "removing the active copy should fall back to the master"
    assert pr.load(key, session.edit_folder, keep).basic.contrast == 30
    with pytest.raises(ValueError):
        session.delete_version(key, MASTER)
    assert_originals_untouched(before, library)


# --------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------- #

def test_versions_names_and_the_active_one_survive_reopening(library):
    session = _open(library)
    key = _first(session)
    copy = session.create_version(key, "Moody Edit")
    session.recipe(key, copy).basic.clarity = 25
    session.touch(key, copy)
    neutral = session.create_version(key, "Untouched", recipe=pr.Recipe())
    session.set_active_version(key, copy)
    session.save_all()

    again = _open(library)
    assert [v.name for v in again.versions(key)] == ["Master", "Moody Edit", "Untouched"]
    assert again.active_version(key) == copy
    assert again.recipe(key).basic.clarity == 25
    assert again.recipe(key, neutral).is_neutral, "a copy with no edits yet was lost on reopening"


def test_a_lost_session_file_loses_only_the_names(library):
    session = _open(library)
    key = _first(session)
    copy = session.create_version(key, "Moody Edit")
    session.recipe(key, copy).basic.clarity = 25
    session.touch(key, copy)
    session.save_all()
    (session.edit_folder / STATE_FILENAME).write_bytes(b"\x00\xffnot json")

    again = _open(library)
    assert [v.name for v in again.versions(key)] == ["Master", f"Copy {copy}"]
    assert again.recipe(key, copy).basic.clarity == 25


def test_damaged_version_entries_are_ignored(library):
    session = _open(library)
    key = _first(session)
    name = Path(key).name
    state = {"version": 1, "versions": {
        name: {"names": {"1": "Good", "x": "bad id", "-2": "negative", "3": 7}, "active": "1"},
        "missing.jpg": {"names": {"1": "Ghost"}},
        Path(session.keys()[1]).name: "not an object",
    }}
    (session.edit_folder / STATE_FILENAME).write_text(json.dumps(state), encoding="utf-8")
    again = _open(library)
    assert [v.name for v in again.versions(key)] == ["Master", "Good", "Copy 3"]
    assert again.active_version(key) == MASTER, "an active id of the wrong type was trusted"
    assert again.version_count(session.keys()[1]) == 1


# --------------------------------------------------------------------- #
# VRS-04: style versions
# --------------------------------------------------------------------- #

def test_three_style_versions_in_one_action(library):
    session = _open(library)
    key = _first(session)
    master = session.recipe(key, MASTER)
    master.geometry.crop = [0.1, 0.1, 0.8, 0.8]
    master.basic.exposure = 0.3
    session.touch(key, MASTER)

    ids = session.create_style_versions(key)
    assert [session.version_name(key, i) for i in ids] == ["Cinematic", "Warm Analog", "Clean B&W"]
    assert session.active_version(key) == ids[0]
    recipes = [session.recipe(key, i) for i in ids]
    assert all(r.geometry.crop == [0.1, 0.1, 0.8, 0.8] for r in recipes), "a style dropped the master's crop"
    assert all(r.basic.exposure == pytest.approx(0.3) for r in recipes)
    assert len({pr.dumps(r) for r in recipes}) == 3

    source = dp.open_source(Path(key))
    renders = {name: dp.render(source, r).astype(np.int16)
               for name, r in zip(("cinematic", "warm", "mono"), recipes)}
    mono = renders["mono"]
    assert np.abs(mono[:, :, 0] - mono[:, :, 1]).max() <= 2 and np.abs(mono[:, :, 1] - mono[:, :, 2]).max() <= 2, \
        "Clean B&W is not monochrome"
    warm = renders["warm"].reshape(-1, 3).mean(axis=0)
    assert warm[0] > warm[2], "Warm Analog is not warmer than it is blue"
    assert np.abs(renders["cinematic"] - renders["warm"]).mean() > 3


def test_style_names_are_checked():
    with pytest.raises(KeyError):
        pr.styled("Vaporwave")
    for name in pr.VERSION_STYLES:
        assert pr.from_dict(pr.to_dict(pr.styled(name))) == pr.styled(name), f"{name} does not round-trip"


# --------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------- #

def test_version_tags_for_file_names():
    # Numbered like the grid chips: the master is v1, copy 1 is v2.
    assert export_presets.version_tag(MASTER, "Master") == ""
    assert export_presets.version_tag(1, "Moody Edit") == "v2-moody-edit"
    assert export_presets.version_tag(2, "Clean B&W") == "v3-clean-b-w"
    assert export_presets.version_tag(3, "!!!") == "v4"


# --------------------------------------------------------------------- #
# The interface
# --------------------------------------------------------------------- #

def _settle(harness, workspace) -> None:
    for _ in range(2):
        for job in ("loader_job", "renderer"):
            runner = getattr(workspace, job, None)
            if runner is not None:
                runner.wait_idle(harness.app)


def _key(harness, name: str) -> str:
    return str(harness.folder / name)


@pytest.fixture
def before(library):
    return snapshot(library)


@pytest.fixture
def develop(harness):
    w = harness.window
    w.open_in_develop(_key(harness, "IMG_0003.jpg"))
    _settle(harness, w.develop)
    return w.develop


def test_version_pills_branch_and_switch_without_losing_work(harness, develop):
    w = harness.window
    bar = develop.version_bar
    assert list(bar.pills) == [MASTER] and bar.pills[MASTER].isChecked()

    develop.sliders[("basic", "contrast")].set_value(35, announce=True)
    develop.sliders[("basic", "contrast")].released.emit()
    QTest.mouseClick(bar.add_button, Qt.MouseButton.LeftButton)
    _settle(harness, develop)
    copy = w.session.active_version(develop.key)
    assert copy != MASTER and develop.version == copy
    assert [p.text() for p in bar.pills.values()] == ["Master", "Copy 1"] and bar.pills[copy].isChecked()
    assert "(Copy 1)" in develop.frame_label.text()
    assert develop.sliders[("basic", "contrast")].value() == 35, "the copy did not start from the master"

    develop.sliders[("basic", "saturation")].set_value(-100, announce=True)
    develop.sliders[("basic", "saturation")].released.emit()
    QTest.mouseClick(bar.pills[MASTER], Qt.MouseButton.LeftButton)
    _settle(harness, develop)
    assert develop.version == MASTER and develop.sliders[("basic", "saturation")].value() == 0
    QTest.keyClick(develop, Qt.Key.Key_BracketRight)
    _settle(harness, develop)
    assert develop.version == copy and develop.sliders[("basic", "saturation")].value() == -100
    assert develop.undo() is None and develop.recipe.basic.saturation == -100, "undo crossed versions"


def test_compare_versions_side_by_side(harness, develop):
    w = harness.window
    QTest.keyClick(develop, Qt.Key.Key_V)
    assert not develop.sibling_canvas.isVisibleTo(develop), "compare opened with only one version"
    develop.version_bar.create_styles()                  # Cinematic is now active
    _settle(harness, develop)
    QTest.keyClick(develop, Qt.Key.Key_V)
    _settle(harness, develop)
    assert develop.versions_button.isChecked() and develop.sibling_canvas.has_image
    shown = develop.sibling_caption.text()
    assert shown == "Warm Analog" and "Cinematic" in develop.main_caption.text()
    QTest.keyClick(develop, Qt.Key.Key_V, Qt.KeyboardModifier.ShiftModifier)
    _settle(harness, develop)
    assert develop.sibling_caption.text() == "Clean B&W"
    left = develop._last_render.astype(np.int16)
    assert np.abs(left[:, :, 0] - left[:, :, 2]).mean() > 1, "the edited side is not the colour version"
    # Zoom on one side follows on the other.
    develop.canvas.set_zoom(2.0, (0.3, 0.4))
    assert develop.sibling_canvas.view() == pytest.approx(develop.canvas.view())
    QTest.keyClick(develop, Qt.Key.Key_V)
    assert not develop.versions_button.isChecked()


def test_grid_badge_and_right_click_menu(harness):
    w = harness.window
    key = _key(harness, "IMG_0001.jpg")
    cell = w.cull.cells[key]
    assert cell.version_tag.isHidden()
    w.session.set_current(key)
    w.cull.setFocus()
    QTest.keyClick(w.cull, Qt.Key.Key_Apostrophe, Qt.KeyboardModifier.ControlModifier)
    assert w.session.version_count(key) == 2
    # The parent says v1; the copy is a tile of its own, saying v2.
    assert not cell.version_tag.isHidden() and cell.version_tag.text() == "v1"
    copy_cell = w.cull.copy_cells[(key, 1)]
    assert copy_cell.version_tag.text() == "v2"
    menu = w.cull.version_menu(key)
    labels = [a.text() for a in menu.actions()]
    assert "Duplicate / Create Virtual Copy" in labels and "Create 3 Style Versions" in labels
    show = next(a.menu() for a in menu.actions() if a.text() == "Show Version")
    next(a for a in show.actions() if a.text() == "Master").trigger()
    assert w.session.active_version(key) == MASTER and cell.version_tag.text() == "v1"
    harness.select("IMG_0003.jpg", "IMG_0004.jpg")
    assert w.cull.create_style_versions(w.cull._mark_targets()) == 6
    assert w.session.version_count(_key(harness, "IMG_0004.jpg")) == 4


def test_retouch_and_design_follow_the_active_version(harness, develop):
    w = harness.window
    key = develop.key
    copy = develop.version_bar.create_version("Moody")
    w.switch_to(w.design)
    _settle(harness, w.design)
    assert w.design.version == copy and "(Moody)" in w.design.frame_label.text()
    assert w.design.version_bar.pills[copy].isChecked()
    w.design.version_bar.switch_to(MASTER)
    _settle(harness, w.design)
    assert w.design.version == MASTER
    w.switch_to(w.retouch)
    _settle(harness, w.retouch)
    assert w.retouch.version == MASTER
    w.session.set_active_version(key, copy)
    _settle(harness, w.retouch)
    assert w.retouch.version == copy and "(Moody)" in w.retouch.frame_label.text()


def test_rename_and_remove_from_the_bar(harness, develop, monkeypatch):
    from PyQt6.QtWidgets import QInputDialog

    w = harness.window
    bar = develop.version_bar
    copy = bar.create_version()
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Bright Edit", True)))
    assert bar.rename_version() == "Bright Edit"
    assert bar.pills[copy].text() == "Bright Edit"
    assert bar.delete_version() is True                 # the harness answers Yes
    _settle(harness, develop)
    assert list(bar.pills) == [MASTER] and develop.version == MASTER
    assert any("Remove Version" in m for _, m in harness.messages)


def test_every_version_is_exported_as_its_own_file(harness, before):
    w = harness.window
    key = _key(harness, "IMG_0001.jpg")
    w.session.create_version(key, "Moody Edit")
    mono = w.session.recipe(key).copy()
    mono.basic.saturation = -100
    w.session.set_recipe(key, mono)
    harness.select("IMG_0001.jpg", "IMG_0003.jpg")
    w.switch_to(w.export)
    combo = w.export.preset_combo
    combo.setCurrentIndex(combo.findData("full"))
    assert "1 extra version" in w.export.files_count.text()
    harness.run(w.export._on_export_files)
    names = sorted(p.name for p in harness.out_dir.iterdir())
    assert names == ["IMG_0001.jpg", "IMG_0001_v2-moody-edit.jpg", "IMG_0003.jpg"]
    copy = cv2.imread(str(harness.out_dir / "IMG_0001_v2-moody-edit.jpg")).astype(np.int16)
    master = cv2.imread(str(harness.out_dir / "IMG_0001.jpg")).astype(np.int16)
    assert np.abs(copy[:, :, 0] - copy[:, :, 2]).mean() < 3 < np.abs(master[:, :, 0] - master[:, :, 2]).mean()

    for p in harness.out_dir.iterdir():
        p.unlink()
    w.export.every_version.setChecked(False)
    harness.run(w.export._on_export_files)
    assert sorted(p.name for p in harness.out_dir.iterdir()) == ["IMG_0001.jpg", "IMG_0003.jpg"]
    only = cv2.imread(str(harness.out_dir / "IMG_0001.jpg")).astype(np.int16)
    assert np.abs(only[:, :, 0] - only[:, :, 2]).mean() < 3, "the active version was not the one exported"
    assert_originals_untouched(before, harness.folder)
