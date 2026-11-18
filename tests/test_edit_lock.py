"""The edit lock: rejected and removed photos cannot be edited.

EL-01  is_editable: keepers yes; rejected and removed no
EL-02  set_recipe, touch, apply_recipe and sync_settings skip locked photos
EL-03  nothing is saved for a locked photo
EL-04  Cull will not open a locked photo in Develop; the filmstrip leaves it out
EL-05  Develop, Retouch and Design lock their controls and offer Restore
"""

from __future__ import annotations

from pathlib import Path

from src.engine import parametric_recipe as pr
from src.engine.session import LibrarySession
from src.engine.universal_loader import UniversalLoader
from src.ui.widgets.edit_lock import LOCKED_TEXT, OPEN_REFUSED
from tests.conftest import pump


def _session(library: Path) -> LibrarySession:
    session = LibrarySession()
    session.open(library, UniversalLoader())
    return session


def _bright() -> pr.Recipe:
    recipe = pr.Recipe()
    recipe.basic.exposure = 1.0
    return recipe


# --------------------------------------------------------------------- #
# EL-01 .. EL-03: the session
# --------------------------------------------------------------------- #

def test_is_editable_is_false_for_rejected_and_removed_photos(library):
    session = _session(library)
    keeper, rejected, removed = (library / n for n in ("IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg"))
    session.set_rejected([rejected, removed], True)
    session.purge([removed])
    assert session.is_editable(keeper)
    assert not session.is_editable(rejected)
    assert not session.is_editable(removed)
    assert not session.is_editable(None)
    assert str(rejected) not in session.editable_keys() and str(keeper) in session.editable_keys()


def test_sync_settings_only_changes_the_keepers(library):
    session = _session(library)
    keepers = [library / "IMG_0001.jpg", library / "IMG_0003.jpg"]
    rejects = [library / "IMG_0002.jpg", library / "IMG_0004.jpg"]
    session.set_rejected(rejects, True)
    target = session.recipe(keepers[1])
    target.geometry.rotation = 3.0
    synced = session.sync_settings(_bright(), keepers + rejects, label="Synced")
    assert synced == [str(p) for p in keepers]
    assert all(session.recipe(p).basic.exposure == 1.0 for p in keepers)
    assert all(session.recipe(p).basic.exposure == 0.0 for p in rejects)
    assert session.recipe(keepers[1]).geometry.rotation == 3.0, "a target keeps its own crop"


def test_apply_recipe_skips_locked_photos(library):
    session = _session(library)
    session.set_rejected([library / "IMG_0002.jpg"], True)
    applied = session.apply_recipe([library / "IMG_0001.jpg", library / "IMG_0002.jpg"], _bright())
    assert applied == [str(library / "IMG_0001.jpg")]


def test_saving_a_recipe_for_a_rejected_photo_is_blocked(library, caplog):
    session = _session(library)
    path = library / "IMG_0002.jpg"
    session.set_rejected([path], True)
    with caplog.at_level("WARNING"):
        assert session.set_recipe(path, _bright()) is False
    assert "marked for removal" in caplog.text
    session.touch(path)
    assert session.unsaved == 0
    session.save_all()
    assert not any(p.name.startswith("IMG_0002.jpg.granite") for p in session.edit_folder.iterdir())
    assert not session.has_edits(path)


def test_a_removed_photo_is_locked_too(library):
    session = _session(library)
    path = library / "IMG_0003.jpg"
    session.purge([path])
    assert session.set_recipe(path, _bright()) is False


def test_restoring_unlocks(library):
    session = _session(library)
    path = library / "IMG_0002.jpg"
    session.set_rejected([path], True)
    session.purge([path])
    session.restore_photo(path)
    assert session.is_editable(path) and str(path) in session.keys()
    assert session.set_recipe(path, _bright()) is True


def test_an_edit_made_before_rejecting_is_kept(library):
    session = _session(library)
    path = library / "IMG_0001.jpg"
    session.set_recipe(path, _bright())
    session.set_rejected([path], True)
    session.save_all()
    reopened = _session(library)
    assert reopened.recipe(path).basic.exposure == 1.0, "locking stops new edits; it does not undo old ones"


# --------------------------------------------------------------------- #
# EL-04: opening and navigating
# --------------------------------------------------------------------- #

def test_enter_on_a_rejected_photo_stays_in_cull(harness):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    w = harness.window
    key = str(harness.folder / "IMG_0002.jpg")
    harness.reject("IMG_0002.jpg")
    w.session.set_current(key)
    w.cull.setFocus()
    QTest.keyClick(w.cull, Qt.Key.Key_Return)
    assert w.current_workspace is w.cull
    assert w.toast.text() == OPEN_REFUSED
    assert not w.toast.isHidden()
    assert w.open_in_develop(key) is False
    assert w.open_in_develop(str(harness.folder / "IMG_0001.jpg")) is True
    assert w.current_workspace is w.develop


def test_the_filmstrip_leaves_out_rejected_photos(harness):
    w = harness.window
    harness.reject("IMG_0002.jpg", "IMG_0003.jpg")
    w.session.set_current(str(harness.folder / "IMG_0001.jpg"))
    w.switch_to(w.develop)
    tiles = {Path(t).name for t in w.develop.filmstrip._tiles}
    assert "IMG_0002.jpg" not in tiles and "IMG_0003.jpg" not in tiles and "IMG_0001.jpg" in tiles


def test_arrow_keys_in_develop_skip_rejected_photos(harness):
    w = harness.window
    harness.reject("IMG_0002.jpg", "IMG_0003.jpg")
    w.open_in_develop(str(harness.folder / "IMG_0001.jpg"))
    w.develop._step(1)
    assert Path(w.session.current).name == "IMG_0004.jpg"
    w.develop._step(-1)
    assert Path(w.session.current).name == "IMG_0001.jpg"


# --------------------------------------------------------------------- #
# EL-05: the workspaces
# --------------------------------------------------------------------- #

def _show_locked(harness, workspace):
    w = harness.window
    key = str(harness.folder / "IMG_0002.jpg")
    w.session.set_current(key)
    harness.reject("IMG_0002.jpg")
    w.switch_to(workspace)
    pump(harness.app, lambda: workspace.key == key)
    return key


def test_develop_locks_its_controls_and_restore_unlocks_them(harness):
    w = harness.window
    develop = w.develop
    key = _show_locked(harness, develop)
    assert not develop.lock_banner.isHidden()
    assert develop.lock_banner.label.text().startswith(LOCKED_TEXT)
    assert not develop.controls_container.isEnabled() and not develop.version_bar.isEnabled()
    # Nothing a shortcut does is kept.
    before = pr.to_dict(w.session.recipe(key))
    develop.recipe.basic.exposure = 2.0
    develop._commit()
    develop.undo()
    assert pr.to_dict(w.session.recipe(key)) == before
    develop.lock_banner.restore_button.click()
    assert develop.lock_banner.isHidden()
    assert develop.controls_container.isEnabled() and develop.version_bar.isEnabled()
    assert key not in w.session.rejected
    develop.recipe.basic.exposure = 0.5
    develop._commit()
    assert w.session.recipe(key).basic.exposure == 0.5


def test_rejecting_the_open_photo_locks_develop_at_once(harness):
    w = harness.window
    develop = w.develop
    w.open_in_develop(str(harness.folder / "IMG_0001.jpg"))
    assert develop.lock_banner.isHidden() and develop.controls_container.isEnabled()
    harness.reject("IMG_0001.jpg")
    assert not develop.lock_banner.isHidden() and not develop.controls_container.isEnabled()


def test_develop_sync_skips_rejected_frames(harness):
    w = harness.window
    develop = w.develop
    w.open_in_develop(str(harness.folder / "IMG_0001.jpg"))
    harness.select("IMG_0003.jpg", "IMG_0004.jpg")
    harness.reject("IMG_0004.jpg")
    develop.recipe.basic.exposure = 0.7
    develop._commit()
    develop._on_sync()
    assert w.session.recipe(harness.folder / "IMG_0003.jpg").basic.exposure == 0.7
    assert w.session.recipe(harness.folder / "IMG_0004.jpg").basic.exposure == 0.0
    assert "1 rejected frame(s) were skipped" in w.status_label.text()


def test_retouch_and_design_lock_too(harness):
    w = harness.window
    retouch = w.retouch
    key = _show_locked(harness, retouch)
    assert not retouch.lock_banner.isHidden()
    assert not retouch.rail_host.isEnabled() and not retouch.controls_container.isEnabled()
    layers = list(w.session.recipe(key).layers)
    retouch._on_press(0.5, 0.5, __import__("PyQt6.QtCore", fromlist=["Qt"]).Qt.MouseButton.LeftButton)
    assert w.session.recipe(key).layers == layers
    w.switch_to(w.design)
    pump(harness.app, lambda: w.design.key == key)
    assert not w.design.lock_banner.isHidden() and not w.design.left_host.isEnabled()
    w.design.lock_banner.restore_button.click()
    assert w.design.left_host.isEnabled() and w.session.is_editable(key)
