"""Delete Photo: one photo (or the selected ones) leaves the workspace.

DP-01  the photo leaves the working set; the file on disk is untouched (INV-1)
DP-02  its ratings, edits and virtual copies are kept, so Restore brings it back
DP-03  the other photos are completely unaffected
DP-04  the next photo becomes current (the one before it at the end)
DP-05  the dialog: thumbnail, file name and size, "the file is not deleted"
DP-06  Cancel changes nothing
DP-07  Cull: Shift+Delete and the tile menu's red item
DP-08  Develop, Retouch, Design: Shift+Delete and the filmstrip's tile menu
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.engine import parametric_recipe as pr
from src.engine.session import LibrarySession
from src.engine.universal_loader import UniversalLoader
from tests.conftest import assert_originals_untouched, pump, snapshot


def _session(library: Path) -> LibrarySession:
    session = LibrarySession()
    session.open(library, UniversalLoader())
    return session


def _bright(stops: float = 1.0) -> pr.Recipe:
    recipe = pr.Recipe()
    recipe.basic.exposure = stops
    return recipe


def _confirm(monkeypatch, answer: int = 1) -> list:
    """Answer the Delete Photo dialog; returns the dialogs shown."""
    from src.ui.dialogs import delete_photo_dialog as dd

    shown = []

    def exec_(self):
        shown.append(self)
        return answer

    monkeypatch.setattr(dd.DeletePhotoDialog, "exec", exec_)
    return shown


# --------------------------------------------------------------------- #
# DP-01 .. DP-04: the session
# --------------------------------------------------------------------- #

def test_deleting_leaves_the_file_on_disk_and_restore_brings_it_back(library):
    before = snapshot(library)
    session = _session(library)
    target = str(library / "IMG_0003.jpg")
    session.set_rating([target], 4)
    session.set_recipe(target, _bright(0.7))
    copy = session.create_virtual_copy(target)
    session.save_all()

    assert session.delete_photos_from_workspace([target]) == 1
    assert target not in session.keys() and target in session.purged
    assert (library / "IMG_0003.jpg").is_file()
    session.save_all()
    assert_originals_untouched(before, library)

    # Reopening the folder remembers the deletion; restoring brings it all back.
    again = _session(library)
    assert target not in again.keys()
    assert again.restore_purged_images([target]) == 1
    assert target in again.keys()
    assert again.ratings.get(target) == 4
    assert again.recipe(target).basic.exposure == pytest.approx(0.7)
    assert copy.version in [v.id for v in again.versions(target)]
    assert_originals_untouched(before, library)


def test_other_photos_are_completely_unaffected(library):
    session = _session(library)
    keys = session.keys()
    target, others = keys[2], [k for k in keys if k != keys[2]]
    for i, key in enumerate(others):
        session.set_recipe(key, _bright(0.1 * (i + 1)))
    session.set_rating(others[:2], 5)
    session.set_rejected(others[-1:], True)
    session.save_all()
    # Every other photo's own files in the edit folder (the folder-wide
    # session file records the deletion, so it is left out).
    edit_before = {p.name: p.read_bytes() for p in session.edit_folder.iterdir()
                   if p.is_file() and any(p.name.startswith(Path(k).name) for k in others)}
    assert len(edit_before) >= len(others)
    marks_before = {k: (session.ratings.get(k), k in session.rejected, k in session.picks) for k in others}
    recipes_before = {k: pr.to_dict(session.recipe(k)) for k in others}

    session.delete_photos_from_workspace([target])
    session.save_all()

    assert session.keys() == others
    assert {k: (session.ratings.get(k), k in session.rejected, k in session.picks) for k in others} == marks_before
    assert {k: pr.to_dict(session.recipe(k)) for k in others} == recipes_before
    edit_after = {p.name: p.read_bytes() for p in session.edit_folder.iterdir() if p.name in edit_before}
    assert edit_after == edit_before


def test_the_next_photo_becomes_current(library):
    session = _session(library)
    keys = session.keys()
    session.set_current(keys[1])
    session.delete_photos_from_workspace([keys[1]])
    assert session.current == keys[2]

    # At the end, the one before it.
    session.set_current(keys[-1])
    session.delete_photos_from_workspace([keys[-1]])
    assert session.current == keys[-2]

    # In the grid's order, not the folder's.
    order = list(reversed(session.keys()))
    session.set_current(order[0])
    session.delete_photos_from_workspace([order[0]], order)
    assert session.current == order[1]

    # Deleting a photo that is not shown leaves the current one alone.
    shown = session.current
    other = next(k for k in session.keys() if k != shown)
    session.delete_photos_from_workspace([other])
    assert session.current == shown


def test_deleting_several_skips_past_all_of_them(library):
    session = _session(library)
    keys = session.keys()
    session.set_current(keys[2])
    assert session.delete_photos_from_workspace([keys[2], keys[3], keys[4]]) == 3
    assert session.current == keys[5]
    assert session.delete_photos_from_workspace([keys[2]]) == 0      # already gone


# --------------------------------------------------------------------- #
# DP-05, DP-06: the dialog and the window
# --------------------------------------------------------------------- #

def test_the_dialog_says_what_happens(qapp, library):
    from src.ui.dialogs.delete_photo_dialog import DeletePhotoDialog, file_size_label

    path = library / "IMG_0001.jpg"
    dialog = DeletePhotoDialog([str(path)])
    assert "IMG_0001.jpg" in dialog.windowTitle() + dialog.details.text()
    assert file_size_label(path) in dialog.details.text() and "KB" in file_size_label(path)
    assert "not deleted" in dialog.note.text() and "Restore Removed Images" in dialog.note.text()
    assert "Workspace" in dialog.delete_button.text()
    assert file_size_label(library / "missing.jpg") == "missing"
    several = DeletePhotoDialog([str(library / "IMG_0001.jpg"), str(library / "IMG_0002.jpg")])
    assert "2 Photos" in several.windowTitle()


def test_the_window_asks_then_deletes_and_logs(harness, monkeypatch):
    w = harness.window
    before = snapshot(harness.folder)
    keys = w.session.keys()

    shown = _confirm(monkeypatch, answer=0)              # Cancel
    assert w.delete_photos([keys[0]]) == 0
    assert len(shown) == 1 and keys[0] in w.session.keys()

    shown = _confirm(monkeypatch, answer=1)
    w.session.set_current(keys[0])
    assert w.delete_photos([keys[0]]) == 1
    assert keys[0] not in w.session.keys() and w.session.current == keys[1]
    assert "Deleted IMG_0001.jpg from the workspace" in w.status_label.text()
    assert "still on disk" in w.status_label.text()
    assert any("Deleted IMG_0001.jpg" in e["description"] for e in w._pending_log)
    assert_originals_untouched(before, harness.folder)


# --------------------------------------------------------------------- #
# DP-07: Cull
# --------------------------------------------------------------------- #

def test_shift_delete_in_cull_deletes_the_selected_photos(harness, monkeypatch):
    from PyQt6.QtGui import QShortcut

    w = harness.window
    cull = w.cull
    shortcuts = {s.key().toString(): s for s in cull.findChildren(QShortcut)}
    assert {"Shift+Del", "Shift+Backspace"} <= set(shortcuts)

    _confirm(monkeypatch)
    keys = w.session.keys()
    w.session.set_current(keys[0])
    shortcuts["Shift+Del"].activated.emit()               # nothing selected: the photo shown
    assert keys[0] not in w.session.keys()
    harness.select("IMG_0003.jpg", "IMG_0004.jpg")
    shortcuts["Shift+Backspace"].activated.emit()
    assert not {keys[2], keys[3]} & set(w.session.keys())
    assert not w.session.rejected & {keys[2], keys[3]}    # deleted, not rejected
    assert all((harness.folder / n).is_file() for n in ("IMG_0001.jpg", "IMG_0003.jpg", "IMG_0004.jpg"))


def test_the_tile_menu_has_a_red_delete_item(harness, monkeypatch):
    from PyQt6.QtWidgets import QWidgetAction

    from src.ui.dialogs.delete_photo_dialog import DANGER, DELETE_TEXT

    w = harness.window
    keys = w.session.keys()
    menu = w.cull.version_menu(keys[4])
    action = next(a for a in menu.actions() if a.text() == DELETE_TEXT)
    assert isinstance(action, QWidgetAction)
    assert DANGER in action.defaultWidget().styleSheet()

    shown = _confirm(monkeypatch)
    action.trigger()
    assert shown and keys[4] not in w.session.keys()
    pump(harness.app, lambda: keys[4] not in w.cull.cells)


# --------------------------------------------------------------------- #
# DP-08: the editing workspaces
# --------------------------------------------------------------------- #

def test_develop_deletes_the_photo_shown_and_moves_on(harness, monkeypatch):
    from PyQt6.QtGui import QShortcut

    w = harness.window
    keys = w.session.keys()
    assert w.open_in_develop(keys[1])
    develop = w.develop
    pump(harness.app, lambda: develop.key == keys[1])
    develop.recipe.basic.exposure = 0.4                  # an edit not yet committed
    _confirm(monkeypatch)

    shortcut = next(s for s in develop.findChildren(QShortcut) if s.key().toString() == "Shift+Del")
    shortcut.activated.emit()
    assert keys[1] not in w.session.keys()
    pump(harness.app, lambda: develop.key == keys[2])
    # The edit in progress was saved before the photo left, so Restore keeps it.
    w.session.restore_purged_images([keys[1]])
    assert w.session.recipe(keys[1]).basic.exposure == pytest.approx(0.4)


@pytest.mark.parametrize("name", ["develop", "retouch", "design"])
def test_the_filmstrip_menu_deletes_from_each_editing_workspace(harness, monkeypatch, name):
    from PyQt6.QtGui import QShortcut

    from src.ui.dialogs.delete_photo_dialog import DELETE_TEXT

    w = harness.window
    workspace = getattr(w, name)
    assert {"Shift+Del", "Shift+Backspace"} <= {s.key().toString() for s in workspace.findChildren(QShortcut)}
    keys = w.session.keys()
    w.session.set_current(keys[0])
    w.switch_to(workspace)
    workspace.filmstrip.rebuild()
    entry = next(e for e in w.session.entries() if e.key == keys[3])
    menu = workspace.filmstrip.tile_menu(entry)
    _confirm(monkeypatch)
    next(a for a in menu.actions() if a.text() == DELETE_TEXT).trigger()
    assert keys[3] not in w.session.keys()
    assert w.session.current == keys[0]                  # another photo: the one shown stays
    assert keys[3] not in workspace.filmstrip.order()
