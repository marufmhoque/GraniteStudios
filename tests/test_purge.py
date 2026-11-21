"""Remove Images: rejected photos leave the working set; files never change."""

from __future__ import annotations

import pytest

from src.engine import parametric_recipe as pr
from src.engine import session as topics
from src.engine.session import LibrarySession
from src.engine.universal_loader import UniversalLoader
from tests.conftest import assert_originals_untouched, snapshot


@pytest.fixture
def opened(library):
    session = LibrarySession()
    session.open(library, UniversalLoader())
    return session


def test_purging_hides_the_rejected_and_leaves_every_file_alone(library, opened):
    before = snapshot(library)
    total = opened.total_images
    doomed = opened.keys()[1:3]
    opened.set_rejected(doomed, True)
    told = []
    opened.subscribe(told.append)

    assert opened.purge_rejected_images() == 2
    assert opened.total_images == total - 2
    assert not set(doomed) & set(opened.keys())
    assert opened.purged_paths() == [opened.all_paths[1], opened.all_paths[2]]
    assert len(opened.all_paths) == total
    assert topics.PURGED in told
    assert_originals_untouched(before, library)       # hashes, names and count unchanged


def test_nothing_rejected_means_nothing_removed(opened):
    assert opened.purge_rejected_images() == 0
    assert not opened.purged


def test_removed_photos_leave_selection_exports_and_the_current_frame(opened):
    keys = opened.keys()
    opened.selected = set(keys[:3])
    opened.set_current(keys[0])
    opened.set_rejected([keys[0]], True)
    opened.purge_rejected_images()
    assert keys[0] not in opened.selected and opened.selected == set(keys[1:3])
    assert opened.current == opened.keys()[0] != keys[0]
    assert all(str(p) != keys[0] for p in opened.exportable(opened.image_paths))


def test_restoring_brings_them_back_with_ratings_and_edits(library, opened):
    keys = opened.keys()
    target = keys[2]
    recipe = pr.Recipe()
    recipe.basic.exposure = 1.25
    opened.set_recipe(target, recipe)
    opened.set_rating([target], 4)
    opened.set_color_label([target], "Green")
    copy = opened.create_version(target, "Moody")
    opened.set_rejected([target], True)
    opened.purge_rejected_images()
    assert target not in opened.keys()

    assert opened.restore_purged_images() == 1
    assert opened.keys() == keys                                  # back in folder order
    assert opened.ratings[target] == 4 and opened.color_labels[target] == "Green"
    assert opened.recipe(target, pr.MASTER).basic.exposure == 1.25
    assert copy in [v.id for v in opened.versions(target)]
    assert target in opened.rejected, "restoring is not un-rejecting"


def test_restore_can_bring_back_just_some(opened):
    keys = opened.keys()
    opened.set_rejected(keys[:3], True)
    opened.purge_rejected_images()
    assert opened.restore_purged_images([keys[1]]) == 1
    assert keys[1] in opened.keys() and keys[0] not in opened.keys() and keys[2] not in opened.keys()
    assert opened.restore_purged_images(["not a photo"]) == 0


def test_removals_survive_reopening_the_folder(library, opened):
    keys = opened.keys()
    opened.set_rating([keys[1]], 2)
    opened.set_rejected([keys[1]], True)
    opened.purge_rejected_images()
    opened.save_all()

    again = LibrarySession()
    again.open(library, UniversalLoader())
    assert keys[1] not in again.keys() and again.purged_paths() == [again.all_paths[1]]
    assert again.ratings[keys[1]] == 2, "a removed photo's marks must load too"
    again.restore_purged_images()
    again.save_all()
    third = LibrarySession()
    third.open(library, UniversalLoader())
    assert third.keys() == keys


def test_a_restored_photo_can_be_removed_again(opened):
    key = opened.keys()[0]
    opened.set_rejected([key], True)
    opened.purge_rejected_images()
    opened.restore_purged_images()
    assert opened.purge_rejected_images() == 1


# --------------------------------------------------------------------- #
# The Cull workspace
# --------------------------------------------------------------------- #

def _grid_keys(cull) -> list[str]:
    """The keys of the cells laid out in the grid, in order."""
    found = []
    for index in range(cull.grid.count()):
        widget = cull.grid.itemAt(index).widget()
        key = next((k for k, c in cull.cells.items() if c is widget), None)
        if key is not None:
            found.append(key)
    return found


def test_remove_images_takes_rejected_photos_out_of_the_grid(harness, library, monkeypatch):
    from tests.conftest import assert_originals_untouched, snapshot

    before = snapshot(library)
    w = harness.window
    cull = w.cull
    total = len(cull.cells)
    doomed = w.session.keys()[:2]
    w.session.set_rejected(doomed, True)
    asked = []
    monkeypatch.setattr(type(cull), "_confirm_remove", lambda self, n: asked.append(n) or True)

    assert cull.remove_images() == 2
    assert asked == [2]
    assert len(cull.cells) == total - 2 and not set(doomed) & set(cull.cells)
    assert _grid_keys(cull) == w.session.keys()
    assert all(cull._parked[k].isHidden() for k in doomed)
    assert w.status_label.text().startswith(f"Showing {total - 2} keepers (2 culled photos removed)")
    assert "2 removed" in (cull._refresh_status() or w.status_label.text())
    assert_originals_untouched(before, library)


def test_nothing_rejected_says_so_and_cancel_changes_nothing(harness, monkeypatch):
    w = harness.window
    cull = w.cull
    monkeypatch.setattr(type(cull), "_confirm_remove", lambda self, n: pytest.fail("asked with nothing to remove"))
    assert cull.remove_images() == 0
    assert w.status_label.text() == "No rejected images to remove."

    w.session.set_rejected(w.session.keys()[:1], True)
    monkeypatch.setattr(type(cull), "_confirm_remove", lambda self, n: False)
    assert cull.remove_images() == 0 and not w.session.purged


def test_ctrl_shift_delete_is_the_shortcut(harness, monkeypatch):
    """Remove Images moved to Ctrl+Shift+Delete in 2.09; Shift+Delete now
    deletes just the selected photo (see test_delete_photo.py)."""
    from PyQt6.QtGui import QShortcut

    cull = harness.window.cull
    slots = {s.key().toString(): s for s in cull.findChildren(QShortcut)}
    assert {"Ctrl+Shift+Del", "Ctrl+Shift+Backspace"} <= set(slots)
    assert "Remove Images" in cull.remove_button.text()
    assert "Ctrl+Shift+Delete" in cull.remove_button.toolTip()


def test_restore_dialog_brings_chosen_photos_back(harness, monkeypatch):
    from src.ui.widgets import restore_removed

    w = harness.window
    cull = w.cull
    keys = w.session.keys()
    w.session.set_rating([keys[1]], 5)
    w.session.set_rejected(keys[:3], True)
    monkeypatch.setattr(type(cull), "_confirm_remove", lambda self, n: True)
    cull.remove_images()

    def pick_second(dialog):
        assert dialog.list.count() == 3 and dialog.chosen() == keys[:3]
        dialog._set_all(restore_removed.Qt.CheckState.Unchecked)
        assert not dialog.restore_button.isEnabled()
        dialog.list.item(1).setCheckState(restore_removed.Qt.CheckState.Checked)
        return True

    monkeypatch.setattr(restore_removed.RestoreRemovedDialog, "exec", pick_second)
    w.restore_removed_images()                          # View > Restore Removed Images...
    assert w.current_workspace is cull
    assert keys[1] in w.session.keys() and keys[0] not in w.session.keys()
    assert keys[1] in cull.cells and not cull.cells[keys[1]].isHidden()
    assert _grid_keys(cull) == w.session.keys()
    assert w.session.ratings[keys[1]] == 5
    assert "Restored 1 photo" in w.status_label.text()


def test_removed_photos_leave_the_filmstrips_and_develop(harness, monkeypatch):
    from tests.test_workspaces import _settle

    w = harness.window
    keys = w.session.keys()
    w.session.set_current(keys[0])
    w.switch_to(w.develop)
    _settle(harness, w.develop)
    w.session.set_rejected([keys[0]], True)
    monkeypatch.setattr(type(w.cull), "_confirm_remove", lambda self, n: True)
    w.cull.remove_images()
    _settle(harness, w.develop)
    assert w.session.current != keys[0]
    assert keys[0] not in w.develop.filmstrip._tiles
