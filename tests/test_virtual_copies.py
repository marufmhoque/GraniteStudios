"""Virtual copies as tiles of their own, and the Smart Priority sort.

VC-01  a virtual copy duplicates the edit recipe only: no image data on disk
VC-02  each copy is its own tile, right after its parent, chipped v2, v3
VC-03  copies are edited and exported independently
VC-04  Ctrl+' duplicates in Cull and Develop; right-click offers it on every tile
SP-01  Smart Priority puts the keepers first and sinks the culled and rejected
SP-02  a divider marks where the culled and flagged photos begin
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QShortcut

from src.engine import parametric_recipe as pr
from src.engine import session as topics
from src.engine.session import MASTER, LibrarySession, VirtualImage
from src.engine.universal_loader import UniversalLoader
from tests.conftest import assert_originals_untouched, snapshot
from tests.test_workspaces import _settle


@pytest.fixture
def opened(library):
    session = LibrarySession()
    session.open(library, UniversalLoader())
    return session


def _key(folder: Path, name: str) -> str:
    return str(folder / name)


# --------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------- #

def test_a_virtual_copy_never_touches_or_duplicates_the_media(library, opened):
    before = snapshot(library)
    sizes = sum(p.stat().st_size for p in library.rglob("*") if p.is_file())
    key = opened.keys()[0]
    copy = opened.create_virtual_copy(key)
    edited = opened.recipe(key, copy.version).copy()
    edited.basic.exposure = 0.75
    opened.set_recipe(key, edited, copy.version)
    opened.save_all()

    assert_originals_untouched(before, library)                     # hashes, names and count
    assert sorted(p.name for p in library.iterdir() if p.is_file()) == sorted(before)
    new_files = [p for p in library.rglob("*") if p.is_file() and p.name not in before]
    assert all(p.suffix == ".json" for p in new_files), "only small JSON recipes and history may appear"
    added = sum(p.stat().st_size for p in library.rglob("*") if p.is_file()) - sizes
    assert added < 64_000, "no pixels may be copied"
    assert pr.sidecar_path(key, opened.edit_folder, copy.version).is_file()


def test_the_copy_is_a_tile_named_and_placed_beside_its_parent(library, opened):
    first, second = opened.keys()[:2]
    made = opened.create_virtual_copy(first)
    again = opened.create_virtual_copy(first)
    assert isinstance(made, VirtualImage)
    assert (made.chip, again.chip) == ("v2", "v3")
    assert made.variant_key == "IMG_0001_v2" and made.tile_id == f"{first}#v{made.version}"
    tiles = [(e.key, e.version) for e in opened.entries()]
    assert tiles[:4] == [(first, MASTER), (first, made.version), (first, again.version), (second, MASTER)]
    assert opened.active_version(first) == again.version, "the newest copy is the one shown"


def test_a_copy_starts_from_the_version_it_was_made_from_and_then_goes_its_own_way(opened):
    key = opened.keys()[0]
    warm = opened.recipe(key).copy()
    warm.basic.temperature = 40
    opened.set_recipe(key, warm)
    copy = opened.create_virtual_copy(key, MASTER)
    assert opened.recipe(key, copy.version).basic.temperature == 40
    changed = opened.recipe(key, copy.version).copy()
    changed.basic.temperature = -30
    opened.set_recipe(key, changed, copy.version)
    assert opened.recipe(key, MASTER).basic.temperature == 40
    # A copy of the copy carries the copy's settings.
    third = opened.create_virtual_copy(key, copy.version)
    assert opened.recipe(key, third.version).basic.temperature == -30


def test_copies_and_their_selection_survive_and_clear_properly(library, opened):
    key = opened.keys()[1]
    copy = opened.create_virtual_copy(key)
    opened.selected_copies.add((key, copy.version))
    assert opened.selected_photo_keys() == {key}
    assert [e.tile_id for e in opened.selected_entries()] == [copy.tile_id]

    again = LibrarySession()
    again.open(library, UniversalLoader())
    assert [e.chip for e in again.entries() if e.key == key] == ["v1", "v2"]
    assert not again.selected_copies

    opened.delete_version(key, copy.version)
    assert not opened.selected_copies
    assert not pr.sidecar_path(key, opened.edit_folder, copy.version).exists()


# --------------------------------------------------------------------- #
# Smart Priority
# --------------------------------------------------------------------- #

def _frame(path, starred=False, duplicate=False, blink=False):
    return SimpleNamespace(path=Path(path), is_starred=starred, is_duplicate=duplicate, eyes_closed=blink)


def test_smart_priority_puts_keepers_first_and_rejects_last(opened):
    k = opened.keys()          # IMG_0001 .. IMG_0007
    opened.set_rejected([k[0]], True)                    # rejected for no other reason
    opened.set_rating([k[1]], 3)
    opened.set_pick([k[2]], True)
    opened.set_rating([k[3]], 5)
    opened.last_cull_frames = [_frame(k[4], duplicate=True), _frame(k[5], blink=True)]
    opened.blur_reports = {k[6]: SimpleNamespace(is_blurry=True)}

    assert [opened.priority(x) for x in k] == [6, 1, 1, 1, 3, 4, 5]
    order = opened.sorted_keys(topics.SORT_SMART_PRIORITY)
    # Picks first, then 5 stars before 3; then the sunk ones by reason; the rejected last.
    assert order == [k[2], k[3], k[1], k[4], k[5], k[6], k[0]]
    assert opened.sunk_count() == 4
    assert opened.sorted_keys(topics.SORT_CAPTURE) == k, "capture order is untouched"


def test_unrated_keepers_sit_between_the_marked_and_the_sunk(opened):
    k = opened.keys()
    opened.set_rating([k[6]], 1)
    opened.set_rejected([k[3]], True)
    order = opened.sorted_keys(topics.SORT_SMART_PRIORITY)
    assert order[0] == k[6] and order[-1] == k[3]
    assert order[1:-1] == [x for x in k if x not in (k[3], k[6])], "unrated keepers keep capture order"


def test_a_rejected_photo_never_counts_as_a_keeper(opened):
    key = opened.keys()[0]
    opened.set_rating([key], 5)
    opened.set_rejected([key], True)
    assert opened.priority(key) == topics.PRIORITY_REJECTED


# --------------------------------------------------------------------- #
# The Cull grid
# --------------------------------------------------------------------- #

def _grid_tiles(cull) -> list:
    """What the grid shows, row by row: photo keys, (key, copy) tiles and 'divider'."""
    placed = []
    for index in range(cull.grid.count()):
        item = cull.grid.itemAt(index)
        widget = item.widget()
        row, column, _, _ = cull.grid.getItemPosition(index)
        if widget is cull.divider:
            placed.append((row, column, "divider"))
            continue
        key = next((k for k, c in cull.cells.items() if c is widget), None)
        if key is None:
            key = next((t for t, c in cull.copy_cells.items() if c is widget), None)
        if key is not None:
            placed.append((row, column, key))
    return [tile for _, _, tile in sorted(placed)]


def test_copies_appear_as_tiles_right_after_their_parent(harness):
    w = harness.window
    cull = w.cull
    first, second = w.session.keys()[:2]
    w.session.set_current(first)
    made = cull.create_virtual_copies()
    assert len(made) == 1 and made[0].chip == "v2"
    tile = (first, made[0].version)
    assert tile in cull.copy_cells
    assert _grid_tiles(cull)[:3] == [first, tile, second]
    cell = cull.copy_cells[tile]
    assert cell.version_tag.text() == "v2" and not cell.version_tag.isHidden()
    assert cull.cells[first].version_tag.text() == "v1"
    assert "+1 virtual copy" in w.status_label.text() or "IMG_0001_v2" in w.status_label.text()
    # The filmstrips show it too.
    assert made[0].tile_id in w.develop.filmstrip._tiles


def test_ctrl_apostrophe_is_the_duplicate_shortcut_in_cull_and_develop(harness):
    """Ctrl+' makes a virtual copy (as in Lightroom); since 2.08 Ctrl+D is
    Select None in Cull."""
    w = harness.window
    for workspace in (w.cull, w.develop):
        shortcuts = {s.key().toString(): s for s in workspace.findChildren(QShortcut)}
        assert "Ctrl+'" in shortcuts, workspace.title
    cull_keys = {s.key().toString() for s in w.cull.findChildren(QShortcut)}
    assert {"Ctrl+A", "Ctrl+D"} <= cull_keys
    before = len(w.session.entries())
    w.session.set_current(w.session.keys()[0])
    w.selected = set()
    {s.key().toString(): s for s in w.cull.findChildren(QShortcut)}["Ctrl+'"].activated.emit()
    assert len(w.session.entries()) == before + 1
    w.selected = set(w.session.keys())
    {s.key().toString(): s for s in w.cull.findChildren(QShortcut)}["Ctrl+D"].activated.emit()
    assert not w.selected
    labels = [a.text() for a in w.cull.version_menu(w.session.keys()[0]).actions()]
    assert "Duplicate / Create Virtual Copy" in labels


def test_clicking_a_copy_tile_shows_it_and_selects_it_for_export(harness):
    w = harness.window
    cull = w.cull
    key = w.session.keys()[2]
    copy = w.session.create_virtual_copy(key)
    cull.copy_cells[(key, copy.version)].clicked.emit(key)
    assert w.session.current == key and w.session.active_version(key) == copy.version
    assert (key, copy.version) in w.session.selected_copies
    cull.cells[key].clicked.emit(key)                        # the parent's own tile
    assert w.session.active_version(key) == MASTER


def test_copy_tile_menu_renames_and_deletes(harness, monkeypatch):
    from PyQt6.QtWidgets import QInputDialog

    w = harness.window
    key = w.session.keys()[0]
    copy = w.session.create_virtual_copy(key)
    labels = [a.text() for a in w.cull.version_menu(key, copy.version).actions()]
    assert {"Rename Copy…", "Delete Virtual Copy"} <= set(labels)
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Moody", True)))
    assert w.cull.rename_copy(key, copy.version) == "Moody"
    assert w.cull.delete_copy(key, copy.version)
    assert (key, copy.version) not in w.cull.copy_cells
    assert not w.session.version_count(key) > 1


def test_develop_ctrl_d_branches_the_version_being_edited(harness):
    w = harness.window
    key = w.session.keys()[2]
    w.open_in_develop(key)
    _settle(harness, w.develop)
    w.develop.recipe.basic.exposure = 0.5
    w.develop._commit()
    entry = w.develop.duplicate()
    _settle(harness, w.develop)
    assert entry.chip == "v2" and w.develop.version == entry.version
    assert w.session.recipe(key, entry.version).basic.exposure == 0.5


def test_a_selected_copy_tile_exports_as_its_own_file(harness, library):
    before = snapshot(library)
    w = harness.window
    key = _key(library, "IMG_0003.jpg")
    copy = w.session.create_virtual_copy(key)
    mono = w.session.recipe(key, copy.version).copy()
    mono.basic.saturation = -100
    w.session.set_recipe(key, mono, copy.version)
    w.session.set_active_version(key, MASTER)
    w.session.selected = {key}
    w.session.selected_copies = {(key, copy.version)}
    w.switch_to(w.export)
    w.export.every_version.setChecked(False)
    combo = w.export.preset_combo
    combo.setCurrentIndex(combo.findData("full"))
    assert "1 selected virtual copy" in w.export.files_count.text()
    harness.run(w.export._on_export_files)
    names = sorted(p.name for p in harness.out_dir.iterdir())
    assert names == ["IMG_0003.jpg", "IMG_0003_v2-copy-1.jpg"]
    copy_pixels = cv2.imread(str(harness.out_dir / "IMG_0003_v2-copy-1.jpg")).astype(np.int16)
    master = cv2.imread(str(harness.out_dir / "IMG_0003.jpg")).astype(np.int16)
    assert np.abs(copy_pixels[:, :, 0] - copy_pixels[:, :, 2]).mean() < 3 < \
        np.abs(master[:, :, 0] - master[:, :, 2]).mean()

    # Only the copy's tile selected: just the copy.
    for p in harness.out_dir.iterdir():
        p.unlink()
    w.session.selected = set()
    w.export._refresh_counts()
    harness.run(w.export._on_export_files)
    assert sorted(p.name for p in harness.out_dir.iterdir()) == ["IMG_0003_v2-copy-1.jpg"]
    assert_originals_untouched(before, library)


def test_smart_priority_sinks_the_culled_below_a_divider(harness):
    w = harness.window
    cull = w.cull
    k = w.session.keys()
    w.session.set_pick([k[5]], True)
    w.session.set_rejected([k[0], k[2]], True)
    copy = w.session.create_virtual_copy(k[0])
    cull.sort_combo.setCurrentIndex(cull.sort_combo.findData(topics.SORT_SMART_PRIORITY))
    tiles = _grid_tiles(cull)
    assert tiles[0] == k[5]
    divider = tiles.index("divider")
    assert tiles[divider + 1:] == [k[0], (k[0], copy.version), k[2]], "copies sink with their parent"
    assert "Culled & Flagged for Removal (2 items)" in cull.divider.text()
    assert not cull.divider.isHidden()
    # The divider starts its own row.
    _row, column, _rows, _span = cull.grid.getItemPosition(cull.grid.indexOf(cull.divider))
    assert column == 0
    # Arrow keys follow the order shown.
    assert cull._visible_keys()[0] == k[5]

    # Keeping one moves it back up, and the count follows.
    w.session.set_rejected([k[2]], False)
    assert "(1 item)" in cull.divider.text()
    cull.sort_combo.setCurrentIndex(cull.sort_combo.findData(topics.SORT_CAPTURE))
    assert "divider" not in _grid_tiles(cull) and cull.divider.isHidden()
    assert [t for t in _grid_tiles(cull) if isinstance(t, str)] == k
