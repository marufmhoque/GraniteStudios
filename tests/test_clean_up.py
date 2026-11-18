"""Smart misfire selection and the Clean Up Shoot assistant.

CU-01  duplicates, blurry photos, blinks and rejects are found, once each
CU-02  the assistant counts them, lets groups be unticked, and removes them
CU-03  removing never touches the files (INV-1) and can be undone
CU-04  the Cull Select menu and the Clean Up button
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.engine.session import LibrarySession
from src.engine.universal_loader import UniversalLoader
from tests.conftest import assert_originals_untouched, make_scene, snapshot, write_jpeg

KEEPERS = [f"IMG_{i:04d}.jpg" for i in range(1, 11)]
DUPLICATES = [f"IMG_{i:04d}.jpg" for i in range(11, 16)]
BLURRY = [f"IMG_{i:04d}.jpg" for i in range(16, 19)]
BLINKS = [f"IMG_{i:04d}.jpg" for i in range(19, 21)]


def _flag(session, folder: Path, duplicates=(), blinks=(), blurry=()):
    """What a Smart Cull and Find Blurry would leave on the session."""
    session.last_cull_frames = [
        SimpleNamespace(path=folder / name, is_duplicate=name in duplicates, eyes_closed=name in blinks,
                        is_starred=False)
        for name in sorted(set(duplicates) | set(blinks))
    ]
    session.blur_reports = {str(folder / name): SimpleNamespace(is_blurry=True, verdict="out_of_focus")
                            for name in blurry}


@pytest.fixture
def shoot(tmp_path) -> Path:
    folder = tmp_path / "Shoot"
    folder.mkdir()
    for index, name in enumerate(KEEPERS + DUPLICATES + BLURRY + BLINKS):
        write_jpeg(folder / name, make_scene(index, (160, 120)))
    return folder


@pytest.fixture
def session(shoot) -> LibrarySession:
    session = LibrarySession()
    session.open(shoot, UniversalLoader())
    _flag(session, shoot, DUPLICATES, BLINKS, BLURRY)
    return session


def _names(paths) -> list[str]:
    return [Path(p).name for p in paths]


# --------------------------------------------------------------------- #
# CU-01: the selection helpers
# --------------------------------------------------------------------- #

def test_each_kind_of_misfire_is_found(session):
    assert session.total_images == 20
    assert _names(session.get_duplicate_images()) == DUPLICATES
    assert _names(session.get_blurry_images()) == BLURRY
    assert _names(session.get_blink_images()) == BLINKS
    assert session.get_rejected_images() == []


def test_all_misfires_are_exactly_the_ten_bad_photos(session):
    assert _names(session.get_all_misfires()) == DUPLICATES + BLURRY + BLINKS


def test_motion_blur_counts_as_blurry(session, shoot):
    session.blur_reports[str(shoot / "IMG_0001.jpg")] = SimpleNamespace(is_blurry=False, verdict="motion_blur")
    assert "IMG_0001.jpg" in _names(session.get_blurry_images())


def test_rejects_are_misfires_and_each_photo_counts_once(session, shoot):
    session.set_rejected([shoot / "IMG_0001.jpg", shoot / "IMG_0011.jpg"], True)
    session.blur_reports[str(shoot / "IMG_0012.jpg")] = SimpleNamespace(is_blurry=True, verdict="out_of_focus")
    misfires = _names(session.get_all_misfires())
    assert misfires == ["IMG_0001.jpg"] + DUPLICATES + BLURRY + BLINKS
    assert len(misfires) == len(set(misfires))


def test_a_photo_you_picked_or_starred_is_not_a_misfire(session, shoot):
    session.set_pick([shoot / "IMG_0011.jpg"], True)
    session.set_rating([shoot / "IMG_0019.jpg"], 4)
    assert "IMG_0011.jpg" not in _names(session.get_duplicate_images())
    assert "IMG_0019.jpg" not in _names(session.get_blink_images())
    session.set_rejected([shoot / "IMG_0019.jpg"], True)
    assert "IMG_0019.jpg" in _names(session.get_all_misfires()), "unless you rejected it after all"


def test_removed_photos_are_not_misfires_any_more(session, shoot):
    session.purge([shoot / "IMG_0011.jpg"])
    assert "IMG_0011.jpg" not in _names(session.get_all_misfires())
    assert len(session.get_all_misfires()) == 9


# --------------------------------------------------------------------- #
# CU-02, CU-03: the assistant
# --------------------------------------------------------------------- #

def test_the_assistant_counts_every_group(qapp, session):
    from src.ui.dialogs.clean_up_dialog import CleanUpAssistantDialog

    dialog = CleanUpAssistantDialog(session)
    texts = {key: check.text() for key, check in dialog.checks.items()}
    assert texts["duplicates"].startswith("Duplicates  (5)") and texts["blurry"].startswith("Blurry  (3)")
    assert texts["blinks"].startswith("Blinks  (2)") and texts["rejected"].startswith("Rejected  (0)")
    assert not dialog.checks["rejected"].isEnabled(), "an empty group cannot be ticked"
    assert dialog.total_label.text() == "Total selected for removal: 10 photos"
    assert dialog.keepers_label.text() == "Keepers remaining: 10 photos"


def test_unticking_a_group_keeps_it(qapp, session, shoot):
    from src.ui.dialogs.clean_up_dialog import CleanUpAssistantDialog

    dialog = CleanUpAssistantDialog(session)
    dialog.checks["blurry"].setChecked(False)
    assert dialog.total_label.text() == "Total selected for removal: 7 photos"
    assert dialog.keepers_label.text() == "Keepers remaining: 13 photos"
    assert dialog.remove_selected() == 7
    assert session.total_images == 13
    assert all(str(shoot / name) in session.keys() for name in BLURRY)


def test_removing_takes_the_misfires_out_and_leaves_the_files(qapp, session, shoot):
    from src.ui.dialogs.clean_up_dialog import REMOVE, CleanUpAssistantDialog

    before = snapshot(shoot)
    dialog = CleanUpAssistantDialog(session)
    assert dialog.remove_selected() == 10
    assert dialog.action == REMOVE
    assert session.total_images == 10
    assert _names(session.image_paths) == KEEPERS
    assert set(_names(session.purged_paths())) == set(DUPLICATES + BLURRY + BLINKS)
    assert all(str(shoot / n) in session.rejected for n in DUPLICATES + BLURRY + BLINKS)
    assert_originals_untouched(before, shoot)
    assert sorted(p.name for p in shoot.iterdir() if p.is_file()) == sorted(before), "nothing moved or deleted"
    # Undone exactly.
    assert session.restore_purged_images() == 10
    assert session.total_images == 20


def test_removal_is_remembered(qapp, session, shoot):
    from src.ui.dialogs.clean_up_dialog import CleanUpAssistantDialog

    CleanUpAssistantDialog(session).remove_selected()
    session.save_all()
    reopened = LibrarySession()
    reopened.open(shoot, UniversalLoader())
    assert reopened.total_images == 10


def test_inspect_closes_without_changing_anything(qapp, session):
    from src.ui.dialogs.clean_up_dialog import INSPECT, CleanUpAssistantDialog

    dialog = CleanUpAssistantDialog(session)
    dialog.inspect()
    assert dialog.action == INSPECT and session.total_images == 20 and not session.rejected
    assert _names(dialog.chosen()) == DUPLICATES + BLURRY + BLINKS


# --------------------------------------------------------------------- #
# CU-04: Cull
# --------------------------------------------------------------------- #

def _harness_flags(harness):
    w = harness.window
    _flag(w.session, harness.folder, duplicates=["IMG_0002.jpg"], blinks=["IMG_0006.jpg"], blurry=["IMG_0004.jpg"])
    return w


def test_the_select_menu_counts_and_selects(harness):
    w = _harness_flags(harness)
    cull = w.cull
    cull._fill_select_menu()
    texts = [a.text() for a in cull.select_menu.actions() if a.text()]
    assert texts[:3] == ["Select All\tCtrl+A", "Select None\tCtrl+D", "Invert Selection"]
    assert "Select All Duplicates (1)" in texts and "Select All Blurry (1)" in texts
    assert "Select All Blinks (1)" in texts and any(t.startswith("\U0001F3AF Select All Misfires") and t.endswith("(3)")
                                                     for t in texts)
    harness.reject("IMG_0007.png")
    cull._fill_select_menu()
    misfires = next(a for a in cull.select_menu.actions() if "Misfires" in a.text())
    assert misfires.text().endswith("(4)")
    misfires.trigger()
    assert {Path(k).name for k in w.selected} == {"IMG_0002.jpg", "IMG_0004.jpg", "IMG_0006.jpg", "IMG_0007.png"}
    assert w.session.current.endswith("IMG_0002.jpg"), "the grid goes to the first one"
    blurry = next(a for a in cull.select_menu.actions() if "Blurry" in a.text())
    blurry.trigger()
    assert {Path(k).name for k in w.selected} == {"IMG_0004.jpg"}


def test_selecting_a_group_clears_a_filter_that_would_hide_it(harness):
    w = _harness_flags(harness)
    w.cull.view_filter.setCurrentText("Rejected")
    w.cull.select_group(w.session.get_duplicate_images(), "duplicates")
    assert w.cull.view_filter.currentText() == "All"
    assert not w.cull.cells[str(harness.folder / "IMG_0002.jpg")].isHidden()


def test_clean_up_from_cull_removes_and_says_so(harness, monkeypatch):
    from src.ui.dialogs.clean_up_dialog import CleanUpAssistantDialog

    w = _harness_flags(harness)
    before = snapshot(harness.folder)
    monkeypatch.setattr(CleanUpAssistantDialog, "exec", lambda self: self.remove_selected())
    assert w.cull.clean_up_button.text().startswith("\U0001F9F9 Clean Up Misfires")
    w.cull.clean_up_button.click()
    assert w.session.total_images == 4
    assert w.toast.text() == "Removed 3 misfires. Showing 4 keepers."
    assert str(harness.folder / "IMG_0002.jpg") not in w.cull.cells
    assert_originals_untouched(before, harness.folder)


def test_inspect_in_grid_selects_the_misfires(harness, monkeypatch):
    from src.ui.dialogs.clean_up_dialog import CleanUpAssistantDialog

    w = _harness_flags(harness)
    monkeypatch.setattr(CleanUpAssistantDialog, "exec", lambda self: self.inspect())
    w.cull.clean_up()
    assert {Path(k).name for k in w.selected} == {"IMG_0002.jpg", "IMG_0004.jpg", "IMG_0006.jpg"}
    assert w.session.total_images == 7
