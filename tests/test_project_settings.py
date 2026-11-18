"""Project settings: renaming, moving the project file, relinking photo folders.

PS-01  rename: the file, the name inside it and the recent list all change
PS-02  move the project file; the recent list follows
PS-03  relink a moved photo folder: every path, the removed photos, the selected
       photo and the edit folder (recipes, marks, virtual copies) come along
PS-04  nothing here ever modifies, moves or removes an original (INV-1)
PS-05  the settings dialog applies everything or nothing
PS-06  the window offers to locate a missing folder; File > Project Settings
PS-07  the Project Hub's right-click menu
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from src.engine import project_manager as pm
from src.engine.session import LibrarySession
from src.engine.universal_loader import UniversalLoader
from tests.conftest import assert_originals_untouched, snapshot


def _worked_project(library: Path, tmp_path: Path):
    """A project with real work in it: an edit, a rating, a virtual copy, a
    removed photo and a selected photo."""
    session = LibrarySession()
    session.open(library, UniversalLoader())
    project = pm.create_project("Shoot", library, tmp_path / "projects")
    keys = session.keys()
    recipe = session.recipe(keys[0]).copy()
    recipe.basic.exposure = 1.25
    session.set_recipe(keys[0], recipe)
    session.set_rating([keys[0]], 4)
    copy = session.create_virtual_copy(keys[0])
    session.set_rejected([keys[1]], True)
    session.purge_rejected_images()
    session.set_current(keys[2])
    project.save_project(session, workspace="Develop")
    return project, keys, copy


def _reopen(project_path: Path) -> tuple[pm.GraniteProject, LibrarySession]:
    project = pm.load_project(project_path)
    session = LibrarySession()
    session.open(project.folder, UniversalLoader())
    project.apply_to_session(session)
    return project, session


# --------------------------------------------------------------------- #
# Rename
# --------------------------------------------------------------------- #

def test_renaming_changes_the_file_the_name_inside_and_the_recent_list(library, tmp_path):
    recent = pm.RecentProjects(tmp_path / "recent.json")
    project = pm.create_project("Shoot", library, tmp_path / "projects")
    recent.remember(project, 7, 7)
    old = project.path

    project.rename_project("Smith Wedding: Ceremony", recent=recent)

    assert project.project_name == "Smith Wedding: Ceremony"
    assert project.path == tmp_path / "projects" / "Smith Wedding Ceremony.granite"
    assert project.path.is_file() and not old.exists()
    assert json.loads(project.path.read_text(encoding="utf-8"))["project_name"] == "Smith Wedding: Ceremony"
    entry = pm.RecentProjects(tmp_path / "recent.json").entries[0]
    assert (entry.title, entry.path) == ("Smith Wedding: Ceremony", str(project.path))
    assert not list((tmp_path / "projects").glob("*.tmp"))


def test_a_rename_keeps_everything_else_the_project_knew(library, tmp_path):
    project, keys, _copy = _worked_project(library, tmp_path)
    before = pm.load_project(project.path)
    project.rename_project("Renamed")
    after = pm.load_project(project.path)
    assert after.project_name == "Renamed"
    for field in ("created_at", "source_directories", "image_manifest", "purged_paths",
                  "active_workspace", "last_selected_image", "audit_log"):
        assert getattr(after, field) == getattr(before, field), field


def test_a_bad_rename_changes_nothing(library, tmp_path):
    project = pm.create_project("Shoot", library, tmp_path / "projects")
    other = pm.create_project("Taken", library, tmp_path / "projects")
    original = project.path
    with pytest.raises(pm.ProjectError, match="needs a name"):
        project.rename_project("   ")
    with pytest.raises(pm.ProjectError, match="already exists"):
        project.rename_project("Taken")
    assert project.project_name == "Shoot" and project.path == original and original.is_file()
    assert pm.load_project(other.path).project_name == "Taken"


def test_a_rename_that_only_changes_the_case_works(library, tmp_path):
    project = pm.create_project("shoot", library, tmp_path / "projects")
    project.rename_project("SHOOT")
    files = [p.name for p in (tmp_path / "projects").glob("*.granite")]
    assert files == ["SHOOT.granite"] and pm.load_project(project.path).project_name == "SHOOT"


def test_the_recent_entry_keeps_its_place_and_its_thumbnail(library, tmp_path):
    recent = pm.RecentProjects(tmp_path / "recent.json")
    one = pm.create_project("One", library, tmp_path)
    two = pm.create_project("Two", library, tmp_path)
    thumb = recent.thumbnail_path(two)
    thumb.parent.mkdir(parents=True)
    thumb.write_bytes(b"jpg")
    recent.remember(two, 7, 7, thumbnail=thumb)
    recent.remember(one, 7, 7)                           # One is now first
    two.rename_project("Second", recent=recent)
    entries = pm.RecentProjects(tmp_path / "recent.json").entries
    assert [e.title for e in entries] == ["One", "Second"]
    moved = entries[1]
    assert moved.thumbnail == str(recent.thumbnail_path(two)) and Path(moved.thumbnail).read_bytes() == b"jpg"
    assert not thumb.exists()


# --------------------------------------------------------------------- #
# Moving the project file
# --------------------------------------------------------------------- #

def test_the_project_file_can_be_moved(library, tmp_path):
    recent = pm.RecentProjects(tmp_path / "recent.json")
    project = pm.create_project("Shoot", library, tmp_path / "projects")
    recent.remember(project, 7, 7)
    old = project.path
    archive = tmp_path / "archive" / "2026"                # does not exist yet
    new = project.relocate_project_file(archive, recent=recent)
    assert new == archive / "Shoot.granite" == project.path
    assert new.is_file() and not old.exists()
    assert pm.load_project(new).source_directories == project.source_directories
    assert pm.RecentProjects(tmp_path / "recent.json").entries[0].path == str(new)
    assert project.relocate_project_file(archive, recent=recent) == new       # already there: a no-op


def test_a_move_never_replaces_another_project(library, tmp_path):
    project = pm.create_project("Shoot", library, tmp_path / "one")
    pm.create_project("Shoot", library, tmp_path / "two")
    with pytest.raises(pm.ProjectError, match="already exists"):
        project.relocate_project_file(tmp_path / "two")
    assert project.path == tmp_path / "one" / "Shoot.granite" and project.path.is_file()


# --------------------------------------------------------------------- #
# Relinking a photo folder
# --------------------------------------------------------------------- #

def test_relinking_a_moved_folder_keeps_every_edit_and_mark(library, tmp_path):
    project, keys, copy = _worked_project(library, tmp_path)
    moved = tmp_path / "Drive2" / "Wedding"                # moved, and under a new name
    moved.parent.mkdir()
    shutil.move(str(library), str(moved))
    assert not project.verify_media_accessibility()[0]

    found = project.relink_source_folder(library, moved)

    assert found == 7
    new_root = moved.resolve()
    assert project.source_directories == [str(new_root)]
    assert all(Path(k).parent == new_root for k in project.image_manifest) and len(project.image_manifest) == 7
    assert project.purged_paths == [str(new_root / "IMG_0002.jpg")]
    assert project.last_selected_image == str(new_root / "IMG_0003.jpg")
    assert project.verify_media_accessibility() == (True, [])
    assert pm.load_project(project.path).source_directories == [str(new_root)]     # saved

    # The edit folder came with the photos and now carries the new folder's name.
    assert (moved / "Wedding-edit").is_dir() and not (moved / "Shoot-edit").exists()
    reopened, session = _reopen(project.path)
    first = str(new_root / "IMG_0001.jpg")
    assert session.recipe(first).basic.exposure == pytest.approx(1.25)
    assert session.ratings[first] == 4
    assert [v.id for v in session.versions(first)] == [0, copy.version]
    assert session.recipe(first, copy.version).basic.exposure == pytest.approx(1.25)
    assert session.purged_paths() == [new_root / "IMG_0002.jpg"]
    assert session.current == str(new_root / "IMG_0003.jpg")
    assert [s.label for s in session.history(first, 0).steps][:2] == ["Original", "Exposure +1.25"]
    # The photo IDs did not change.
    assert reopened.image_manifest[first] == pm.load_project(project.path).image_manifest[first]


def test_photos_moved_without_their_edit_folder_get_a_copy_of_it(library, tmp_path):
    project, keys, _copy = _worked_project(library, tmp_path)
    card = tmp_path / "Card"
    card.mkdir()
    for photo in [p for p in library.iterdir() if p.is_file()]:
        shutil.move(str(photo), str(card / photo.name))
    assert (library / "Shoot-edit").is_dir()

    assert project.relink_source_folder(library, card) == 7
    assert (card / "Card-edit").is_dir()
    assert (library / "Shoot-edit").is_dir(), "the old edit folder is left where it was"
    _project, session = _reopen(project.path)
    assert session.recipe(str(card.resolve() / "IMG_0001.jpg")).basic.exposure == pytest.approx(1.25)


def test_a_new_folder_that_already_has_its_own_edit_folder_is_left_alone(library, tmp_path):
    project, keys, _copy = _worked_project(library, tmp_path)
    other = tmp_path / "Other"
    shutil.copytree(library, other)
    (other / "Other-edit").mkdir()
    (other / "Other-edit" / "mine.txt").write_text("keep me", encoding="utf-8")
    project.relink_source_folder(library, other)
    assert (other / "Other-edit" / "mine.txt").read_text(encoding="utf-8") == "keep me"
    assert not (other / "Other-edit" / "IMG_0001.jpg.granite.json").exists()


def test_relinking_re_indexes_and_counts_only_the_photos_present(library, tmp_path):
    project, keys, _copy = _worked_project(library, tmp_path)
    moved = tmp_path / "Moved"
    shutil.copytree(library, moved)
    (moved / "IMG_0007.png").unlink()                       # not on this drive
    shutil.copy(moved / "IMG_0006.jpg", moved / "IMG_0008.jpg")        # new since the project
    ids = dict(project.image_manifest)
    assert project.relink_source_folder(library, moved) == 7          # 6 of the old seven, and the new one
    root = moved.resolve()
    assert str(root / "IMG_0008.jpg") in project.image_manifest      # re-indexed
    assert project.image_manifest[str(root / "IMG_0001.jpg")] == ids[str(library.resolve() / "IMG_0001.jpg")],         "a photo keeps its ID when its folder moves"
    assert len(set(project.image_manifest.values())) == len(project.image_manifest)

    never_opened = pm.create_project("Fresh", library, tmp_path / "projects")      # no photo list yet
    assert never_opened.relink_source_folder(library, moved) == 7


def test_relinking_to_a_missing_folder_or_a_stranger_changes_nothing(library, tmp_path):
    project, keys, _copy = _worked_project(library, tmp_path)
    before = pm.load_project(project.path).to_dict()
    with pytest.raises(pm.ProjectError, match="not a folder"):
        project.relink_source_folder(library, tmp_path / "nowhere")
    with pytest.raises(pm.ProjectError, match="not one of this project's folders"):
        project.relink_source_folder(tmp_path / "Elsewhere", tmp_path)
    assert project.to_dict() == before == pm.load_project(project.path).to_dict()


def test_verify_media_accessibility_lists_the_missing_folders(library, tmp_path):
    project = pm.create_project("Shoot", library, tmp_path)
    assert project.verify_media_accessibility() == (True, [])
    gone = tmp_path / "Gone"
    project.source_directories = [str(library), str(gone)]
    ok, missing = project.verify_media_accessibility()
    assert not ok and missing == [gone]


# --------------------------------------------------------------------- #
# INV-1
# --------------------------------------------------------------------- #

def test_renaming_moving_and_relinking_never_touch_an_original(library, tmp_path):
    project, keys, _copy = _worked_project(library, tmp_path)
    before = snapshot(library)
    names = sorted(before)

    project.rename_project("Renamed")
    project.relocate_project_file(tmp_path / "archive")
    assert_originals_untouched(before, library)

    elsewhere = tmp_path / "Elsewhere"
    shutil.copytree(library, elsewhere)
    project.relink_source_folder(library, elsewhere)
    assert_originals_untouched(before, library)                    # where they were
    assert_originals_untouched(before, elsewhere)                  # where they are now
    assert sorted(p.name for p in library.iterdir() if p.is_file()) == names
    assert sorted(p.name for p in elsewhere.iterdir() if p.is_file()) == names
    # Nothing the project wrote is beside the photos but the app's own edit folder.
    assert {p.name for p in elsewhere.iterdir() if p.is_dir()} == {"Elsewhere-edit"}
    assert not list(elsewhere.glob("*.granite"))


# --------------------------------------------------------------------- #
# The settings dialog
# --------------------------------------------------------------------- #

def test_the_dialog_applies_everything_or_nothing(qapp, library, tmp_path, monkeypatch):
    from src.ui.dialogs.project_settings_dialog import CONNECTED, MISSING, ProjectSettingsDialog

    shown = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: shown.append(a[2])))
    recent = pm.RecentProjects(tmp_path / "recent.json")
    project = pm.create_project("Card", library, tmp_path / "projects")
    pm.create_project("Taken", library, tmp_path / "projects")
    recent.remember(project, 7, 7)
    moved = tmp_path / "Moved"
    shutil.copytree(library, moved)
    shutil.rmtree(library)

    dialog = ProjectSettingsDialog(project, recent)
    assert dialog.table.item(0, 0).text() == MISSING
    assert dialog.table.item(0, 1).text() == str(library.resolve())
    dialog.stage_relink(0, moved)
    assert dialog.table.item(0, 0).text().startswith(CONNECTED) and "relinked on Save" in dialog.table.item(0, 0).text()
    dialog.name_edit.setText("Taken")                       # collides with another project: nothing may change
    assert dialog.save() is False and "already exists" in shown[0]
    assert project.project_name == "Card" and project.source_directories == [str(library.resolve())]
    assert (tmp_path / "projects" / "Card.granite").is_file()
    assert pm.RecentProjects(tmp_path / "recent.json").entries[0].title == "Card"
    assert pm.load_project(project.path).source_directories == [str(library.resolve())]

    dialog.stage_destination(tmp_path / "archive")
    assert str(tmp_path / "archive" / "Taken.granite") in dialog.file_label.text()
    dialog.name_edit.setText("Wedding")
    assert dialog.save() is True and dialog.result() == dialog.DialogCode.Accepted
    assert project.path == tmp_path / "archive" / "Wedding.granite" and project.path.is_file()
    assert not (tmp_path / "projects" / "Card.granite").exists()
    assert pm.load_project(project.path).source_directories == [str(moved.resolve())]
    entry = pm.RecentProjects(tmp_path / "recent.json").entries[0]
    assert (entry.title, entry.path) == ("Wedding", str(project.path))
    assert dialog.relinked == 7


def test_the_dialog_will_not_save_without_a_name_and_a_cancel_changes_nothing(qapp, library, tmp_path):
    from src.ui.dialogs.project_settings_dialog import ProjectSettingsDialog

    project = pm.create_project("Shoot", library, tmp_path)
    dialog = ProjectSettingsDialog(project, pm.RecentProjects(tmp_path / "recent.json"))
    dialog.name_edit.setText("  ")
    assert not dialog.save_button.isEnabled()
    dialog.name_edit.setText("Other")
    dialog.stage_destination(tmp_path / "elsewhere")
    dialog.reject()
    assert project.project_name == "Shoot" and project.path == tmp_path / "Shoot.granite" and project.path.is_file()
    dialog.stage_relink(0, library)                            # back to the folder it already has: not a change
    assert dialog.relinks == {}


def test_the_dialog_buttons_use_the_file_pickers(qapp, library, tmp_path, monkeypatch):
    from src.ui.dialogs.project_settings_dialog import ProjectSettingsDialog

    project = pm.create_project("Shoot", library, tmp_path)
    dialog = ProjectSettingsDialog(project, pm.RecentProjects(tmp_path / "recent.json"))
    picks = iter([str(tmp_path / "archive"), str(tmp_path / "Moved"), ""])
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: next(picks)))
    dialog.move_button.click()
    assert dialog.destination == tmp_path / "archive"
    dialog.table.cellWidget(0, 2).click()
    assert dialog.relinks == {project.source_directories[0]: str(tmp_path / "Moved")}
    dialog.table.cellWidget(0, 2).click()                      # a cancelled picker changes nothing
    assert len(dialog.relinks) == 1


# --------------------------------------------------------------------- #
# The window
# --------------------------------------------------------------------- #

def test_opening_a_project_whose_folder_moved_offers_to_locate_it(harness, library, tmp_path, monkeypatch):
    w = harness.window
    card = tmp_path / "Card"
    shutil.copytree(library, card)
    project = pm.create_project("Card", card, tmp_path / "projects")
    moved = tmp_path / "Card2"
    shutil.move(str(card), str(moved))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(moved)))

    assert w.open_project(project.path)
    assert any("could not be found. Would you like to locate it now?" in text and str(card) in text
               for _, text in harness.messages)
    assert w.session.current_folder == moved.resolve() and w.session.total_images == 7
    assert pm.load_project(project.path).source_directories == [str(moved.resolve())]
    assert "Relinked" in w.status_label.text()


def test_declining_to_locate_the_folder_leaves_the_project_alone(harness, library, tmp_path, monkeypatch):
    w = harness.window
    card = tmp_path / "Card"
    shutil.copytree(library, card)
    project = pm.create_project("Card", card, tmp_path / "projects")
    shutil.rmtree(card)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: ""))
    assert not w.open_project(project.path)                    # the picker was cancelled
    assert any("can no longer be found" in text for _, text in harness.messages)
    assert pm.load_project(project.path).source_directories == [str(card.resolve())]


def test_project_settings_is_in_the_file_menu(harness):
    w = harness.window
    assert w.project_settings_action.shortcut().toString() == "Ctrl+Shift+P"
    assert w.project_settings_action.text().startswith("Project Settings")
    assert not w.project_settings()                            # no project open
    assert "Open a project first" in w.status_label.text()


def test_renaming_the_open_project_updates_the_title_file_and_recent_list(harness, library, tmp_path, monkeypatch):
    from src.ui.dialogs import project_settings_dialog as psd

    w = harness.window
    projects = tmp_path / "projects"
    w.create_project("Shoot", library, projects)
    old = w.project.path

    def rename(dialog):
        dialog.name_edit.setText("Smith Wedding")
        return dialog.save()

    monkeypatch.setattr(psd.ProjectSettingsDialog, "exec", rename)
    assert w.project_settings()
    assert w.windowTitle().endswith("— Smith Wedding")
    assert w.project.path == projects / "Smith Wedding.granite" and not old.exists()
    assert pm.RecentProjects(w.recent.path).entries[0].path == str(w.project.path)
    assert w.save_project()                                    # saves go to the new file, not a revived old one
    assert [p.name for p in projects.glob("*.granite")] == ["Smith Wedding.granite"]


def test_relinking_the_open_folder_from_settings_reopens_it_from_the_new_place(harness, library, tmp_path, monkeypatch):
    from src.ui.dialogs import project_settings_dialog as psd

    w = harness.window
    w.create_project("Shoot", library, tmp_path / "projects")
    keys = w.session.keys()
    w.session.set_rating([keys[0]], 5)
    w.session.save_all()                                   # the folder is copied after its marks are saved
    moved = tmp_path / "Moved"
    shutil.copytree(library, moved)

    def relink(dialog):
        dialog.stage_relink(0, moved)
        return dialog.save()

    monkeypatch.setattr(psd.ProjectSettingsDialog, "exec", relink)
    assert w.project_settings()
    assert w.session.current_folder == moved.resolve()
    assert w.project is not None and w.project.source_directories == [str(moved.resolve())]
    assert pm.load_project(w.project.path).source_directories == [str(moved.resolve())]
    assert w.session.ratings[str(moved.resolve() / "IMG_0001.jpg")] == 5, "the rating came with the folder"


# --------------------------------------------------------------------- #
# The Project Hub's right-click menu
# --------------------------------------------------------------------- #

def _menu(hub, row=0):
    menu = hub.context_menu(hub.list.item(row))
    return menu, {a.text(): a for a in menu.actions()}


def _by(actions, word):
    return next(a for text, a in actions.items() if word in text)


def test_the_hub_menu_renames_moves_and_forgets(qapp, library, tmp_path, monkeypatch):
    from src.ui.widgets.welcome_hub import WelcomeHub

    recent = pm.RecentProjects(tmp_path / "recent.json")
    project = pm.create_project("Alpha", library, tmp_path / "p")
    recent.remember(project, 7, 5)
    hub = WelcomeHub(recent)
    menu, actions = _menu(hub)
    assert [t.split(" ", 1)[1] for t in actions] == [
        "Rename…", "Relink Folders…", "Move Project File…", "Show in Explorer / Finder",
        "Remove from Recent List"]
    assert all(a.isEnabled() for a in actions.values())

    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Beta", True)))
    _by(actions, "Rename").trigger()
    assert hub.list.item(0).text().startswith("Beta\n")
    renamed = tmp_path / "p" / "Beta.granite"
    assert renamed.is_file() and not (tmp_path / "p" / "Alpha.granite").exists()
    assert pm.RecentProjects(tmp_path / "recent.json").entries[0].path == str(renamed)

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(tmp_path / "archive")))
    _by(_menu(hub)[1], "Move Project File").trigger()
    assert (tmp_path / "archive" / "Beta.granite").is_file() and not renamed.exists()

    _by(_menu(hub)[1], "Remove from Recent").trigger()
    assert hub.list.count() == 0 and pm.RecentProjects(tmp_path / "recent.json").entries == []
    assert (tmp_path / "archive" / "Beta.granite").is_file(), "the list forgets a project; the file stays"


def test_the_hub_menu_opens_the_settings_dialog_to_relink(qapp, library, tmp_path, monkeypatch):
    from src.ui.dialogs import project_settings_dialog as psd
    from src.ui.widgets.welcome_hub import WelcomeHub

    recent = pm.RecentProjects(tmp_path / "recent.json")
    project = pm.create_project("Alpha", library, tmp_path / "p")
    recent.remember(project, 7, 5)
    opened = []
    monkeypatch.setattr(psd.ProjectSettingsDialog, "exec", lambda self: opened.append(self.project.project_name) or True)
    hub = WelcomeHub(recent)
    _by(_menu(hub)[1], "Relink Folders").trigger()
    assert opened == ["Alpha"]


def test_the_hub_menu_is_careful_with_the_open_project_and_missing_files(qapp, library, tmp_path):
    from src.ui.widgets.welcome_hub import WelcomeHub

    recent = pm.RecentProjects(tmp_path / "recent.json")
    open_one = pm.create_project("Open", library, tmp_path / "p")
    gone = pm.create_project("Gone", library, tmp_path / "p")
    recent.remember(gone, 1, 1)
    recent.remember(open_one, 1, 1)
    gone.path.unlink()
    hub = WelcomeHub(recent, locked_path=str(open_one.path))

    _menu_open, actions = _menu(hub, 0)                         # the project the window has open
    assert not _by(actions, "Rename").isEnabled() and "File" in _by(actions, "Rename").toolTip()
    assert not _by(actions, "Relink").isEnabled() and not _by(actions, "Move").isEnabled()
    assert _by(actions, "Show in").isEnabled() and _by(actions, "Remove").isEnabled()

    _menu_gone, actions = _menu(hub, 1)                         # its file is gone
    assert not _by(actions, "Rename").isEnabled() and not _by(actions, "Show in").isEnabled()
    assert _by(actions, "Remove").isEnabled()


def test_the_hub_has_a_visible_rename_button_and_f2(qapp, library, tmp_path, monkeypatch):
    from src.ui.widgets.welcome_hub import WelcomeHub

    recent = pm.RecentProjects(tmp_path / "recent.json")
    project = pm.create_project("Alpha", library, tmp_path / "p")
    recent.remember(project, 7, 5)
    hub = WelcomeHub(recent)
    assert hub.rename_button.isEnabled() and hub.more_button.isEnabled()
    names = iter(["Beta", "Gamma"])
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: (next(names), True)))
    hub.rename_button.click()
    assert hub.list.item(0).text().startswith("Beta\n") and (tmp_path / "p" / "Beta.granite").is_file()
    assert hub.rename_selected()                                  # what F2 calls
    assert hub.list.item(0).text().startswith("Gamma\n")

    locked = WelcomeHub(pm.RecentProjects(tmp_path / "recent.json"), locked_path=str(tmp_path / "p" / "Gamma.granite"))
    assert not locked.rename_button.isEnabled() and "File" in locked.rename_button.toolTip()
    assert not locked.rename_selected()
    empty = WelcomeHub(pm.RecentProjects(tmp_path / "none.json"))
    assert not empty.rename_button.isEnabled() and not empty.more_button.isEnabled()


def test_show_in_explorer_opens_the_right_file_manager(library, tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(pm.subprocess, "Popen", lambda args, **k: started.append(args))
    target = tmp_path / "p" / "Alpha.granite"
    for system, expected in (
        ("Windows", ["explorer", f"/select,{target}"]),
        ("Darwin", ["open", "-R", str(target)]),
        ("Linux", ["xdg-open", str(target.parent)]),
    ):
        monkeypatch.setattr(pm.platform, "system", lambda s=system: s)
        assert pm.reveal_in_file_manager(target) is True
        assert started[-1] == expected
    monkeypatch.setattr(pm.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(OSError("no manager")))
    assert pm.reveal_in_file_manager(target) is False


def test_the_file_menu_can_check_for_updates(harness, monkeypatch):
    w = harness.window
    asked = []
    monkeypatch.setattr(w, "check_for_updates", lambda manual=True: asked.append(manual))
    assert w.check_updates_action.text().startswith("Check for Updates")
    w.check_updates_action.trigger()
    assert asked == [True]
