""".granite projects and the recent-projects list."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.engine import project_manager as pm
from src.engine.session import LibrarySession
from src.engine.universal_loader import UniversalLoader
from tests.conftest import assert_originals_untouched, snapshot


@pytest.fixture
def opened(library):
    session = LibrarySession()
    session.open(library, UniversalLoader())
    return session


def test_a_new_project_is_written_where_asked_and_never_beside_the_photos(library, tmp_path):
    before = snapshot(library)
    project = pm.create_project("Smith Wedding: Ceremony", library, tmp_path / "projects")
    assert project.path == tmp_path / "projects" / "Smith Wedding Ceremony.granite"
    assert project.path.is_file()
    assert project.project_name == "Smith Wedding: Ceremony"
    assert project.source_directories == [str(library.resolve())]
    assert not list(tmp_path.glob("projects/*.tmp")), "the atomic write left its temporary file"
    assert_originals_untouched(before, library)


def test_a_project_saves_and_reloads_everything(library, tmp_path, opened):
    project = pm.create_project("Shoot", library, tmp_path)
    opened.set_current(opened.keys()[2])
    opened.set_rejected([opened.keys()[0]], True)
    opened.purge_rejected_images()
    project.save_project(opened, workspace="Develop")

    again = pm.load_project(project.path)
    assert again.project_name == "Shoot" and again.created_at == project.created_at
    assert again.source_directories == project.source_directories
    assert again.active_workspace == "Develop"
    assert again.last_selected_image == opened.current
    assert again.purged_paths == [str(opened.all_paths[0])]
    assert set(again.image_manifest) == {str(p) for p in opened.all_paths}
    assert len(set(again.image_manifest.values())) == len(opened.all_paths)    # one UUID each


def test_get_active_images_leaves_out_the_removed(library, tmp_path, opened):
    project = pm.create_project("Shoot", library, tmp_path)
    project.update_manifest(opened.all_paths)
    project.purged_paths = [str(opened.all_paths[1]), str(opened.all_paths[3])]
    active = project.get_active_images()
    assert len(active) == len(opened.all_paths) - 2
    assert opened.all_paths[1] not in active and opened.all_paths[3] not in active
    assert active[0] == opened.all_paths[0]                      # folder order kept


def test_photo_ids_stay_the_same_across_saves(library, tmp_path, opened):
    project = pm.create_project("Shoot", library, tmp_path)
    project.save_project(opened)
    first = dict(project.image_manifest)
    project.save_project(opened)
    assert pm.load_project(project.path).image_manifest == first


def test_reopening_a_project_restores_its_removals_and_place(library, tmp_path, opened):
    project = pm.create_project("Shoot", library, tmp_path)
    opened.set_rejected([opened.keys()[1]], True)
    opened.purge_rejected_images()
    opened.set_current(opened.keys()[3])
    project.save_project(opened)
    removed, place = opened.purged_paths(), opened.current

    # The folder's own record is lost: the project still knows.
    (opened.edit_folder / "session.granite.json").unlink()
    fresh = LibrarySession()
    fresh.open(library, UniversalLoader())
    assert not fresh.purged
    pm.load_project(project.path).apply_to_session(fresh)
    assert fresh.purged_paths() == removed and fresh.current == place


@pytest.mark.parametrize("content,reason", [
    ("{not json", "damaged"),
    ("[1, 2]", "not a Granite Studios project"),
    ('{"source_directories": []}', "does not say which folder"),
])
def test_a_bad_project_file_says_why(tmp_path, content, reason):
    path = tmp_path / "Bad.granite"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(pm.ProjectError, match=reason):
        pm.load_project(path)


def test_unknown_or_wrong_fields_fall_back(tmp_path, library):
    path = tmp_path / "Odd.granite"
    path.write_text(json.dumps({
        "source_directories": [str(library), 7], "project_name": 12, "purged_paths": "all",
        "image_manifest": {"a.jpg": "id", "b.jpg": 3}, "active_workspace": None, "future_field": True,
    }), encoding="utf-8")
    project = pm.load_project(path)
    assert project.project_name == "Odd" and project.source_directories == [str(library)]
    assert project.purged_paths == [] and project.image_manifest == {"a.jpg": "id"}
    assert project.active_workspace == "Cull"


def test_a_project_needs_a_real_folder(tmp_path):
    with pytest.raises(pm.ProjectError):
        pm.create_project("Nowhere", tmp_path / "missing", tmp_path)


def test_names_become_safe_file_names():
    assert pm.safe_filename('a/b\\c:d*e?"f<g>h|') == "a b c d e f g h"
    assert pm.safe_filename("   ") == "Untitled Project"
    assert pm.safe_filename("trailing dots...") == "trailing dots"


# --------------------------------------------------------------------- #
# Recent projects
# --------------------------------------------------------------------- #

def test_recent_projects_keep_the_newest_first_and_survive_reloading(library, tmp_path):
    store = tmp_path / "recent_projects.json"
    recent = pm.RecentProjects(store)
    one = pm.create_project("One", library, tmp_path)
    two = pm.create_project("Two", library, tmp_path)
    recent.remember(one, 7, 5)
    recent.remember(two, 7, 6, thumbnail=tmp_path / "t.jpg")
    recent.remember(one, 7, 4)
    again = pm.RecentProjects(store)
    assert [e.title for e in again.entries] == ["One", "Two"]
    assert (again.entries[0].photo_count, again.entries[0].keeper_count) == (7, 4)
    assert again.entries[1].thumbnail == str(tmp_path / "t.jpg")
    assert again.entries[0].last_opened


def test_the_list_is_capped_and_skips_missing_projects(library, tmp_path):
    recent = pm.RecentProjects(tmp_path / "recent.json")
    projects = [pm.create_project(f"P{i}", library, tmp_path) for i in range(pm.RECENT_LIMIT + 3)]
    for project in projects:
        recent.remember(project, 1, 1)
    assert len(recent.entries) == pm.RECENT_LIMIT
    projects[-1].path.unlink()
    assert recent.last().path == str(projects[-2].path)


def test_reopen_last_is_remembered_and_a_damaged_list_is_empty(tmp_path):
    store = tmp_path / "recent.json"
    recent = pm.RecentProjects(store)
    assert recent.reopen_last is False and recent.entries == []
    recent.reopen_last = True
    recent.save()
    assert pm.RecentProjects(store).reopen_last is True
    store.write_text("garbage", encoding="utf-8")
    assert pm.RecentProjects(store).entries == []


def test_the_recent_list_lives_in_the_platform_folder(monkeypatch, tmp_path):
    monkeypatch.delenv("GRANITE_STUDIOS_HOME", raising=False)
    monkeypatch.setattr(pm.platform, "system", lambda: "Windows")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert pm.recent_store_path() == tmp_path / "Granite Studios" / "recent_projects.json"
    monkeypatch.setattr(pm.platform, "system", lambda: "Darwin")
    assert pm.recent_store_path() == Path.home() / ".config" / "granite-studios" / "recent_projects.json"


# --------------------------------------------------------------------- #
# The window: File menu, hub and start-up
# --------------------------------------------------------------------- #

def test_the_file_menu_has_the_project_commands(harness):
    w = harness.window
    assert w.new_project_action.shortcut().toString() == "Ctrl+N"
    assert w.open_project_action.shortcut().toString() == "Ctrl+O"
    assert w.save_project_action.shortcut().toString() == "Ctrl+S"
    assert w.recent_menu.title() == "Recent Projects"
    assert w.restore_removed_action.text().startswith("Restore Removed Images")


def test_a_project_made_from_the_open_folder_reopens_where_work_stopped(harness, library, tmp_path, monkeypatch):
    from tests.test_workspaces import _settle

    w = harness.window
    project = w.create_project("Shoot", library, tmp_path)
    assert project is not None and w.project is project
    assert w.windowTitle().endswith("\u2014 Shoot")
    keys = w.session.keys()
    w.session.set_rejected([keys[0]], True)
    monkeypatch.setattr(type(w.cull), "_confirm_remove", lambda self, n: True)
    w.cull.remove_images()                                     # saved automatically
    assert pm.load_project(project.path).purged_paths == [keys[0]]
    w.session.set_current(keys[3])
    w.switch_to(w.develop)
    _settle(harness, w.develop)
    assert w.save_project()
    entry = w.recent.entries[0]
    assert entry.path == str(project.path) and entry.photo_count == len(keys) - 1
    assert entry.keeper_count == len(keys) - 1

    w.cull.open_folder(tmp_path)                               # another folder ends the project
    assert w.project is None and not w.windowTitle().endswith("Shoot")
    assert w.open_project(project.path)
    assert w.session.keys() == keys[1:] and w.session.current == keys[3]
    assert w.current_workspace is w.develop


def test_a_missing_photo_folder_is_explained(harness, tmp_path):
    folder = tmp_path / "Card"
    folder.mkdir()
    project = pm.create_project("Card", folder, tmp_path)
    folder.rmdir()
    w = harness.window
    assert not w.open_project(project.path)
    assert any("can no longer be found" in text for _, text in harness.messages)


def test_save_with_no_project_offers_to_make_one(harness, library, tmp_path, monkeypatch):
    from src.ui.widgets import welcome_hub

    w = harness.window
    seen = {}

    def fill(dialog):
        seen["folder"], seen["locked"] = dialog.folder_edit.text(), dialog.folder_edit.isReadOnly()
        dialog.location_edit.setText(str(tmp_path))
        return True

    monkeypatch.setattr(welcome_hub.NewProjectDialog, "exec", fill)
    assert w.save_project()
    assert seen == {"folder": str(library), "locked": True}
    assert w.project is not None and w.project.path.parent == tmp_path


def test_the_hub_lists_recent_projects_and_remembers_the_reopen_setting(qapp, library, tmp_path):
    from src.ui.widgets.welcome_hub import RECENT, WelcomeHub

    recent = pm.RecentProjects(tmp_path / "recent.json")
    kept = pm.create_project("Kept", library, tmp_path)
    gone = pm.create_project("Gone", library, tmp_path / "elsewhere")
    recent.remember(gone, 3, 2)
    recent.remember(kept, 7, 5)
    gone.path.unlink()
    hub = WelcomeHub(recent)
    assert hub.list.count() == 2
    assert hub.list.item(0).text().startswith("Kept\n7 photos \u00b7 5 keepers")
    assert "(file not found)" in hub.list.item(1).text()
    assert not hub.list.item(1).flags() & pm_flags_enabled()
    hub.reopen_check.setChecked(True)
    assert pm.RecentProjects(tmp_path / "recent.json").reopen_last is True
    hub._open_item(hub.list.item(0))
    assert hub.choice == (RECENT, str(kept.path))


def pm_flags_enabled():
    from PyQt6.QtCore import Qt

    return Qt.ItemFlag.ItemIsEnabled


def test_start_up_reopens_the_last_project_without_the_hub(harness, library, tmp_path, monkeypatch):
    from src.ui.widgets import welcome_hub

    w = harness.window
    project = w.create_project("Last", library, tmp_path)
    w.recent.reopen_last = True
    w.recent.save()
    w._leave_project()
    monkeypatch.setattr(welcome_hub.WelcomeHub, "exec", lambda self: pytest.fail("the hub was shown"))
    w._recent = None                                   # as on a fresh launch
    w.start_up()
    assert w.project is not None and w.project.path == project.path


def test_start_up_shows_the_hub_otherwise(harness, monkeypatch):
    from src.ui.widgets import welcome_hub

    shown = []
    monkeypatch.setattr(welcome_hub.WelcomeHub, "exec", lambda self: shown.append(self) or False)
    harness.window.start_up()
    assert len(shown) == 1
