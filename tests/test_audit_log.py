"""The project audit trail, the History drawer and per-photo edit history.

AUD-01  batch operations are logged with a time and persist in the .granite file
AUD-02  the workflow checklist: which stages are done, Develop's progress
AUD-03  the 📋 History drawer shows the checklist and the timeline
HIS-01  every finished recipe change is a labelled step, kept in the edit folder
HIS-02  clicking a step restores it, as a new step; nothing later is lost
"""

from __future__ import annotations

import json

import pytest

from src.engine import edit_history as eh
from src.engine import parametric_recipe as pr
from src.engine import project_manager as pm
from src.engine.session import MASTER, LibrarySession
from src.engine.universal_loader import UniversalLoader
from tests.conftest import assert_originals_untouched, snapshot
from tests.test_workspaces import _settle


@pytest.fixture
def opened(library):
    session = LibrarySession()
    session.open(library, UniversalLoader())
    return session


# --------------------------------------------------------------------- #
# The audit log
# --------------------------------------------------------------------- #

def test_actions_are_logged_and_mark_their_stage_done(library, tmp_path):
    project = pm.create_project("Shoot", library, tmp_path)
    entry = project.log_action("cull", "Smart Cull: 420 keepers starred")
    assert entry["stage"] == "cull" and entry["description"].startswith("Smart Cull") and entry["time"]
    project.log_action("blur_triage", "Blur Triage: 45 demoted")
    assert project.stage_status == {"cull": pm.DONE, "blur_triage": pm.DONE}
    assert [e["description"] for e in project.audit_log] == ["Smart Cull: 420 keepers starred",
                                                             "Blur Triage: 45 demoted"]


def test_the_log_persists_across_save_and_reload(library, tmp_path, opened):
    project = pm.create_project("Shoot", library, tmp_path)
    project.log_action("auto_level", "Auto-Level: 380 leveled")
    project.log_action("cull", "Removed 1,200 culled photos")
    project.save_project(opened)
    again = pm.load_project(project.path)
    assert again.audit_log == project.audit_log
    assert again.stage_status == {"auto_level": pm.DONE, "cull": pm.DONE}


def test_a_damaged_log_falls_back_and_the_log_is_capped(library, tmp_path):
    path = tmp_path / "Odd.granite"
    path.write_text(json.dumps({
        "source_directories": [str(library)],
        "audit_log": [{"time": "t", "stage": "cull", "description": "kept"}, {"time": 3}, "junk"],
        "stage_status": {"cull": "Done", "made_up": "Done", "export": "maybe"},
    }), encoding="utf-8")
    project = pm.load_project(path)
    assert [e["description"] for e in project.audit_log] == ["kept"]
    assert project.stage_status == {"cull": pm.DONE}
    path.write_text(json.dumps({"source_directories": [str(library)], "audit_log": "all"}), encoding="utf-8")
    assert pm.load_project(path).audit_log == []
    for index in range(pm.AUDIT_LIMIT + 5):
        project.log_action("cull", f"step {index}")
    assert len(project.audit_log) == pm.AUDIT_LIMIT and project.audit_log[-1]["description"].endswith(
        str(pm.AUDIT_LIMIT + 4))


def test_the_checklist_follows_the_work(library, tmp_path, opened):
    project = pm.create_project("Shoot", library, tmp_path)
    rows = {stage: (status, detail) for stage, _, status, detail in project.workflow_checklist(opened)}
    assert rows["ingest"] == (pm.DONE, "7 photos")
    assert rows["cull"][0] == pm.NOT_STARTED and rows["export"][0] == pm.PENDING
    assert rows["develop"] == (pm.NOT_STARTED, "0/7 completed")

    keys = opened.keys()
    opened.set_rejected(keys[:2], True)
    for key in keys[2:4]:
        recipe = opened.recipe(key).copy()
        recipe.basic.exposure = 0.3
        opened.set_recipe(key, recipe)
    project.log_action("export", "Exported 5 images")
    rows = {stage: (status, detail) for stage, _, status, detail in project.workflow_checklist(opened)}
    assert rows["develop"] == (pm.IN_PROGRESS, "2/5 completed")
    assert rows["export"][0] == pm.DONE
    names = [name for _, name, _, _ in project.workflow_checklist(opened)]
    assert names == ["Ingest", "Cull", "Blur Triage", "Auto-Level", "Develop", "Export"]


# --------------------------------------------------------------------- #
# The window logs the batch operations
# --------------------------------------------------------------------- #

def test_the_window_logs_batches_into_the_project(harness, library, tmp_path, monkeypatch):
    w = harness.window
    project = w.create_project("Shoot", library, tmp_path)
    assert project.audit_log[-1]["stage"] == "ingest"
    keys = w.session.keys()
    w.session.set_rejected(keys[:2], True)
    monkeypatch.setattr(type(w.cull), "_confirm_remove", lambda self, n: True)
    w.cull.remove_images()
    harness.run(w.cull._on_find_blurry)
    saved = pm.load_project(project.path)
    descriptions = [e["description"] for e in saved.audit_log]
    assert "Removed 2 culled photos from the project" in descriptions
    assert any(d.startswith("Blur Triage:") for d in descriptions)
    assert saved.stage_status.get("blur_triage") == pm.DONE and saved.stage_status.get("cull") == pm.DONE


def test_work_before_a_project_is_kept_by_a_project_made_from_that_folder(harness, library, tmp_path):
    w = harness.window
    assert w.project is None
    w.log_action("auto_level", "Auto-Level: 3 leveled")
    project = w.create_project("Later", library, tmp_path)
    descriptions = [e["description"] for e in project.audit_log]
    assert descriptions[0] == "Auto-Level: 3 leveled" and project.stage_status["auto_level"] == pm.DONE
    assert descriptions[-1].startswith("Project created")


def test_the_history_drawer_shows_the_checklist_and_timeline(harness, library, tmp_path):
    w = harness.window
    assert w.history_button.text().endswith("History")
    w.log_action("cull", "Smart Cull: 5 keepers starred")
    assert w.toggle_history() and w.history_drawer.is_open and w.history_button.isChecked()
    drawer = w.history_drawer
    assert "Create a project" in drawer.note.text()
    assert "Smart Cull: 5 keepers starred" in drawer.timeline.item(0).text()
    assert drawer.rows["cull"].property("state") == pm.DONE
    assert drawer.rows["blur_triage"].property("state") == pm.NOT_STARTED

    w.create_project("Shoot", library, tmp_path)                 # refreshes while open
    assert drawer.title.text() == "Shoot" and drawer.timeline.item(0).text().endswith("photos imported")
    w.log_action("export", "Exported 4 images")
    assert drawer.timeline.item(0).text().endswith("Exported 4 images")
    assert drawer.rows["export"].property("state") == pm.DONE
    assert not w.toggle_history() and not drawer.is_open and not w.history_button.isChecked()


# --------------------------------------------------------------------- #
# Per-photo edit history
# --------------------------------------------------------------------- #

def test_changes_are_described_readably():
    base = pr.to_dict(pr.Recipe())
    brighter = pr.Recipe()
    brighter.basic.exposure = 0.5
    assert eh.describe_change(base, pr.to_dict(brighter)) == "Exposure +0.50"
    several = brighter.copy()
    several.basic.contrast = 10
    several.basic.highlights = -20
    several.geometry.rotation = 2.0
    label = eh.describe_change(pr.to_dict(brighter), pr.to_dict(several))
    assert label == "Contrast +10, Highlights −20 +1 more"
    curved = pr.Recipe()
    curved.curve.master = [[0.0, 0.0], [0.5, 0.6], [1.0, 1.0]]
    assert eh.describe_change(base, pr.to_dict(curved)) == "Tone curve"


def test_every_change_is_a_step_kept_in_the_edit_folder(library, opened):
    before = snapshot(library)
    key = opened.keys()[0]
    for exposure in (0.25, 0.75):
        recipe = opened.recipe(key).copy()
        recipe.basic.exposure = exposure
        opened.set_recipe(key, recipe)
    same = opened.recipe(key).copy()
    opened.set_recipe(key, same)                                # no change: no step
    opened.set_recipe(key, pr.Recipe())
    steps = opened.history(key).steps
    assert [s.label for s in steps] == ["Original", "Exposure +0.25", "Exposure +0.50", "Reset"]
    opened.save_all()
    stored = eh.history_path(key, opened.edit_folder)
    assert stored.parent == opened.edit_folder / "history" and stored.is_file()

    again = LibrarySession()
    again.open(library, UniversalLoader())
    assert [s.label for s in again.history(key).steps] == [s.label for s in steps]
    assert_originals_untouched(before, library)
    assert pr.version_files(opened.edit_folder) == {}, "history files are not mistaken for copies"


def test_restoring_a_step_adds_a_step_and_keeps_the_later_ones(opened):
    key = opened.keys()[1]
    for exposure in (0.5, 1.0, 1.5):
        recipe = opened.recipe(key).copy()
        recipe.basic.exposure = exposure
        opened.set_recipe(key, recipe)
    restored = opened.restore_history_step(key, None, 1)
    assert restored.basic.exposure == 0.5 and opened.recipe(key).basic.exposure == 0.5
    labels = [s.label for s in opened.history(key).steps]
    assert labels[-1] == "Back to: Exposure +0.50" and len(labels) == 5


def test_a_copy_has_its_own_history_from_the_moment_it_is_made(opened):
    key = opened.keys()[0]
    copy = opened.create_virtual_copy(key)
    assert [s.label for s in opened.history(key, copy.version).steps] == ["Virtual copy of Master"]
    opened.delete_version(key, copy.version)
    assert not eh.history_path(key, opened.edit_folder, copy.version).exists()


def test_a_damaged_history_file_is_an_empty_history(tmp_path):
    path = tmp_path / "x.json"
    path.write_text("{oops", encoding="utf-8")
    assert len(eh.EditHistory.load(path)) == 0
    path.write_text(json.dumps({"steps": [{"label": "ok", "recipe": {}, "time": "t"}, {"label": 3}]}),
                    encoding="utf-8")
    assert [s.label for s in eh.EditHistory.load(path).steps] == ["ok"]


def test_the_develop_history_panel_lists_steps_and_goes_back(harness):
    w = harness.window
    key = w.session.keys()[2]
    w.open_in_develop(key)
    develop = w.develop
    _settle(harness, develop)
    for exposure in (0.4, 0.8):
        develop.recipe.basic.exposure = exposure
        develop._commit()
    develop.recipe.basic.contrast = 25
    develop._commit()
    _settle(harness, develop)
    texts = [develop.history_list.item(i).text() for i in range(develop.history_list.count())]
    assert [t.split("   ")[0] for t in texts] == ["Original", "Exposure +0.40", "Exposure +0.40", "Contrast +25"]
    assert develop.history_list.currentRow() == 3

    develop.history_list.itemClicked.emit(develop.history_list.item(1))
    _settle(harness, develop)
    assert develop.recipe.basic.exposure == 0.4 and develop.recipe.basic.contrast == 0
    assert develop.sliders[("basic", "exposure")].value() == pytest.approx(0.4)
    assert develop.history_list.item(develop.history_list.count() - 1).text().startswith("Back to: Exposure")
    develop.undo()                                              # the restore can be undone
    _settle(harness, develop)
    assert develop.recipe.basic.contrast == 25
    assert develop.history_list.item(develop.history_list.count() - 1).text().startswith("Undo:")
