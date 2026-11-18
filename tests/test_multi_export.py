"""The person-based export queue: who is in which photo, bundles, and running them.

MX-01  ANY / ALL / SOLO / ALL_PROJECT pick the right photos
MX-02  removed and rejected photos are never picked
MX-03  bundles land in their own subfolders; a shared photo is rendered once
MX-04  originals are never modified, renamed or written beside (INV-1)
MX-05  the queue builder: cards, counts, the picker, running, saving
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest

from src.ai_tools import person_clustering as pc
from src.ai_tools.person_clustering import FaceRef, PeopleIndex
from src.engine import export_presets as ep
from src.engine.export_presets import ExportBundle, MultiExportRunner
from src.engine.session import LibrarySession
from src.engine.universal_loader import UniversalLoader
from tests.conftest import assert_originals_untouched, pump, snapshot

JOHN, JANE, SAM = 0, 1, 2


def _index(folder: Path, extra_stranger_in: str | None = None) -> PeopleIndex:
    """Photo 1: John. Photo 2: Jane. Photo 3: John and Jane. Photo 4: Sam."""
    def ref(name, box=(40, 40, 120, 120), face=0):
        return FaceRef(str(folder / name), face, box)

    index = PeopleIndex(
        people={
            JOHN: [ref("IMG_0001.jpg"), ref("IMG_0003.jpg")],
            JANE: [ref("IMG_0002.jpg"), ref("IMG_0003.jpg", (300, 40, 120, 120), 1)],
            SAM: [ref("IMG_0004.jpg")],
        },
        names={JOHN: "John", JANE: "Jane", SAM: "Sam"},
    )
    if extra_stranger_in:
        index.unassigned.append(ref(extra_stranger_in, (400, 200, 60, 60), 1))
    return index


def _names(paths) -> list[str]:
    return [Path(p).name for p in paths]


def _session(library: Path) -> LibrarySession:
    session = LibrarySession()
    session.open(library, UniversalLoader())
    session.people = _index(library)
    return session


# --------------------------------------------------------------------- #
# MX-01, MX-02: the person query
# --------------------------------------------------------------------- #

def test_any_all_and_all_project_pick_the_right_photos(library):
    index = _index(library)
    photos = [library / f"IMG_000{i}.jpg" for i in range(1, 5)]
    assert _names(index.get_images_for_people([JOHN, JANE], "ANY")) == ["IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg"]
    assert _names(index.get_images_for_people([JOHN, JANE], "ALL")) == ["IMG_0003.jpg"]
    assert _names(index.get_images_for_people([], "ALL_PROJECT", all_paths=photos)) == _names(photos)
    assert _names(index.get_images_for_people([JOHN], "ALL_PROJECT")) == _names(photos), "the index's own photos"


def test_person_ids_may_be_strings(library):
    index = _index(library)
    assert _names(index.get_images_for_people(["0", "1"], "ALL")) == ["IMG_0003.jpg"]
    assert _names(index.get_images_for_people(["2"], "ANY")) == ["IMG_0004.jpg"]


def test_solo_means_one_person_alone(library):
    index = _index(library, extra_stranger_in="IMG_0004.jpg")
    assert _names(index.get_images_for_people([JOHN], "SOLO")) == ["IMG_0001.jpg"], "photo 3 has Jane in it too"
    assert _names(index.get_images_for_people([JOHN, JANE], "SOLO")) == ["IMG_0001.jpg", "IMG_0002.jpg"]
    assert index.get_images_for_people([SAM], "SOLO") == [], "an unrecognised face is somebody else"


def test_nobody_chosen_matches_nothing_and_a_bad_mode_is_refused(library):
    index = _index(library)
    assert index.get_images_for_people([], "ANY") == []
    with pytest.raises(ValueError):
        index.get_images_for_people([JOHN], "SOMETIMES")


def test_the_working_set_orders_and_limits_the_result(library):
    index = _index(library)
    order = [library / "IMG_0003.jpg", library / "IMG_0001.jpg"]          # photo 2 is not in it
    assert _names(index.get_images_for_people([JOHN, JANE], "ANY", all_paths=order)) == ["IMG_0003.jpg", "IMG_0001.jpg"]


def test_purged_photos_are_left_out(library):
    index = _index(library)
    purged = {str(library / "IMG_0003.jpg")}
    assert _names(index.get_images_for_people([JOHN, JANE], "ANY", exclude=purged)) == ["IMG_0001.jpg", "IMG_0002.jpg"]
    assert "IMG_0003.jpg" not in _names(index.get_images_for_people([], "ALL_PROJECT", exclude=purged))


def test_the_saved_manifest_is_read(library, tmp_path):
    edit = tmp_path / "edit"
    _index(library).save(edit)
    assert _names(pc.get_images_for_people(edit, ["0", "1"], "ALL")) == ["IMG_0003.jpg"]
    assert pc.get_images_for_people(tmp_path / "nothing-here", ["0"], "ANY") == []


def test_the_session_leaves_out_rejected_and_removed_photos(library):
    session = _session(library)
    assert _names(session.images_for_people(["0", "1"], "ANY")) == ["IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg"]
    session.set_rejected([library / "IMG_0002.jpg"], True)
    session.purge([str(library / "IMG_0001.jpg")])
    assert _names(session.images_for_people(["0", "1"], "ANY")) == ["IMG_0003.jpg"]
    everyone = _names(session.images_for_people([], "ALL_PROJECT"))
    assert "IMG_0001.jpg" not in everyone and "IMG_0002.jpg" not in everyone and "IMG_0007.png" in everyone


def test_everyone_works_before_anyone_is_recognised(library):
    session = LibrarySession()
    session.open(library, UniversalLoader())
    assert len(session.images_for_people([], "ALL_PROJECT")) == 7
    assert session.images_for_people(["0"], "ANY") == []


# --------------------------------------------------------------------- #
# Bundles
# --------------------------------------------------------------------- #

def test_a_bundle_with_nobody_in_it_is_everyone():
    bundle = ExportBundle("Everyone", [], "ANY")
    assert bundle.mode == "ALL_PROJECT"
    assert ExportBundle("Sam", ["2"], "SOLO").mode == "SOLO"


def test_bundle_names_and_folders():
    assert ep.bundle_title(["John", "Jane"]) == "John & Jane"
    assert ep.bundle_title(["Ann", "Bo", "Cy"]) == "Ann, Bo & Cy"
    assert ep.bundle_title([]) == "Everyone"
    assert ep.folder_slug("John & Jane") == "John_and_Jane"
    assert ep.folder_slug('Sam: "VIP" / Joe') == "Sam_VIP_Joe"
    assert ExportBundle("John & Jane VIP").subfolder == "John_and_Jane_VIP"


def test_the_queue_presets_are_known_by_their_names():
    for name, key in (("Web JPEG", "web"), ("Full-Size JPEG", "full"), ("16-bit TIFF", "tiff16"),
                      ("Print 300 DPI", "print"), ("Instagram 4:5", "ig45"), ("web", "web")):
        assert ep.preset_key(name) == key
    with pytest.raises(KeyError):
        ep.preset_key("Polaroid")


def test_the_queue_is_saved_with_the_folder(tmp_path):
    bundles = [ExportBundle("John & Jane", ["0", "1"], "ALL", "Print 300 DPI", "signature", "John_and_Jane", "M. Hoque"),
               ExportBundle()]
    ep.save_queue(tmp_path, bundles, tmp_path / "out")
    loaded, base = ep.load_queue(tmp_path)
    assert loaded == bundles and base == str(tmp_path / "out")
    (tmp_path / ep.QUEUE_FILENAME).write_text("{ not json", encoding="utf-8")
    assert ep.load_queue(tmp_path) == ([], "")


def test_workers_follow_the_ram_budget():
    assert ep.export_workers(8, budget_gb=0.5) == 1
    assert ep.export_workers(16, budget_gb=5.0) <= ep.export_workers(8, budget_gb=5.0)
    assert ep.export_workers(8, budget_gb=64) <= 6


# --------------------------------------------------------------------- #
# MX-03, MX-04: running the queue
# --------------------------------------------------------------------- #

def test_two_bundles_land_in_their_own_folders(library, tmp_path):
    before = snapshot(library)
    session = _session(library)
    exports = tmp_path / "Exports"
    calls = []
    runner = MultiExportRunner(session, workers=2)
    result = runner.run_bundles(
        [ExportBundle("John", ["0"], "ANY", "Web JPEG"), ExportBundle("Sam", ["2"], "ANY", "Web JPEG")],
        exports, lambda name, done, total: calls.append((name, done, total)))
    assert sorted(p.name for p in (exports / "John").iterdir()) == ["IMG_0001_web.jpg", "IMG_0003_web.jpg"]
    assert sorted(p.name for p in (exports / "Sam").iterdir()) == ["IMG_0004_web.jpg"]
    assert result.written == 3 and not result.failures
    assert calls == [("John", 1, 3), ("John", 2, 3), ("Sam", 3, 3)]
    assert runner.progress.describe() == "Exporting Bundle 2 of 2 (Sam): photo 1 of 1…"
    assert cv2.imread(str(exports / "Sam" / "IMG_0004_web.jpg")) is not None
    assert_originals_untouched(before, library)
    assert sorted(p.name for p in library.iterdir() if p.is_file()) == sorted(before), "nothing new beside the photos"


def test_a_photo_two_bundles_share_is_rendered_once(library, tmp_path, monkeypatch):
    session = _session(library)
    rendered = []
    real = ep.render_frame
    monkeypatch.setattr(ep, "render_frame", lambda job, bits=8: rendered.append(job.path.name) or real(job, bits))
    result = MultiExportRunner(session, workers=2).run_bundles(
        [ExportBundle("Everyone", [], "ALL_PROJECT", "web"),
         ExportBundle("John & Jane", ["0", "1"], "ALL", "web"),
         ExportBundle("John print", ["0"], "ANY", "print")],
        tmp_path / "out")
    assert sorted(rendered) == sorted(_names(session.keepers()) + ["IMG_0001.jpg", "IMG_0003.jpg"]), \
        "the shared photo is copied for the same preset, rendered again for another"
    assert result.copied == 1 and result.rendered == 9
    copied = tmp_path / "out" / "John_and_Jane" / "IMG_0003_web.jpg"
    assert copied.read_bytes() == (tmp_path / "out" / "Everyone" / "IMG_0003_web.jpg").read_bytes()
    assert (tmp_path / "out" / "John_print" / "IMG_0003_print.jpg").is_file()


def test_a_watermark_is_part_of_what_makes_two_renders_the_same(qapp, library, tmp_path):
    # qapp: a watermark's text is drawn with Qt's fonts, as it is in the app.
    session = _session(library)
    result = MultiExportRunner(session, workers=1).run_bundles(
        [ExportBundle("A", ["2"], "ANY", "web"),
         ExportBundle("B", ["2"], "ANY", "web", "exif_strip")], tmp_path / "out")
    assert result.copied == 0 and result.rendered == 2


def test_an_unreadable_photo_does_not_stop_the_queue(library, tmp_path):
    session = _session(library)
    session.people.people[SAM].append(FaceRef(str(library / "IMG_0006.jpg"), 0, (0, 0, 10, 10)))
    (library / "IMG_0006.jpg").write_bytes(b"not a photo")
    result = MultiExportRunner(session, workers=2).run_bundles(
        [ExportBundle("Sam", ["2"], "ANY", "web"), ExportBundle("John", ["0"], "ANY", "web")], tmp_path / "out")
    assert [p.name for p, _ in result.failures] == ["IMG_0006.jpg"]
    assert result.written == 3


def test_a_bundle_never_writes_into_the_photo_folder(library, tmp_path):
    before = snapshot(library)
    session = _session(library)
    result = MultiExportRunner(session).run_bundles(
        [ExportBundle("Into the shoot", [], "ALL_PROJECT", "Full-Size JPEG", destination_subfolder=library.name),
         ExportBundle("Sam", ["2"], "ANY", "Full-Size JPEG")],
        library.parent)
    assert len(result.failures) == 7 and "photo folder" in result.failures[0][1]
    assert (library.parent / "Sam" / "IMG_0004.jpg").is_file()
    assert_originals_untouched(before, library)
    assert sorted(p.name for p in library.iterdir() if p.is_file()) == sorted(before)


def test_the_instagram_preset_is_exactly_4_by_5(library, tmp_path):
    session = _session(library)
    MultiExportRunner(session).run_bundles([ExportBundle("Sam", ["2"], "ANY", "Instagram 4:5")], tmp_path)
    image = cv2.imread(str(tmp_path / "Sam" / "IMG_0004_ig.jpg"))
    assert image.shape[:2] == (1350, 1080)


def test_an_edit_is_exported_and_the_original_is_not_touched(library, tmp_path):
    before = snapshot(library)
    session = _session(library)
    recipe = session.recipe(library / "IMG_0004.jpg")
    recipe.basic.exposure = 1.0
    session.set_recipe(library / "IMG_0004.jpg", recipe)
    MultiExportRunner(session).run_bundles([ExportBundle("Sam", ["2"], "ANY", "web")], tmp_path)
    edited = cv2.imread(str(tmp_path / "Sam" / "IMG_0004_web.jpg")).mean()
    original = cv2.imread(str(library / "IMG_0004.jpg")).mean()
    assert edited > original + 10
    assert_originals_untouched(before, library)


# --------------------------------------------------------------------- #
# MX-05: the queue builder in the Export workspace
# --------------------------------------------------------------------- #

def _queue(harness):
    w = harness.window
    w.session.people = _index(harness.folder)
    w.switch_to(w.export)
    queue = w.export.queue
    queue.load_for_folder()
    return w, queue


def test_the_queue_starts_with_everyone(harness):
    w, queue = _queue(harness)
    assert len(queue.cards) == 1
    card = queue.cards[0]
    assert card.bundle.mode == "ALL_PROJECT" and card.title_edit.text() == "Everyone"
    assert "Matches: 7 photos" in card.badge.text()
    assert "7 total files across 1 bundle" in queue.run_button.text()
    assert not card.mode_buttons["ALL"].isEnabled(), "a mode for people needs people"


def test_choosing_people_names_the_bundle_and_counts_its_photos(harness):
    w, queue = _queue(harness)
    card = queue.add_bundle()
    card.set_people(["0", "1"])
    assert card.bundle.match_mode == "ANY" and card.title_edit.text() == "John & Jane"
    assert card.folder_edit.text() == "John_and_Jane"
    assert "Matches: 3 photos" in card.badge.text()
    card.mode_buttons["ALL"].click()
    assert card.bundle.match_mode == "ALL" and "Matches: 1 photo" in card.badge.text()
    card.mode_buttons["SOLO"].click()
    assert "Matches: 2 photos" in card.badge.text()
    assert "9 total files across 2 bundles" in queue.run_button.text()
    chips = [card.chips.itemAt(i).widget().text() for i in range(card.chips.count())]
    assert any("John" in c for c in chips) and any("Jane" in c for c in chips)

    card.title_edit.setText("VIP")
    card.title_edit.textEdited.emit("VIP")
    card.set_people(["0"])
    assert card.bundle.bundle_name == "VIP", "a typed title is kept"
    assert card.bundle.destination_subfolder == "VIP"


def test_rejecting_a_photo_updates_the_counts(harness):
    w, queue = _queue(harness)
    harness.reject("IMG_0005.jpg")
    assert "Matches: 6 photos" in queue.cards[0].badge.text()


def test_the_people_picker_searches_and_ticks(harness):
    from PyQt6.QtCore import Qt

    w, queue = _queue(harness)
    card = queue.add_bundle()
    picker = card.open_picker()
    names = [picker.list.item(i).data(Qt.ItemDataRole.UserRole + 1) for i in range(picker.list.count())]
    assert set(names) == {"John", "Jane", "Sam"}
    picker.search.setText("sa")
    visible = [picker.list.item(i).data(Qt.ItemDataRole.UserRole + 1)
               for i in range(picker.list.count()) if not picker.list.item(i).isHidden()]
    assert visible == ["Sam"]
    sam = next(picker.list.item(i) for i in range(picker.list.count()) if not picker.list.item(i).isHidden())
    sam.setCheckState(Qt.CheckState.Checked)
    assert card.bundle.person_ids == ["2"] and card.title_edit.text() == "Sam"
    picker.close()


def test_removing_a_bundle(harness):
    w, queue = _queue(harness)
    card = queue.add_bundle()
    card.remove_button.click()
    assert card not in queue.cards and len(queue.cards) == 1


def test_run_all_exports_every_bundle_into_its_folder(harness, library):
    before = snapshot(library)
    w, queue = _queue(harness)
    queue.cards[0].preset_combo.setCurrentIndex(queue.cards[0].preset_combo.findData("web"))
    john = queue.add_bundle()
    john.set_people(["0"])
    sam = queue.add_bundle()
    sam.set_people(["2"])
    queue.set_base_dir(harness.out_dir)
    progress = []
    queue._signals.progress.connect(lambda text, done, total: progress.append((text, done, total)))
    harness.run(queue.run_all)
    pump(harness.app, lambda: queue.last_result is not None)
    result = queue.last_result
    assert result.written == 10 and result.copied == 3 and not result.failures
    assert sorted(p.name for p in (harness.out_dir / "John").iterdir()) == ["IMG_0001_web.jpg", "IMG_0003_web.jpg"]
    assert sorted(p.name for p in (harness.out_dir / "Sam").iterdir()) == ["IMG_0004_web.jpg"]
    assert len(list((harness.out_dir / "Everyone").iterdir())) == 7
    assert progress[-1][1:] == (10, 10) and progress[-1][0].startswith("Exporting Bundle 3 of 3 (Sam)")
    assert "Exported 10 file(s) across 3 bundle(s)" in queue.queue_status.text()
    assert any("Export queue: 10 files" in e["description"] for e in (w.project.audit_log if w.project else [])) \
        or w.project is None
    assert_originals_untouched(before, library)


def test_the_queue_comes_back_with_the_folder(harness):
    w, queue = _queue(harness)
    card = queue.add_bundle()
    card.set_people(["0", "1"])
    card.mode_buttons["ALL"].click()
    queue.set_base_dir(harness.out_dir)
    queue.save()
    saved = json.loads((w.edit_folder / ep.QUEUE_FILENAME).read_text(encoding="utf-8"))
    assert [b["bundle_name"] for b in saved["bundles"]] == ["Everyone", "John & Jane"]
    queue._loaded_for = None                       # as if the folder were opened again
    queue.load_for_folder()
    assert [c.bundle.bundle_name for c in queue.cards] == ["Everyone", "John & Jane"]
    assert queue.cards[1].bundle.match_mode == "ALL" and queue.base_dir == str(harness.out_dir)


def test_a_signature_needs_a_name(harness):
    w, queue = _queue(harness)
    card = queue.cards[0]
    card.watermark_combo.setCurrentIndex(card.watermark_combo.findData("signature"))
    queue.set_base_dir(harness.out_dir)
    assert not queue.run_all()
    assert any("watermark" in m.lower() for _, m in harness.messages)
