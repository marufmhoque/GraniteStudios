"""The working state of one open folder, shared by every workspace.

The interface has several workspaces looking at the same shoot. They must
agree on what is selected, what is rejected and how each frame is rated, so
that state lives here once rather than in any one of them. Moving from
culling to developing therefore keeps the selection, and a frame rejected
in one place is rejected everywhere.

What is kept between sessions
-----------------------------
Ratings, colour labels, picks, rejections, category labels and the
publish-to-web marks are written to ``session.granite.json`` in the edit
folder, keyed by filename, and read back when the folder is opened again.
Recipes are separate files (see parametric_recipe). Nothing is ever written
beside the originals.

"Rejected" keeps the meaning it has always had: the frame is left out of
everything that is exported. The file itself is never moved or deleted.

Removed photos
--------------
**Remove Images** (:meth:`LibrarySession.purge_rejected_images`) takes the
rejected photos out of the working set: the grid, the filmstrips, every
count and every export stop seeing them. It is the strongest form of
"rejected", and still only a mark: the files stay exactly where they are
(INV-1), and a removed photo keeps its rating, labels, versions and edits,
so :meth:`restore_purged_images` brings it back as it was. ``all_paths`` is
everything in the folder; ``image_paths`` is what remains in the set.

Versions
--------
A photograph can have several independent edits (see parametric_recipe,
"Versions"). One of them is *active*: that is the one every workspace shows
and edits, so switching versions in Develop is seen in Retouch and Design
too. Version names and which one is active are kept in the session file;
the copies themselves are found from their recipe files as well, so a lost
session file loses only the names. Ratings, labels, picks and rejection
belong to the photograph and are shared by all of its versions.

Virtual copies
--------------
Every copy is also shown as its own tile, right after its parent
(:meth:`LibrarySession.entries`): ``DSC_0001.jpg`` is v1 and its first copy
is ``DSC_0001_v2``. Selecting a copy's tile selects that copy for export
(``selected_copies``); choosing it makes it the active version. Only the
recipe is copied; no image data is ever duplicated on disk.

Edit history
------------
Every finished change to a recipe, from any workspace, is recorded as a step
in the photo version's history (see edit_history), kept in the edit folder
and listed by the Develop History panel.

Smart Priority sort
-------------------
:meth:`LibrarySession.sorted_keys` with :data:`SORT_SMART_PRIORITY` puts the
keepers first and sinks what the cull, blink and blur passes flagged, and the
rejected, to the bottom (see :meth:`LibrarySession.priority`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.engine import edit_history as eh
from src.engine import parametric_recipe as pr
from src.engine.parametric_recipe import Recipe

logger = logging.getLogger(__name__)

STATE_FILENAME = "session.granite.json"
COLOR_LABELS = ("Red", "Yellow", "Green", "Blue", "Purple")

# What each kind of change is called when listeners are told about it.
SELECTION, MARKS, LIBRARY, RECIPE, CURRENT = "selection", "marks", "library", "recipe", "current"
VERSIONS = "versions"           # a version was made, renamed, removed or switched to
PURGED = "purged"               # photos were removed from, or restored to, the working set

MASTER = pr.MASTER
MASTER_NAME = "Master"
NAME_LIMIT = 40

# Grid order.
SORT_CAPTURE = "capture"                # folder order
SORT_SMART_PRIORITY = "smart"           # keepers first, the culled and rejected sink
# Smart Priority ranks, best first.
PRIORITY_KEEPER_MARKED = 1              # picked, rated or AI-starred
PRIORITY_KEEPER = 2                     # an unrated keeper
PRIORITY_DUPLICATE = 3
PRIORITY_BLINK = 4
PRIORITY_BLURRY = 5
PRIORITY_REJECTED = 6
SINK_FROM = PRIORITY_DUPLICATE          # from here down: culled and flagged for removal


@dataclass(frozen=True)
class Version:
    id: int
    name: str

    @property
    def is_master(self) -> bool:
        return self.id == MASTER


@dataclass(frozen=True)
class VirtualImage:
    """One tile in the grid: a photograph's master, or one virtual copy of it."""

    path: Path
    version: int = MASTER
    name: str = MASTER_NAME

    @property
    def key(self) -> str:
        return str(self.path)

    @property
    def is_copy(self) -> bool:
        return self.version != MASTER

    @property
    def number(self) -> int:
        """The master is v1, the first copy v2."""
        return self.version + 1

    @property
    def chip(self) -> str:
        return f"v{self.number}"

    @property
    def variant_key(self) -> str:
        """``DSC_0001`` for the master, ``DSC_0001_v2`` for its first copy."""
        stem = Path(self.path).stem
        return f"{stem}_{self.chip}" if self.is_copy else stem

    @property
    def tile_id(self) -> str:
        return self.key if not self.is_copy else f"{self.key}#v{self.version}"


class LibrarySession:
    """Everything the app knows about the folder that is open."""

    def __init__(self) -> None:
        self.current_folder: Path | None = None
        self.edit_folder: Path | None = None
        self.all_paths: list[Path] = []            # every photo in the folder
        self.image_paths: list[Path] = []          # the working set: all_paths less the removed
        self.purged: set[str] = set()              # removed from the working set; files untouched
        self.selected: set[str] = set()
        self.selected_copies: set[tuple[str, int]] = set()   # (photo, copy id) tiles selected
        self.rejected: set[str] = set()
        self.picks: set[str] = set()
        self.ratings: dict[str, int] = {}
        self.color_labels: dict[str, str] = {}
        self.labels: dict[str, str] = {}           # free-text category, for the gallery
        self.web_selected: set[str] = set()
        self.metadata: dict = {}
        self.last_cull_frames: list = []
        self.blur_reports: dict = {}
        self.people = None                         # person_clustering.PeopleIndex
        self.current: str | None = None            # the frame open in Develop and Retouch
        self._recipes: dict[tuple[str, int], Recipe] = {}
        self._dirty: set[tuple[str, int]] = set()
        self._copies: dict[str, dict[int, str]] = {}   # photo -> {copy id: name}
        self._active: dict[str, int] = {}
        self._histories: dict[tuple[str, int], eh.EditHistory] = {}
        self._history_dirty: set[tuple[str, int]] = set()
        self._baseline: dict[tuple[str, int], dict] = {}   # each recipe as last recorded
        self._listeners: list[Callable[[str], None]] = []

    # -- listeners --------------------------------------------------------- #

    def subscribe(self, listener: Callable[[str], None]) -> None:
        self._listeners.append(listener)

    def notify(self, topic: str) -> None:
        for listener in list(self._listeners):
            try:
                listener(topic)
            except Exception:       # one broken view must not stop the others
                logger.exception("A session listener failed on %r", topic)

    # -- opening a folder -------------------------------------------------- #

    def open(self, folder: str | Path, loader) -> int:
        """Load a folder. Returns how many photographs it holds."""
        self.save_all()
        self.current_folder = Path(folder)
        self.edit_folder = loader.get_edit_folder(self.current_folder)
        self.all_paths = loader.scan_folder(self.current_folder)
        self.image_paths = list(self.all_paths)
        for store in (self.selected, self.rejected, self.picks, self.web_selected, self._dirty, self.purged,
                      self.selected_copies, self._history_dirty):
            store.clear()
        for store in (self.ratings, self.color_labels, self.labels, self.metadata,
                      self.blur_reports, self._recipes, self._copies, self._active,
                      self._histories, self._baseline):
            store.clear()
        self.last_cull_frames = []
        self.people = None
        self.load_state()
        self._apply_purge()
        self.current = str(self.image_paths[0]) if self.image_paths else None
        self.notify(LIBRARY)
        return len(self.image_paths)

    @property
    def is_open(self) -> bool:
        return self.current_folder is not None and bool(self.image_paths)

    def keys(self) -> list[str]:
        return [str(p) for p in self.image_paths]

    @property
    def total_images(self) -> int:
        """Photos in the working set (removed ones are not counted)."""
        return len(self.image_paths)

    # -- removing photos from the working set ------------------------------------ #

    def _apply_purge(self) -> None:
        known = {str(p) for p in self.all_paths}
        self.purged &= known
        self.image_paths = [p for p in self.all_paths if str(p) not in self.purged]

    def purged_paths(self) -> list[Path]:
        """The removed photos, in folder order."""
        return [p for p in self.all_paths if str(p) in self.purged]

    def purge_rejected_images(self) -> int:
        """Remove every rejected photo from the working set. Returns how many.

        Whatever rejected them (a manual reject, blur triage, blinks,
        duplicates, a cull to a target) they leave the grid, the filmstrips
        and every export. Nothing on disk changes: the files stay where they
        are and every mark and edit is kept, so :meth:`restore_purged_images`
        undoes it exactly.
        """
        return self.purge(key for key in self.keys() if key in self.rejected)

    def purge(self, paths) -> int:
        """Remove ``paths`` from the working set (see
        :meth:`purge_rejected_images`). Returns how many were removed."""
        active = set(self.keys())
        targets = [key for key in dict.fromkeys(str(p) for p in paths) if key in active]
        if not targets:
            return 0
        self.purged.update(targets)
        self.selected.difference_update(targets)
        self.selected_copies = {(k, v) for k, v in self.selected_copies if k not in self.purged}
        self._apply_purge()
        self.save_state()
        self.notify(PURGED)
        if self.current in self.purged:
            self.current = None
            self.set_current(self.image_paths[0] if self.image_paths else None)
        self.notify(MARKS)
        return len(targets)

    def restore_purged_images(self, paths=None) -> int:
        """Put removed photos back (all of them, or ``paths``). Returns how
        many. They come back as they were: still rejected, with their
        ratings, labels, versions and edits."""
        targets = self.purged if paths is None else {str(p) for p in paths} & self.purged
        count = len(targets)
        if not count:
            return 0
        self.purged = set() if paths is None else self.purged - targets
        self._apply_purge()
        self.save_state()
        self.notify(PURGED)
        if self.current is None and self.image_paths:
            self.set_current(self.image_paths[0])
        self.notify(MARKS)
        return count

    # -- selection and the current frame ----------------------------------- #

    def selected_paths(self) -> list[Path]:
        return [p for p in self.image_paths if str(p) in self.selected]

    def exportable(self, paths) -> list:
        """``paths`` without the rejected ones: the meaning of "rejected"."""
        return [p for p in paths if str(p) not in self.rejected]

    def set_current(self, path: str | Path | None) -> None:
        key = str(path) if path is not None else None
        if key != self.current:
            self.current = key
            self.notify(CURRENT)

    def step(self, offset: int, among: list[str] | None = None) -> str | None:
        """Move the current frame forwards or backwards. Stops at the ends."""
        keys = among if among is not None else self.keys()
        if not keys:
            return None
        try:
            index = keys.index(self.current) if self.current in keys else -1
        except ValueError:
            index = -1
        index = min(len(keys) - 1, max(0, index + offset)) if index >= 0 else 0
        self.set_current(keys[index])
        return self.current

    # -- marks --------------------------------------------------------------- #

    def set_rejected(self, paths, rejected: bool) -> None:
        for path in paths:
            (self.rejected.add if rejected else self.rejected.discard)(str(path))
            if rejected:
                self.picks.discard(str(path))
        self.notify(MARKS)

    def set_rating(self, paths, stars: int) -> None:
        stars = int(min(5, max(0, stars)))
        for path in paths:
            if stars:
                self.ratings[str(path)] = stars
            else:
                self.ratings.pop(str(path), None)
        self.notify(MARKS)

    def set_color_label(self, paths, label: str | None) -> None:
        """Apply a colour label; applying the label a frame already has clears it."""
        if label is not None and label not in COLOR_LABELS:
            raise ValueError(f"Unknown colour label {label!r}")
        keys = [str(p) for p in paths]
        clearing = label is None or all(self.color_labels.get(k) == label for k in keys)
        for key in keys:
            if clearing:
                self.color_labels.pop(key, None)
            else:
                self.color_labels[key] = label
        self.notify(MARKS)

    def set_pick(self, paths, picked: bool) -> None:
        for path in paths:
            key = str(path)
            (self.picks.add if picked else self.picks.discard)(key)
            if picked:
                self.rejected.discard(key)
        self.notify(MARKS)

    # -- versions -------------------------------------------------------------- #

    def versions(self, path: str | Path) -> list[Version]:
        """The master, then each copy in the order it was made."""
        copies = self._copies.get(str(path), {})
        return [Version(MASTER, MASTER_NAME)] + [Version(i, copies[i]) for i in sorted(copies)]

    def version_count(self, path: str | Path) -> int:
        return 1 + len(self._copies.get(str(path), {}))

    def active_version(self, path: str | Path) -> int:
        key = str(path)
        active = self._active.get(key, MASTER)
        return active if active == MASTER or active in self._copies.get(key, {}) else MASTER

    def version_name(self, path: str | Path, version: int | None = None) -> str:
        version = self.active_version(path) if version is None else version
        if version == MASTER:
            return MASTER_NAME
        try:
            return self._copies[str(path)][version]
        except KeyError:
            raise KeyError(f"{Path(path).name} has no version {version}") from None

    def display_name(self, path: str | Path, version: int | None = None) -> str:
        """``IMG_1.CR2`` for the master, ``IMG_1.CR2 (Copy 1)`` for a copy."""
        version = self.active_version(path) if version is None else version
        name = Path(path).name
        return name if version == MASTER else f"{name} ({self.version_name(path, version)})"

    def set_active_version(self, path: str | Path, version: int) -> None:
        key = str(path)
        if version != MASTER and version not in self._copies.get(key, {}):
            raise KeyError(f"{Path(key).name} has no version {version}")
        if self.active_version(key) != version:
            if version == MASTER:
                self._active.pop(key, None)
            else:
                self._active[key] = version
            self.notify(VERSIONS)

    def step_version(self, path: str | Path, offset: int) -> int:
        """Make the previous or next version active, wrapping round."""
        ids = [v.id for v in self.versions(path)]
        index = ids.index(self.active_version(path))
        self.set_active_version(path, ids[(index + offset) % len(ids)])
        return self.active_version(path)

    @staticmethod
    def _clean_name(name: str | None) -> str:
        return " ".join(str(name or "").split())[:NAME_LIMIT]

    def create_version(self, path: str | Path, name: str | None = None, recipe: Recipe | None = None,
                       source: int | None = None, activate: bool = True) -> int:
        """Branch a new version of a photograph and return its id.

        Only the settings are copied (from ``source``, by default the active
        version, or ``recipe`` when given); no pixels are duplicated. The
        new version is written with the next save, like any edit.
        """
        key = str(path)
        copies = self._copies.setdefault(key, {})
        new_id = max(copies, default=0) + 1
        taken = {n.casefold() for n in copies.values()} | {MASTER_NAME.casefold()}
        wanted = self._clean_name(name)
        if wanted.casefold() in taken:
            number = 2
            while f"{wanted} {number}".casefold() in taken:
                number += 1
            wanted = f"{wanted} {number}"
        elif not wanted:
            number = new_id
            while f"Copy {number}".casefold() in taken:
                number += 1
            wanted = f"Copy {number}"
        copies[new_id] = wanted
        origin = self.active_version(key) if source is None else source
        if recipe is None:
            recipe = self.recipe(key, origin)
        slot = (key, new_id)
        self._recipes[slot] = recipe.copy()
        self._dirty.add(slot)
        snapshot = pr.to_dict(self._recipes[slot])
        self._baseline[slot] = snapshot
        history = eh.EditHistory()
        try:
            source_name = self.version_name(key, origin)
        except KeyError:
            source_name = MASTER_NAME
        history.record(f"Virtual copy of {source_name}", snapshot)
        self._histories[slot] = history
        self._history_dirty.add(slot)
        if activate:
            self._active[key] = new_id
        self.notify(VERSIONS)
        return new_id

    def create_virtual_copy(self, source_path: str | Path, version: int | None = None) -> VirtualImage:
        """Duplicate a photo as a virtual copy and return its tile.

        The copy points at the same file; only the edit recipe of
        ``version`` (by default the active one) is copied into the copy's own
        recipe file. It becomes the active version, and is written straight
        away. No image data is ever copied.
        """
        key = str(source_path)
        new_id = self.create_version(key, source=version)
        self.save_all()
        return VirtualImage(Path(key), new_id, self.version_name(key, new_id))

    def entry(self, path: str | Path, version: int = MASTER) -> VirtualImage:
        return VirtualImage(Path(path), version, self.version_name(path, version))

    def entries(self, paths=None) -> list[VirtualImage]:
        """Every tile: each photo, followed straight away by its copies."""
        out = []
        for path in (self.image_paths if paths is None else paths):
            key = str(path)
            out.append(VirtualImage(Path(key)))
            for copy_id, name in sorted(self._copies.get(key, {}).items()):
                out.append(VirtualImage(Path(key), copy_id, name))
        return out

    def copy_count(self, paths=None) -> int:
        keys = self.keys() if paths is None else [str(p) for p in paths]
        return sum(len(self._copies.get(k, {})) for k in keys)

    def selected_entries(self) -> list[VirtualImage]:
        """The selected tiles, in grid order."""
        return [e for e in self.entries()
                if (e.key in self.selected if not e.is_copy else (e.key, e.version) in self.selected_copies)]

    def selected_photo_keys(self) -> set[str]:
        """Photos with any tile selected: the master's or a copy's."""
        return set(self.selected) | {k for k, _ in self.selected_copies}

    def create_style_versions(self, path: str | Path, styles=None, activate: bool = True) -> list[int]:
        """One copy per style, each built on the master. Returns their ids."""
        styles = list(styles or pr.VERSION_STYLES)
        master = self.recipe(path, MASTER)
        ids = [self.create_version(path, name, pr.styled(name, master), activate=False) for name in styles]
        if activate and ids:
            self.set_active_version(path, ids[0])
        return ids

    def rename_version(self, path: str | Path, version: int, name: str) -> str:
        """Rename a copy. The master keeps its name. Returns the name used."""
        key = str(path)
        if version == MASTER:
            raise ValueError("the master version keeps its name")
        copies = self._copies.get(key, {})
        if version not in copies:
            raise KeyError(f"{Path(key).name} has no version {version}")
        cleaned = self._clean_name(name)
        if not cleaned:
            raise ValueError("a version needs a name")
        if any(n.casefold() == cleaned.casefold() for i, n in copies.items() if i != version) \
                or cleaned.casefold() == MASTER_NAME.casefold():
            raise ValueError(f"another version is already called {cleaned!r}")
        copies[version] = cleaned
        self.notify(VERSIONS)
        return cleaned

    def delete_version(self, path: str | Path, version: int) -> None:
        """Remove a copy and its recipe file. The master cannot be removed,
        and the photograph itself is never touched."""
        key = str(path)
        if version == MASTER:
            raise ValueError("the master version cannot be removed")
        if version not in self._copies.get(key, {}):
            raise KeyError(f"{Path(key).name} has no version {version}")
        del self._copies[key][version]
        if not self._copies[key]:
            del self._copies[key]
        self._recipes.pop((key, version), None)
        self._dirty.discard((key, version))
        self._histories.pop((key, version), None)
        self._history_dirty.discard((key, version))
        self._baseline.pop((key, version), None)
        self.selected_copies.discard((key, version))
        if self._active.get(key) == version:
            self._active.pop(key, None)
        if self.edit_folder is not None:
            eh.discard(key, self.edit_folder, version)
            try:
                pr.discard(key, self.edit_folder, version)
            except OSError as exc:
                logger.warning("Could not remove the recipe of %s: %s", self.display_name(key, version), exc)
        self.save_state()
        self.notify(VERSIONS)

    # -- recipes --------------------------------------------------------------- #

    def _version(self, path: str | Path, version: int | None) -> int:
        return self.active_version(path) if version is None else version

    def recipe(self, path: str | Path, version: int | None = None) -> Recipe:
        """A version's recipe (the active one by default), read from its
        sidecar the first time."""
        slot = (str(path), self._version(path, version))
        if slot not in self._recipes:
            self._recipes[slot] = (
                pr.load(slot[0], self.edit_folder, slot[1]) if self.edit_folder is not None else Recipe()
            )
            self._baseline.setdefault(slot, pr.to_dict(self._recipes[slot]))
        return self._recipes[slot]

    def set_recipe(self, path: str | Path, recipe: Recipe, version: int | None = None,
                   label: str | None = None) -> None:
        """Replace a version's recipe. The change is recorded in its history,
        as ``label`` or, by default, a description of what changed."""
        slot = (str(path), self._version(path, version))
        if slot not in self._baseline:
            self.recipe(*slot)                 # what it was, for the history
        self._recipes[slot] = recipe
        self._dirty.add(slot)
        self._record(slot, label)
        self.notify(RECIPE)

    def touch(self, path: str | Path, version: int | None = None, label: str | None = None) -> None:
        """Record that a recipe was edited in place."""
        slot = (str(path), self._version(path, version))
        self._dirty.add(slot)
        if slot in self._recipes:
            self._record(slot, label)
        self.notify(RECIPE)

    # -- edit history ------------------------------------------------------------ #

    def history(self, path: str | Path, version: int | None = None) -> eh.EditHistory:
        """The recorded steps of a version (the active one by default)."""
        slot = (str(path), self._version(path, version))
        if slot not in self._histories:
            self._histories[slot] = (eh.EditHistory.load(eh.history_path(slot[0], self.edit_folder, slot[1]))
                                     if self.edit_folder is not None else eh.EditHistory())
        return self._histories[slot]

    def _record(self, slot: tuple[str, int], label: str | None) -> None:
        after = pr.to_dict(self._recipes[slot])
        neutral = pr.to_dict(Recipe())
        history = self.history(*slot)
        if not history.steps:
            before = self._baseline.get(slot) or neutral
            if before == after:
                return
            history.record("Original" if before == neutral else "Starting point", before)
        if not label:
            label = "Reset" if after == neutral else eh.describe_change(history.latest, after)
        if history.record(label, after):
            self._history_dirty.add(slot)
        self._baseline[slot] = after

    def restore_history_step(self, path: str | Path, version: int | None, index: int) -> Recipe:
        """Put a version's recipe back as it was at step ``index``. Recorded
        as a new step, so the steps after it are kept."""
        version = self._version(path, version)
        step = self.history(path, version).steps[index]
        recipe = pr.from_dict(step.recipe)
        self.set_recipe(path, recipe, version, label=f"Back to: {step.label}")
        return recipe

    def save_histories(self) -> int:
        if self.edit_folder is None:
            return 0
        written = 0
        for key, version in sorted(self._history_dirty):
            if version != MASTER and version not in self._copies.get(key, {}):
                continue
            self._histories[(key, version)].save(eh.history_path(key, self.edit_folder, version))
            written += 1
        self._history_dirty.clear()
        return written

    # -- Smart Priority sort ---------------------------------------------------- #

    def _frames_by_key(self) -> dict:
        return {str(getattr(f, "path", "")): f for f in (self.last_cull_frames or [])}

    def priority(self, path: str | Path, frames: dict | None = None) -> int:
        """Where a photo sorts under Smart Priority, 1 (best) to 6.

        1 picked, rated or AI-starred keepers; 2 unrated keepers; 3
        duplicates; 4 blinks; 5 blurry or out of focus; 6 rejected for no
        other reason. A rejected photo never counts as a keeper, whatever
        its stars.
        """
        key = str(path)
        frame = (self._frames_by_key() if frames is None else frames).get(key)
        rejected = key in self.rejected
        if not rejected and (key in self.picks or self.ratings.get(key) or getattr(frame, "is_starred", False)):
            return PRIORITY_KEEPER_MARKED
        if getattr(frame, "is_duplicate", False):
            return PRIORITY_DUPLICATE
        if getattr(frame, "eyes_closed", False):
            return PRIORITY_BLINK
        report = self.blur_reports.get(key)
        if report is not None and getattr(report, "is_blurry", False):
            return PRIORITY_BLURRY
        return PRIORITY_REJECTED if rejected else PRIORITY_KEEPER

    def sorted_keys(self, mode: str = SORT_CAPTURE, keys=None) -> list[str]:
        """``keys`` (the working set by default) in the order ``mode`` asks.

        Smart Priority is stable: within the top rank picks come first, then
        5 stars down to 1; otherwise capture order is kept.
        """
        keys = self.keys() if keys is None else [str(k) for k in keys]
        if mode != SORT_SMART_PRIORITY:
            return keys
        frames = self._frames_by_key()
        order = {k: i for i, k in enumerate(keys)}

        def rank(key: str):
            level = self.priority(key, frames)
            if level == PRIORITY_KEEPER_MARKED:
                return (level, 0 if key in self.picks else 1, -self.ratings.get(key, 0), order[key])
            return (level, 0, 0, order[key])

        return sorted(keys, key=rank)

    def sunk_count(self, keys=None) -> int:
        """How many photos Smart Priority sinks below the divider."""
        frames = self._frames_by_key()
        return sum(1 for k in (self.keys() if keys is None else keys) if self.priority(k, frames) >= SINK_FROM)

    def has_edits(self, path: str | Path, version: int | None = None) -> bool:
        slot = (str(path), self._version(path, version))
        if slot in self._recipes:
            return not self._recipes[slot].is_neutral
        return self.edit_folder is not None and pr.sidecar_path(path, self.edit_folder, slot[1]).exists()

    def save_recipes(self) -> int:
        """Write every edited recipe to its sidecar. Returns how many."""
        if self.edit_folder is None:
            return 0
        written = 0
        for key, version in sorted(self._dirty):
            if version != MASTER and version not in self._copies.get(key, {}):
                continue                      # removed since it was edited
            try:
                pr.save(self._recipes[(key, version)], key, self.edit_folder, version)
                written += 1
            except OSError as exc:
                logger.warning("Could not save the recipe for %s: %s", self.display_name(key, version), exc)
        self._dirty.clear()
        return written

    @property
    def unsaved(self) -> int:
        return len(self._dirty)

    # -- persistence ------------------------------------------------------------- #

    def _state_path(self) -> Path | None:
        return self.edit_folder / STATE_FILENAME if self.edit_folder is not None else None

    def save_state(self) -> None:
        path = self._state_path()
        if path is None:
            return
        name = lambda key: Path(key).name  # noqa: E731
        state = {
            "version": 1,
            "rejected": sorted(name(k) for k in self.rejected),
            "picks": sorted(name(k) for k in self.picks),
            "web": sorted(name(k) for k in self.web_selected),
            "purged": sorted(name(k) for k in self.purged),
            "ratings": {name(k): v for k, v in sorted(self.ratings.items())},
            "colors": {name(k): v for k, v in sorted(self.color_labels.items())},
            "labels": {name(k): v for k, v in sorted(self.labels.items())},
            "versions": {
                name(k): {"names": {str(i): n for i, n in sorted(copies.items())},
                          "active": self.active_version(k)}
                for k, copies in sorted(self._copies.items()) if copies
            },
        }
        empty = not any(state[k] for k in ("rejected", "picks", "web", "purged", "ratings", "colors",
                                           "labels", "versions"))
        try:
            if empty:
                if path.exists():
                    path.unlink()
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state, indent=1), encoding="utf-8")
            temporary.replace(path)
        except OSError as exc:
            logger.warning("Could not save the session state: %s", exc)

    def load_state(self) -> None:
        """Read saved marks. A missing or damaged file leaves everything unmarked."""
        path = self._state_path()
        by_name = {p.name: str(p) for p in self.all_paths}
        if path is None or not path.exists():
            self._load_versions({}, by_name)
            return
        try:
            state = json.loads(path.read_bytes().decode("utf-8", errors="replace"))
            if not isinstance(state, dict):
                raise ValueError("not an object")
        except (OSError, ValueError):
            logger.warning("The saved session state could not be read; starting unmarked.")
            self._load_versions({}, by_name)
            return

        def keys(field: str) -> set[str]:
            raw = state.get(field)
            return {by_name[n] for n in raw if isinstance(n, str) and n in by_name} if isinstance(raw, list) else set()

        def mapping(field: str) -> dict:
            raw = state.get(field)
            return {by_name[n]: v for n, v in raw.items() if n in by_name} if isinstance(raw, dict) else {}

        self.rejected, self.picks, self.web_selected = keys("rejected"), keys("picks"), keys("web")
        self.purged = keys("purged")
        self.ratings = {
            k: int(v) for k, v in mapping("ratings").items() if isinstance(v, int) and 1 <= v <= 5
        }
        self.color_labels = {k: v for k, v in mapping("colors").items() if v in COLOR_LABELS}
        self.labels = {k: str(v) for k, v in mapping("labels").items() if isinstance(v, str)}
        self._load_versions(mapping("versions"), by_name)

    def _load_versions(self, saved: dict, by_name: dict[str, str]) -> None:
        """Copies named in the session file, plus any found only as recipe files."""
        self._copies, self._active = {}, {}
        for key, entry in saved.items():
            names = entry.get("names") if isinstance(entry, dict) else None
            if not isinstance(names, dict):
                continue
            copies = {}
            for raw_id, raw_name in names.items():
                if str(raw_id).isdigit() and int(raw_id) > 0:
                    cleaned = self._clean_name(raw_name) if isinstance(raw_name, str) else ""
                    copies[int(raw_id)] = cleaned or f"Copy {int(raw_id)}"
            if copies:
                self._copies[key] = copies
                active = entry.get("active")
                if isinstance(active, int) and active in copies:
                    self._active[key] = active
        if self.edit_folder is not None:
            for filename, ids in pr.version_files(self.edit_folder).items():
                key = by_name.get(filename)
                if key is None:
                    continue
                copies = self._copies.setdefault(key, {})
                for version in ids:
                    copies.setdefault(version, f"Copy {version}")

    def save_all(self) -> None:
        """Write recipes and marks. Safe to call at any time, including on exit."""
        if self.edit_folder is None:
            return
        self.save_recipes()
        self.save_histories()
        self.save_state()
