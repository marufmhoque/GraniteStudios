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

Versions
--------
A photograph can have several independent edits (see parametric_recipe,
"Versions"). One of them is *active*: that is the one every workspace shows
and edits, so switching versions in Develop is seen in Retouch and Design
too. Version names and which one is active are kept in the session file;
the copies themselves are found from their recipe files as well, so a lost
session file loses only the names. Ratings, labels, picks and rejection
belong to the photograph and are shared by all of its versions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.engine import parametric_recipe as pr
from src.engine.parametric_recipe import Recipe

logger = logging.getLogger(__name__)

STATE_FILENAME = "session.granite.json"
COLOR_LABELS = ("Red", "Yellow", "Green", "Blue", "Purple")

# What each kind of change is called when listeners are told about it.
SELECTION, MARKS, LIBRARY, RECIPE, CURRENT = "selection", "marks", "library", "recipe", "current"
VERSIONS = "versions"           # a version was made, renamed, removed or switched to

MASTER = pr.MASTER
MASTER_NAME = "Master"
NAME_LIMIT = 40


@dataclass(frozen=True)
class Version:
    id: int
    name: str

    @property
    def is_master(self) -> bool:
        return self.id == MASTER


class LibrarySession:
    """Everything the app knows about the folder that is open."""

    def __init__(self) -> None:
        self.current_folder: Path | None = None
        self.edit_folder: Path | None = None
        self.image_paths: list[Path] = []
        self.selected: set[str] = set()
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
        self.image_paths = loader.scan_folder(self.current_folder)
        for store in (self.selected, self.rejected, self.picks, self.web_selected, self._dirty):
            store.clear()
        for store in (self.ratings, self.color_labels, self.labels, self.metadata,
                      self.blur_reports, self._recipes, self._copies, self._active):
            store.clear()
        self.last_cull_frames = []
        self.people = None
        self.current = str(self.image_paths[0]) if self.image_paths else None
        self.load_state()
        self.notify(LIBRARY)
        return len(self.image_paths)

    @property
    def is_open(self) -> bool:
        return self.current_folder is not None and bool(self.image_paths)

    def keys(self) -> list[str]:
        return [str(p) for p in self.image_paths]

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
        if recipe is None:
            recipe = self.recipe(key, self.active_version(key) if source is None else source)
        self._recipes[(key, new_id)] = recipe.copy()
        self._dirty.add((key, new_id))
        if activate:
            self._active[key] = new_id
        self.notify(VERSIONS)
        return new_id

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
        if self._active.get(key) == version:
            self._active.pop(key, None)
        if self.edit_folder is not None:
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
        return self._recipes[slot]

    def set_recipe(self, path: str | Path, recipe: Recipe, version: int | None = None) -> None:
        slot = (str(path), self._version(path, version))
        self._recipes[slot] = recipe
        self._dirty.add(slot)
        self.notify(RECIPE)

    def touch(self, path: str | Path, version: int | None = None) -> None:
        """Record that a recipe was edited in place."""
        self._dirty.add((str(path), self._version(path, version)))
        self.notify(RECIPE)

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
            "ratings": {name(k): v for k, v in sorted(self.ratings.items())},
            "colors": {name(k): v for k, v in sorted(self.color_labels.items())},
            "labels": {name(k): v for k, v in sorted(self.labels.items())},
            "versions": {
                name(k): {"names": {str(i): n for i, n in sorted(copies.items())},
                          "active": self.active_version(k)}
                for k, copies in sorted(self._copies.items()) if copies
            },
        }
        empty = not any(state[k] for k in ("rejected", "picks", "web", "ratings", "colors", "labels", "versions"))
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
        by_name = {p.name: str(p) for p in self.image_paths}
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
        self.save_state()
