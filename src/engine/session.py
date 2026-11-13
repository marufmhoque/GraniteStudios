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
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

from src.engine import parametric_recipe as pr
from src.engine.parametric_recipe import Recipe

logger = logging.getLogger(__name__)

STATE_FILENAME = "session.granite.json"
COLOR_LABELS = ("Red", "Yellow", "Green", "Blue", "Purple")

# What each kind of change is called when listeners are told about it.
SELECTION, MARKS, LIBRARY, RECIPE, CURRENT = "selection", "marks", "library", "recipe", "current"


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
        self._recipes: dict[str, Recipe] = {}
        self._dirty: set[str] = set()
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
                      self.blur_reports, self._recipes):
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

    # -- recipes --------------------------------------------------------------- #

    def recipe(self, path: str | Path) -> Recipe:
        """The recipe for a photograph, read from its sidecar the first time."""
        key = str(path)
        if key not in self._recipes:
            self._recipes[key] = (
                pr.load(path, self.edit_folder) if self.edit_folder is not None else Recipe()
            )
        return self._recipes[key]

    def set_recipe(self, path: str | Path, recipe: Recipe) -> None:
        key = str(path)
        self._recipes[key] = recipe
        self._dirty.add(key)
        self.notify(RECIPE)

    def touch(self, path: str | Path) -> None:
        """Record that a recipe was edited in place."""
        self._dirty.add(str(path))
        self.notify(RECIPE)

    def has_edits(self, path: str | Path) -> bool:
        key = str(path)
        if key in self._recipes:
            return not self._recipes[key].is_neutral
        return self.edit_folder is not None and pr.sidecar_path(path, self.edit_folder).exists()

    def save_recipes(self) -> int:
        """Write every edited recipe to its sidecar. Returns how many."""
        if self.edit_folder is None:
            return 0
        written = 0
        for key in sorted(self._dirty):
            try:
                pr.save(self._recipes[key], key, self.edit_folder)
                written += 1
            except OSError as exc:
                logger.warning("Could not save the recipe for %s: %s", Path(key).name, exc)
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
        }
        empty = not any(state[k] for k in ("rejected", "picks", "web", "ratings", "colors", "labels"))
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
        if path is None or not path.exists():
            return
        try:
            state = json.loads(path.read_bytes().decode("utf-8", errors="replace"))
            if not isinstance(state, dict):
                return
        except (OSError, ValueError):
            logger.warning("The saved session state could not be read; starting unmarked.")
            return
        by_name = {p.name: str(p) for p in self.image_paths}

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

    def save_all(self) -> None:
        """Write recipes and marks. Safe to call at any time, including on exit."""
        if self.edit_folder is None:
            return
        self.save_recipes()
        self.save_state()
