"""The edit history of one photograph (or one virtual copy of it).

Every finished change to a recipe -- a slider let go in Develop, a batch
grade in Cull, a mask in Retouch -- becomes a *step*: when it happened, a
short readable label ("Exposure +0.50") and the whole recipe as it stood
afterwards. The Develop History panel lists the steps; clicking one puts the
recipe back exactly as it was then, as a new step, so nothing later is lost.

Steps are kept in ``<Folder>-edit/history/``, one small JSON file per
photograph version, never beside the originals. A damaged file reads as an
empty history; only the last :data:`HISTORY_LIMIT` steps are kept.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

HISTORY_FOLDER = "history"
HISTORY_LIMIT = 100
MASTER = 0

# Readable names for each section of a recipe.
SECTION_NAMES = {
    "curve": "Tone curve",
    "hsl": "Colour (HSL)",
    "grade": "Colour grading",
    "geometry": "Crop and straighten",
    "lens": "Lens corrections",
    "detail": "Detail",
    "deblur": "De-blur",
    "look": "Look",
    "lut": "LUT",
    "layers": "Masks and layers",
    "vectors": "Overlays",
}
# Basic sliders show their change as a number; these use two decimals.
DECIMAL_SETTINGS = {"exposure"}
LABEL_PARTS = 2


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def history_path(source: str | Path, edit_folder: str | Path, version: int = MASTER) -> Path:
    """Where the history of one version of ``source`` is kept."""
    name = Path(source).name
    suffix = ".json" if version == MASTER else f".v{int(version)}.json"
    return Path(edit_folder) / HISTORY_FOLDER / f"{name}{suffix}"


def _title(attr: str) -> str:
    return attr.replace("_", " ").capitalize()


def _signed(value: float, decimals: int) -> str:
    text = f"{abs(value):.{decimals}f}"
    return ("+" if value >= 0 else "−") + text


def describe_change(before: dict | None, after: dict | None) -> str:
    """A short label for what changed between two recipe dicts."""
    before, after = before or {}, after or {}
    parts: list[str] = []
    basic_before, basic_after = before.get("basic") or {}, after.get("basic") or {}
    for attr, value in basic_after.items():
        old = basic_before.get(attr, 0.0)
        if isinstance(value, (int, float)) and isinstance(old, (int, float)) and value != old:
            decimals = 2 if attr in DECIMAL_SETTINGS else 0
            delta = float(value) - float(old)
            if round(delta, decimals) == 0:
                decimals = 2
            parts.append(f"{_title(attr)} {_signed(delta, decimals)}")
    for section, name in SECTION_NAMES.items():
        if before.get(section) != after.get(section):
            if section == "layers":
                old, new = len(before.get("layers") or []), len(after.get("layers") or [])
                if new > old:
                    name = "Added a mask" if new - old == 1 else f"Added {new - old} masks"
                elif new < old:
                    name = "Removed a mask" if old - new == 1 else f"Removed {old - new} masks"
            parts.append(name)
    if not parts:
        return "No change"
    label = ", ".join(parts[:LABEL_PARTS])
    if len(parts) > LABEL_PARTS:
        label += f" +{len(parts) - LABEL_PARTS} more"
    return label


@dataclass
class Step:
    time: str
    label: str
    recipe: dict

    def to_dict(self) -> dict:
        return {"time": self.time, "label": self.label, "recipe": self.recipe}


class EditHistory:
    """The steps of one photograph version, oldest first."""

    def __init__(self, steps: list[Step] | None = None) -> None:
        self.steps: list[Step] = list(steps or [])

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def latest(self) -> dict | None:
        return self.steps[-1].recipe if self.steps else None

    def record(self, label: str, recipe: dict, when: str | None = None) -> bool:
        """Add a step. A recipe identical to the latest one adds nothing."""
        if self.steps and self.steps[-1].recipe == recipe:
            return False
        self.steps.append(Step(when or _now(), str(label), recipe))
        del self.steps[:-HISTORY_LIMIT]
        return True

    # -- the file ----------------------------------------------------------- #

    @classmethod
    def load(cls, path: Path) -> "EditHistory":
        """Read a history file. Missing or damaged means empty."""
        try:
            data = json.loads(Path(path).read_bytes().decode("utf-8", errors="replace"))
        except (OSError, ValueError):
            return cls()
        raw = data.get("steps") if isinstance(data, dict) else None
        steps = []
        for item in raw if isinstance(raw, list) else []:
            if (isinstance(item, dict) and isinstance(item.get("recipe"), dict)
                    and isinstance(item.get("label"), str)):
                steps.append(Step(str(item.get("time") or ""), item["label"], item["recipe"]))
        return cls(steps[-HISTORY_LIMIT:])

    def save(self, path: Path) -> None:
        """Write atomically; an empty history removes the file."""
        path = Path(path)
        try:
            if not self.steps:
                if path.exists():
                    path.unlink()
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps({"version": 1, "steps": [s.to_dict() for s in self.steps]}),
                                 encoding="utf-8")
            os.replace(temporary, path)
        except OSError as exc:
            logger.warning("Could not save the edit history %s: %s", path.name, exc)


def discard(source: str | Path, edit_folder: str | Path, version: int) -> None:
    """Remove a version's history file, if it has one."""
    path = history_path(source, edit_folder, version)
    try:
        if path.exists():
            path.unlink()
    except OSError as exc:
        logger.warning("Could not remove the edit history %s: %s", path.name, exc)
