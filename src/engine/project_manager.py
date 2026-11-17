"""Projects: a named, reopenable shoot, and the list of recent ones.

A project is a small JSON file, ``<Project Name>.granite``, saved wherever
the user chooses (by default ``Documents/Granite Studios/Projects``). It
points at the photo folder and remembers where work stopped: which
workspace was open, which photo was selected, and which photos were removed
from the working set. It holds no pixels and no edits: ratings, marks and
recipes stay in the folder's ``-edit`` subfolder exactly as without a
project, so a folder opened on its own still shows all its work.

Nothing here writes beside the originals (INV-1). Removing a photo from a
project (``purged_paths``) hides it; the file is never touched.

A project wraps one folder for now. ``source_directories`` is a list so
the format can grow to several folders without a new version.

Audit trail
-----------
``audit_log`` records each batch operation with its time ("Smart Cull: 420
keepers starred", "Removed 1,200 culled photos"), and ``stage_status``
which workflow stages are done. :func:`workflow_checklist` turns them, with
the session, into the checklist the History drawer shows.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_SUFFIX = ".granite"
FORMAT_VERSION = 1
RECENT_LIMIT = 12
NAME_LIMIT = 80
AUDIT_LIMIT = 1000

# The workflow, in order, with the name each stage is shown under.
STAGES = {
    "ingest": "Ingest",
    "cull": "Cull",
    "blur_triage": "Blur Triage",
    "auto_level": "Auto-Level",
    "develop": "Develop",
    "export": "Export",
}
DONE, IN_PROGRESS, NOT_STARTED, PENDING = "Done", "In Progress", "Not Started", "Pending"


class ProjectError(Exception):
    """A project file that cannot be read or written, with a reason fit to show."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_entry(stage: str, description: str, when: str | None = None) -> dict:
    """One line of the audit log."""
    return {"time": when or _now(), "stage": str(stage), "description": str(description)}


def workflow_checklist(session, stage_status: dict | None = None) -> list[tuple[str, str, str, str]]:
    """``(stage, name, status, detail)`` for each workflow stage, in order.

    Cull, Blur Triage and Auto-Level are done once they have run; Develop is
    worked out from the photos (keepers with edits out of all keepers);
    Export is pending until something has been exported.
    """
    done = {k for k, v in (stage_status or {}).items() if v == DONE}
    keys = session.keys() if session is not None else []
    keepers = [k for k in keys if k not in session.rejected] if keys else []
    rows = []
    for stage, name in STAGES.items():
        detail = ""
        if stage == "ingest":
            status = DONE if keys or stage in done else NOT_STARTED
            if keys:
                detail = f"{len(keys):,} photo{'s' if len(keys) != 1 else ''}"
        elif stage == "develop":
            edited = sum(1 for k in keepers if any(session.has_edits(k, v.id) for v in session.versions(k)))
            if not edited:
                status = NOT_STARTED
            elif edited >= len(keepers):
                status = DONE
            else:
                status = IN_PROGRESS
            if keepers:
                detail = f"{edited:,}/{len(keepers):,} completed"
        elif stage == "export":
            status = DONE if stage in done else PENDING
        else:
            status = DONE if stage in done else NOT_STARTED
        rows.append((stage, name, status, detail))
    return rows


def safe_filename(name: str) -> str:
    """A project name made safe as a file name on Windows and macOS."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", name).strip().rstrip(".")
    cleaned = re.sub(r"\s+", " ", cleaned)[:NAME_LIMIT]
    return cleaned or "Untitled Project"


# Redirects the recent list and the default projects folder (the tests use
# it so they never touch a real user's files).
HOME_OVERRIDE = "GRANITE_STUDIOS_HOME"


def default_projects_folder() -> Path:
    override = os.environ.get(HOME_OVERRIDE)
    if override:
        return Path(override) / "Projects"
    return Path.home() / "Documents" / "Granite Studios" / "Projects"


def _atomic_write(path: Path, data: dict) -> None:
    """Write JSON so a crash mid-write leaves the old file, never half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(temporary, path)


@dataclass
class GraniteProject:
    project_name: str
    path: Path                                       # the .granite file itself
    created_at: str = field(default_factory=_now)
    last_opened: str = field(default_factory=_now)
    source_directories: list[str] = field(default_factory=list)
    image_manifest: dict[str, str] = field(default_factory=dict)   # photo path -> UUID
    purged_paths: list[str] = field(default_factory=list)
    active_workspace: str = "Cull"
    last_selected_image: str | None = None
    audit_log: list[dict] = field(default_factory=list)            # {time, stage, description}
    stage_status: dict[str, str] = field(default_factory=dict)     # stage -> "Done"

    # -- the photos ---------------------------------------------------------- #

    @property
    def folder(self) -> Path | None:
        return Path(self.source_directories[0]) if self.source_directories else None

    def update_manifest(self, paths) -> None:
        """Give every photo a stable ID; IDs already given are kept."""
        for path in paths:
            self.image_manifest.setdefault(str(path), str(uuid.uuid4()))

    def get_active_images(self) -> list[Path]:
        """Every photo in the project except those removed from it."""
        purged = set(self.purged_paths)
        return [Path(p) for p in self.image_manifest if p not in purged]

    # -- the session ----------------------------------------------------------- #

    def sync_from_session(self, session, workspace: str | None = None) -> None:
        """Take the session's working state into the project."""
        self.update_manifest(session.all_paths)
        self.purged_paths = [str(p) for p in session.purged_paths()]
        self.last_selected_image = session.current
        if workspace:
            self.active_workspace = workspace

    def apply_to_session(self, session) -> None:
        """After the folder is open: remove what the project had removed
        (the folder's own record is merged with the project's), and return
        to the photo that was selected."""
        session.purge(p for p in self.purged_paths if p not in session.purged)
        if self.last_selected_image in session.keys():
            session.set_current(self.last_selected_image)
        elif session.current in session.purged:
            session.set_current(session.image_paths[0] if session.image_paths else None)

    # -- the audit trail ---------------------------------------------------------- #

    def log_action(self, stage: str, description: str, when: str | None = None) -> dict:
        """Record a batch operation and mark its stage done. Saving is the
        caller's job."""
        entry = make_entry(stage, description, when)
        self.audit_log.append(entry)
        del self.audit_log[:-AUDIT_LIMIT]
        if stage in STAGES:
            self.stage_status[stage] = DONE
        return entry

    def adopt(self, entries) -> None:
        """Take log entries made before the project existed."""
        for entry in entries:
            self.log_action(entry["stage"], entry["description"], entry.get("time"))

    def workflow_checklist(self, session) -> list[tuple[str, str, str, str]]:
        return workflow_checklist(session, self.stage_status)

    # -- the file -------------------------------------------------------------- #

    def to_dict(self) -> dict:
        return {
            "format": "granite-project",
            "version": FORMAT_VERSION,
            "project_name": self.project_name,
            "created_at": self.created_at,
            "last_opened": self.last_opened,
            "source_directories": list(self.source_directories),
            "image_manifest": dict(self.image_manifest),
            "purged_paths": list(self.purged_paths),
            "active_workspace": self.active_workspace,
            "last_selected_image": self.last_selected_image,
            "audit_log": list(self.audit_log),
            "stage_status": dict(self.stage_status),
        }

    def save_project(self, session=None, workspace: str | None = None) -> Path:
        """Write the project file atomically. With ``session``, first take
        its state and write the folder's own settings files (recipes and
        marks in the ``-edit`` subfolder), so the two agree."""
        if session is not None:
            self.sync_from_session(session, workspace)
            session.save_all()
        try:
            _atomic_write(self.path, self.to_dict())
        except OSError as exc:
            raise ProjectError(f"The project could not be saved to {self.path}: {exc}") from exc
        return self.path


def create_project(name: str, base_folder: str | Path, location: str | Path | None = None) -> GraniteProject:
    """A new project for the photos in ``base_folder``, saved as
    ``<location>/<name>.granite`` (location defaults to
    :func:`default_projects_folder`)."""
    folder = Path(base_folder)
    if not folder.is_dir():
        raise ProjectError(f"{folder} is not a folder.")
    title = " ".join(str(name).split())[:NAME_LIMIT] or folder.name
    target = Path(location) if location is not None else default_projects_folder()
    project = GraniteProject(project_name=title, path=target / (safe_filename(title) + PROJECT_SUFFIX),
                             source_directories=[str(folder.resolve())])
    project.save_project()
    return project


def load_project(project_path: str | Path) -> GraniteProject:
    """Read a project file. Damaged or unknown fields fall back to defaults;
    a file that is not a project at all raises :class:`ProjectError`."""
    path = Path(project_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ProjectError(f"The project file {path.name} could not be opened: {exc}") from exc
    except ValueError as exc:
        raise ProjectError(f"{path.name} is damaged and could not be read.") from exc
    if not isinstance(data, dict) or not isinstance(data.get("source_directories"), list):
        raise ProjectError(f"{path.name} is not a Granite Studios project.")

    def text(key: str, default: str | None) -> str | None:
        value = data.get(key)
        return value if isinstance(value, str) else default

    folders = [d for d in data["source_directories"] if isinstance(d, str) and d]
    if not folders:
        raise ProjectError(f"{path.name} does not say which folder it holds.")
    manifest = data.get("image_manifest")
    purged = data.get("purged_paths")
    log = data.get("audit_log")
    stages = data.get("stage_status")
    return GraniteProject(
        project_name=(text("project_name", "") or path.stem)[:NAME_LIMIT],
        path=path,
        created_at=text("created_at", _now()),
        last_opened=text("last_opened", _now()),
        source_directories=folders,
        image_manifest={k: v for k, v in manifest.items() if isinstance(k, str) and isinstance(v, str)}
        if isinstance(manifest, dict) else {},
        purged_paths=[p for p in purged if isinstance(p, str)] if isinstance(purged, list) else [],
        active_workspace=text("active_workspace", "Cull"),
        last_selected_image=text("last_selected_image", None),
        audit_log=[make_entry(e["stage"], e["description"], e["time"]) for e in log
                   if isinstance(e, dict) and all(isinstance(e.get(k), str) for k in ("time", "stage", "description"))
                   ][-AUDIT_LIMIT:] if isinstance(log, list) else [],
        stage_status={k: v for k, v in stages.items() if k in STAGES and v == DONE}
        if isinstance(stages, dict) else {},
    )


# --------------------------------------------------------------------- #
# Recent projects
# --------------------------------------------------------------------- #

def recent_store_path() -> Path:
    """``%LOCALAPPDATA%/Granite Studios/recent_projects.json`` on Windows,
    ``~/.config/granite-studios/recent_projects.json`` elsewhere."""
    override = os.environ.get(HOME_OVERRIDE)
    if override:
        return Path(override) / "recent_projects.json"
    if platform.system() == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Granite Studios"
    else:
        base = Path.home() / ".config" / "granite-studios"
    return base / "recent_projects.json"


@dataclass
class RecentProject:
    title: str
    path: str
    thumbnail: str | None = None        # a small JPEG proxy beside the recent list
    photo_count: int = 0
    keeper_count: int = 0
    last_opened: str = ""

    @property
    def exists(self) -> bool:
        return Path(self.path).is_file()


class RecentProjects:
    """The recent-projects list and the "reopen the last project" setting."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or recent_store_path()
        self.entries: list[RecentProject] = []
        self.reopen_last = False
        self._load()

    @property
    def thumbnail_folder(self) -> Path:
        return self.path.parent / "project-thumbnails"

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        self.reopen_last = data.get("reopen_last") is True
        for raw in data.get("projects", []) if isinstance(data.get("projects"), list) else []:
            if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
                continue
            count = lambda key: raw.get(key) if isinstance(raw.get(key), int) and raw.get(key) >= 0 else 0  # noqa: E731
            self.entries.append(RecentProject(
                title=str(raw.get("title") or Path(raw["path"]).stem)[:NAME_LIMIT],
                path=raw["path"],
                thumbnail=raw.get("thumbnail") if isinstance(raw.get("thumbnail"), str) else None,
                photo_count=count("photo_count"),
                keeper_count=count("keeper_count"),
                last_opened=str(raw.get("last_opened") or ""),
            ))
        self.entries = self.entries[:RECENT_LIMIT]

    def save(self) -> None:
        data = {
            "reopen_last": self.reopen_last,
            "projects": [vars(entry) for entry in self.entries[:RECENT_LIMIT]],
        }
        try:
            _atomic_write(self.path, data)
        except OSError as exc:
            logger.warning("Could not save the recent projects list: %s", exc)

    def remember(self, project: GraniteProject, photo_count: int, keeper_count: int,
                 thumbnail: str | Path | None = None) -> RecentProject:
        """Put a project at the top of the list."""
        key = str(project.path)
        old = next((e for e in self.entries if e.path == key), None)
        entry = RecentProject(
            title=project.project_name, path=key,
            thumbnail=str(thumbnail) if thumbnail else (old.thumbnail if old else None),
            photo_count=photo_count, keeper_count=keeper_count, last_opened=_now(),
        )
        self.entries = [entry] + [e for e in self.entries if e.path != key]
        self.entries = self.entries[:RECENT_LIMIT]
        self.save()
        return entry

    def forget(self, path: str | Path) -> None:
        self.entries = [e for e in self.entries if e.path != str(path)]
        self.save()

    def last(self) -> RecentProject | None:
        """The most recently opened project that still exists."""
        return next((e for e in self.entries if e.exists), None)

    def thumbnail_path(self, project: GraniteProject) -> Path:
        import hashlib

        digest = hashlib.sha1(str(project.path).encode("utf-8")).hexdigest()[:16]
        return self.thumbnail_folder / f"{digest}.jpg"
