"""Granite Studios main window: a bar of task workspaces over one shared folder.

    CULL      the grid, the AI cull, ratings, rejects, survey and face loupe
    DEVELOP   one frame with every tone and colour setting
    RETOUCH   layers, masks, healing and portrait work
    DESIGN    text, shapes and watermarks over the finished frame
    EXPORT    everything that writes files
    VIDEO     the FFmpeg editor

Ctrl+1 to Ctrl+6 switch between them. The **📋 History** button opens the
project's activity drawer: the workflow checklist and a timeline of every
batch operation (see :meth:`MainWindow.log_action`). All of them look at the same
:class:`~src.engine.session.LibrarySession`, so a selection, a rating or a
rejection made in one is seen in every other, and the filmstrip along the
bottom of the editing workspaces is the same set of frames as the grid.

Older code and tests talked to a single window that had every control on
it. Attribute lookups this window does not answer itself are passed on to
the workspaces, so ``window.cull_button`` still finds the Cull workspace's
button.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QApplication, QButtonGroup, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow, QMenu, QMessageBox,
    QPushButton, QStackedWidget, QToolButton, QVBoxLayout, QWidget,
)

from src.ai_tools.background_removal import BackgroundRemover
from src.ai_tools.color_engine import ColorEngine
from src.ai_tools.culling import SmartCull
from src import __version__
from src.engine import project_manager as pm
from src.engine import session as session_topics
from src.engine import updates
from src.engine.hardware import get_hardware_summary
from src.engine.resources import resource_path
from src.engine.session import LibrarySession
from src.engine.universal_loader import UniversalLoader
from src.social.ig_sync import IgFormatter
from src.social.reel_maker import ScrlReelMaker
from src.ui.common import TaskWorker
from src.ui.theme import CRIMSON_GRANITE_STYLESHEET, NIKON_RED
from src.ui.workspaces.cull import CullWorkspace
from src.ui.workspaces.design import DesignWorkspace
from src.ui.workspaces.develop import DevelopWorkspace
from src.ui.workspaces.export import ExportWorkspace
from src.ui.workspaces.retouch import RetouchWorkspace
from src.ui.workspaces.video import VideoWorkspace

logger = logging.getLogger(__name__)

WORKSPACE_STYLE = """
QFrame#WorkspaceBar { background-color: #161616; border-bottom: 1px solid #2a2a2a; }
QPushButton#WorkspaceButton {
    background: transparent; border: none; border-bottom: 3px solid transparent;
    padding: 12px 18px; color: #8f8f94; font-size: 13px; font-weight: 700; letter-spacing: 2px;
}
QPushButton#WorkspaceButton:hover { color: #ffffff; }
QPushButton#WorkspaceButton:checked { color: #ffffff; border-bottom: 3px solid #E31B23; }
QPushButton#SectionHeader {
    background: #1b1b1b; border: none; border-radius: 6px; padding: 7px 10px;
    text-align: left; color: #e8e8ea; font-weight: 600;
}
QPushButton#SectionHeader:hover { background: #232323; }
QFrame#StatusBar { background-color: #141414; border-top: 1px solid #2a2a2a; }
QToolButton#VersionButton {
    background: transparent; border: none; color: #6a6b72; font-size: 11px; padding: 4px 10px;
}
QToolButton#VersionButton:hover { color: #ffffff; }
QToolButton#VersionButton::menu-indicator { image: none; width: 0; }
QMenuBar { background-color: #121212; color: #c8c8cc; border-bottom: 1px solid #222; }
QMenuBar::item { padding: 4px 10px; background: transparent; }
QMenuBar::item:selected { background: #2a2a2a; color: #ffffff; }
"""

class _UpdateSignals(QObject):
    """Carries an update check's answer from its thread to the window."""
    result = pyqtSignal(object, bool)     # (version on GitHub or None, asked by hand)


# Workspaces whose controls older callers may reach for, in lookup order.
_DELEGATES = ("cull", "export", "develop", "retouch", "design", "video")


def _session_field(name: str):
    return property(
        lambda self: getattr(self.session, name),
        lambda self, value: setattr(self.session, name, value),
    )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        from src import __version__

        self.setWindowTitle(f"Granite Studios v{__version__}")
        self.resize(1440, 900)
        self.setStyleSheet(CRIMSON_GRANITE_STYLESHEET + WORKSPACE_STYLE)
        icon_path = resource_path("icon.png")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Engines, created once and shared by every workspace.
        self.loader = UniversalLoader()
        self.culler = SmartCull(loader=self.loader)
        self.color_engine = ColorEngine()
        self.bg_remover = BackgroundRemover()
        self.reel_maker = ScrlReelMaker(loader=self.loader)
        self.ig_formatter = IgFormatter(loader=self.loader)
        self.session = LibrarySession()
        self._task_worker: TaskWorker | None = None
        self.available_version: str | None = None
        self.project: pm.GraniteProject | None = None
        self._recent: pm.RecentProjects | None = None
        self._pending_log: list[dict] = []        # batch operations before a project exists
        self._update_signals = _UpdateSignals(self)
        self._update_signals.result.connect(self._on_update_result)

        self.status_label = QLabel("No folder loaded. Click “Load Folder” to begin.")
        self.status_label.setObjectName("PageSubtitle")
        self.status_label.setWordWrap(True)
        self._build_ui()
        self._build_menus()
        self.session.subscribe(self._on_session_for_project)

    # -- state the workspaces share, readable as if it were the window's -------- #
    current_folder = _session_field("current_folder")
    edit_folder = _session_field("edit_folder")
    image_paths = _session_field("image_paths")
    selected = _session_field("selected")
    rejected = _session_field("rejected")
    last_cull_frames = _session_field("last_cull_frames")
    blur_reports = _session_field("blur_reports")
    web_selected = _session_field("web_selected")
    metadata = _session_field("metadata")
    labels = _session_field("labels")

    def __getattr__(self, name: str):
        # Only reached when normal lookup fails. Guarded so that a lookup
        # made before the workspaces exist cannot recurse.
        if name.startswith("__") or "workspaces" not in self.__dict__:
            raise AttributeError(name)
        for key in _DELEGATES:
            workspace = self.__dict__.get(key)
            if workspace is not None and hasattr(type(workspace), name) or (
                workspace is not None and name in workspace.__dict__
            ):
                return getattr(workspace, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    # -- layout ------------------------------------------------------------------- #

    def _build_ui(self) -> None:
        root = QWidget()
        column = QVBoxLayout(root)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        self.setCentralWidget(root)

        bar = QFrame()
        bar.setObjectName("WorkspaceBar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(14, 0, 14, 0)
        row.setSpacing(2)
        icon_path = resource_path("icon.png")
        if icon_path.exists():
            brand = QLabel()
            brand.setPixmap(QPixmap(str(icon_path)).scaled(
                28, 28, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            row.addWidget(brand)
        logo = QLabel("GRANITE STUDIOS")
        font = QFont()
        font.setPointSize(11)
        font.setBold(True)
        logo.setFont(font)
        logo.setStyleSheet("color: #ffffff; letter-spacing: 2px; padding: 0 18px 0 6px;")
        row.addWidget(logo)

        self.pages = QStackedWidget()
        self.cull = CullWorkspace(self)
        self.develop = DevelopWorkspace(self)
        self.retouch = RetouchWorkspace(self)
        self.design = DesignWorkspace(self)
        self.export = ExportWorkspace(self)
        self.video = VideoWorkspace(self)
        self.workspaces = [self.cull, self.develop, self.retouch, self.design, self.export, self.video]

        self.nav_group = QButtonGroup(self)
        self.nav_buttons: list[QPushButton] = []
        for index, workspace in enumerate(self.workspaces):
            self.pages.addWidget(workspace)
            button = QPushButton(workspace.title.upper())
            button.setObjectName("WorkspaceButton")
            button.setCheckable(True)
            button.setToolTip(f"{workspace.title}  (Ctrl+{index + 1})")
            button.clicked.connect(lambda _c=False, i=index: self.switch_to(i))
            self.nav_group.addButton(button, index)
            row.addWidget(button)
            self.nav_buttons.append(button)
            QShortcut(QKeySequence(f"Ctrl+{index + 1}"), self, activated=lambda i=index: self.switch_to(i))
        row.addStretch()
        hw = get_hardware_summary()
        engine = QLabel(f"Engine: {hw['tier']}")
        engine.setStyleSheet("color: #6a6b72; font-size: 11px; padding-right: 12px;")
        row.addWidget(engine)
        self._build_version_controls(row)
        self.history_button = QPushButton("\U0001F4CB History")
        self.history_button.setObjectName("SecondaryButton")
        self.history_button.setCheckable(True)
        self.history_button.setToolTip("The project's workflow checklist and activity timeline")
        self.history_button.clicked.connect(self.toggle_history)
        row.addWidget(self.history_button)
        self.load_button = QPushButton("Load Folder")
        self.load_button.setObjectName("ActionButton")
        self.load_button.clicked.connect(lambda: self.cull._on_load_folder())
        row.addWidget(self.load_button)
        column.addWidget(bar)
        column.addWidget(self.pages, stretch=1)

        status = QFrame()
        status.setObjectName("StatusBar")
        status_row = QHBoxLayout(status)
        status_row.setContentsMargins(16, 6, 16, 6)
        status_row.addWidget(self.status_label, stretch=1)
        column.addWidget(status)
        self._bar, self._status_bar = bar, status

        from src.ui.widgets.activity_log_drawer import ActivityLogDrawer

        self.history_drawer = ActivityLogDrawer(root)
        self.history_drawer.closed.connect(lambda: self.history_button.setChecked(False))

        self.nav_buttons[0].setChecked(True)
        self._current_index = 0

    def _build_version_controls(self, row: QHBoxLayout) -> None:
        """The version, with a menu to check for updates, and a button that
        appears when GitHub has a newer one."""
        self.version_button = QToolButton()
        self.version_button.setObjectName("VersionButton")
        self.version_button.setText(f"v{__version__}")
        self.version_button.setToolTip("Granite Studios version. Click to check for updates.")
        self.version_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.version_button)
        check_now = menu.addAction("Check for Updates Now")
        check_now.triggered.connect(lambda: self.check_for_updates(manual=True))
        self.auto_update_action = menu.addAction("Check Automatically at Startup")
        self.auto_update_action.setCheckable(True)
        self.auto_update_action.setChecked(updates.load_prefs()["auto_check"])
        self.auto_update_action.setToolTip("Once a day, read the latest version number from GitHub. "
                                           "Nothing is sent.")
        self.auto_update_action.toggled.connect(self._on_auto_update)
        self.version_button.setMenu(menu)
        row.addWidget(self.version_button)

        self.update_button = QPushButton("Update")
        self.update_button.setObjectName("ActionButton")
        self.update_button.clicked.connect(self._on_update_clicked)
        self.update_button.hide()
        row.addWidget(self.update_button)

    # -- menus ------------------------------------------------------------------- #

    def _build_menus(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        self.new_project_action = file_menu.addAction("New Project\u2026")
        self.new_project_action.setShortcut(QKeySequence("Ctrl+N"))
        self.new_project_action.triggered.connect(self.new_project)
        self.open_project_action = file_menu.addAction("Open Project\u2026")
        self.open_project_action.setShortcut(QKeySequence("Ctrl+O"))
        self.open_project_action.triggered.connect(self.open_project_dialog)
        self.save_project_action = file_menu.addAction("Save Project")
        self.save_project_action.setShortcut(QKeySequence("Ctrl+S"))
        self.save_project_action.triggered.connect(self.save_project)
        self.project_settings_action = file_menu.addAction("Project Settings\u2026")
        self.project_settings_action.setShortcut(QKeySequence("Ctrl+Shift+P"))   # Cmd+Shift+P on macOS
        self.project_settings_action.triggered.connect(self.project_settings)
        self.recent_menu = file_menu.addMenu("Recent Projects")
        self.recent_menu.aboutToShow.connect(self._fill_recent_menu)
        file_menu.addSeparator()
        open_folder = file_menu.addAction("Open Folder Without a Project\u2026")
        open_folder.triggered.connect(lambda: self.cull._on_load_folder())
        file_menu.addAction("Show Project Hub\u2026").triggered.connect(self.show_hub)
        file_menu.addSeparator()
        self.check_updates_action = file_menu.addAction("Check for Updates\u2026")
        self.check_updates_action.setToolTip("Ask GitHub whether a newer version of Granite Studios is available")
        self.check_updates_action.triggered.connect(lambda: self.check_for_updates(manual=True))
        view_menu = bar.addMenu("&View")
        self.restore_removed_action = view_menu.addAction("Restore Removed Images\u2026")
        self.restore_removed_action.triggered.connect(self.restore_removed_images)

    def _fill_recent_menu(self) -> None:
        self.recent_menu.clear()
        entries = self.recent.entries
        for entry in entries:
            action = self.recent_menu.addAction(entry.title + ("" if entry.exists else "  (not found)"))
            action.setToolTip(entry.path)
            action.setEnabled(entry.exists)
            action.triggered.connect(lambda _c=False, p=entry.path: self.open_project(p))
        if not entries:
            self.recent_menu.addAction("No recent projects").setEnabled(False)
        else:
            self.recent_menu.addSeparator()
            self.recent_menu.addAction("Clear Recent Projects").triggered.connect(self._clear_recent)

    def _clear_recent(self) -> None:
        self.recent.entries = []
        self.recent.save()

    def restore_removed_images(self) -> None:
        self.switch_to(self.cull)
        self.cull.restore_removed()

    # -- projects ---------------------------------------------------------------- #

    @property
    def recent(self) -> pm.RecentProjects:
        if self._recent is None:
            self._recent = pm.RecentProjects()
        return self._recent

    def _update_title(self) -> None:
        title = f"Granite Studios v{__version__}"
        self.setWindowTitle(f"{title} \u2014 {self.project.project_name}" if self.project else title)

    def start_up(self) -> None:
        """At launch: reopen the last project if asked to, else show the hub."""
        last = self.recent.last()
        if self.recent.reopen_last and last is not None and self.open_project(last.path):
            return
        self.show_hub()

    def show_hub(self) -> None:
        from src.ui.widgets.welcome_hub import FOLDER, NEW, OPEN, RECENT, WelcomeHub

        hub = WelcomeHub(self.recent, self, locked_path=str(self.project.path) if self.project else None)
        if not hub.exec() or hub.choice is None:
            return
        action, path = hub.choice
        if action == NEW:
            self.new_project()
        elif action == OPEN:
            self.open_project_dialog()
        elif action == RECENT and path:
            self.open_project(path)
        elif action == FOLDER:
            self.cull._on_load_folder()

    def new_project(self, folder=None) -> pm.GraniteProject | None:
        """Ask for a name, a photo folder and a place; create and open it."""
        from src.ui.widgets.welcome_hub import NewProjectDialog

        start = folder or (self.session.current_folder if self.project is None else None)
        dialog = NewProjectDialog(start, parent=self, lock_folder=folder is not None)
        if not dialog.exec():
            return None
        name, photos, location = dialog.values()
        return self.create_project(name, photos, location)

    def create_project(self, name: str, folder, location=None) -> pm.GraniteProject | None:
        try:
            project = pm.create_project(name, folder, location)
        except pm.ProjectError as exc:
            QMessageBox.warning(self, "New Project", str(exc))
            return None
        # What was done to this folder before the project existed comes with it.
        try:
            same = self.session.current_folder is not None and \
                Path(self.session.current_folder).resolve() == Path(folder).resolve()
        except OSError:
            same = False
        earlier = list(self._pending_log) if same and self.project is None else []
        if not self.open_project(project.path):
            return None
        self.project.adopt(earlier)
        self.project.log_action("ingest", f"Project created: {len(self.session.all_paths):,} photos imported")
        self.save_project(quiet=True)
        self._refresh_history_drawer()
        return self.project

    def open_project_dialog(self) -> None:
        start = pm.default_projects_folder()
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", str(start if start.exists() else Path.home()),
                                              "Granite Studios projects (*.granite)")
        if path:
            self.open_project(path)

    def open_project(self, path) -> bool:
        """Open a .granite project: its folder, its removals, where work stopped."""
        try:
            project = pm.load_project(path)
        except pm.ProjectError as exc:
            QMessageBox.warning(self, "Open Project", str(exc))
            return False
        relinked = self._locate_missing_folders(project)
        if relinked is None:
            return False
        folder = project.folder
        self._leave_project()
        self.cull.open_folder(folder)
        project.last_opened = pm._now()
        project.apply_to_session(self.session)
        self.project = project
        for workspace in self.workspaces:
            if workspace.title == project.active_workspace and workspace is not self.cull:
                self.switch_to(workspace)
                break
        self.save_project(quiet=True)
        self._update_title()
        self._refresh_history_drawer()
        self.status_label.setText(f"Opened project {project.project_name}: {self.session.total_images} photos"
                                  + (f" ({len(self.session.purged)} removed)" if self.session.purged else "") + "."
                                  + (f" Relinked {relinked:,} photos to {folder}." if relinked else ""))
        return True

    def _locate_missing_folders(self, project: pm.GraniteProject) -> int | None:
        """Offer to find each photo folder that cannot be reached. Returns
        how many photos were relinked, or None if a folder stayed missing."""
        ok, missing = project.verify_media_accessibility()
        relinked = 0
        for folder in ([] if ok else missing):
            answer = QMessageBox.question(
                self, "Locate Folder",
                f"The folder '{folder}' could not be found. Would you like to locate it now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
            chosen = ""
            if answer == QMessageBox.StandardButton.Yes:
                start = folder.parent if folder.parent.is_dir() else Path.home()
                chosen = QFileDialog.getExistingDirectory(self, f"Locate {folder.name}", str(start))
            if not chosen:
                QMessageBox.warning(self, "Open Project",
                                    f"The photos for {project.project_name} were in {folder}, which can no "
                                    "longer be found. Reconnect the drive, or move the folder back, and try again.")
                return None
            try:
                relinked += project.relink_source_folder(folder, chosen, recent=self.recent)
            except pm.ProjectError as exc:
                QMessageBox.warning(self, "Open Project", str(exc))
                return None
        return relinked

    def project_settings(self) -> bool:
        """File > Project Settings: rename the project, move its file, relink
        its folders. True if anything was saved."""
        from src.ui.dialogs.project_settings_dialog import ProjectSettingsDialog

        if self.project is None:
            self.status_label.setText("Open a project first (File \u25b8 Open Project, or Ctrl+N to make one).")
            return False
        self.save_project(quiet=True)                   # the file on disk is current before it is changed
        folder = self.project.folder
        dialog = ProjectSettingsDialog(self.project, self.recent, self)
        if not dialog.exec():
            return False
        self._update_title()
        if self.project.folder != folder:
            # The open folder moved: reopen from the file, so nothing keeps the old paths.
            path = self.project.path
            self.session.save_all()
            self.project = None
            self.open_project(path)
        else:
            self.status_label.setText(f"Saved the settings of {self.project.project_name}.")
        self._refresh_history_drawer()
        return True

    def save_project(self, quiet: bool = False) -> bool:
        """Save the open project; with none open, offer to make one from the
        open folder."""
        if self.project is None:
            if quiet:
                return False
            if not self.session.is_open and not self.session.purged:
                self.status_label.setText("Open a folder or a project first.")
                return False
            return self.new_project(folder=self.session.current_folder) is not None
        try:
            self.project.save_project(self.session, workspace=self.current_workspace.title)
        except pm.ProjectError as exc:
            if not quiet:
                QMessageBox.warning(self, "Save Project", str(exc))
            return False
        keepers = sum(1 for key in self.session.keys() if key not in self.session.rejected)
        self.recent.remember(self.project, self.session.total_images, keepers, self._project_thumbnail())
        if not quiet:
            self.status_label.setText(f"Saved project {self.project.project_name}.")
        return True

    def _project_thumbnail(self) -> Path | None:
        """A small JPEG of the selected (or first) photo for the recent list."""
        keys = self.session.keys()
        key = self.session.current if self.session.current in keys else (keys[0] if keys else None)
        pixmap = self.thumbnail_for(key) if key else None
        if pixmap is None or self.project is None:
            return None
        target = self.recent.thumbnail_path(self.project)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            small = pixmap.scaled(240, 160, Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation)
            return target if small.save(str(target), "JPG", 85) else None
        except OSError:
            return None

    def _leave_project(self) -> None:
        if self.project is not None:
            self.save_project(quiet=True)
            self.project = None
            self._update_title()

    def project_folder_changing(self, folder) -> None:
        """Cull is about to open ``folder``: a project for another folder ends."""
        if self.project is None:
            return
        current = self.project.folder
        try:
            same = current is not None and Path(folder).resolve() == current.resolve()
        except OSError:
            same = False
        if not same:
            self._leave_project()

    def _on_session_for_project(self, topic: str) -> None:
        if topic == session_topics.PURGED and self.project is not None:
            self.save_project(quiet=True)
        elif topic == session_topics.LIBRARY:
            self._pending_log = []                 # another folder: start its log afresh
            self._refresh_history_drawer()

    # -- the audit trail ------------------------------------------------------------ #

    def log_action(self, stage: str, description: str) -> dict:
        """Record a batch operation in the project's activity log and save.
        With no project open it is kept until one is made from this folder."""
        if self.project is not None:
            entry = self.project.log_action(stage, description)
            self.save_project(quiet=True)
        else:
            entry = pm.make_entry(stage, description)
            self._pending_log.append(entry)
            del self._pending_log[:-pm.AUDIT_LIMIT]
        self._refresh_history_drawer()
        return entry

    def _refresh_history_drawer(self) -> None:
        drawer = self.__dict__.get("history_drawer")
        if drawer is not None and drawer.is_open:
            drawer.refresh(self.project, self.session, self._pending_log)

    def _dock_history_drawer(self) -> None:
        drawer = self.__dict__.get("history_drawer")
        if drawer is not None:
            drawer.top, drawer.bottom = self._bar.height(), self._status_bar.height()
            drawer.reposition()

    def toggle_history(self) -> bool:
        """Slide the History drawer in or out."""
        drawer = self.history_drawer
        if not drawer.is_open:
            drawer.refresh(self.project, self.session, self._pending_log)
            self._dock_history_drawer()
        opened = drawer.toggle()
        self.history_button.setChecked(opened)
        return opened

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        self._dock_history_drawer()

    # -- updates ----------------------------------------------------------------- #

    def check_for_updates(self, manual: bool = True) -> None:
        """Ask GitHub for the latest version on a background thread."""
        if manual:
            self.status_label.setText("Checking GitHub for a newer version\u2026")
        signals = self._update_signals

        def work() -> None:
            version = updates.latest_version()
            prefs = updates.load_prefs()
            prefs["last_check"] = time.time()
            updates.save_prefs(prefs)
            try:
                signals.result.emit(version, manual)
            except RuntimeError:          # the window closed meanwhile
                pass

        threading.Thread(target=work, name="update-check", daemon=True).start()

    def check_for_updates_if_due(self) -> None:
        """The startup check: at most once a day, and only if it is on."""
        if updates.check_due(updates.load_prefs()):
            self.check_for_updates(manual=False)

    def _on_update_result(self, version, manual: bool) -> None:
        if updates.is_newer(version):
            self.available_version = version
            self.update_button.setText(f"Update to v{version}")
            self.update_button.setToolTip(f"You have v{__version__}. Install v{version} from GitHub.")
            self.update_button.show()
            if manual:
                self.status_label.setText(f"Version v{version} is available: click Update to v{version}.")
        elif manual:
            self.status_label.setText(
                f"You have the latest version (v{__version__})." if version
                else "Could not reach GitHub to check for updates. Check your connection and try again.")

    def _on_auto_update(self, checked: bool) -> None:
        prefs = updates.load_prefs()
        prefs["auto_check"] = bool(checked)
        updates.save_prefs(prefs)

    def _on_update_clicked(self) -> None:
        version = self.available_version or "the latest version"
        if not updates.can_self_update():
            updates.launch_update()
            self.status_label.setText("Opened the installer download in your browser. Run it to update.")
            return
        answer = QMessageBox.question(
            self, "Update Granite Studios",
            f"Granite Studios will close, install v{version} and reopen.\n\n"
            "Your photos and edits are untouched.")
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            started = updates.launch_update()
        except OSError as exc:
            QMessageBox.warning(self, "Update Granite Studios", f"The updater could not start: {exc}")
            return
        if started:
            self.close()

    # -- navigation ---------------------------------------------------------------- #

    def switch_to(self, index_or_workspace) -> None:
        index = (index_or_workspace if isinstance(index_or_workspace, int)
                 else self.workspaces.index(index_or_workspace))
        if index == self._current_index:
            return
        self.workspaces[self._current_index].on_leave()
        self._current_index = index
        self.pages.setCurrentIndex(index)
        self.nav_buttons[index].setChecked(True)
        self.workspaces[index].on_enter()

    def _switch_page(self, index: int) -> None:     # the 1.x name
        self.switch_to(index)

    @property
    def current_workspace(self):
        return self.workspaces[self._current_index]

    def open_in_develop(self, path) -> bool:
        """Enter or a double-click: edit a photo in Develop. A rejected or
        removed photo is locked, so it stays in the grid and a toast says
        why. Returns True if Develop was opened."""
        if path is None:
            return False
        if not self.session.is_editable(path):
            from src.ui.widgets.edit_lock import OPEN_REFUSED

            self.show_toast(OPEN_REFUSED)
            return False
        self.session.set_current(str(path))
        self.switch_to(self.develop)
        return True

    def show_toast(self, text: str) -> None:
        """A short message over the bottom of the window (and in the status bar)."""
        from src.ui.widgets.toast import Toast

        if getattr(self, "toast", None) is None:
            self.toast = Toast(self.centralWidget() or self)
        self.toast.show_message(text)
        self.status_label.setText(text)

    def delete_photos(self, keys, order=None) -> int:
        """Delete Photo: confirm, then take ``keys`` out of the workspace and
        move on to the next photo in ``order``. The files stay on disk
        (INV-1); Restore Removed Images brings them back. Returns how many."""
        from src.ui.dialogs.delete_photo_dialog import DeletePhotoDialog

        active = set(self.session.keys())
        keys = [k for k in dict.fromkeys(str(k) for k in keys) if k in active]
        if not keys:
            return 0
        dialog = DeletePhotoDialog(keys, self.thumbnail_for, self)
        if not dialog.exec():
            return 0
        deleted = self.session.delete_photos_from_workspace(keys, order)
        if deleted:
            name = Path(keys[0]).name if deleted == 1 else f"{deleted} photos"
            self.log_action("cull", f"Deleted {name} from the workspace")
            self.show_toast(f"Deleted {name} from the workspace. The file{'s are' if deleted != 1 else ' is'} "
                            f"still on disk; View ▸ Restore Removed Images brings "
                            f"{'them' if deleted != 1 else 'it'} back.")
        return deleted

    def thumbnail_for(self, key: str):
        """The grid's thumbnail for a frame, for the filmstrips."""
        cell = self.cull.cells.get(key)
        pixmap = cell.pixmap() if cell is not None else None
        return pixmap if pixmap is not None and not pixmap.isNull() else None

    # -- background tasks ----------------------------------------------------------- #

    def _run_task(self, fn, on_done, busy_button: QPushButton) -> None:
        if self._task_worker is not None and self._task_worker.isRunning():
            QMessageBox.information(self, "Busy", "Another task is still running.")
            return
        busy_button.setEnabled(False)

        def _finish(result):
            busy_button.setEnabled(True)
            on_done(result)

        def _fail(message: str):
            busy_button.setEnabled(True)
            QMessageBox.critical(self, "Task failed", message)

        self._task_worker = TaskWorker(fn, parent=self)
        self._task_worker.done.connect(_finish)
        self._task_worker.failed.connect(_fail)
        self._task_worker.start()

    def with_models(self, specs, purpose: str, then, busy_button: QPushButton) -> None:
        """Run ``then()`` once the given models are on this computer.

        If any are missing the user is told what they are for and how large
        they are, and asked. Nothing is fetched without a yes; a no leaves
        the feature unavailable rather than half-working.
        """
        from src.ai_tools import model_store

        missing = [spec for spec in specs if not model_store.is_cached(spec)]
        if not missing:
            then()
            return
        size = model_store.total_download_mb(missing)
        answer = QMessageBox.question(
            self,
            "Download a model?",
            f"{purpose} needs {len(missing)} model file(s), about {size:.0f} MB in total.\n\n"
            "They are downloaded once, checked against a known fingerprint and kept on this "
            "computer. Nothing is uploaded, and your photos stay on this machine.\n\n"
            "Download now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self.status_label.setText(f"{purpose} needs a model that was not downloaded.")
            return
        self.status_label.setText(f"Downloading {size:.0f} MB for {purpose.lower()}…")

        def work():
            return model_store.ensure_models(missing, download=True) is not None

        def done(ok):
            if ok:
                then()
            else:
                QMessageBox.warning(self, "Download failed",
                                    "The model could not be fetched. Check the connection and try again.")

        self._run_task(work, done, busy_button)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self.current_workspace.on_leave()
        if self.project is not None:
            self.save_project(quiet=True)
        self.session.save_all()
        for workspace in self.workspaces:
            workspace.shutdown()
        if self._task_worker is not None and self._task_worker.isRunning():
            self._task_worker.wait(5000)
        super().closeEvent(event)


def run() -> int:
    """Create the QApplication and run the main window."""
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = QApplication(sys.argv)
    app.setApplicationName("Granite Studios")
    app_font = QFont()
    app_font.setFamilies(["Inter", "Segoe UI", "Roboto", "sans-serif"])
    app_font.setPointSize(10)
    app.setFont(app_font)
    window = MainWindow()
    window.show()
    return app.exec()
