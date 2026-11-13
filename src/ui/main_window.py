"""Granite Studios main window: a bar of task workspaces over one shared folder.

    CULL      the grid, the AI cull, ratings, rejects, survey and face loupe
    DEVELOP   one frame with every tone and colour setting
    RETOUCH   layers, masks, healing and portrait work
    DESIGN    text, shapes and watermarks over the finished frame
    EXPORT    everything that writes files
    VIDEO     the FFmpeg editor

Ctrl+1 to Ctrl+6 switch between them. All of them look at the same
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
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QIcon, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QApplication, QButtonGroup, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QStackedWidget, QVBoxLayout, QWidget,
)

from src.ai_tools.background_removal import BackgroundRemover
from src.ai_tools.color_engine import ColorEngine
from src.ai_tools.culling import SmartCull
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
"""

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
        self.setWindowTitle("Granite Studios")
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

        self.status_label = QLabel("No folder loaded. Click “Load Folder” to begin.")
        self.status_label.setObjectName("PageSubtitle")
        self.status_label.setWordWrap(True)
        self._build_ui()

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

        self.nav_buttons[0].setChecked(True)
        self._current_index = 0

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

    def open_in_develop(self, path) -> None:
        if path is None:
            return
        self.session.set_current(str(path))
        self.switch_to(self.develop)

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

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self.current_workspace.on_leave()
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
