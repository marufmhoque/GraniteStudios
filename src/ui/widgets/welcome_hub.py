"""The Project Hub shown at launch, and the New Project dialog.

The hub offers a new project, an existing one, or the recent list (with a
thumbnail, when it was last opened, and how many photos and keepers it
has), plus the setting that skips the hub and reopens the last project.
Choosing something closes the hub; the window does the work.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QPushButton, QVBoxLayout, QWidget,
)

from src import __version__
from src.engine import project_manager as pm
from src.engine.resources import resource_path
from src.ui.theme import CRIMSON_GRANITE_STYLESHEET

NEW, OPEN, RECENT, FOLDER = "new", "open", "recent", "folder"

HUB_STYLE = """
QFrame#HubSide { background: #161616; border-right: 1px solid #2a2a2a; }
QFrame#HubSide QLabel { background: transparent; }
QPushButton#HubAction {
    text-align: left; padding: 12px 16px; border-radius: 8px; font-size: 14px; font-weight: 600;
    background: #232326; color: #f2f3f5; border: 1px solid #2f2f33;
}
QPushButton#HubAction:hover { background: #2c2c30; border-color: #E31B23; }
QPushButton#HubLink { background: transparent; border: none; color: #8f8f94; text-align: left; padding: 4px 2px; }
QPushButton#HubLink:hover { color: #ffffff; }
QListWidget#RecentList { background: transparent; border: none; }
QListWidget#RecentList::item { padding: 8px; border-radius: 8px; margin: 2px 0; }
QListWidget#RecentList::item:selected { background: #2a2a2e; }
QListWidget#RecentList::item:hover { background: #222225; }
"""


def describe_when(stamp: str) -> str:
    """'Opened 15 Nov 2026' (or today's time) from an ISO timestamp."""
    try:
        when = datetime.fromisoformat(stamp).astimezone()
    except (TypeError, ValueError):
        return ""
    now = datetime.now().astimezone()
    if when.date() == now.date():
        return f"Opened today at {when:%H:%M}"
    return f"Opened {when.day} {when:%b %Y}"


def recent_text(entry: pm.RecentProject) -> str:
    counts = f"{entry.photo_count:,} photo{'s' if entry.photo_count != 1 else ''} · {entry.keeper_count:,} keepers"
    when = describe_when(entry.last_opened)
    missing = "" if entry.exists else "   (file not found)"
    return f"{entry.title}{missing}\n{counts}" + (f" · {when}" if when else "")


class WelcomeHub(QDialog):
    """Start a project, open one, or reopen a recent one."""

    def __init__(self, recent: pm.RecentProjects, parent=None) -> None:
        super().__init__(parent)
        self.recent = recent
        self.choice: tuple[str, str | None] | None = None
        self.setWindowTitle("Granite Studios")
        # Its own copy of the theme, so it is dark even before the window is.
        self.setStyleSheet(CRIMSON_GRANITE_STYLESHEET + HUB_STYLE)
        self.resize(900, 560)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        body = QHBoxLayout()
        body.setSpacing(0)
        outer.addLayout(body, stretch=1)

        side = QFrame()
        side.setObjectName("HubSide")
        side.setFixedWidth(300)
        left = QVBoxLayout(side)
        left.setContentsMargins(28, 30, 24, 24)
        left.setSpacing(10)
        logo = QLabel()
        icon = resource_path("icon.png")
        if icon.exists():
            logo.setPixmap(QPixmap(str(icon)).scaled(72, 72, Qt.AspectRatioMode.KeepAspectRatio,
                                                     Qt.TransformationMode.SmoothTransformation))
        left.addWidget(logo)
        title = QLabel("Granite Studios")
        title.setStyleSheet("font-size: 22px; font-weight: 800; color: #ffffff;")
        left.addWidget(title)
        version = QLabel(f"v{__version__}")
        version.setStyleSheet("color: #6a6b72;")
        left.addWidget(version)
        left.addSpacing(18)
        self.new_button = self._action("New Project…", "Start a project from a folder of photos",
                                       lambda: self._choose(NEW))
        self.open_button = self._action("Open Existing Project…", "Open a .granite project file",
                                        lambda: self._choose(OPEN))
        left.addWidget(self.new_button)
        left.addWidget(self.open_button)
        self.folder_link = QPushButton("Just open a folder, without a project")
        self.folder_link.setObjectName("HubLink")
        self.folder_link.clicked.connect(lambda: self._choose(FOLDER))
        left.addWidget(self.folder_link)
        left.addStretch()
        body.addWidget(side)

        right_host = QWidget()
        right = QVBoxLayout(right_host)
        right.setContentsMargins(26, 28, 26, 16)
        heading = QLabel("Recent Projects")
        heading.setStyleSheet("font-size: 16px; font-weight: 700; color: #f2f3f5;")
        right.addWidget(heading)
        self.list = QListWidget()
        self.list.setObjectName("RecentList")
        self.list.setIconSize(QSize(96, 64))
        self.list.itemActivated.connect(self._open_item)
        right.addWidget(self.list, stretch=1)
        self.empty_note = QLabel("Projects you open appear here.")
        self.empty_note.setObjectName("PageSubtitle")
        right.addWidget(self.empty_note)
        row = QHBoxLayout()
        row.addStretch()
        self.open_recent_button = QPushButton("Open")
        self.open_recent_button.setObjectName("ActionButton")
        self.open_recent_button.clicked.connect(lambda: self._open_item(self.list.currentItem()))
        row.addWidget(self.open_recent_button)
        right.addLayout(row)
        body.addWidget(right_host, stretch=1)

        footer = QFrame()
        footer.setStyleSheet("QFrame { border-top: 1px solid #2a2a2a; }")
        bottom = QHBoxLayout(footer)
        bottom.setContentsMargins(20, 10, 20, 10)
        self.reopen_check = QCheckBox("Always reopen last active project on startup")
        self.reopen_check.setChecked(recent.reopen_last)
        self.reopen_check.toggled.connect(self._on_reopen_toggled)
        bottom.addWidget(self.reopen_check)
        bottom.addStretch()
        outer.addWidget(footer)

        self._fill()

    def _action(self, text: str, tip: str, slot) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("HubAction")
        button.setToolTip(tip)
        button.clicked.connect(slot)
        return button

    def _fill(self) -> None:
        self.list.clear()
        for entry in self.recent.entries:
            item = QListWidgetItem(recent_text(entry))
            item.setData(Qt.ItemDataRole.UserRole, entry.path)
            item.setToolTip(entry.path)
            if entry.thumbnail and Path(entry.thumbnail).is_file():
                item.setIcon(QIcon(QPixmap(entry.thumbnail)))
            if not entry.exists:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.list.addItem(item)
        has = self.list.count() > 0
        self.empty_note.setVisible(not has)
        self.open_recent_button.setEnabled(has)
        first = next((i for i in range(self.list.count()) if self.list.item(i).flags() & Qt.ItemFlag.ItemIsEnabled), None)
        if first is not None:
            self.list.setCurrentRow(first)

    def _on_reopen_toggled(self, checked: bool) -> None:
        self.recent.reopen_last = bool(checked)
        self.recent.save()

    def _open_item(self, item) -> None:
        if item is not None and item.flags() & Qt.ItemFlag.ItemIsEnabled:
            self._choose(RECENT, item.data(Qt.ItemDataRole.UserRole))

    def _choose(self, action: str, path: str | None = None) -> None:
        self.choice = (action, path)
        self.accept()


class NewProjectDialog(QDialog):
    """Name a project, pick its photo folder and where the .granite file goes."""

    def __init__(self, folder: str | Path | None = None, name: str = "", parent=None,
                 lock_folder: bool = False) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setStyleSheet(CRIMSON_GRANITE_STYLESHEET)
        self.resize(560, 220)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit(name or (Path(folder).name if folder else ""))
        self.name_edit.setPlaceholderText("e.g. Smith Wedding")
        form.addRow("Project name", self.name_edit)
        self.folder_edit = QLineEdit(str(folder) if folder else "")
        self.folder_edit.setReadOnly(lock_folder)
        form.addRow("Photos", self._with_browse(self.folder_edit, "Choose the folder of photos",
                                                enabled=not lock_folder))
        self.location_edit = QLineEdit(str(pm.default_projects_folder()))
        form.addRow("Save project in", self._with_browse(self.location_edit, "Where to save the project file"))
        layout.addLayout(form)
        note = QLabel("The project file only points at the photos. Nothing is written beside them.")
        note.setObjectName("PageSubtitle")
        note.setWordWrap(True)
        layout.addWidget(note)
        row = QHBoxLayout()
        row.addStretch()
        cancel = QPushButton("Cancel")
        cancel.setObjectName("SecondaryButton")
        cancel.clicked.connect(self.reject)
        self.create_button = QPushButton("Create Project")
        self.create_button.setObjectName("ActionButton")
        self.create_button.clicked.connect(self.accept)
        row.addWidget(cancel)
        row.addWidget(self.create_button)
        layout.addLayout(row)
        for edit in (self.name_edit, self.folder_edit, self.location_edit):
            edit.textChanged.connect(self._validate)
        self._validate()

    def _with_browse(self, edit: QLineEdit, caption: str, enabled: bool = True) -> QWidget:
        host = QWidget()
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, stretch=1)
        button = QPushButton("Browse…")
        button.setObjectName("SecondaryButton")
        button.setEnabled(enabled)

        def browse() -> None:
            chosen = QFileDialog.getExistingDirectory(self, caption, edit.text())
            if chosen:
                edit.setText(chosen)
                if edit is self.folder_edit and not self.name_edit.text().strip():
                    self.name_edit.setText(Path(chosen).name)
        button.clicked.connect(browse)
        row.addWidget(button)
        return host

    def _validate(self) -> None:
        ok = bool(self.name_edit.text().strip()) and Path(self.folder_edit.text()).is_dir() \
            and bool(self.location_edit.text().strip())
        self.create_button.setEnabled(ok)

    def values(self) -> tuple[str, Path, Path]:
        return self.name_edit.text().strip(), Path(self.folder_edit.text()), Path(self.location_edit.text())
