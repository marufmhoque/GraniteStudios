"""Project Settings: rename a project, move its file, find its photo folders again.

Nothing happens until **Save**. The choices are staged in the dialog (the
file path and the folder table show what they will become) and applied in
one step by :meth:`GraniteProject.apply_changes`, which checks everything
first and leaves the project as it was if anything fails.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from src.engine import project_manager as pm
from src.ui.theme import CRIMSON_GRANITE_STYLESHEET

CONNECTED, MISSING = "\U0001F7E2 Connected", "\U0001F534 Missing"


class ProjectSettingsDialog(QDialog):
    """Edit a project's name, where its file lives and where its photos are."""

    def __init__(self, project: pm.GraniteProject, recent: pm.RecentProjects | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        self.project = project
        self.recent = recent
        self.destination: Path | None = None          # a staged "move the file here"
        self.relinks: dict[str, str] = {}             # a staged relink: the folder now -> where it went
        self.relinked = 0
        self.setWindowTitle("Project Settings")
        self.setStyleSheet(CRIMSON_GRANITE_STYLESHEET)
        self.resize(720, 430)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.name_edit = QLineEdit(project.project_name)
        self.name_edit.setPlaceholderText("e.g. Smith Wedding")
        self.name_edit.textChanged.connect(self._on_name_changed)
        form.addRow("Project name", self.name_edit)

        file_row = QWidget()
        row = QHBoxLayout(file_row)
        row.setContentsMargins(0, 0, 0, 0)
        self.file_label = QLabel()
        self.file_label.setWordWrap(True)
        self.file_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(self.file_label, stretch=1)
        self.move_button = QPushButton("Move…")
        self.move_button.setObjectName("SecondaryButton")
        self.move_button.setToolTip("Move the .granite project file to another folder. The photos stay where they are.")
        self.move_button.clicked.connect(self.choose_destination)
        row.addWidget(self.move_button)
        form.addRow("Project file", file_row)
        layout.addLayout(form)

        heading = QLabel("Photo folders")
        heading.setStyleSheet("font-weight: 700; color: #f2f3f5;")
        layout.addWidget(heading)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Status", "Folder", ""])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table, stretch=1)
        note = QLabel("Moved a photo folder? Relink it. Your ratings, edits and virtual copies are kept "
                      "with the folder and are found again. Nothing changes until you press Save, and "
                      "the photos themselves are never touched.")
        note.setObjectName("PageSubtitle")
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.setObjectName("SecondaryButton")
        cancel.clicked.connect(self.reject)
        self.save_button = QPushButton("Save")
        self.save_button.setObjectName("ActionButton")
        self.save_button.setDefault(True)
        self.save_button.clicked.connect(self.save)
        buttons.addWidget(cancel)
        buttons.addWidget(self.save_button)
        layout.addLayout(buttons)
        self._refresh()

    # -- what the dialog shows -------------------------------------------------- #

    def planned_path(self) -> Path:
        """Where the project file will be after Save."""
        folder = self.destination if self.destination is not None else self.project.path.parent
        return folder / (pm.safe_filename(self.name_edit.text().strip() or self.project.project_name)
                         + pm.PROJECT_SUFFIX)

    def folder_now(self, original: str) -> str:
        """A project folder as it will be after Save."""
        return self.relinks.get(original, original)

    def status_of(self, folder: str) -> str:
        return CONNECTED if Path(folder).is_dir() else MISSING

    def _refresh(self) -> None:
        planned = self.planned_path()
        moving = str(planned) != str(self.project.path)
        self.file_label.setText(str(planned) + ("   (will be moved on Save)" if moving else ""))
        if self.table.rowCount() != len(self.project.source_directories):
            self.table.setRowCount(len(self.project.source_directories))
            for row in range(self.table.rowCount()):             # the buttons are made once
                button = QPushButton("Relink Folder…")
                button.setObjectName("SecondaryButton")
                button.clicked.connect(lambda _c=False, r=row: self.choose_relink(r))
                self.table.setCellWidget(row, 2, button)
        for row, original in enumerate(self.project.source_directories):
            shown = self.folder_now(original)
            status = self.status_of(shown) + ("  (relinked on Save)" if shown != original else "")
            self.table.setItem(row, 0, QTableWidgetItem(status))
            item = QTableWidgetItem(shown)
            item.setToolTip(shown if shown == original else f"Was {original}")
            self.table.setItem(row, 1, item)
        self.table.resizeColumnToContents(0)
        self.save_button.setEnabled(bool(self.name_edit.text().strip()))

    def _on_name_changed(self, _text: str) -> None:
        self._refresh()

    # -- staging ------------------------------------------------------------------ #

    def stage_destination(self, folder: str | Path) -> None:
        self.destination = Path(folder)
        self._refresh()

    def stage_relink(self, row: int, folder: str | Path) -> None:
        original = self.project.source_directories[row]
        if str(folder) == original:
            self.relinks.pop(original, None)
        else:
            self.relinks[original] = str(folder)
        self._refresh()

    def choose_destination(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Move the project file to", str(self.planned_path().parent))
        if chosen:
            self.stage_destination(chosen)

    def choose_relink(self, row: int) -> None:
        original = self.project.source_directories[row]
        start = Path(self.folder_now(original))
        chosen = QFileDialog.getExistingDirectory(
            self, f"Locate {start.name}", str(start if start.is_dir() else start.parent))
        if chosen:
            self.stage_relink(row, chosen)

    # -- Save ------------------------------------------------------------------------ #

    def save(self) -> bool:
        """Apply the name, the file's place and the relinks together. On
        failure say why and leave the project as it was."""
        name = self.name_edit.text().strip()
        try:
            self.relinked = self.project.apply_changes(
                name=name if name != self.project.project_name else None,
                destination=self.destination,
                relinks=list(self.relinks.items()),
                recent=self.recent,
            )
        except pm.ProjectError as exc:
            QMessageBox.warning(self, "Project Settings", str(exc))
            return False
        self.accept()
        return True
