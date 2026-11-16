"""Restore Removed Images: pick removed photos to bring back into the project."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout,
)


class RestoreRemovedDialog(QDialog):
    """Every removed photo, ticked; untick any to leave out."""

    def __init__(self, paths: list[str], thumbnail: Callable[[str], object] | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Restore Removed Images")
        self.resize(560, 520)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        heading = QLabel(f"{len(paths)} photo{'s' if len(paths) != 1 else ''} removed from the project")
        heading.setStyleSheet("font-size: 16px; font-weight: 700;")
        layout.addWidget(heading)
        note = QLabel("Ticked photos come back into the grid with their ratings and edits. "
                      "They are still marked rejected until you keep them.")
        note.setWordWrap(True)
        note.setObjectName("PageSubtitle")
        layout.addWidget(note)

        self.list = QListWidget()
        self.list.setIconSize(QSize(72, 72))
        for path in paths:
            item = QListWidgetItem(Path(path).name)
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            pixmap = thumbnail(path) if thumbnail is not None else None
            if pixmap is not None:
                item.setIcon(QIcon(pixmap))
            self.list.addItem(item)
        layout.addWidget(self.list, stretch=1)

        row = QHBoxLayout()
        for label, state in (("Tick All", Qt.CheckState.Checked), ("Untick All", Qt.CheckState.Unchecked)):
            button = QPushButton(label)
            button.setObjectName("SecondaryButton")
            button.clicked.connect(lambda _c=False, s=state: self._set_all(s))
            row.addWidget(button)
        row.addStretch()
        layout.addLayout(row)

        buttons = QDialogButtonBox()
        self.restore_button = buttons.addButton("Restore", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.list.itemChanged.connect(lambda _i: self._update_button())
        self._update_button()

    def _set_all(self, state) -> None:
        for index in range(self.list.count()):
            self.list.item(index).setCheckState(state)

    def chosen(self) -> list[str]:
        return [self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.CheckState.Checked]

    def _update_button(self) -> None:
        count = len(self.chosen())
        self.restore_button.setText(f"Restore {count}" if count else "Restore")
        self.restore_button.setEnabled(bool(count))
