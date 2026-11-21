"""Delete Photo: take one photo (or the selected ones) out of the workspace.

Shift+Delete (Shift+Backspace on a Mac) in Cull, Develop, Retouch and Design,
or **🗑️ Delete Photo from Workspace…** on a tile's right-click menu, opens
:class:`DeletePhotoDialog`: the thumbnail, the file name and size, and what
will happen. Confirming removes the photo from the project and moves on to
the next one.

The file itself is never deleted, moved or changed (INV-1), the same as for
culled and blurry photos: it stays on disk with its ratings and edits, and
View ▸ Restore Removed Images brings it back.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QMenu, QPushButton, QVBoxLayout, QWidgetAction,
)

from src.ui.theme import CRIMSON_GRANITE_STYLESHEET

DELETE_TEXT = "\U0001F5D1️ Delete Photo from Workspace…"
DANGER = "#ff5a5f"


def file_size_label(path) -> str:
    """``24.3 MB``, or ``missing`` when the file cannot be found."""
    try:
        size = Path(path).stat().st_size
    except OSError:
        return "missing"
    for unit in ("bytes", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:,.0f} {unit}" if unit == "bytes" else f"{size:,.1f} {unit}"
        size /= 1024.0
    return ""


def add_delete_action(menu: QMenu, slot, text: str = DELETE_TEXT) -> QWidgetAction:
    """Add the red Delete Photo item to ``menu``. A plain QAction cannot be
    coloured, so the item is a label; clicking it (or Enter on it) triggers
    the action and closes the menu, as any other item does."""
    action = QWidgetAction(menu)
    action.setText(text)
    label = QLabel(text)
    label.setObjectName("DangerMenuItem")
    label.setStyleSheet(
        f"QLabel#DangerMenuItem {{ color: {DANGER}; padding: 5px 24px 5px 22px; font-weight: 600; }}"
        "QLabel#DangerMenuItem:hover { background: #3a1a12; }")

    def clicked(_event) -> None:
        menu.close()
        action.trigger()

    label.mouseReleaseEvent = clicked
    action.setDefaultWidget(label)
    action.triggered.connect(lambda _checked=False: slot())
    menu.addAction(action)
    return action


class DeletePhotoDialog(QDialog):
    """Confirm deleting ``keys`` from the workspace. ``thumbnail(key)``
    returns the grid's pixmap for a photo, or None."""

    def __init__(self, keys, thumbnail=None, parent=None) -> None:
        super().__init__(parent)
        self.keys = [str(k) for k in keys]
        count = len(self.keys)
        first = Path(self.keys[0])
        self.setWindowTitle("Delete Photo" if count == 1 else f"Delete {count} Photos")
        self.setStyleSheet(CRIMSON_GRANITE_STYLESHEET)
        self.setMinimumWidth(460)

        outer = QVBoxLayout(self)
        outer.setSpacing(12)
        row = QHBoxLayout()
        row.setSpacing(14)
        self.thumb = QLabel()
        self.thumb.setFixedSize(150, 150)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setStyleSheet("background: #161616; border: 1px solid #2a2a2a; border-radius: 6px;")
        pixmap = thumbnail(self.keys[0]) if thumbnail is not None else None
        if pixmap is not None and not pixmap.isNull():
            self.thumb.setPixmap(pixmap.scaled(146, 146, Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation))
        else:
            self.thumb.setText(first.suffix.upper().lstrip(".") or "?")
        row.addWidget(self.thumb, alignment=Qt.AlignmentFlag.AlignTop)

        facts = QVBoxLayout()
        title = QLabel(f"Delete {first.name} from the workspace?" if count == 1
                       else f"Delete {count} photos from the workspace?")
        title.setWordWrap(True)
        title.setStyleSheet("font-size: 15px; font-weight: 700;")
        facts.addWidget(title)
        if count == 1:
            self.details = QLabel(f"{first.name}  ·  {file_size_label(first)}\n{first.parent}")
        else:
            names = ", ".join(Path(k).name for k in self.keys[:3]) + (" …" if count > 3 else "")
            self.details = QLabel(f"{names}\nFirst: {first.name}  ·  {file_size_label(first)}")
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.details.setStyleSheet("color: #9a9a9a;")
        facts.addWidget(self.details)
        facts.addStretch()
        row.addLayout(facts, stretch=1)
        outer.addLayout(row)

        note = QFrame()
        note.setStyleSheet("QFrame { background: #15221a; border: 1px solid #2e5a3c; border-radius: 6px; }"
                           "QLabel { background: transparent; color: #cfe8d6; }")
        note_row = QVBoxLayout(note)
        self.note = QLabel(
            ("It leaves the grid, the filmstrips and every export. " if count == 1 else
             "They leave the grid, the filmstrips and every export. ")
            + "The file is not deleted: it stays on disk exactly as it is, with its ratings, "
            "edits and virtual copies, and View ▸ Restore Removed Images brings it back.")
        self.note.setWordWrap(True)
        note_row.addWidget(self.note)
        outer.addWidget(note)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.cancel_button)
        self.delete_button = QPushButton("\U0001F5D1️ Delete from Workspace")
        self.delete_button.setObjectName("ActionButton")
        self.delete_button.setDefault(True)
        self.delete_button.clicked.connect(self.accept)
        buttons.addWidget(self.delete_button)
        outer.addLayout(buttons)
