"""The edit lock: rejected and removed photos cannot be edited.

:class:`EditLockBanner` sits above the picture in Develop, Retouch and
Design. When the photo shown is locked (see
:meth:`LibrarySession.is_editable`) it says so, the workspace's editing
controls are disabled, and **↺ Restore Photo** brings the photo back (no
longer rejected, and back in the project if it was removed), which unlocks
everything again.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QPushButton

LOCKED_TEXT = "⚠️ Marked for Removal — Editing Disabled"
OPEN_REFUSED = "Editing disabled: This image is marked for removal. Click 'Restore' to edit."


class EditLockBanner(QFrame):
    """The warning shown over a locked photo, with its Restore button."""

    restore_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("EditLockBanner")
        self.setStyleSheet(
            "QFrame#EditLockBanner { background: #3a1a12; border: 1px solid #E31B23; border-radius: 8px; }"
            "QFrame#EditLockBanner QLabel { background: transparent; color: #ffd9cf; font-weight: 700; }")
        row = QHBoxLayout(self)
        row.setContentsMargins(12, 6, 8, 6)
        self.label = QLabel(LOCKED_TEXT)
        row.addWidget(self.label, stretch=1)
        self.restore_button = QPushButton("↺ Restore Photo")
        self.restore_button.setObjectName("ActionButton")
        self.restore_button.setToolTip("Un-reject this photo (and bring it back into the project if it was "
                                       "removed) so it can be edited")
        self.restore_button.clicked.connect(self.restore_requested.emit)
        row.addWidget(self.restore_button)
        self.setVisible(False)


def apply_lock(session, key, banner: EditLockBanner, controls) -> bool:
    """Show or hide the banner and enable or disable ``controls`` for the
    photo ``key``. Returns True when it may be edited."""
    editable = key is None or session.is_editable(key)
    banner.setVisible(not editable)
    if not editable:
        banner.label.setText(LOCKED_TEXT + ("   (removed from the project)" if str(key) in session.purged
                                            else "   (rejected)"))
    for widget in controls:
        widget.setEnabled(editable)
        # Dimmed as well as disabled: the theme's sliders look the same either way.
        if editable:
            widget.setGraphicsEffect(None)
        elif not isinstance(widget.graphicsEffect(), QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect(widget)
            effect.setOpacity(0.4)
            widget.setGraphicsEffect(effect)
    return editable
