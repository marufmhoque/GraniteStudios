"""Clean Up Shoot: review every misfire, then remove them in one step.

The assistant counts what the cull and blur triage flagged (duplicates,
blurry photos, blinks) and what was rejected, lets you untick any group
(keep the blurry ones for now, remove the duplicates), and shows how many
photos go and how many keepers stay. **Inspect in Grid** selects them in
Cull to look through first; **Remove Selected Photos** rejects them and
takes them out of the project.

Removing never touches the files: they stay where they are on disk, with
their ratings and edits, and View ▸ Restore Removed Images brings them back
(INV-1).
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
)

from src.ui.theme import CRIMSON_GRANITE_STYLESHEET

INSPECT, REMOVE = "inspect", "remove"

CATEGORIES = (
    # key, label, what it means
    ("duplicates", "Duplicates", "Weaker frames of a burst the cull found a better one of"),
    ("blurry", "Blurry", "Out of focus or blurred by motion (Find Blurry)"),
    ("blinks", "Blinks", "Someone's eyes are closed (Smart Cull with Skip blinks)"),
    ("rejected", "Rejected", "Photos you, or a batch action, marked rejected"),
)


def misfire_groups(session) -> dict[str, list[Path]]:
    """Each category's photos, from the session's selection helpers."""
    return {
        "duplicates": session.get_duplicate_images(),
        "blurry": session.get_blurry_images(),
        "blinks": session.get_blink_images(),
        "rejected": session.get_rejected_images(),
    }


class CleanUpAssistantDialog(QDialog):
    """Choose which misfires to remove from the project."""

    def __init__(self, session, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self.action: str | None = None
        self.removed = 0
        self.groups = misfire_groups(session)
        self.setWindowTitle("Clean Up Shoot")
        self.setStyleSheet(CRIMSON_GRANITE_STYLESHEET + """
            QFrame#CleanUpCard { background: #1c1c1f; border: 1px solid #2c2c30; border-radius: 10px; }
            QFrame#CleanUpCard QLabel { background: transparent; }
            QCheckBox { font-size: 14px; font-weight: 600; }
        """)
        self.setMinimumWidth(500)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(12)
        title = QLabel("\U0001F9F9 Clean Up Shoot")
        title.setStyleSheet("font-size: 20px; font-weight: 800; color: #ffffff;")
        layout.addWidget(title)
        intro = QLabel("Untick anything you want to keep for now. Everything ticked leaves the grid, "
                       "the filmstrips and every export.")
        intro.setObjectName("PageSubtitle")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        card = QFrame()
        card.setObjectName("CleanUpCard")
        rows = QVBoxLayout(card)
        rows.setContentsMargins(16, 12, 16, 12)
        rows.setSpacing(8)
        self.checks: dict[str, QCheckBox] = {}
        for key, label, tip in CATEGORIES:
            count = len(self.groups[key])
            check = QCheckBox(f"{label}  ({count:,})")
            check.setToolTip(tip)
            check.setChecked(count > 0)
            check.setEnabled(count > 0)
            check.toggled.connect(lambda _c: self._refresh())
            hint = QLabel(tip)
            hint.setObjectName("PageSubtitle")
            row = QHBoxLayout()
            row.addWidget(check)
            row.addStretch()
            row.addWidget(hint)
            rows.addLayout(row)
            self.checks[key] = check
        layout.addWidget(card)

        self.total_label = QLabel()
        self.total_label.setStyleSheet("font-size: 15px; font-weight: 700; color: #ffffff;")
        layout.addWidget(self.total_label)
        self.keepers_label = QLabel()
        self.keepers_label.setStyleSheet("font-size: 14px; color: #8fe39b;")
        layout.addWidget(self.keepers_label)
        safe = QLabel("\U0001F512 Removed from the project only: the files stay safe on disk, untouched. "
                      "View ▸ Restore Removed Images brings them back at any time.")
        safe.setObjectName("PageSubtitle")
        safe.setWordWrap(True)
        layout.addWidget(safe)

        buttons = QHBoxLayout()
        cancel = QPushButton("Cancel")
        cancel.setObjectName("SecondaryButton")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        buttons.addStretch()
        self.inspect_button = QPushButton("Inspect in Grid")
        self.inspect_button.setObjectName("SecondaryButton")
        self.inspect_button.setToolTip("Select them in Cull to look through before removing anything")
        self.inspect_button.clicked.connect(self.inspect)
        buttons.addWidget(self.inspect_button)
        self.remove_button = QPushButton("\U0001F5D1️ Remove Selected Photos")
        self.remove_button.setObjectName("ActionButton")
        self.remove_button.clicked.connect(self.remove_selected)
        buttons.addWidget(self.remove_button)
        layout.addLayout(buttons)
        self._refresh()

    # -- what is chosen ----------------------------------------------------------- #

    def chosen(self) -> list[Path]:
        """The photos in the ticked groups, once each, in folder order."""
        wanted = {str(p) for key, check in self.checks.items() if check.isChecked() for p in self.groups[key]}
        return [p for p in self.session.image_paths if str(p) in wanted]

    def keepers_remaining(self) -> int:
        """Photos left after removal that are not rejected."""
        going = {str(p) for p in self.chosen()}
        return sum(1 for p in self.session.image_paths
                   if str(p) not in going and str(p) not in self.session.rejected)

    def _refresh(self) -> None:
        count = len(self.chosen())
        self.total_label.setText(f"Total selected for removal: {count:,} photo{'s' if count != 1 else ''}")
        keepers = self.keepers_remaining()
        self.keepers_label.setText(f"Keepers remaining: {keepers:,} photo{'s' if keepers != 1 else ''}")
        self.remove_button.setEnabled(count > 0)
        self.inspect_button.setEnabled(count > 0)

    # -- the two actions ---------------------------------------------------------- #

    def inspect(self) -> None:
        """Close, so Cull selects them and shows the first."""
        self.action = INSPECT
        self.accept()

    def remove_selected(self) -> int:
        """Reject the chosen photos and take them out of the project.
        Returns how many were removed; the files are not touched."""
        chosen = self.chosen()
        if chosen:
            self.session.set_rejected(chosen, True)
            self.removed = self.session.purge(chosen)
        self.action = REMOVE
        self.accept()
        return self.removed
