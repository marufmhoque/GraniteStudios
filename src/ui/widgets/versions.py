"""The version bar: one pill per version of the open photograph.

``Master``, ``Copy 1``, ``Cinematic``... Clicking a pill makes that version
the active one, which every workspace then shows and edits. Beside the pills:

    + Version    branch a copy of the version being shown
    + 3 Styles   branch Cinematic, Warm Analog and Clean B&W from the master
    ⋯            rename or remove the active copy

Right-clicking a pill offers the same rename and remove for that version.
Removing a copy deletes only its small recipe file in the edit folder; the
photograph is never touched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup, QHBoxLayout, QInputDialog, QLabel, QMenu, QMessageBox, QPushButton, QToolButton, QWidget,
)

from src.engine import parametric_recipe as pr
from src.engine import session as topics
from src.ui.theme import NIKON_RED

STYLE = f"""
QPushButton#VersionPill {{
    background: #1b1d23; color: #c9cad0; border: 1px solid #2e3139;
    border-radius: 11px; padding: 2px 12px; font-size: 12px; min-height: 18px;
}}
QPushButton#VersionPill:hover {{ border-color: {NIKON_RED}; }}
QPushButton#VersionPill:checked {{ background: {NIKON_RED}; color: white; border-color: {NIKON_RED}; font-weight: 600; }}
QPushButton#VersionAction, QToolButton#VersionAction {{
    background: transparent; color: #9a9ba1; border: 1px dashed #3a3d46;
    border-radius: 11px; padding: 2px 10px; font-size: 12px; min-height: 18px;
}}
QPushButton#VersionAction:hover, QToolButton#VersionAction:hover {{ color: white; border-color: {NIKON_RED}; }}
"""


def menu_text(name: str) -> str:
    """Qt reads "&" in button and menu text as a shortcut marker; show it as written."""
    return name.replace("&", "&&")


class VersionBar(QWidget):
    """Pills for the versions of one photograph, kept in step with the session."""

    def __init__(self, session, before_change: Callable[[], None] | None = None, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self.before_change = before_change or (lambda: None)
        self.key: str | None = None
        self.pills: dict[int, QPushButton] = {}
        self.setStyleSheet(STYLE)
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(0, 2, 0, 2)
        self._row.setSpacing(6)
        caption = QLabel("Versions")
        caption.setStyleSheet("color: #7d7f87; font-size: 11px;")
        self._row.addWidget(caption)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._pill_row = QHBoxLayout()
        self._pill_row.setSpacing(6)
        self._row.addLayout(self._pill_row)

        self.add_button = self._action("+ Version", self.create_version,
                                       "Branch a copy of this version's settings (no pixels are copied)")
        self.styles_button = self._action(
            "+ 3 Styles", self.create_styles,
            "Branch " + ", ".join(pr.VERSION_STYLES) + " from the master in one go")
        self.more_button = QToolButton()
        self.more_button.setObjectName("VersionAction")
        self.more_button.setText("⋯")
        self.more_button.setToolTip("Rename or remove the active version")
        self.more_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.more_button)
        self.rename_action = menu.addAction("Rename Version…", lambda: self.rename_version())
        self.delete_action = menu.addAction("Remove Version…", lambda: self.delete_version())
        self.more_button.setMenu(menu)
        self._row.addWidget(self.add_button)
        self._row.addWidget(self.styles_button)
        self._row.addWidget(self.more_button)
        self._row.addStretch()
        session.subscribe(self._on_session)
        self.refresh()

    def _action(self, text: str, slot, tip: str) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("VersionAction")
        button.setToolTip(tip)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(lambda _checked=False: slot())
        return button

    # -- keeping in step ----------------------------------------------------- #

    def set_photo(self, key: str | None) -> None:
        self.key = key
        self.refresh()

    def _on_session(self, topic: str) -> None:
        if topic == topics.LIBRARY:
            self.key = None
        if topic in (topics.VERSIONS, topics.LIBRARY):
            self.refresh()

    def refresh(self) -> None:
        for pill in self.pills.values():
            self._group.removeButton(pill)
            self._pill_row.removeWidget(pill)
            pill.setParent(None)        # gone now, not at the next turn of the event loop
            pill.deleteLater()
        self.pills.clear()
        enabled = self.key is not None
        for widget in (self.add_button, self.styles_button, self.more_button):
            widget.setEnabled(enabled)
        if not enabled:
            return
        active = self.session.active_version(self.key)
        for index, version in enumerate(self.session.versions(self.key), start=1):
            pill = QPushButton(menu_text(version.name))
            pill.setObjectName("VersionPill")
            pill.setCheckable(True)
            pill.setChecked(version.id == active)
            pill.setCursor(Qt.CursorShape.PointingHandCursor)
            pill.setToolTip(f"v{index}: {self.session.display_name(self.key, version.id)}"
                            + ("" if version.is_master else "\nRight-click to rename or remove"))
            pill.clicked.connect(lambda _checked=False, v=version.id: self.switch_to(v))
            pill.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            pill.customContextMenuRequested.connect(
                lambda pos, v=version.id, p=pill: self._pill_menu(v, p.mapToGlobal(pos)))
            self._group.addButton(pill)
            self._pill_row.addWidget(pill)
            self.pills[version.id] = pill
        is_copy = active != pr.MASTER
        self.rename_action.setEnabled(is_copy)
        self.delete_action.setEnabled(is_copy)

    def _pill_menu(self, version: int, where) -> None:
        if version == pr.MASTER:
            return
        menu = QMenu(self)
        menu.addAction("Rename…", lambda: self.rename_version(version))
        menu.addAction("Remove…", lambda: self.delete_version(version))
        menu.exec(where)

    # -- actions --------------------------------------------------------------- #

    def switch_to(self, version: int) -> None:
        if self.key is None:
            return
        if version != self.session.active_version(self.key):
            self.before_change()
            self.session.set_active_version(self.key, version)
        else:
            self.refresh()          # keep the clicked pill checked

    def create_version(self, name: str | None = None) -> int | None:
        if self.key is None:
            return None
        self.before_change()
        return self.session.create_version(self.key, name)

    def create_styles(self) -> list[int]:
        if self.key is None:
            return []
        self.before_change()
        return self.session.create_style_versions(self.key)

    def rename_version(self, version: int | None = None, name: str | None = None) -> str | None:
        if self.key is None:
            return None
        version = self.session.active_version(self.key) if version is None else version
        if version == pr.MASTER:
            return None
        if name is None:
            name, ok = QInputDialog.getText(self, "Rename Version", "Name this version (for example Moody Edit):",
                                            text=self.session.version_name(self.key, version))
            if not ok:
                return None
        try:
            return self.session.rename_version(self.key, version, name)
        except ValueError as exc:
            QMessageBox.information(self, "Rename Version", str(exc).capitalize() + ".")
            return None

    def delete_version(self, version: int | None = None, confirm: bool = True) -> bool:
        if self.key is None:
            return False
        version = self.session.active_version(self.key) if version is None else version
        if version == pr.MASTER:
            return False
        if confirm:
            answer = QMessageBox.question(
                self, "Remove Version",
                f"Remove {self.session.display_name(self.key, version)}?\n\n"
                f"Its edit settings are deleted. {Path(self.key).name} itself and its other "
                f"versions are not touched.")
            if answer != QMessageBox.StandardButton.Yes:
                return False
        self.before_change()
        self.session.delete_version(self.key, version)
        return True
