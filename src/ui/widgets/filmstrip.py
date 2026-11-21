"""A strip of the working set along the bottom of an editing workspace.

Shows the frames that can be edited, in order, each followed by its virtual
copies (with a small ``v2`` chip), outlines the one being edited, and moves to a tile when it is clicked. A tile's right-click menu
deletes the photo from the workspace (``delete_requested``). It borrows the
thumbnails the Cull grid has already decoded, so it costs nothing to keep up
to date.

Rejected photos are locked against editing, so by default they are left out
(``hide_locked``); the photo being shown stays, so you can see where you are.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QMenu, QScrollArea, QWidget

STRIP_THUMB = 84
MASTER = 0


class Filmstrip(QScrollArea):
    delete_requested = pyqtSignal(str)        # Delete Photo chosen on a tile's menu

    def __init__(self, session, thumbnails, parent=None) -> None:
        """``thumbnails(key)`` returns a QPixmap for a frame, or None."""
        super().__init__(parent)
        self.session = session
        self._thumbnails = thumbnails
        self.setWidgetResizable(True)
        self.setFixedHeight(STRIP_THUMB + 26)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._host = QWidget()
        self._row = QHBoxLayout(self._host)
        self._row.setContentsMargins(4, 4, 4, 4)
        self._row.setSpacing(4)
        self._row.addStretch()
        self.setWidget(self._host)
        self._tiles: dict[str, QLabel] = {}          # tile id -> tile (a photo's key for its own)
        self._entries: dict[str, object] = {}
        self.hide_locked = True                      # rejected photos are not offered for editing
        # Copies made or removed anywhere, and switching between them, show here.
        session.subscribe(lambda topic: self.refresh() if topic == "versions" else None)

    def _choose(self, entry) -> None:
        self.session.set_current(entry.key)
        if self.session.version_count(entry.key) > 1:
            self.session.set_active_version(entry.key, entry.version)

    def tile_menu(self, entry):
        """A tile's right-click menu: Delete Photo from Workspace."""
        from src.ui.dialogs.delete_photo_dialog import add_delete_action

        menu = QMenu(self)
        add_delete_action(menu, lambda: self.delete_requested.emit(entry.key))
        return menu

    def _menu(self, entry, where) -> None:
        self.tile_menu(entry).exec(where)

    def order(self) -> list[str]:
        """The photos in the strip, in its order (each once)."""
        return list(dict.fromkeys(e.key for e in self.entries()))

    def entries(self) -> list:
        """The tiles to show: every entry, less the locked ones by default."""
        entries = self.session.entries()
        if not self.hide_locked:
            return entries
        current = self.session.current
        return [e for e in entries if e.key == current or self.session.is_editable(e.key)]

    def rebuild(self) -> None:
        while self._row.count():
            item = self._row.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._tiles.clear()
        self._entries.clear()
        for entry in self.entries():
            tile = QLabel()
            tile.setFixedSize(QSize(STRIP_THUMB, STRIP_THUMB))
            tile.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tile.setToolTip(Path(entry.key).name if not entry.is_copy
                            else f"{entry.variant_key} — virtual copy ({entry.name})")
            tile.setCursor(Qt.CursorShape.PointingHandCursor)
            tile.mousePressEvent = lambda _event, e=entry: self._choose(e)
            tile.contextMenuEvent = lambda event, e=entry: self._menu(e, event.globalPos())
            if entry.is_copy:
                chip = QLabel(entry.chip, tile)
                chip.setGeometry(STRIP_THUMB - 30, 4, 26, 16)
                chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
                chip.setStyleSheet("background: rgba(10,11,14,0.85); color: #fff; border: 1px solid #E31B23;"
                                   " border-radius: 4px; font-size: 9px; font-weight: 700;")
            self._row.addWidget(tile)
            self._tiles[entry.tile_id] = tile
            self._entries[entry.tile_id] = entry
        self._row.addStretch()
        self.refresh()

    def refresh(self) -> None:
        """Pick up newly decoded thumbnails, the marks and the copies."""
        if set(self._tiles) != {e.tile_id for e in self.entries()}:
            self.rebuild()
            return
        current = self.session.current
        shown = None
        if current is not None:
            version = self.session.active_version(current)
            shown = current if version == MASTER else f"{current}#v{version}"
        for tile_id, tile in self._tiles.items():
            key = self._entries[tile_id].key
            if tile.pixmap() is None or tile.pixmap().isNull():
                pixmap = self._thumbnails(key)
                if pixmap is not None and not pixmap.isNull():
                    tile.setPixmap(pixmap.scaled(
                        STRIP_THUMB - 4, STRIP_THUMB - 4, Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation))
                else:
                    tile.setText(Path(key).stem[:10])
            border = "2px solid #ffffff" if tile_id == shown else "1px solid #2a2a2a"
            dim = "background: #2a0a0c;" if key in self.session.rejected else "background: #161616;"
            tile.setStyleSheet(f"QLabel {{ border: {border}; border-radius: 4px; {dim} color: #888; }}")
        tile = self._tiles.get(shown or "")
        if tile is not None:
            self.ensureWidgetVisible(tile, 40, 0)
