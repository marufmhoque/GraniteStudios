"""A strip of the working set along the bottom of an editing workspace.

Shows every frame in the folder in order, dims the rejected ones, outlines
the current one, and moves the current frame when one is clicked. It
borrows the thumbnails the Cull grid has already decoded, so it costs
nothing to keep up to date.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QWidget

STRIP_THUMB = 84


class Filmstrip(QScrollArea):
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
        self._tiles: dict[str, QLabel] = {}

    def rebuild(self) -> None:
        while self._row.count():
            item = self._row.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._tiles.clear()
        for key in self.session.keys():
            tile = QLabel()
            tile.setFixedSize(QSize(STRIP_THUMB, STRIP_THUMB))
            tile.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tile.setToolTip(Path(key).name)
            tile.setCursor(Qt.CursorShape.PointingHandCursor)
            tile.mousePressEvent = lambda _event, k=key: self.session.set_current(k)
            self._row.addWidget(tile)
            self._tiles[key] = tile
        self._row.addStretch()
        self.refresh()

    def refresh(self) -> None:
        """Pick up newly decoded thumbnails and the current marks."""
        if set(self._tiles) != set(self.session.keys()):
            self.rebuild()
            return
        current = self.session.current
        for key, tile in self._tiles.items():
            if tile.pixmap() is None or tile.pixmap().isNull():
                pixmap = self._thumbnails(key)
                if pixmap is not None and not pixmap.isNull():
                    tile.setPixmap(pixmap.scaled(
                        STRIP_THUMB - 4, STRIP_THUMB - 4, Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation))
                else:
                    tile.setText(Path(key).stem[:10])
            border = "2px solid #ffffff" if key == current else "1px solid #2a2a2a"
            dim = "background: #2a0a0c;" if key in self.session.rejected else "background: #161616;"
            tile.setStyleSheet(f"QLabel {{ border: {border}; border-radius: 4px; {dim} color: #888; }}")
        tile = self._tiles.get(current or "")
        if tile is not None:
            self.ensureWidgetVisible(tile, 40, 0)
