"""A short message that floats over the bottom of the window and fades."""

from __future__ import annotations

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QLabel

TOAST_MS = 4000


class Toast(QLabel):
    """``show_message(text)`` shows it for a few seconds over its parent."""

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.setObjectName("Toast")
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setStyleSheet("QLabel#Toast { background: rgba(24, 24, 27, 235); color: #ffffff; "
                           "border: 1px solid #E31B23; border-radius: 10px; padding: 10px 18px; "
                           "font-size: 13px; font-weight: 600; }")
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)
        self.hide()

    def show_message(self, text: str, ms: int = TOAST_MS) -> None:
        self.setText(text)
        parent = self.parentWidget()
        width = min(560, max(240, parent.width() - 80)) if parent is not None else 420
        self.setFixedWidth(width)
        self.adjustSize()
        if parent is not None:
            self.move((parent.width() - self.width()) // 2, parent.height() - self.height() - 56)
        self.show()
        self.raise_()
        self._timer.start(ms)
