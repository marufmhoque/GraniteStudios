"""The Project Activity drawer: the workflow checklist and the activity timeline.

It slides in from the right edge of the window when the **📋 History**
button is pressed. The checklist shows which pipeline stages are done
(Ingest, Cull, Blur Triage, Auto-Level, Develop with its progress, Export);
the timeline lists every batch operation with when it happened, newest first.
With no project open it shows what has been done since the folder was
opened, which a new project made from that folder keeps.
"""

from __future__ import annotations

from datetime import datetime

from PyQt6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget,
)

from src.engine import project_manager as pm

DRAWER_WIDTH = 360
SLIDE_MS = 180
MARKS = {pm.DONE: "✓", pm.IN_PROGRESS: "◐", pm.NOT_STARTED: "○", pm.PENDING: "○"}
COLOURS = {pm.DONE: "#4cc38a", pm.IN_PROGRESS: "#f0b429", pm.NOT_STARTED: "#6a6b72", pm.PENDING: "#6a6b72"}

DRAWER_STYLE = """
QFrame#ActivityDrawer { background: #17171a; border-left: 1px solid #2f2f33; }
QFrame#ActivityDrawer QLabel { background: transparent; }
QLabel#DrawerHeading { color: #8f8f94; font-size: 11px; font-weight: 700; letter-spacing: 1px; }
QListWidget#Timeline { background: transparent; border: none; color: #d8d9de; }
QListWidget#Timeline::item { padding: 6px 2px; border-bottom: 1px solid #242428; }
QPushButton#DrawerClose { background: transparent; border: none; color: #8f8f94; font-size: 16px; }
QPushButton#DrawerClose:hover { color: #ffffff; }
"""


def when_text(stamp: str) -> str:
    """'15 Nov 14:02' in local time, from an ISO timestamp."""
    try:
        return datetime.fromisoformat(stamp).astimezone().strftime("%d %b %H:%M")
    except (TypeError, ValueError):
        return ""


class ActivityLogDrawer(QFrame):
    """An overlay panel docked to the right edge of its parent."""

    closed = pyqtSignal()

    def __init__(self, parent: QWidget, top: int = 0, bottom: int = 0) -> None:
        super().__init__(parent)
        self.setObjectName("ActivityDrawer")
        self.setStyleSheet(DRAWER_STYLE)
        self.setFixedWidth(DRAWER_WIDTH)
        self.top, self.bottom = top, bottom
        self.is_open = False
        self._animation = QPropertyAnimation(self, b"pos", self)
        self._animation.setDuration(SLIDE_MS)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._animation.finished.connect(self._on_slid)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 16, 16)
        layout.setSpacing(10)
        head = QHBoxLayout()
        self.title = QLabel("Project History")
        self.title.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: 800;")
        head.addWidget(self.title, stretch=1)
        close = QPushButton("✕")
        close.setObjectName("DrawerClose")
        close.setToolTip("Close (Esc)")
        close.clicked.connect(self.close_drawer)
        head.addWidget(close)
        layout.addLayout(head)
        self.subtitle = QLabel()
        self.subtitle.setObjectName("PageSubtitle")
        self.subtitle.setWordWrap(True)
        layout.addWidget(self.subtitle)

        heading = QLabel("WORKFLOW CHECKLIST")
        heading.setObjectName("DrawerHeading")
        layout.addWidget(heading)
        self.checklist_host = QWidget()
        self.checklist = QVBoxLayout(self.checklist_host)
        self.checklist.setContentsMargins(0, 0, 0, 0)
        self.checklist.setSpacing(4)
        layout.addWidget(self.checklist_host)
        self.rows: dict[str, QLabel] = {}
        for stage in pm.STAGES:
            row = QLabel()
            row.setTextFormat(Qt.TextFormat.RichText)
            self.checklist.addWidget(row)
            self.rows[stage] = row

        heading = QLabel("ACTIVITY TIMELINE")
        heading.setObjectName("DrawerHeading")
        layout.addSpacing(6)
        layout.addWidget(heading)
        self.timeline = QListWidget()
        self.timeline.setObjectName("Timeline")
        self.timeline.setWordWrap(True)
        layout.addWidget(self.timeline, stretch=1)
        self.note = QLabel()
        self.note.setObjectName("PageSubtitle")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)

        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=self.close_drawer,
                  context=Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.hide()

    # -- content ------------------------------------------------------------- #

    def refresh(self, project, session, pending: list[dict] | None = None) -> None:
        """Show ``project``'s checklist and log, or, with none, ``pending``."""
        if project is not None:
            entries, status = project.audit_log, project.stage_status
            self.title.setText(project.project_name)
            self.subtitle.setText("Project history")
            self.note.setText("")
        else:
            entries = list(pending or [])
            status = {e["stage"]: pm.DONE for e in entries if e["stage"] in pm.STAGES}
            self.title.setText("Project History")
            self.subtitle.setText("No project is open.")
            self.note.setText("Create a project (Ctrl+N) to keep this history."
                              if session is not None and session.is_open else "Open a project to see its history.")
        for stage, name, state, detail in pm.workflow_checklist(session, status):
            extra = f" ({detail})" if detail and state != pm.NOT_STARTED else (f" · {detail}" if detail else "")
            self.rows[stage].setText(
                f"<span style='color:{COLOURS[state]}; font-weight:700'>{MARKS[state]}</span>"
                f"&nbsp;&nbsp;<span style='color:#f2f3f5'>{name}</span>"
                f"&nbsp;&nbsp;<span style='color:{COLOURS[state]}'>{state}{extra}</span>")
            self.rows[stage].setProperty("state", state)
        self.timeline.clear()
        for entry in reversed(entries):
            when = when_text(entry.get("time", ""))
            item = QListWidgetItem((f"{when}  ·  " if when else "") + entry["description"])
            item.setToolTip(pm.STAGES.get(entry["stage"], entry["stage"]))
            self.timeline.addItem(item)
        if not entries:
            self.timeline.addItem(QListWidgetItem("Nothing yet. Culls, blur triage, auto-level, removals "
                                                  "and exports are listed here as they happen."))

    # -- sliding -------------------------------------------------------------- #

    def _open_pos(self) -> QPoint:
        return QPoint(self.parentWidget().width() - DRAWER_WIDTH, self.top)

    def _closed_pos(self) -> QPoint:
        return QPoint(self.parentWidget().width(), self.top)

    def reposition(self) -> None:
        """Keep docked to the right edge, full height, as the window resizes."""
        parent = self.parentWidget()
        self.setFixedHeight(max(100, parent.height() - self.top - self.bottom))
        if self._animation.state() != QPropertyAnimation.State.Running:
            self.move(self._open_pos() if self.is_open else self._closed_pos())

    def open_drawer(self) -> None:
        if self.is_open:
            return
        self.is_open = True
        self.reposition()
        self.move(self._closed_pos())
        self.show()
        self.raise_()
        self._slide(self._open_pos())
        self.setFocus()

    def close_drawer(self) -> None:
        if not self.is_open:
            return
        self.is_open = False
        self._slide(self._closed_pos())
        self.closed.emit()

    def toggle(self) -> bool:
        (self.close_drawer if self.is_open else self.open_drawer)()
        return self.is_open

    def _slide(self, target: QPoint) -> None:
        self._animation.stop()
        self._animation.setStartValue(self.pos())
        self._animation.setEndValue(target)
        self._animation.start()

    def _on_slid(self) -> None:
        if not self.is_open:
            self.hide()
