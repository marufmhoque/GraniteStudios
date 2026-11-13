"""Widgets and workers shared by more than one workspace."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QDialog,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.engine.hardware import get_hardware_summary
from src.engine.resources import resource_path
from src.engine.universal_loader import UniversalLoader, ImageMetadata
from src.ai_tools import blur_analysis
from src.ai_tools import orientation
from src.ai_tools import pose_analysis
from src.ai_tools import xmp_sidecar
from src.ai_tools.culling import SmartCull
from src.ai_tools.color_engine import ColorEngine
from src.ai_tools.background_removal import BackgroundRemover
from src.social.reel_maker import ScrlReelMaker
from src.social.ig_sync import IgFormatter
from src.web.publisher import generate_static_gallery
from src.video import audio as video_audio
from src.video import color as video_color
from src.video import export as video_export
from src.video import spatial as video_spatial
from src.video import timing as video_timing
from src.video.ffmpeg_utils import FFmpegError, probe as video_probe

from src.ui.theme import NIKON_RED, NIKON_RED_DARK, THUMB_SIZE, pil_to_pixmap

logger = logging.getLogger(__name__)


class ThumbnailWorker(QThread):
    """Decode previews off the UI thread and stream them back."""

    thumb_ready = pyqtSignal(str, QPixmap)
    meta_ready = pyqtSignal(str, object)
    progress = pyqtSignal(int, int)
    finished_all = pyqtSignal()

    def __init__(self, loader: UniversalLoader, paths: list[Path], parent=None):
        super().__init__(parent)
        self.loader = loader
        self.paths = paths
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        total = len(self.paths)
        for i, path in enumerate(self.paths, start=1):
            if self._cancelled:
                break
            thumb = self.loader.load_thumbnail(path, size=THUMB_SIZE * 2)
            if thumb is not None and not self._cancelled:
                thumb.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)
                self.thumb_ready.emit(str(path), pil_to_pixmap(thumb))
            if not self._cancelled:
                # EXIF gear/exposure info for the Tech Info overlay.
                self.meta_ready.emit(str(path), self.loader.read_metadata(path))
            self.progress.emit(i, total)
        self.finished_all.emit()


class TaskWorker(QThread):
    """Run a long operation (cull / reel / export) off the UI thread."""

    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            self.done.emit(self._fn())
        except Exception as exc:  # surface every failure to the UI
            logger.exception("Task failed")
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------- #
# Thumbnail widget
# ---------------------------------------------------------------------- #


def _dim_effect(widget) -> QGraphicsOpacityEffect:
    """A reusable 'this frame is out' dimming effect."""
    effect = QGraphicsOpacityEffect(widget)
    effect.setOpacity(0.35)
    return effect


class ThumbnailCell(QLabel):
    """One selectable thumbnail in the grid."""

    clicked = pyqtSignal(str)
    opened = pyqtSignal(str)                # double-click: open in Develop
    web_toggled = pyqtSignal(str, bool)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path
        self.selected = False
        self.rejected = False
        # Remembered so the view filter can show just picks or duplicates.
        self.starred = False
        self.duplicate = False
        self.blur_verdict = None
        self.rating, self.colour_label, self.picked, self.edited = 0, None, False, False
        self.metadata: ImageMetadata | None = None
        self.label_text = ""
        self._badge = ""
        self._tech_visible = False
        self.setObjectName("Thumb")
        self.setFixedSize(QSize(THUMB_SIZE + 12, THUMB_SIZE + 34))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText(Path(path).name)
        self.setWordWrap(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        w = THUMB_SIZE + 12
        h = THUMB_SIZE + 34

        # Top-left: star / duplicate badge from the AI cull pass.
        self.badge_label = QLabel(self)
        self.badge_label.setGeometry(6, 6, 130, 22)
        self.badge_label.hide()

        # Top-right: assigned category label.
        self.label_tag = QLabel(self)
        self.label_tag.setGeometry(w - 96, 6, 90, 22)
        self.label_tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_tag.setStyleSheet(
            "background: rgba(61,109,242,0.92); color: white; border-radius: 6px;"
            " padding: 2px 6px; font-size: 11px; font-weight: 600;"
        )
        self.label_tag.hide()

        # Bottom-left: "Publish to Web" toggle — always visible, independent
        # of the tech-info overlay so it stays reachable regardless of that
        # toggle's state.
        self.web_checkbox = QCheckBox(self)
        self.web_checkbox.setGeometry(4, h - 26, 20, 20)
        self.web_checkbox.setToolTip("Publish to Web")
        self.web_checkbox.setStyleSheet(
            "QCheckBox { background: transparent; }"
            "QCheckBox::indicator { width: 18px; height: 18px; border-radius: 4px;"
            " border: 1.5px solid #E31B23; background-color: rgba(10,11,14,0.85); }"
            "QCheckBox::indicator:checked { background-color: #E31B23; }"
        )
        self.web_checkbox.toggled.connect(lambda checked: self.web_toggled.emit(self.path, checked))

        # Bottom-right: the photographer's own marks -- stars, colour
        # label, pick flag -- and whether the frame has been edited.
        self.marks_tag = QLabel(self)
        self.marks_tag.setGeometry(w - 112, h - 26, 106, 20)
        self.marks_tag.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.marks_tag.setTextFormat(Qt.TextFormat.RichText)
        self.marks_tag.setStyleSheet(
            "background: rgba(10,11,14,0.80); border-radius: 6px; padding: 1px 6px; font-size: 11px;"
        )
        self.marks_tag.hide()
        # Under the star badge: what the blur analysis decided.
        self.blur_tag = QLabel(self)
        self.blur_tag.setGeometry(6, 30, 140, 20)
        self.blur_tag.hide()

        # Centre: rejection marker. Deliberately loud, because rejecting is
        # the one action a photographer needs to be certain about at a glance.
        self.reject_tag = QLabel("REJECTED", self)
        self.reject_tag.setGeometry(6, h // 2 - 16, w - 12, 30)
        self.reject_tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.reject_tag.setStyleSheet(
            "background: rgba(227,27,35,0.92); color: white; border-radius: 6px;"
            " padding: 4px; font-size: 13px; font-weight: 800; letter-spacing: 1px;"
        )
        self.reject_tag.hide()

        # Bottom strip: EXIF tech info (camera | lens + exposure). Narrowed
        # to leave the bottom-left corner clear for the web checkbox above.
        self.tech_label = QLabel(self)
        self.tech_label.setGeometry(28, h - 44, w - 34, 38)
        self.tech_label.setWordWrap(True)
        self.tech_label.setStyleSheet(
            "background: rgba(10,11,14,0.82); color: #d8d9de; border-radius: 6px;"
            " padding: 3px 6px; font-size: 10px;"
        )
        self.tech_label.hide()

    def _raise_overlays(self) -> None:
        for overlay in (self.badge_label, self.label_tag, self.tech_label,
                        self.web_checkbox, self.reject_tag, self.blur_tag, self.marks_tag):
            overlay.raise_()

    def set_web_selected(self, checked: bool) -> None:
        """Set the web checkbox without re-emitting web_toggled (for syncing
        UI state from the backend model)."""
        self.web_checkbox.blockSignals(True)
        self.web_checkbox.setChecked(checked)
        self.web_checkbox.blockSignals(False)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self.setPixmap(
            pixmap.scaled(
                THUMB_SIZE,
                THUMB_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._raise_overlays()

    def _rebuild_tooltip(self) -> None:
        parts = [Path(self.path).name]
        if self.metadata is not None:
            if self.metadata.gear_label:
                parts.append(self.metadata.gear_label)
            if self.metadata.settings_label:
                parts.append(self.metadata.settings_label)
        if self.label_text:
            parts.append(f"Label: {self.label_text}")
        if self._badge:
            parts.append(self._badge)
        self.setToolTip("\n".join(parts))

    def set_badge(self, text: str) -> None:
        self._badge = text
        self._rebuild_tooltip()

    def set_metadata(self, metadata: ImageMetadata | None) -> None:
        self.metadata = metadata
        self._rebuild_tooltip()
        self._refresh_tech()

    def set_tech_visible(self, visible: bool) -> None:
        self._tech_visible = visible
        self._refresh_tech()

    def _refresh_tech(self) -> None:
        if self._tech_visible and self.metadata is not None:
            combined = "\n".join(
                t for t in (self.metadata.gear_label, self.metadata.settings_label) if t
            )
            if combined:
                self.tech_label.setText(combined)
                self.tech_label.show()
                self.tech_label.raise_()
                return
        self.tech_label.hide()

    def set_label(self, label: str) -> None:
        self.label_text = label or ""
        if self.label_text:
            self.label_tag.setText(self.label_text)
            self.label_tag.show()
            self.label_tag.raise_()
        else:
            self.label_tag.hide()
        self._rebuild_tooltip()

    def set_pick_state(self, starred: bool, duplicate: bool, detail: str = "") -> None:
        self.starred = starred
        self.duplicate = duplicate
        if starred:
            self.badge_label.setText("★ STARRED")
            self.badge_label.setStyleSheet(
                "background: rgba(20,22,28,0.9); color: #ffd24a; border-radius: 6px;"
                " padding: 2px 6px; font-size: 11px; font-weight: 700;"
            )
            self.badge_label.show()
            self.badge_label.raise_()
        elif duplicate:
            self.badge_label.setText("DUPLICATE")
            self.badge_label.setStyleSheet(
                "background: rgba(20,22,28,0.9); color: #9a9ba1; border-radius: 6px;"
                " padding: 2px 6px; font-size: 11px; font-weight: 600;"
            )
            self.badge_label.show()
            self.badge_label.raise_()
        else:
            self.badge_label.hide()
        if detail:
            self.set_badge(detail)

    def set_blur_state(self, report) -> None:
        """Show what the blur analysis found, if anything."""
        self.blur_verdict = report.verdict
        if report.is_blurry:
            colour = "rgba(201,88,30,0.92)" if report.verdict == "motion_blur" else "rgba(120,80,160,0.92)"
            text = "MOTION BLUR" if report.verdict == "motion_blur" else "OUT OF FOCUS"
            self.blur_tag.setText(text)
            self.blur_tag.setStyleSheet(
                f"background: {colour}; color: white; border-radius: 6px;"
                " padding: 2px 6px; font-size: 10px; font-weight: 700;"
            )
            self.blur_tag.show()
            self.blur_tag.raise_()
        else:
            self.blur_tag.hide()
        self.setToolTip(report.summary)

    def set_rejected(self, rejected: bool) -> None:
        """Mark this frame as rejected — excluded from the edit folder.

        Rejection is purely a flag on the app's side. The original file is
        never moved or deleted; it simply stops being copied into the
        -edit subfolder when anything is exported.
        """
        self.rejected = rejected
        if rejected:
            self.reject_tag.show()
            self.reject_tag.raise_()
            # Dim the frame so a rejected shot reads as "out" at a glance.
            self.setGraphicsEffect(_dim_effect(self))
        else:
            self.reject_tag.hide()
            self.setGraphicsEffect(None)
        self._rebuild_tooltip()

    _COLOURS = {"Red": "#e74c3c", "Yellow": "#f1c40f", "Green": "#2ecc71",
                "Blue": "#3498db", "Purple": "#9b59b6"}

    def set_marks(self, rating: int = 0, colour: str | None = None, pick: bool = False,
                  edited: bool = False) -> None:
        """Show the photographer's marks. Nothing is shown when there are none."""
        self.rating, self.colour_label, self.picked, self.edited = rating, colour, pick, edited
        parts = []
        if edited:
            parts.append("<span style='color:#9ad0ff'>✎</span>")
        if pick:
            parts.append("<span style='color:#ffffff'>⚑</span>")
        if colour in self._COLOURS:
            parts.append(f"<span style='color:{self._COLOURS[colour]}'>●</span>")
        if rating:
            parts.append(f"<span style='color:#ffd24a'>{'★' * rating}</span>")
        if parts:
            self.marks_tag.setText(" ".join(parts))
            self.marks_tag.show()
            self.marks_tag.raise_()
        else:
            self.marks_tag.hide()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.path)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.button() == Qt.MouseButton.LeftButton:
            self.opened.emit(self.path)

    def set_selected(self, selected: bool) -> None:
        self.selected = selected
        self.setObjectName("ThumbSelected" if selected else "Thumb")
        # Re-polish so the stylesheet re-applies with the new object name.
        self.style().unpolish(self)
        self.style().polish(self)


class RejectsDialog(QDialog):
    """Review rejected frames before they are excluded from the edit folder.

    This window exists so rejecting is never a surprise: nothing leaves the
    selection silently. It is explicitly *not* a delete confirmation --
    originals are never touched -- so the wording avoids the word "delete"
    except to say what does not happen.
    """

    def __init__(self, paths: list[str], loader, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rejected photos")
        self.resize(900, 620)
        self.restored: set[str] = set()
        self._loader = loader

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        heading = QLabel(f"{len(paths)} photo(s) rejected")
        heading.setStyleSheet("font-size: 17px; font-weight: 700; color: #f2f3f5;")
        layout.addWidget(heading)

        explain = QLabel(
            "These will not be copied into the -edit subfolder. "
            "Your original files are never deleted, moved or changed — "
            "they stay exactly where they are. Untick any you want to keep."
        )
        explain.setWordWrap(True)
        explain.setStyleSheet("color: #a8a8ac; font-size: 12px;")
        layout.addWidget(explain)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        grid = QGridLayout(host)
        grid.setSpacing(10)
        grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        self._checks: dict[str, QCheckBox] = {}
        columns = 5
        for index, path in enumerate(paths):
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(4, 4, 4, 4)
            cell_layout.setSpacing(4)

            thumb = QLabel()
            thumb.setFixedSize(150, 150)
            thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb.setStyleSheet("background: #121318; border-radius: 6px;")
            thumb.setText(Path(path).name)
            thumb.setWordWrap(True)
            try:
                preview = loader.load_preview(path)
                if preview is not None:
                    image = QImage(
                        preview.tobytes("raw", "RGB"),
                        preview.width,
                        preview.height,
                        preview.width * 3,
                        QImage.Format.Format_RGB888,
                    )
                    thumb.setPixmap(
                        QPixmap.fromImage(image).scaled(
                            150,
                            150,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )
            except Exception:
                # A thumbnail that will not decode must not stop the review.
                pass
            cell_layout.addWidget(thumb)

            check = QCheckBox("Reject")
            check.setChecked(True)
            check.setToolTip("Untick to keep this photo in the edit folder.")
            self._checks[path] = check
            cell_layout.addWidget(check)

            name = QLabel(Path(path).name)
            name.setStyleSheet("color: #8b8c93; font-size: 10px;")
            name.setWordWrap(True)
            cell_layout.addWidget(name)

            grid.addWidget(cell, index // columns, index % columns)

        scroll.setWidget(host)
        layout.addWidget(scroll, stretch=1)

        buttons = QHBoxLayout()
        keep_all = QPushButton("Keep all (undo rejection)")
        keep_all.setObjectName("SecondaryButton")
        keep_all.clicked.connect(self._keep_all)
        buttons.addWidget(keep_all)
        buttons.addStretch()

        cancel = QPushButton("Cancel")
        cancel.setObjectName("SecondaryButton")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)

        confirm = QPushButton("Confirm")
        confirm.setObjectName("ActionButton")
        confirm.clicked.connect(self.accept)
        buttons.addWidget(confirm)
        layout.addLayout(buttons)

    def _keep_all(self) -> None:
        for check in self._checks.values():
            check.setChecked(False)

    def still_rejected(self) -> set[str]:
        """Paths the user left ticked, i.e. still rejected."""
        return {path for path, check in self._checks.items() if check.isChecked()}


# ---------------------------------------------------------------------- #
# Main window
# ---------------------------------------------------------------------- #
