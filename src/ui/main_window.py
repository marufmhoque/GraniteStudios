"""Granite Studios main window — dark-themed PyQt6 professional UI.

Layout: sidebar navigation (Library, AI Cull, Color/Filters, Reel Maker,
Web Export) + a central thumbnail grid inside a QScrollArea. Loading and
thumbnail decoding run on a worker thread so the UI never freezes, even
with folders of RAW files on modest hardware.
"""

from __future__ import annotations

import logging
from pathlib import Path

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

logger = logging.getLogger(__name__)

THUMB_SIZE = 200

# --- "Crimson Granite" theme — Nikon-red accents on deep charcoal --- #
NIKON_RED = "#E31B23"
NIKON_RED_DARK = "#8B0000"

CRIMSON_GRANITE_STYLESHEET = """
* { font-family: "Inter", "Segoe UI", "Roboto", sans-serif; }

QMainWindow, QWidget { background-color: #121212; color: #e8e8ea; font-size: 13px; }

QFrame#Sidebar { background-color: #1a1a1a; border-right: 1px solid #2a2a2a; }

QPushButton#NavButton {
    background: transparent; border: none; border-left: 3px solid transparent;
    border-radius: 8px; padding: 12px 16px; text-align: left;
    color: #a8a8ac; font-size: 14px;
}
QPushButton#NavButton:hover { background-color: #232323; color: #ffffff; }
QPushButton#NavButton:checked {
    background-color: rgba(227, 27, 35, 32);
    border-left: 3px solid #E31B23;
    color: #ffffff; font-weight: 600;
}

QPushButton#ActionButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #E31B23, stop:1 #8B0000);
    border: none; border-radius: 12px;
    padding: 10px 22px; color: #ffffff; font-weight: 700;
}
QPushButton#ActionButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #FF3A42, stop:1 #B00000);
}
QPushButton#ActionButton:pressed {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #C21620, stop:1 #6E0000);
}
QPushButton#ActionButton:disabled { background: #2a2a2a; color: #6a6a6e; }

QPushButton#SecondaryButton {
    background-color: #1e1e1e; border: 1px solid #333333; border-radius: 12px;
    padding: 9px 18px; color: #e8e8ea;
}
QPushButton#SecondaryButton:hover { background-color: #262626; border-color: #E31B23; }

QScrollArea { border: none; }
QLabel#PageTitle { font-size: 22px; font-weight: 700; color: #ffffff; }
QLabel#PageSubtitle { color: #949498; font-size: 13px; }

QLabel#Thumb {
    background-color: #1a1a1a; border: 1px solid #333333; border-radius: 12px;
}
QLabel#Thumb:hover { border: 1px solid #E31B23; }
QLabel#ThumbSelected {
    border: 2px solid #E31B23; border-radius: 12px; background-color: #1a1a1a;
}

QComboBox {
    background-color: #1e1e1e; border: 1px solid #333333; border-radius: 8px;
    padding: 6px 12px; color: #e8e8ea;
}
QComboBox:hover { border-color: #E31B23; }
QComboBox QAbstractItemView {
    background-color: #1a1a1a; color: #e8e8ea; selection-background-color: #E31B23;
    border-radius: 8px;
}

QProgressBar {
    background-color: #1a1a1a; border: none; border-radius: 6px; height: 8px; text-align: center;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #E31B23, stop:1 #8B0000);
    border-radius: 6px;
}

QCheckBox { color: #a8a8ac; spacing: 8px; }
QCheckBox::indicator {
    width: 16px; height: 16px; border-radius: 4px; border: 1px solid #333333;
    background-color: #1a1a1a;
}
QCheckBox::indicator:checked { background-color: #E31B23; border-color: #E31B23; }

QScrollBar:vertical { background: #121212; width: 10px; }
QScrollBar::handle:vertical { background: #333333; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #E31B23; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
"""

# Backward-compatible alias — kept in case other modules import the old name.
DARK_STYLESHEET = CRIMSON_GRANITE_STYLESHEET


def pil_to_pixmap(image: Image.Image) -> QPixmap:
    """Convert a PIL RGB image into a QPixmap."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    data = image.tobytes("raw", "RGB")
    qimage = QImage(data, image.width, image.height, image.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


# ---------------------------------------------------------------------- #
# Worker threads
# ---------------------------------------------------------------------- #


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
            preview = self.loader.load_preview(path)
            if preview is not None and not self._cancelled:
                thumb = preview.copy()
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


class ThumbnailCell(QLabel):
    """One selectable thumbnail in the grid."""

    clicked = pyqtSignal(str)
    web_toggled = pyqtSignal(str, bool)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path
        self.selected = False
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
        for overlay in (self.badge_label, self.label_tag, self.tech_label, self.web_checkbox):
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

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.path)
        super().mousePressEvent(event)

    def set_selected(self, selected: bool) -> None:
        self.selected = selected
        self.setObjectName("ThumbSelected" if selected else "Thumb")
        # Re-polish so the stylesheet re-applies with the new object name.
        self.style().unpolish(self)
        self.style().polish(self)


# ---------------------------------------------------------------------- #
# Main window
# ---------------------------------------------------------------------- #

NAV_ITEMS = ["Library", "AI Cull", "Color / Filters", "Reel Maker", "Web Export", "Video Editor"]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Granite Studios")
        self.resize(1360, 860)
        self.setStyleSheet(CRIMSON_GRANITE_STYLESHEET)

        # Resolve the app icon via resource_path, which checks sys._MEIPASS
        # so this works identically from source and inside a packaged
        # .exe / .app bundle. Optional — the app runs fine without it.
        icon_path = resource_path("icon.png")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Engine
        self.loader = UniversalLoader()
        self.culler = SmartCull(loader=self.loader)
        self.color_engine = ColorEngine()
        self.bg_remover = BackgroundRemover()
        self.reel_maker = ScrlReelMaker(loader=self.loader)
        self.ig_formatter = IgFormatter(loader=self.loader)

        # State
        self.current_folder: Path | None = None
        self.edit_folder: Path | None = None
        self.image_paths: list[Path] = []
        self.selected: set[str] = set()
        self.web_selected: set[str] = set()
        self.cells: dict[str, ThumbnailCell] = {}
        self.metadata: dict[str, ImageMetadata] = {}
        self.labels: dict[str, str] = {}
        self._thumb_worker: ThumbnailWorker | None = None
        self._task_worker: TaskWorker | None = None
        self._page_fade_anim: QPropertyAnimation | None = None

        # Video Editor state — kept separate from the photo library above.
        # video_source_path is the untouched original; video_current_path is
        # the working file that each operation reads from and overwrites,
        # so successive edits chain (trim -> speed -> color -> ...) without
        # ever writing back over the source, mirroring the photo side's
        # edit-safe -edit subfolder convention.
        self.video_source_path: Path | None = None
        self.video_current_path: Path | None = None
        self.video_edit_folder: Path | None = None
        self._api_server = None

        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root)

        root_layout.addWidget(self._build_sidebar())

        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_library_page())
        self.pages.addWidget(self._build_cull_page())
        self.pages.addWidget(self._build_color_page())
        self.pages.addWidget(self._build_reel_page())
        self.pages.addWidget(self._build_web_page())
        self.pages.addWidget(self._build_video_page())
        root_layout.addWidget(self.pages, stretch=1)

        self.nav_buttons[0].setChecked(True)
        self._set_nav_glow(self.nav_buttons[0], True)

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(230)
        layout = QVBoxLayout(sidebar)
        # Extra top padding (>= 20px) so the brand icon has room to breathe
        # instead of sitting flush against the window's top edge.
        layout.setContentsMargins(14, 28, 14, 20)
        layout.setSpacing(4)

        icon_path = resource_path("icon.png")
        if icon_path.exists():
            brand_icon = QLabel()
            brand_icon.setPixmap(
                QPixmap(str(icon_path)).scaled(
                    48,
                    48,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            layout.addWidget(brand_icon)
            layout.addSpacing(12)

        logo = QLabel("GRANITE\nSTUDIOS")
        logo_font = QFont()
        logo_font.setPointSize(15)
        logo_font.setBold(True)
        logo.setFont(logo_font)
        logo.setStyleSheet("color: #ffffff; letter-spacing: 2px;")
        layout.addWidget(logo)
        layout.addSpacing(24)

        self.nav_buttons: list[QPushButton] = []
        for i, name in enumerate(NAV_ITEMS):
            btn = QPushButton(name)
            btn.setObjectName("NavButton")
            btn.setCheckable(True)
            btn.clicked.connect(lambda _checked, idx=i: self._switch_page(idx))
            layout.addWidget(btn)
            self.nav_buttons.append(btn)

        layout.addStretch()

        hw = get_hardware_summary()
        hw_label = QLabel(f"Engine: {hw['tier']}")
        hw_label.setStyleSheet("color: #6a6b72; font-size: 11px;")
        hw_label.setWordWrap(True)
        layout.addWidget(hw_label)

        return sidebar

    @staticmethod
    def _page_header(title: str, subtitle: str) -> QVBoxLayout:
        header = QVBoxLayout()
        title_label = QLabel(title)
        title_label.setObjectName("PageTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("PageSubtitle")
        header.addWidget(title_label)
        header.addWidget(subtitle_label)
        return header

    # --- Library -------------------------------------------------------- #

    def _build_library_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)

        top = QHBoxLayout()
        top.addLayout(self._page_header("Library", "Universal import: RAW, HEIC, JPEG, PNG"))
        top.addStretch()
        self.load_button = QPushButton("Load Folder")
        self.load_button.setObjectName("ActionButton")
        self.load_button.clicked.connect(self._on_load_folder)
        top.addWidget(self.load_button)
        layout.addLayout(top)

        # Labeling + tech-info toolbar.
        tools = QHBoxLayout()
        tools.setSpacing(10)
        tools.addWidget(QLabel("Label:"))
        self.label_combo = QComboBox()
        self.label_combo.setEditable(True)
        self.label_combo.addItems(
            ["Graduation", "Nature", "Portraits", "Travel", "Events", "Wildlife"]
        )
        self.label_combo.setMinimumWidth(180)
        tools.addWidget(self.label_combo)

        apply_label_btn = QPushButton("Apply to Selected")
        apply_label_btn.setObjectName("SecondaryButton")
        apply_label_btn.clicked.connect(self._on_apply_label)
        tools.addWidget(apply_label_btn)

        clear_label_btn = QPushButton("Clear Label")
        clear_label_btn.setObjectName("SecondaryButton")
        clear_label_btn.clicked.connect(self._on_clear_label)
        tools.addWidget(clear_label_btn)

        tools.addStretch()

        self.tech_info_check = QCheckBox("Show Tech Info")
        self.tech_info_check.toggled.connect(self._on_toggle_tech)
        tools.addWidget(self.tech_info_check)
        layout.addLayout(tools)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.grid_scroll = QScrollArea()
        self.grid_scroll.setWidgetResizable(True)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setSpacing(12)
        self.grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.grid_scroll.setWidget(self.grid_host)
        layout.addWidget(self.grid_scroll, stretch=1)

        self.status_label = QLabel("No folder loaded. Click “Load Folder” to begin.")
        self.status_label.setObjectName("PageSubtitle")
        layout.addWidget(self.status_label)

        return page

    # --- AI Cull --------------------------------------------------------- #

    def _build_cull_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)

        top = QHBoxLayout()
        top.addLayout(
            self._page_header("AI Cull", "Sharpness scoring + burst grouping, best-of-burst picks")
        )
        top.addStretch()
        self.cull_button = QPushButton("Run Smart Cull")
        self.cull_button.setObjectName("ActionButton")
        self.cull_button.clicked.connect(self._on_run_cull)
        top.addWidget(self.cull_button)
        layout.addLayout(top)

        # Cull to Target: photographers think in deliverable counts ("the best
        # 400 of these 3,000"), not sharpness thresholds.
        target_row = QHBoxLayout()
        target_row.setSpacing(10)
        target_row.addWidget(QLabel("Cull to target:"))
        self.cull_target_input = QSpinBox()
        self.cull_target_input.setRange(1, 100000)
        self.cull_target_input.setValue(400)
        self.cull_target_input.setFixedWidth(110)
        self.cull_target_input.setToolTip(
            "How many keepers you want. The quota is spread across bursts so "
            "no part of the shoot gets dropped wholesale."
        )
        target_row.addWidget(self.cull_target_input)
        self.cull_target_button = QPushButton("Select Top N")
        self.cull_target_button.clicked.connect(self._on_cull_to_target)
        target_row.addWidget(self.cull_target_button)

        target_row.addSpacing(24)
        self.xmp_button = QPushButton("Write XMP Sidecars")
        self.xmp_button.setToolTip(
            "Write star ratings and colour labels to .xmp files beside each "
            "image, so Lightroom, Capture One, Photo Mechanic or Bridge open "
            "with these picks already applied. Your originals are not modified."
        )
        self.xmp_button.clicked.connect(self._on_write_xmp)
        target_row.addWidget(self.xmp_button)

        self.xmp_reject_check = QCheckBox("Mark non-picks as rejected")
        self.xmp_reject_check.setToolTip(
            "Write rating -1 (Lightroom's reject flag) instead of leaving "
            "non-picks unrated."
        )
        target_row.addWidget(self.xmp_reject_check)
        target_row.addStretch()
        layout.addLayout(target_row)

        self.cull_results = QLabel("Load a folder in Library, then run Smart Cull.")
        self.cull_results.setObjectName("PageSubtitle")
        self.cull_results.setWordWrap(True)
        self.cull_results.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        cull_scroll = QScrollArea()
        cull_scroll.setWidgetResizable(True)
        cull_host = QWidget()
        cull_layout = QVBoxLayout(cull_host)
        cull_layout.addWidget(self.cull_results)
        cull_layout.addStretch()
        cull_scroll.setWidget(cull_host)
        layout.addWidget(cull_scroll, stretch=1)

        return page

    # --- Color / Filters -------------------------------------------------- #

    def _build_color_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)

        top = QHBoxLayout()
        top.addLayout(
            self._page_header("Color / Filters", "Selective color isolation and cinematic grades")
        )
        top.addStretch()
        layout.addLayout(top)

        controls = QHBoxLayout()
        controls.setSpacing(12)

        controls.addWidget(QLabel("Mode:"))
        self.color_mode = QComboBox()
        self.color_mode.addItems(["Cinematic Filter", "Selective Color"])
        controls.addWidget(self.color_mode)

        controls.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(ColorEngine.FILTERS)
        controls.addWidget(self.filter_combo)

        controls.addWidget(QLabel("Isolate color:"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(ColorEngine.COLORS)
        controls.addWidget(self.color_combo)

        self.apply_color_button = QPushButton("Apply to Selected")
        self.apply_color_button.setObjectName("ActionButton")
        self.apply_color_button.clicked.connect(self._on_apply_color)
        controls.addWidget(self.apply_color_button)
        controls.addStretch()
        layout.addLayout(controls)

        # Background removal (separate row).
        bg_row = QHBoxLayout()
        bg_row.addWidget(QLabel("Remove background:"))
        self.remove_bg_button = QPushButton("Remove BG & Export as PNG")
        self.remove_bg_button.setObjectName("ActionButton")
        self.remove_bg_button.clicked.connect(self._on_remove_background)
        bg_row.addWidget(self.remove_bg_button)
        bg_row.addStretch()
        layout.addLayout(bg_row)

        self.color_preview = QLabel("Select an image in Library, choose a mode, and apply.")
        self.color_preview.setObjectName("PageSubtitle")
        self.color_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.color_preview.setMinimumHeight(480)

        preview_scroll = QScrollArea()
        preview_scroll.setWidgetResizable(True)
        preview_scroll.setWidget(self.color_preview)
        layout.addWidget(preview_scroll, stretch=1)

        self.color_status = QLabel(
            "Edited images are saved into the folder's -edit subfolder as *_graded.jpg or *_nobg.png — "
            "originals are never modified. Background removal outputs RGBA PNG with transparency."
        )
        self.color_status.setObjectName("PageSubtitle")
        layout.addWidget(self.color_status)

        return page

    # --- Reel Maker -------------------------------------------------------- #

    def _build_reel_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)

        top = QHBoxLayout()
        top.addLayout(
            self._page_header(
                "Reel Maker", "Select 3–7 images in Library → panoramic 9:16 scrolling reel"
            )
        )
        top.addStretch()
        self.reel_button = QPushButton("Create Panoramic Reel")
        self.reel_button.setObjectName("ActionButton")
        self.reel_button.clicked.connect(self._on_create_reel)
        top.addWidget(self.reel_button)
        layout.addLayout(top)

        ig_row = QHBoxLayout()
        ig_row.addWidget(QLabel("IG format for selected images:"))
        self.ig_target = QComboBox()
        self.ig_target.addItems(["feed (4:5)", "story (9:16)"])
        ig_row.addWidget(self.ig_target)
        self.ig_button = QPushButton("Export IG Formats")
        self.ig_button.setObjectName("SecondaryButton")
        self.ig_button.clicked.connect(self._on_export_ig)
        ig_row.addWidget(self.ig_button)
        ig_row.addStretch()
        layout.addLayout(ig_row)

        self.reel_status = QLabel(
            "The reel stitches your selection into one panorama and slides a "
            "1080×1920 viewport across it over 10 seconds."
        )
        self.reel_status.setObjectName("PageSubtitle")
        self.reel_status.setWordWrap(True)
        layout.addWidget(self.reel_status)
        layout.addStretch()

        return page

    # --- Web Export ---------------------------------------------------------- #

    def _build_web_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)

        top = QHBoxLayout()
        top.addLayout(
            self._page_header("Web Export", "Responsive Tailwind masonry gallery, ready to host")
        )
        top.addStretch()
        self.web_button = QPushButton("Generate Gallery")
        self.web_button.setObjectName("ActionButton")
        self.web_button.clicked.connect(self._on_generate_gallery)
        top.addWidget(self.web_button)
        layout.addLayout(top)

        self.web_selection_status = QLabel("0 images marked for web (checked in Library).")
        self.web_selection_status.setObjectName("PageSubtitle")
        layout.addWidget(self.web_selection_status)

        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        title_row.addWidget(QLabel("Gallery Name:"))
        self.gallery_title_input = QLineEdit()
        self.gallery_title_input.setPlaceholderText("e.g. Vacation 2024 Highlights")
        title_row.addWidget(self.gallery_title_input, stretch=1)
        layout.addLayout(title_row)

        self.web_group_by_label = QCheckBox("Organize into category tabs by assigned label")
        self.web_group_by_label.setChecked(True)
        layout.addWidget(self.web_group_by_label)

        self.web_auto_fix = QCheckBox(
            "Auto-Edit on export (white balance + histogram equalization)"
        )
        layout.addWidget(self.web_auto_fix)

        self.web_status = QLabel(
            "Check “Publish to Web” on thumbnails in Library to choose what gets "
            "published. Sources are pulled from the folder's -edit subfolder (never "
            "the originals), resized to max 2000px, and published with an index.html "
            "masonry grid (Tailwind CSS via CDN). Camera + lens info appears under "
            "each photo."
        )
        self.web_status.setObjectName("PageSubtitle")
        self.web_status.setWordWrap(True)
        self.web_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.web_status)
        layout.addStretch()

        return page

    # --- Video Editor ---------------------------------------------------- #

    def _build_video_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(28, 24, 28, 24)
        outer.setSpacing(14)

        top = QHBoxLayout()
        top.addLayout(
            self._page_header(
                "Video Editor",
                "Trim, speed-ramp, reframe, stabilize, color-grade and export video "
                "— FFmpeg + OpenCV powered, runs locally",
            )
        )
        top.addStretch()
        self.video_load_button = QPushButton("Load Video")
        self.video_load_button.setObjectName("ActionButton")
        self.video_load_button.clicked.connect(self._on_load_video)
        top.addWidget(self.video_load_button)
        outer.addLayout(top)

        self.video_info_label = QLabel("No video loaded.")
        self.video_info_label.setObjectName("PageSubtitle")
        self.video_info_label.setWordWrap(True)
        outer.addWidget(self.video_info_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        form = QVBoxLayout(inner)
        form.setSpacing(12)

        # --- Timing ---
        timing_row = QHBoxLayout()
        timing_row.addWidget(QLabel("Trim start (s):"))
        self.trim_start_input = QLineEdit("0")
        self.trim_start_input.setFixedWidth(60)
        timing_row.addWidget(self.trim_start_input)
        timing_row.addWidget(QLabel("end (s, blank = clip end):"))
        self.trim_end_input = QLineEdit("")
        self.trim_end_input.setFixedWidth(60)
        timing_row.addWidget(self.trim_end_input)
        trim_btn = QPushButton("Trim")
        trim_btn.setObjectName("SecondaryButton")
        trim_btn.clicked.connect(self._on_video_trim)
        timing_row.addWidget(trim_btn)
        reverse_btn = QPushButton("Reverse")
        reverse_btn.setObjectName("SecondaryButton")
        reverse_btn.clicked.connect(self._on_video_reverse)
        timing_row.addWidget(reverse_btn)
        timing_row.addStretch()
        form.addLayout(timing_row)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed factor (0.1–8.0, 1.0 = normal):"))
        self.speed_factor_input = QLineEdit("1.0")
        self.speed_factor_input.setFixedWidth(50)
        speed_row.addWidget(self.speed_factor_input)
        speed_btn = QPushButton("Change Speed")
        speed_btn.setObjectName("SecondaryButton")
        speed_btn.clicked.connect(self._on_video_speed)
        speed_row.addWidget(speed_btn)

        speed_row.addWidget(QLabel("Timelapse target (s):"))
        self.timelapse_target_input = QLineEdit("30")
        self.timelapse_target_input.setFixedWidth(50)
        speed_row.addWidget(self.timelapse_target_input)
        timelapse_btn = QPushButton("Extract Timelapse")
        timelapse_btn.setObjectName("SecondaryButton")
        timelapse_btn.clicked.connect(self._on_video_timelapse)
        speed_row.addWidget(timelapse_btn)
        speed_row.addStretch()
        form.addLayout(speed_row)

        # --- Spatial ---
        spatial_row = QHBoxLayout()
        spatial_row.addWidget(QLabel("Aspect ratio:"))
        self.aspect_combo = QComboBox()
        self.aspect_combo.addItems(list(video_spatial.ASPECT_PRESETS.keys()))
        self.aspect_combo.setCurrentText("9:16")
        spatial_row.addWidget(self.aspect_combo)
        self.aspect_mode_combo = QComboBox()
        self.aspect_mode_combo.addItems(["crop", "pad"])
        spatial_row.addWidget(self.aspect_mode_combo)
        aspect_btn = QPushButton("Convert Aspect")
        aspect_btn.setObjectName("SecondaryButton")
        aspect_btn.clicked.connect(self._on_video_aspect)
        spatial_row.addWidget(aspect_btn)

        autocrop_btn = QPushButton("Auto-Crop to Subject")
        autocrop_btn.setObjectName("SecondaryButton")
        autocrop_btn.clicked.connect(self._on_video_autocrop)
        spatial_row.addWidget(autocrop_btn)

        stabilize_btn = QPushButton("Stabilize")
        stabilize_btn.setObjectName("SecondaryButton")
        stabilize_btn.clicked.connect(self._on_video_stabilize)
        spatial_row.addWidget(stabilize_btn)
        spatial_row.addStretch()
        form.addLayout(spatial_row)

        # --- Color ---
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Brightness:"))
        self.vbright_input = QLineEdit("0.0")
        self.vbright_input.setFixedWidth(45)
        color_row.addWidget(self.vbright_input)
        color_row.addWidget(QLabel("Contrast:"))
        self.vcontrast_input = QLineEdit("1.0")
        self.vcontrast_input.setFixedWidth(45)
        color_row.addWidget(self.vcontrast_input)
        color_row.addWidget(QLabel("Saturation:"))
        self.vsat_input = QLineEdit("1.0")
        self.vsat_input.setFixedWidth(45)
        color_row.addWidget(self.vsat_input)
        color_btn = QPushButton("Apply Color Correction")
        color_btn.setObjectName("SecondaryButton")
        color_btn.clicked.connect(self._on_video_color)
        color_row.addWidget(color_btn)
        color_row.addStretch()
        form.addLayout(color_row)

        # --- Audio ---
        audio_row = QHBoxLayout()
        denoise_btn = QPushButton("Reduce Audio Noise")
        denoise_btn.setObjectName("SecondaryButton")
        denoise_btn.clicked.connect(self._on_video_denoise)
        audio_row.addWidget(denoise_btn)
        loudness_btn = QPushButton("Normalize Loudness")
        loudness_btn.setObjectName("SecondaryButton")
        loudness_btn.clicked.connect(self._on_video_loudness)
        audio_row.addWidget(loudness_btn)
        audio_row.addStretch()
        form.addLayout(audio_row)

        # --- Export / Reset ---
        export_row = QHBoxLayout()
        export_row.addWidget(QLabel("Export preset:"))
        self.video_preset_combo = QComboBox()
        self.video_preset_combo.addItems(list(video_export.PRESETS.keys()))
        export_row.addWidget(self.video_preset_combo)
        export_btn = QPushButton("Export")
        export_btn.setObjectName("ActionButton")
        export_btn.clicked.connect(self._on_video_export)
        export_row.addWidget(export_btn)
        reset_btn = QPushButton("Reset to Original")
        reset_btn.setObjectName("SecondaryButton")
        reset_btn.clicked.connect(self._on_video_reset)
        export_row.addWidget(reset_btn)
        export_row.addStretch()
        form.addLayout(export_row)

        api_row = QHBoxLayout()
        self.api_server_button = QPushButton("Start Local API Server")
        self.api_server_button.setObjectName("SecondaryButton")
        self.api_server_button.clicked.connect(self._on_toggle_api_server)
        api_row.addWidget(self.api_server_button)
        self.api_server_status = QLabel("REST API not running.")
        self.api_server_status.setObjectName("PageSubtitle")
        api_row.addWidget(self.api_server_status)
        api_row.addStretch()
        form.addLayout(api_row)

        form.addStretch()
        scroll.setWidget(inner)
        outer.addWidget(scroll, stretch=1)

        self.video_status = QLabel(
            "Load a video to begin. Every operation writes a new file into a "
            "<video-name>-edit folder next to the source — the original is "
            "never modified. Operations chain: each one builds on the last "
            "saved result until you Export or Reset."
        )
        self.video_status.setObjectName("PageSubtitle")
        self.video_status.setWordWrap(True)
        outer.addWidget(self.video_status)

        return page

    # ------------------------------------------------------------------ #
    # Navigation & selection
    # ------------------------------------------------------------------ #

    def _switch_page(self, index: int) -> None:
        if index == self.pages.currentIndex():
            return
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)
            self._set_nav_glow(btn, i == index)
        self.pages.setCurrentIndex(index)
        self._fade_in_current_page()

    def _set_nav_glow(self, btn: QPushButton, active: bool) -> None:
        """Give the active nav item a subtle red glow (Nikon-red accent)."""
        if not active:
            btn.setGraphicsEffect(None)
            return
        glow = QGraphicsDropShadowEffect(btn)
        glow.setColor(QColor(NIKON_RED))
        glow.setBlurRadius(28)
        glow.setOffset(0, 0)
        btn.setGraphicsEffect(glow)

    def _fade_in_current_page(self) -> None:
        """Smooth 200ms fade-in when switching between Library/Cull/etc tabs."""
        page = self.pages.currentWidget()
        effect = QGraphicsOpacityEffect(page)
        page.setGraphicsEffect(effect)

        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(200)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.finished.connect(lambda: page.setGraphicsEffect(None))
        # Keep a reference so the animation isn't garbage-collected mid-flight.
        self._page_fade_anim = anim
        anim.start()

    def _on_thumb_clicked(self, path: str) -> None:
        cell = self.cells.get(path)
        if cell is None:
            return
        if path in self.selected:
            self.selected.discard(path)
            cell.set_selected(False)
        else:
            self.selected.add(path)
            cell.set_selected(True)
        self.status_label.setText(
            f"{len(self.image_paths)} images · {len(self.selected)} selected"
        )

    def _selected_paths(self) -> list[Path]:
        return [p for p in self.image_paths if str(p) in self.selected]

    # ------------------------------------------------------------------ #
    # Library actions
    # ------------------------------------------------------------------ #

    def _on_load_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose an image folder")
        if not folder:
            return

        self.current_folder = Path(folder)
        # All edits/exports/keepers land here — originals are never touched.
        self.edit_folder = self.loader.get_edit_folder(self.current_folder)
        self.image_paths = self.loader.scan_folder(self.current_folder)
        self.selected.clear()
        self.web_selected.clear()
        self.metadata.clear()
        self.labels.clear()

        # Clear the grid.
        if self._thumb_worker is not None:
            self._thumb_worker.cancel()
            self._thumb_worker.wait()
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.cells.clear()

        if not self.image_paths:
            self.status_label.setText("No supported images found in that folder.")
            return

        columns = max(1, (self.grid_scroll.viewport().width() - 24) // (THUMB_SIZE + 24))
        for i, path in enumerate(self.image_paths):
            cell = ThumbnailCell(str(path))
            cell.clicked.connect(self._on_thumb_clicked)
            cell.web_toggled.connect(self._on_web_toggle)
            self.grid.addWidget(cell, i // columns, i % columns)
            self.cells[str(path)] = cell

        self.progress.setVisible(True)
        self.progress.setMaximum(len(self.image_paths))
        self.progress.setValue(0)
        self.status_label.setText(
            f"Loading {len(self.image_paths)} images… "
            f"Edits and web keepers will be saved to {self.edit_folder.name}/"
        )

        self._thumb_worker = ThumbnailWorker(self.loader, list(self.image_paths), parent=self)
        self._thumb_worker.thumb_ready.connect(self._on_thumb_ready)
        self._thumb_worker.meta_ready.connect(self._on_meta_ready)
        self._thumb_worker.progress.connect(self._on_thumb_progress)
        self._thumb_worker.finished_all.connect(self._on_thumbs_done)
        self._thumb_worker.start()

    def _on_thumb_ready(self, path: str, pixmap: QPixmap) -> None:
        cell = self.cells.get(path)
        if cell is not None:
            cell.set_pixmap(pixmap)

    def _on_meta_ready(self, path: str, metadata: object) -> None:
        self.metadata[path] = metadata  # type: ignore[assignment]
        cell = self.cells.get(path)
        if cell is not None:
            cell.set_metadata(metadata)  # type: ignore[arg-type]
            cell.set_tech_visible(self.tech_info_check.isChecked())

    def _on_thumb_progress(self, done: int, total: int) -> None:
        self.progress.setValue(done)

    def _on_thumbs_done(self) -> None:
        self.progress.setVisible(False)
        self.status_label.setText(
            f"{len(self.image_paths)} images · {len(self.selected)} selected"
        )

    # ------------------------------------------------------------------ #
    # Labeling & tech info
    # ------------------------------------------------------------------ #

    def _on_apply_label(self) -> None:
        targets = self._selected_paths()
        if not targets:
            QMessageBox.information(
                self, "No selection", "Select one or more images in Library first."
            )
            return
        label = self.label_combo.currentText().strip()
        if not label:
            QMessageBox.information(self, "No label", "Type or choose a label first.")
            return
        for path in targets:
            key = str(path)
            self.labels[key] = label
            cell = self.cells.get(key)
            if cell is not None:
                cell.set_label(label)
        self.status_label.setText(f"Labeled {len(targets)} image(s) as “{label}”.")

    def _on_clear_label(self) -> None:
        targets = self._selected_paths()
        if not targets:
            return
        for path in targets:
            key = str(path)
            self.labels.pop(key, None)
            cell = self.cells.get(key)
            if cell is not None:
                cell.set_label("")
        self.status_label.setText(f"Cleared labels on {len(targets)} image(s).")

    def _on_toggle_tech(self, checked: bool) -> None:
        for cell in self.cells.values():
            cell.set_tech_visible(checked)

    def _on_web_toggle(self, path: str, checked: bool) -> None:
        """Track which images are checked for web publishing.

        Pure state update — no I/O here, so toggling stays instant even
        when batch-checking many thumbnails. The actual "keeper" JPEG in
        the -edit subfolder is materialized lazily at export time (see
        _on_generate_gallery), off the UI thread.
        """
        if checked:
            self.web_selected.add(path)
        else:
            self.web_selected.discard(path)
        self._update_web_status()

    def _update_web_status(self) -> None:
        count = len(self.web_selected)
        self.status_label.setText(
            f"{len(self.image_paths)} images · {len(self.selected)} selected · "
            f"{count} marked for web"
        )
        if hasattr(self, "web_selection_status"):
            noun = "image" if count == 1 else "images"
            self.web_selection_status.setText(f"{count} {noun} marked for web (checked in Library).")

    # ------------------------------------------------------------------ #
    # Task plumbing
    # ------------------------------------------------------------------ #

    def _run_task(self, fn, on_done, busy_button: QPushButton) -> None:
        if self._task_worker is not None and self._task_worker.isRunning():
            QMessageBox.information(self, "Busy", "Another task is still running.")
            return

        busy_button.setEnabled(False)

        def _finish(result):
            busy_button.setEnabled(True)
            on_done(result)

        def _fail(message: str):
            busy_button.setEnabled(True)
            QMessageBox.critical(self, "Task failed", message)

        self._task_worker = TaskWorker(fn, parent=self)
        self._task_worker.done.connect(_finish)
        self._task_worker.failed.connect(_fail)
        self._task_worker.start()

    def _require_library(self) -> bool:
        if not self.image_paths:
            QMessageBox.information(self, "No images", "Load a folder in Library first.")
            return False
        return True

    # ------------------------------------------------------------------ #
    # AI Cull actions
    # ------------------------------------------------------------------ #

    def _on_run_cull(self) -> None:
        if not self._require_library():
            return

        paths = list(self.image_paths)
        self.cull_results.setText(
            "Analyzing sharpness, hashing frames and grouping similar shots…"
        )

        def work():
            # Smart grouping: combine capture time + perceptual hash, then
            # star the top 2 sharpest of each group and flag the rest.
            return self.culler.group_similar(paths, top_count=2)

        def done(groups):
            starred = sum(1 for g in groups for f in g if f.is_starred)
            duplicates = sum(1 for g in groups for f in g if f.is_duplicate)
            lines = [
                f"Found {len(groups)} similar-image group(s) across {len(paths)} images.",
                f"★ {starred} starred · {duplicates} flagged as duplicates.\n",
            ]
            for i, group in enumerate(groups, start=1):
                lines.append(f"Group {i} — {len(group)} frame(s):")
                for frame in group:
                    if frame.is_starred:
                        marker = "★ STAR"
                    elif frame.is_duplicate:
                        marker = "  dup "
                    else:
                        marker = "      "
                    lines.append(
                        f"  {marker}  {frame.path.name}   sharpness {frame.sharpness:,.0f}"
                    )
                lines.append("")
            self.cull_results.setText("\n".join(lines))

            # Reflect star / duplicate state onto the Library grid.
            for group in groups:
                for frame in group:
                    cell = self.cells.get(str(frame.path))
                    if cell is not None:
                        cell.set_pick_state(
                            starred=frame.is_starred,
                            duplicate=frame.is_duplicate,
                            detail=f"sharpness {frame.sharpness:,.0f}",
                        )

        self._run_task(work, done, self.cull_button)

    def _apply_cull_state_to_grid(self, frames) -> None:
        """Mirror star / duplicate state onto the Library thumbnails."""
        for frame in frames:
            cell = self.cells.get(str(frame.path))
            if cell is not None:
                cell.set_pick_state(
                    starred=frame.is_starred,
                    duplicate=frame.is_duplicate,
                    detail=f"sharpness {frame.sharpness:,.0f}",
                )

    def _on_cull_to_target(self) -> None:
        """Select a target number of keepers across the whole folder."""
        if not self._require_library():
            return
        folder = self.current_folder
        target = self.cull_target_input.value()
        self.cull_results.setText(
            f"Scoring every frame and selecting the best {target:,}…"
        )

        def work():
            return self.culler.cull_to_target(folder, target)

        def done(frames):
            self.last_cull_frames = frames
            picks = [f for f in frames if f.is_starred]
            lines = [
                f"Selected {len(picks):,} of {len(frames):,} images "
                f"(target was {target:,}).",
                "The quota is spread across bursts, so each moment keeps its "
                "best frames rather than the sharpest stretch taking everything.\n",
                "★ Keepers:",
            ]
            for f in picks:
                lines.append(f"  ★  {f.path.name}   sharpness {f.sharpness:,.0f}")
            rest = [f for f in frames if not f.is_starred]
            if rest:
                lines.append(f"\nNot selected ({len(rest):,}):")
                for f in rest[:200]:
                    lines.append(f"     {f.path.name}   sharpness {f.sharpness:,.0f}")
                if len(rest) > 200:
                    lines.append(f"     … and {len(rest) - 200:,} more")
            self.cull_results.setText("\n".join(lines))
            self._apply_cull_state_to_grid(frames)

        self._run_task(work, done, self.cull_target_button)

    def _on_write_xmp(self) -> None:
        """Write the current cull decisions out as XMP sidecars."""
        frames = getattr(self, "last_cull_frames", None)
        if not frames:
            QMessageBox.information(
                self,
                "Nothing to write",
                "Run Smart Cull or Select Top N first, so there are picks to write.",
            )
            return

        label_rejects = self.xmp_reject_check.isChecked()
        count = len(frames)

        def work():
            return xmp_sidecar.write_cull_results(
                frames, stars_for_picks=2, label_rejects=label_rejects
            )

        def done(written):
            picks = sum(1 for f in frames if f.is_starred)
            msg = (
                f"Wrote {written:,} XMP sidecar(s) for {count:,} image(s).\n"
                f"{picks:,} keeper(s) rated ★★."
            )
            if label_rejects:
                msg += "\nNon-picks marked as rejected."
            self.cull_results.setText(
                msg
                + "\n\nOpen the folder in Lightroom, Capture One, Photo Mechanic or "
                  "Bridge and the ratings will already be there. Your original "
                  "files were not modified."
            )

        self._run_task(work, done, self.xmp_button)

    # ------------------------------------------------------------------ #
    # Color actions
    # ------------------------------------------------------------------ #

    def _on_apply_color(self) -> None:
        targets = self._selected_paths()
        if not targets:
            QMessageBox.information(
                self, "No selection", "Select one or more images in Library first."
            )
            return
        if self.edit_folder is None:
            QMessageBox.information(self, "No folder", "Load a folder in Library first.")
            return

        mode = self.color_mode.currentText()
        filter_name = self.filter_combo.currentText()
        color_name = self.color_combo.currentText()
        edit_folder = self.edit_folder

        def work():
            outputs: list[Path] = []
            last_preview = None
            for path in targets:
                arr = self.loader.load_array(path)
                if arr is None:
                    continue
                if mode == "Selective Color":
                    result = self.color_engine.selective_color(arr, color_name)
                else:
                    result = self.color_engine.apply_lut_filter(arr, filter_name)
                # Graded output goes into the -edit subfolder, never next to
                # the original — this is also what the web publisher pulls
                # from when this image is checked for web (see Task 3).
                out_path = edit_folder / f"{path.stem}_graded.jpg"
                Image.fromarray(result).save(out_path, "JPEG", quality=94)
                outputs.append(out_path)
                last_preview = Image.fromarray(result)
            return outputs, last_preview

        def done(result):
            outputs, last_preview = result
            if last_preview is not None:
                preview = last_preview.copy()
                preview.thumbnail((1100, 700), Image.Resampling.LANCZOS)
                self.color_preview.setPixmap(pil_to_pixmap(preview))
            self.color_status.setText(
                f"Saved {len(outputs)} graded image(s) to {edit_folder.name}/ "
                "(originals untouched)."
            )

        self._run_task(work, done, self.apply_color_button)

    def _on_remove_background(self) -> None:
        """Remove background from selected images, export as transparent PNG."""
        targets = self._selected_paths()
        if not targets:
            QMessageBox.information(
                self, "No selection", "Select one or more images in Library first."
            )
            return
        if self.edit_folder is None:
            QMessageBox.information(self, "No folder", "Load a folder in Library first.")
            return

        edit_folder = self.edit_folder

        def work():
            outputs: list[Path] = []
            last_preview = None
            for path in targets:
                arr = self.loader.load_array(path)
                if arr is None:
                    continue
                # Remove background using AI segmentation, returns RGBA.
                result_rgba = self.bg_remover.remove_background_array(arr)
                out_path = edit_folder / f"{path.stem}_nobg.png"
                Image.fromarray(result_rgba, "RGBA").save(out_path, "PNG", optimize=True)
                outputs.append(out_path)
                # Preview is the alpha channel so the user sees what was kept.
                if len(result_rgba.shape) == 3 and result_rgba.shape[2] == 4:
                    alpha = result_rgba[:, :, 3]
                    # Convert grayscale alpha to RGB for display.
                    rgb_alpha = np.stack([alpha, alpha, alpha], axis=2)
                    last_preview = Image.fromarray(rgb_alpha.astype(np.uint8))
            return outputs, last_preview

        def done(result):
            outputs, last_preview = result
            if last_preview is not None:
                preview = last_preview.copy()
                preview.thumbnail((1100, 700), Image.Resampling.LANCZOS)
                self.color_preview.setPixmap(pil_to_pixmap(preview))
            self.color_status.setText(
                f"Removed background from {len(outputs)} image(s), saved as *_nobg.png "
                "(RGBA with transparency) to {edit_folder.name}/."
            )

        self._run_task(work, done, self.remove_bg_button)

    # ------------------------------------------------------------------ #
    # Reel / IG actions
    # ------------------------------------------------------------------ #

    def _on_create_reel(self) -> None:
        targets = self._selected_paths()
        if not 3 <= len(targets) <= 7:
            QMessageBox.information(
                self,
                "Selection needed",
                f"Select 3–7 images in Library (currently {len(targets)}).",
            )
            return

        out, _ = QFileDialog.getSaveFileName(
            self, "Save reel as", "panoramic_reel.mp4", "MP4 video (*.mp4)"
        )
        if not out:
            return

        self.reel_status.setText("Rendering reel… this can take a minute.")

        def work():
            return self.reel_maker.create_panoramic_reel(targets, out)

        def done(path):
            self.reel_status.setText(f"Reel saved: {path}")

        self._run_task(work, done, self.reel_button)

    def _on_export_ig(self) -> None:
        targets = self._selected_paths()
        if not targets:
            QMessageBox.information(
                self, "No selection", "Select one or more images in Library first."
            )
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Choose export folder")
        if not out_dir:
            return

        target = "feed" if self.ig_target.currentIndex() == 0 else "story"

        def work():
            return self.ig_formatter.export_batch(targets, out_dir, target=target)

        def done(written):
            self.reel_status.setText(f"Exported {len(written)} IG {target} image(s) to {out_dir}")

        self._run_task(work, done, self.ig_button)

    # ------------------------------------------------------------------ #
    # Web export actions
    # ------------------------------------------------------------------ #

    def _on_generate_gallery(self) -> None:
        if not self._require_library():
            return

        if not self.web_selected:
            QMessageBox.information(
                self,
                "Nothing marked for web",
                'Check "Publish to Web" on one or more thumbnails in Library first.',
            )
            return
        if self.edit_folder is None or self.current_folder is None:
            QMessageBox.information(self, "No folder", "Load a folder in Library first.")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Choose gallery output folder")
        if not out_dir:
            return

        targets = [Path(p) for p in self.web_selected]
        group_by_label = self.web_group_by_label.isChecked()
        auto_fix = self.web_auto_fix.isChecked()
        gallery_title = self.gallery_title_input.text().strip() or "Granite Studios Gallery"
        edit_folder = self.edit_folder
        current_folder = self.current_folder

        self.web_status.setText(f"Publishing {len(targets)} images…")

        def work():
            # Source images always come from the -edit subfolder — the
            # graded version if one exists, otherwise a flattened "keeper"
            # JPEG materialized here on demand. Originals are never read
            # from directly for the website.
            gallery: dict[str, list[dict]] = {}
            for original in targets:
                key = str(original)
                graded = edit_folder / f"{original.stem}_graded.jpg"
                source = graded if graded.exists() else self.loader.export_keeper(
                    original, current_folder
                )

                category = self.labels.get(key, "Unlabeled") if group_by_label else "All Photos"
                meta = self.metadata.get(key)
                gallery.setdefault(category, []).append(
                    {
                        "path": source,
                        "gear": meta.gear_label if isinstance(meta, ImageMetadata) else "",
                    }
                )
            return generate_static_gallery(gallery, out_dir, title=gallery_title, auto_fix=auto_fix)

        def done(index_path):
            self.web_status.setText(
                f"Gallery “{gallery_title}” published: {index_path}\nOpen index.html in a browser."
            )

        self._run_task(work, done, self.web_button)

    # ------------------------------------------------------------------ #
    # Video Editor actions
    # ------------------------------------------------------------------ #

    def _on_load_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose a video",
            "",
            "Video files (*.mp4 *.mov *.avi *.mkv *.webm *.flv);;All files (*.*)",
        )
        if not path:
            return
        self.video_source_path = Path(path)
        self.video_current_path = self.video_source_path
        self.video_edit_folder = self.video_source_path.parent / f"{self.video_source_path.stem}-edit"
        self.video_edit_folder.mkdir(exist_ok=True)
        self._refresh_video_info()
        self.video_status.setText(f"Loaded {self.video_source_path.name}.")

    def _refresh_video_info(self) -> None:
        if self.video_current_path is None:
            self.video_info_label.setText("No video loaded.")
            return
        try:
            info = video_probe(self.video_current_path)
        except FFmpegError as exc:
            self.video_info_label.setText(f"Could not probe video: {exc}")
            return
        self.video_info_label.setText(
            f"{self.video_current_path.name}  ·  {info.width}x{info.height}  ·  "
            f"{info.fps:.2f} fps  ·  {info.duration_label}  ·  codec {info.codec_fourcc}  ·  "
            f"{'audio' if info.has_audio else 'no audio'}"
        )

    def _require_video(self) -> bool:
        if self.video_current_path is None:
            QMessageBox.information(self, "No video", "Load a video first.")
            return False
        return True

    def _video_output_path(self, suffix: str) -> Path:
        stem = self.video_source_path.stem
        return self.video_edit_folder / f"{stem}_{suffix}.mp4"

    def _run_video_op(self, fn, suffix: str, **kwargs) -> None:
        if not self._require_video():
            return
        src = self.video_current_path
        out_path = self._video_output_path(suffix)
        self.video_status.setText(f"Running {suffix}…")

        def work():
            return fn(src, out_path, **kwargs)

        def done(result_path):
            self.video_current_path = Path(result_path)
            self._refresh_video_info()
            self.video_status.setText(f"Saved: {result_path}")

        self._run_task(work, done, self.video_load_button)

    def _on_video_trim(self) -> None:
        try:
            start = float(self.trim_start_input.text() or 0.0)
            end_text = self.trim_end_input.text().strip()
            end = float(end_text) if end_text else None
        except ValueError:
            QMessageBox.warning(self, "Invalid input", "Trim start/end must be numbers (seconds).")
            return
        self._run_video_op(video_timing.trim, "trimmed", start_sec=start, end_sec=end)

    def _on_video_reverse(self) -> None:
        self._run_video_op(video_timing.reverse, "reversed")

    def _on_video_speed(self) -> None:
        try:
            factor = float(self.speed_factor_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid input", "Speed factor must be a number.")
            return
        self._run_video_op(video_timing.change_speed, "speed", factor=factor)

    def _on_video_timelapse(self) -> None:
        try:
            target = float(self.timelapse_target_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid input", "Timelapse target must be a number of seconds.")
            return
        self._run_video_op(video_timing.extract_timelapse, "timelapse", target_duration_sec=target)

    def _on_video_aspect(self) -> None:
        target = self.aspect_combo.currentText()
        mode = self.aspect_mode_combo.currentText()
        self._run_video_op(video_spatial.convert_aspect_ratio, "aspect", target=target, mode=mode)

    def _on_video_autocrop(self) -> None:
        target = self.aspect_combo.currentText()
        self._run_video_op(video_spatial.auto_crop_subject, "autocrop", target_aspect=target)

    def _on_video_stabilize(self) -> None:
        self._run_video_op(video_spatial.stabilize, "stabilized")

    def _on_video_color(self) -> None:
        try:
            brightness = float(self.vbright_input.text())
            contrast = float(self.vcontrast_input.text())
            saturation = float(self.vsat_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid input", "Brightness/contrast/saturation must be numbers.")
            return
        self._run_video_op(
            video_color.color_correct, "color",
            brightness=brightness, contrast=contrast, saturation=saturation,
        )

    def _on_video_denoise(self) -> None:
        self._run_video_op(video_audio.reduce_noise, "denoised")

    def _on_video_loudness(self) -> None:
        self._run_video_op(video_audio.normalize_loudness, "normalized")

    def _on_video_reset(self) -> None:
        if self.video_source_path is None:
            return
        self.video_current_path = self.video_source_path
        self._refresh_video_info()
        self.video_status.setText("Reset to original source video.")

    def _on_video_export(self) -> None:
        if not self._require_video():
            return
        preset = self.video_preset_combo.currentText()
        default_name = f"{self.video_source_path.stem}_export.{video_export.PRESETS[preset].container}"
        out, _ = QFileDialog.getSaveFileName(self, "Export video as", default_name, "Video files (*.*)")
        if not out:
            return
        src = self.video_current_path
        self.video_status.setText(f"Exporting with preset “{preset}”…")

        def work():
            return video_export.export(src, out, preset=preset)

        def done(result_path):
            self.video_status.setText(f"Exported: {result_path}")

        self._run_task(work, done, self.video_load_button)

    def _on_toggle_api_server(self) -> None:
        if self._api_server is not None:
            self._api_server.should_exit = True
            self._api_server = None
            self.api_server_button.setText("Start Local API Server")
            self.api_server_status.setText("REST API not running.")
            return
        try:
            from src.video.api import run_api_server

            self._api_server = run_api_server()
        except ImportError as exc:
            QMessageBox.warning(
                self,
                "Missing dependency",
                f"The local API server needs fastapi + uvicorn.\n\n{exc}",
            )
            return
        self.api_server_button.setText("Stop Local API Server")
        self.api_server_status.setText("REST API running at http://127.0.0.1:8756/docs")

    # ------------------------------------------------------------------ #
    # Shutdown
    # ------------------------------------------------------------------ #

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self._thumb_worker is not None:
            self._thumb_worker.cancel()
            self._thumb_worker.wait()
        if self._task_worker is not None and self._task_worker.isRunning():
            self._task_worker.wait(5000)
        if self._api_server is not None:
            self._api_server.should_exit = True
        super().closeEvent(event)


def run() -> int:
    """Create the QApplication and run the main window."""
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = QApplication(sys.argv)
    app.setApplicationName("Granite Studios")

    app_font = QFont()
    app_font.setFamilies(["Inter", "Segoe UI", "Roboto", "sans-serif"])
    app_font.setPointSize(10)
    app.setFont(app_font)

    window = MainWindow()
    window.show()
    return app.exec()
