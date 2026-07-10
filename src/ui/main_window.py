"""Granite Studios main window — dark-themed PyQt6 professional UI.

Layout: sidebar navigation (Library, AI Cull, Color/Filters, Reel Maker,
Web Export) + a central thumbnail grid inside a QScrollArea. Loading and
thumbnail decoding run on a worker thread so the UI never freezes, even
with folders of RAW files on modest hardware.
"""

from __future__ import annotations

import logging
from pathlib import Path

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
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.engine.hardware import get_hardware_summary
from src.engine.resources import resource_path
from src.engine.universal_loader import UniversalLoader, ImageMetadata
from src.ai_tools.culling import SmartCull
from src.ai_tools.color_engine import ColorEngine
from src.social.reel_maker import ScrlReelMaker
from src.social.ig_sync import IgFormatter
from src.web.publisher import generate_static_gallery

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

        # Bottom strip: EXIF tech info (camera | lens + exposure).
        self.tech_label = QLabel(self)
        self.tech_label.setGeometry(6, h - 44, w - 12, 38)
        self.tech_label.setWordWrap(True)
        self.tech_label.setStyleSheet(
            "background: rgba(10,11,14,0.82); color: #d8d9de; border-radius: 6px;"
            " padding: 3px 6px; font-size: 10px;"
        )
        self.tech_label.hide()

    def _raise_overlays(self) -> None:
        for overlay in (self.badge_label, self.label_tag, self.tech_label):
            overlay.raise_()

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

NAV_ITEMS = ["Library", "AI Cull", "Color / Filters", "Reel Maker", "Web Export"]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Granite Studios")
        self.resize(1360, 860)
        self.setStyleSheet(CRIMSON_GRANITE_STYLESHEET)

        # Resolve the app icon via resource_path so it's found both from
        # source and inside a packaged .exe / .app bundle. Optional — the
        # app runs fine with no icon file present.
        icon_path = resource_path("assets/icon.png")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Engine
        self.loader = UniversalLoader()
        self.culler = SmartCull(loader=self.loader)
        self.color_engine = ColorEngine()
        self.reel_maker = ScrlReelMaker(loader=self.loader)
        self.ig_formatter = IgFormatter(loader=self.loader)

        # State
        self.current_folder: Path | None = None
        self.image_paths: list[Path] = []
        self.selected: set[str] = set()
        self.cells: dict[str, ThumbnailCell] = {}
        self.metadata: dict[str, ImageMetadata] = {}
        self.labels: dict[str, str] = {}
        self._thumb_worker: ThumbnailWorker | None = None
        self._task_worker: TaskWorker | None = None
        self._page_fade_anim: QPropertyAnimation | None = None

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
        root_layout.addWidget(self.pages, stretch=1)

        self.nav_buttons[0].setChecked(True)
        self._set_nav_glow(self.nav_buttons[0], True)

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(230)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 20, 14, 20)
        layout.setSpacing(4)

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

        self.color_preview = QLabel("Select an image in Library, choose a mode, and apply.")
        self.color_preview.setObjectName("PageSubtitle")
        self.color_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.color_preview.setMinimumHeight(480)

        preview_scroll = QScrollArea()
        preview_scroll.setWidgetResizable(True)
        preview_scroll.setWidget(self.color_preview)
        layout.addWidget(preview_scroll, stretch=1)

        self.color_status = QLabel("Edited images are saved next to the originals as *_graded.jpg")
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

        self.web_selected_only = QCheckBox("Export selected images only (otherwise: whole library)")
        layout.addWidget(self.web_selected_only)

        self.web_group_by_label = QCheckBox("Organize into category tabs by assigned label")
        self.web_group_by_label.setChecked(True)
        layout.addWidget(self.web_group_by_label)

        self.web_auto_fix = QCheckBox(
            "Auto-Edit on export (white balance + histogram equalization)"
        )
        layout.addWidget(self.web_auto_fix)

        self.web_status = QLabel(
            "Images are resized to max 2000px and published with an index.html "
            "masonry grid (Tailwind CSS via CDN). Camera + lens info appears under "
            "each photo."
        )
        self.web_status.setObjectName("PageSubtitle")
        self.web_status.setWordWrap(True)
        self.web_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.web_status)
        layout.addStretch()

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
        self.image_paths = self.loader.scan_folder(self.current_folder)
        self.selected.clear()
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
            self.grid.addWidget(cell, i // columns, i % columns)
            self.cells[str(path)] = cell

        self.progress.setVisible(True)
        self.progress.setMaximum(len(self.image_paths))
        self.progress.setValue(0)
        self.status_label.setText(f"Loading {len(self.image_paths)} images…")

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

        mode = self.color_mode.currentText()
        filter_name = self.filter_combo.currentText()
        color_name = self.color_combo.currentText()

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
                out_path = path.with_name(f"{path.stem}_graded.jpg")
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
                f"Saved {len(outputs)} graded image(s) next to the originals."
            )

        self._run_task(work, done, self.apply_color_button)

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

        if self.web_selected_only.isChecked():
            targets = self._selected_paths()
            if not targets:
                QMessageBox.information(
                    self, "No selection", "Select images in Library, or untick the checkbox."
                )
                return
        else:
            targets = list(self.image_paths)

        out_dir = QFileDialog.getExistingDirectory(self, "Choose gallery output folder")
        if not out_dir:
            return

        group_by_label = self.web_group_by_label.isChecked()
        auto_fix = self.web_auto_fix.isChecked()

        # Build the {category: [items]} structure the publisher expects,
        # attaching each image's gear caption from its EXIF metadata.
        gallery: dict[str, list[dict]] = {}
        for path in targets:
            key = str(path)
            category = self.labels.get(key, "Unlabeled") if group_by_label else "All Photos"
            meta = self.metadata.get(key)
            gallery.setdefault(category, []).append(
                {
                    "path": path,
                    "gear": meta.gear_label if isinstance(meta, ImageMetadata) else "",
                }
            )

        self.web_status.setText(
            f"Publishing {len(targets)} images into {len(gallery)} categor"
            f"{'y' if len(gallery) == 1 else 'ies'}…"
        )

        def work():
            return generate_static_gallery(gallery, out_dir, auto_fix=auto_fix)

        def done(index_path):
            self.web_status.setText(
                f"Gallery published: {index_path}\nOpen index.html in a browser."
            )

        self._run_task(work, done, self.web_button)

    # ------------------------------------------------------------------ #
    # Shutdown
    # ------------------------------------------------------------------ #

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self._thumb_worker is not None:
            self._thumb_worker.cancel()
            self._thumb_worker.wait()
        if self._task_worker is not None and self._task_worker.isRunning():
            self._task_worker.wait(5000)
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
