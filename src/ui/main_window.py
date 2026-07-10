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
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
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
from src.engine.universal_loader import UniversalLoader
from src.ai_tools.culling import SmartCull
from src.ai_tools.color_engine import ColorEngine
from src.social.reel_maker import ScrlReelMaker
from src.social.ig_sync import IgFormatter
from src.web.publisher import generate_static_gallery

logger = logging.getLogger(__name__)

THUMB_SIZE = 200

DARK_STYLESHEET = """
QMainWindow, QWidget { background-color: #16171b; color: #e6e6e9; font-size: 13px; }
QFrame#Sidebar { background-color: #101114; border-right: 1px solid #26272c; }
QPushButton#NavButton {
    background: transparent; border: none; border-radius: 8px;
    padding: 12px 16px; text-align: left; color: #b6b7bd; font-size: 14px;
}
QPushButton#NavButton:hover { background-color: #1e1f24; color: #ffffff; }
QPushButton#NavButton:checked { background-color: #2a2c33; color: #ffffff; font-weight: 600; }
QPushButton#ActionButton {
    background-color: #3d6df2; border: none; border-radius: 8px;
    padding: 10px 20px; color: white; font-weight: 600;
}
QPushButton#ActionButton:hover { background-color: #5580f4; }
QPushButton#ActionButton:disabled { background-color: #2a2c33; color: #6a6b72; }
QPushButton#SecondaryButton {
    background-color: #26272c; border: 1px solid #3a3b42; border-radius: 8px;
    padding: 9px 18px; color: #e6e6e9;
}
QPushButton#SecondaryButton:hover { background-color: #303138; }
QScrollArea { border: none; }
QLabel#PageTitle { font-size: 22px; font-weight: 700; color: #ffffff; }
QLabel#PageSubtitle { color: #8f9096; font-size: 13px; }
QLabel#Thumb {
    background-color: #1c1d22; border: 1px solid #2a2b31; border-radius: 10px;
}
QLabel#ThumbSelected { border: 2px solid #3d6df2; border-radius: 10px; background-color: #1c1d22; }
QComboBox {
    background-color: #26272c; border: 1px solid #3a3b42; border-radius: 6px;
    padding: 6px 12px; color: #e6e6e9;
}
QComboBox QAbstractItemView { background-color: #1c1d22; color: #e6e6e9; selection-background-color: #2a2c33; }
QProgressBar {
    background-color: #1c1d22; border: none; border-radius: 6px; height: 8px; text-align: center;
}
QProgressBar::chunk { background-color: #3d6df2; border-radius: 6px; }
QCheckBox { color: #b6b7bd; spacing: 8px; }
QScrollBar:vertical { background: #16171b; width: 10px; }
QScrollBar::handle:vertical { background: #33343b; border-radius: 5px; min-height: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
"""


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
        self.setObjectName("Thumb")
        self.setFixedSize(QSize(THUMB_SIZE + 12, THUMB_SIZE + 34))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText(Path(path).name)
        self.setWordWrap(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._badge = ""

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self.setPixmap(
            pixmap.scaled(
                THUMB_SIZE,
                THUMB_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def set_badge(self, text: str) -> None:
        self._badge = text
        self.setToolTip(f"{Path(self.path).name}\n{text}" if text else Path(self.path).name)

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
        self.setStyleSheet(DARK_STYLESHEET)

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
        self._thumb_worker: ThumbnailWorker | None = None
        self._task_worker: TaskWorker | None = None

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

        self.web_status = QLabel(
            "Images are resized to max 2000px and published with an index.html "
            "masonry grid (Tailwind CSS via CDN)."
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
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)
        self.pages.setCurrentIndex(index)

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
        self._thumb_worker.progress.connect(self._on_thumb_progress)
        self._thumb_worker.finished_all.connect(self._on_thumbs_done)
        self._thumb_worker.start()

    def _on_thumb_ready(self, path: str, pixmap: QPixmap) -> None:
        cell = self.cells.get(path)
        if cell is not None:
            cell.set_pixmap(pixmap)

    def _on_thumb_progress(self, done: int, total: int) -> None:
        self.progress.setValue(done)

    def _on_thumbs_done(self) -> None:
        self.progress.setVisible(False)
        self.status_label.setText(
            f"{len(self.image_paths)} images · {len(self.selected)} selected"
        )

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
        self.cull_results.setText("Analyzing sharpness and grouping bursts…")

        def work():
            return self.culler.group_bursts(paths, time_threshold=2.0)

        def done(groups):
            lines = [f"Found {len(groups)} groups across {len(paths)} images.\n"]
            for i, group in enumerate(groups, start=1):
                lines.append(f"Burst {i} — {len(group.frames)} frame(s):")
                for frame in group.frames:
                    marker = "★ PICK" if frame.is_burst_pick else "      "
                    lines.append(
                        f"  {marker}  {frame.path.name}   sharpness {frame.sharpness:,.0f}"
                    )
                lines.append("")
            self.cull_results.setText("\n".join(lines))

            # Badge the picks in the Library grid.
            for group in groups:
                for frame in group.frames:
                    cell = self.cells.get(str(frame.path))
                    if cell is not None:
                        cell.set_badge(
                            ("★ Burst pick · " if frame.is_burst_pick else "")
                            + f"sharpness {frame.sharpness:,.0f}"
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

        self.web_status.setText(f"Publishing {len(targets)} images…")

        def work():
            return generate_static_gallery(targets, out_dir)

        def done(index_path):
            self.web_status.setText(f"Gallery published: {index_path}\nOpen index.html in a browser.")

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
    window = MainWindow()
    window.show()
    return app.exec()
