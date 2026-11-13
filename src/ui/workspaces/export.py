"""The Export workspace: everything that writes files.

Editing anywhere else in the app changes settings. This is where they are
rendered: file presets (web, full size, 16-bit, print), Instagram formats,
the panoramic reel, the static web gallery and background removal. Every
action reads the selection, leaves out rejected frames and renders each
frame through its recipe, layers and overlays. Originals are only read.
"""

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

from src.ui.common import RejectsDialog, TaskWorker, ThumbnailCell, ThumbnailWorker
from src.ui.theme import NIKON_RED, THUMB_SIZE, pil_to_pixmap
from src.engine import export_presets, layer_stack
from src.engine.export_presets import PRESETS, FrameJob
from src.ui.workspaces.base import Workspace

logger = logging.getLogger(__name__)


class ExportWorkspace(Workspace):
    title = "Export"

    def __init__(self, shell) -> None:
        super().__init__(shell)
        self._segmenter = None
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        column = QVBoxLayout(host)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self._build_files_section())
        column.addWidget(self._build_reel_page())
        column.addWidget(self._build_web_page())
        column.addStretch()
        scroll.setWidget(host)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        self.color_status = self.status_label

    def on_enter(self) -> None:
        self._refresh_counts()

    def _refresh_counts(self) -> None:
        chosen = self._selected_paths()
        kept = [p for p in chosen if str(p) not in self.rejected]
        rejected = len(chosen) - len(kept)
        edited = sum(1 for p in kept if self.session.has_edits(p))
        self.files_count.setText(
            f"{len(kept)} selected frame(s) will be exported, {edited} of them edited"
            + (f"; {rejected} rejected frame(s) left out" if rejected else "")
            + ". Unedited frames export as shot."
        )
        noun = "image" if len(self.web_selected) == 1 else "images"
        self.web_selection_status.setText(f"{len(self.web_selected)} {noun} marked for web.")

    # -- rendering edited frames --------------------------------------------- #

    def _job(self, path) -> FrameJob:
        """How to render one frame: its recipe, plus what its layers need."""
        from src.ai_tools import retouch

        path = Path(path)
        recipe = self.session.recipe(path)
        context = None
        if recipe.layers:
            if self._segmenter is None:
                from src.ai_tools.segmentation import Segmenter

                self._segmenter = Segmenter(download=False)
            context = layer_stack.RenderContext(
                segmenter=self._segmenter,
                assets=layer_stack.assets_folder(path, self.edit_folder),
                metadata=self.metadata.get(str(path)),
            )
        needs_fill = recipe.geometry.fill_blank and not recipe.geometry.constrain_to_image
        inpainter = retouch.Inpainter(download=False) if needs_fill and retouch.Inpainter.is_cached() else None
        return FrameJob(path, recipe, context, inpainter, self.metadata.get(str(path)))

    def _publishable(self, paths) -> list[Path]:
        """The images the publishing tools should read.

        An edited frame is rendered to a full-size JPEG in the edit folder
        first; an unedited one is read as it is.
        """
        out = []
        for path in paths:
            if self.session.has_edits(path):
                out.append(export_presets.delivery_copy(self._job(path), self.edit_folder))
            else:
                out.append(Path(path))
        return out

    # -- file export ------------------------------------------------------------ #

    def _build_files_section(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 8)
        layout.setSpacing(12)
        top = QHBoxLayout()
        top.addLayout(self._page_header(
            "Export", "Render the selected frames with their edits. Rejected frames are left out."))
        top.addStretch()
        self.export_button = QPushButton("Export Selected")
        self.export_button.setObjectName("ActionButton")
        self.export_button.clicked.connect(self._on_export_files)
        top.addWidget(self.export_button)
        layout.addLayout(top)

        row = QHBoxLayout()
        row.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        for key, preset in PRESETS.items():
            self.preset_combo.addItem(preset.name, key)
        row.addWidget(self.preset_combo)
        row.addSpacing(16)
        row.addWidget(QLabel("Watermark:"))
        self.watermark_combo = QComboBox()
        for label, key in (("None", ""), ("Signature", "signature"), ("Seal", "seal"),
                           ("Camera details strip", "exif_strip")):
            self.watermark_combo.addItem(label, key)
        row.addWidget(self.watermark_combo)
        self.watermark_name = QLineEdit()
        self.watermark_name.setPlaceholderText("Name for the signature or seal")
        row.addWidget(self.watermark_name, stretch=1)
        layout.addLayout(row)

        bg_row = QHBoxLayout()
        self.remove_bg_button = QPushButton("Remove Background → PNG")
        self.remove_bg_button.setObjectName("SecondaryButton")
        self.remove_bg_button.setToolTip(
            "Cut the subject out of each selected frame, with its edits, as a "
            "transparent PNG in the edit folder."
        )
        self.remove_bg_button.clicked.connect(self._on_remove_background)
        bg_row.addWidget(self.remove_bg_button)
        bg_row.addStretch()
        layout.addLayout(bg_row)

        self.files_count = QLabel()
        self.files_count.setObjectName("PageSubtitle")
        self.files_count.setWordWrap(True)
        layout.addWidget(self.files_count)
        return page

    def _on_export_files(self) -> None:
        targets = self._drop_rejected(self._selected_paths(), "Export")
        if not targets:
            QMessageBox.information(self, "No selection", "Select the frames to export first.")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Export to which folder?")
        if not out_dir:
            return
        preset = PRESETS[self.preset_combo.currentData()]
        watermark = export_presets.watermark_items(self.watermark_combo.currentData(), self.watermark_name.text())
        if self.watermark_combo.currentData() in ("signature", "seal") and not watermark:
            QMessageBox.information(self, "Watermark", "Type the name to put on the watermark.")
            return
        self.session.save_recipes()
        jobs = [self._job(path) for path in targets]
        self.status_label.setText(f"Exporting {len(jobs)} frame(s) as {preset.name}…")

        def work():
            return export_presets.export_batch(jobs, out_dir, preset, watermark)

        def done(outcome):
            written, failures = outcome
            message = f"Exported {len(written)} frame(s) to {out_dir}."
            if failures:
                message += f" {len(failures)} could not be read: " + ", ".join(p.name for p, _ in failures[:5])
            self.status_label.setText(message + " Originals untouched.")

        self._run_task(work, done, self.export_button)

    # --generated-body-start--
    def _build_reel_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)

        top = QHBoxLayout()
        top.addLayout(
            self._page_header(
                "Reel Maker", "Select 3–7 images → a panoramic 9:16 scrolling reel"
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

        self.web_selection_status = QLabel("0 images marked for web .")
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
            "Tick “Publish to Web” on thumbnails in Cull to choose what gets "
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

    def _on_create_reel(self) -> None:
        targets = self._drop_rejected(self._selected_paths(), "Reel")
        if not 3 <= len(targets) <= 7:
            QMessageBox.information(
                self,
                "Selection needed",
                f"Select 3–7 images (currently {len(targets)}).",
            )
            return

        out, _ = QFileDialog.getSaveFileName(
            self, "Save reel as", "panoramic_reel.mp4", "MP4 video (*.mp4)"
        )
        if not out:
            return

        self.reel_status.setText("Rendering reel… this can take a minute.")

        def work():
            return self.reel_maker.create_panoramic_reel(self._publishable(targets), out)

        def done(path):
            self.reel_status.setText(f"Reel saved: {path}")

        self._run_task(work, done, self.reel_button)

    def _on_export_ig(self) -> None:
        targets = self._drop_rejected(self._selected_paths(), "Instagram export")
        if not targets:
            QMessageBox.information(
                self, "No selection", "Select one or more images first."
            )
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Choose export folder")
        if not out_dir:
            return

        target = "feed" if self.ig_target.currentIndex() == 0 else "story"

        def work():
            return self.ig_formatter.export_batch(self._publishable(targets), out_dir, target=target)

        def done(written):
            self.reel_status.setText(f"Exported {len(written)} IG {target} image(s) to {out_dir}")

        self._run_task(work, done, self.ig_button)

    def _on_generate_gallery(self) -> None:
        if not self._require_library():
            return

        if not self.web_selected:
            QMessageBox.information(
                self,
                "Nothing marked for web",
                'Check "Publish to Web" on one or more thumbnails first.',
            )
            return
        if self.edit_folder is None or self.current_folder is None:
            QMessageBox.information(self, "No folder", "Load a folder first.")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Choose gallery output folder")
        if not out_dir:
            return

        targets = self._drop_rejected([Path(p) for p in self.web_selected], "Gallery")
        if not targets:
            QMessageBox.information(
                self,
                "Everything rejected",
                "Every photo marked for the web is also rejected, so the "
                "gallery would be empty. Keep some of them first.",
            )
            return
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
                graded = edit_folder / f"{original.stem}_graded.jpg"   # from before 2.0
                if self.session.has_edits(original):
                    source = export_presets.delivery_copy(self._job(original), edit_folder)
                elif graded.exists():
                    source = graded
                else:
                    source = self.loader.export_keeper(original, current_folder)

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

    def _on_remove_background(self) -> None:
        """Remove background from selected images, export as transparent PNG."""
        targets = self._drop_rejected(self._selected_paths(), "Background removal")
        if not targets:
            QMessageBox.information(
                self, "No selection", "Select one or more images first."
            )
            return
        if self.edit_folder is None:
            QMessageBox.information(self, "No folder", "Load a folder first.")
            return

        edit_folder = self.edit_folder

        def work():
            outputs: list[Path] = []
            last_preview = None
            for path in targets:
                if self.session.has_edits(path):
                    arr = export_presets.render_frame(self._job(path))
                else:
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
            self.color_status.setText(
                f"Removed background from {len(outputs)} image(s), saved as *_nobg.png "
                f"(RGBA with transparency) to {edit_folder.name}/."
            )

        self._run_task(work, done, self.remove_bg_button)
