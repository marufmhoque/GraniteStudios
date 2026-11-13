"""The Video workspace: the FFmpeg editor, moved here unchanged.

Video still works by chaining files: each operation writes a new clip into
``<name>-edit/`` and the next reads it. That is not the parametric model the
photo side uses, and is recorded as a limitation rather than claimed.
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
from src.ui.workspaces.base import Workspace

logger = logging.getLogger(__name__)


class VideoWorkspace(Workspace):
    title = "Video"

    def __init__(self, shell) -> None:
        super().__init__(shell)
        # Kept separate from the photo library. video_source_path is the
        # untouched original; video_current_path is the working file each
        # operation reads and replaces, so edits chain without ever
        # writing over the source.
        self.video_source_path: Path | None = None
        self.video_current_path: Path | None = None
        self.video_edit_folder: Path | None = None
        self._api_server = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_video_page())

    def shutdown(self) -> None:
        if self._api_server is not None:
            self._api_server.should_exit = True

    # --generated-body-start--
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
