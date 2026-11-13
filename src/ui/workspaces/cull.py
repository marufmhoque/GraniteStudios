"""The Cull workspace: the grid, the AI cull, rejecting and rating.

Keyboard, with the grid focused:

    arrows      move through the frames         Space   select / deselect
    1-5         star rating (0 clears)          6-9     red, yellow, green, blue
    P           pick                            X       reject
    U           clear pick and reject           Enter   open in Develop
    S           survey the selection side by side
    F           face loupe for the selection

Ratings, labels and picks apply to the selection, or to the current frame
when nothing is selected. Every action that used to write a straightened,
sharpened or graded copy now changes the frame's recipe instead; nothing is
written until Export.
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
from src.engine import develop_pipeline as dp
from src.engine import fast_preview
from src.engine import parametric_recipe as pr
from src.engine import session as session_topics
from src.ui.widgets.canvas import ImageCanvas, link_views
from src.ui.workspaces.base import Workspace

logger = logging.getLogger(__name__)


COLOUR_KEYS = {Qt.Key.Key_6: "Red", Qt.Key.Key_7: "Yellow", Qt.Key.Key_8: "Green", Qt.Key.Key_9: "Blue"}
SURVEY_MAX = 6
LOUPE_SIDE = 220


class CullWorkspace(Workspace):
    title = "Cull"

    def __init__(self, shell) -> None:
        super().__init__(shell)
        self.cells: dict[str, ThumbnailCell] = {}
        self._thumb_worker: ThumbnailWorker | None = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_photos_page())
        self.session.subscribe(self.on_session)

    # -- session ------------------------------------------------------------ #

    def on_session(self, topic: str) -> None:
        if topic == session_topics.LIBRARY:
            self._rebuild_grid()
        elif topic in (session_topics.MARKS, session_topics.RECIPE):
            self._refresh_marks()
        elif topic == session_topics.CURRENT:
            self._show_current()

    def open_folder(self, folder) -> None:
        """Open a folder through the shared session; every workspace follows."""
        if self._thumb_worker is not None:
            self._thumb_worker.cancel()
            self._thumb_worker.wait()
        self.session.open(folder, self.loader)

    def shutdown(self) -> None:
        if self._thumb_worker is not None:
            self._thumb_worker.cancel()
            self._thumb_worker.wait()

    def _refresh_marks(self) -> None:
        for key, cell in self.cells.items():
            cell.set_rejected(key in self.rejected)
            cell.set_marks(
                self.session.ratings.get(key, 0), self.session.color_labels.get(key),
                key in self.session.picks, self.session.has_edits(key),
            )
            cell.set_web_selected(key in self.web_selected)
            cell.set_label(self.labels.get(key, ""))
        self._refresh_reject_button()
        self._apply_view_filter()

    def _show_current(self) -> None:
        current = self.session.current
        for key, cell in self.cells.items():
            cell.setProperty("current", key == current)
            frame = "2px solid #ffffff" if key == current else ""
            cell.setStyleSheet(f"QLabel#Thumb, QLabel#ThumbSelected {{ border: {frame}; }}" if frame else "")
        cell = self.cells.get(current or "")
        if cell is not None:
            self.grid_scroll.ensureWidgetVisible(cell, 20, 20)

    # -- keyboard ------------------------------------------------------------- #

    def _visible_keys(self) -> list[str]:
        return [str(p) for p in self._visible_paths()]

    def _mark_targets(self) -> list[str]:
        return sorted(self.selected) if self.selected else ([self.session.current] if self.session.current else [])

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if not self.image_paths:
            return super().keyPressEvent(event)
        key = event.key()
        columns = max(1, self.grid.columnCount())
        moves = {Qt.Key.Key_Right: 1, Qt.Key.Key_Left: -1, Qt.Key.Key_Down: columns, Qt.Key.Key_Up: -columns}
        targets = self._mark_targets()
        if key in moves:
            self.session.step(moves[key], self._visible_keys())
        elif key == Qt.Key.Key_Space and self.session.current:
            self._on_thumb_clicked(self.session.current)
        elif Qt.Key.Key_0 <= key <= Qt.Key.Key_5:
            self.session.set_rating(targets, int(key - Qt.Key.Key_0))
        elif key in COLOUR_KEYS:
            self.session.set_color_label(targets, COLOUR_KEYS[key])
        elif key == Qt.Key.Key_P:
            self.session.set_pick(targets, True)
        elif key == Qt.Key.Key_X:
            self.session.set_rejected(targets, True)
        elif key == Qt.Key.Key_U:
            self.session.set_pick(targets, False)
            self.session.set_rejected(targets, False)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.session.current:
            self.shell.open_in_develop(self.session.current)
        elif key == Qt.Key.Key_S:
            self._toggle_survey()
        elif key == Qt.Key.Key_F:
            self._on_face_loupe()
        elif key == Qt.Key.Key_Escape and self.view_stack.currentIndex():
            self.view_stack.setCurrentIndex(0)
        else:
            return super().keyPressEvent(event)
        event.accept()

    # -- people ------------------------------------------------------------------- #

    def _refresh_people(self) -> None:
        people = self.session.people
        self.person_filter.blockSignals(True)
        current = self.person_filter.currentData()
        self.person_filter.clear()
        self.person_filter.addItem("Everyone", None)
        if people is not None:
            for person in people.by_size():
                self.person_filter.addItem(f"{people.label(person)} ({len(people.paths_of(person))})", person)
        index = self.person_filter.findData(current)
        self.person_filter.setCurrentIndex(max(0, index))
        self.person_filter.blockSignals(False)
        self._apply_view_filter()

    def _on_group_people(self) -> None:
        if not self._require_library():
            return
        from src.ai_tools import face_mesh, person_clustering
        from src.ai_tools.face_analysis import MODEL_SPEC as FACE_SPEC

        paths = list(self.image_paths)

        def run() -> None:
            self.status_label.setText(f"Recognising faces in {len(paths)} photo(s)…")

            def work():
                embedder = person_clustering.FaceEmbedder(download=False)
                frames = self.culler.score_frames(paths, check_eyes=True, embedder=embedder)
                index, matrix = person_clustering.cluster_frames(frames)
                index.save(self.edit_folder, matrix)
                return index

            def done(index):
                self.session.people = index
                self._refresh_people()
                self.status_label.setText(
                    f"Found {len(index.people)} people appearing in more than one photo"
                    + (f"; {len(index.unassigned)} face(s) seen only once" if index.unassigned else "")
                    + ". Pick someone under “Everyone” to see their photos; “Name…” to label them.")

            self._run_task(work, done, self.people_button)

        self.shell.with_models([FACE_SPEC, face_mesh.MESH_SPEC, person_clustering.SFACE_SPEC],
                               "Grouping by person", run, self.people_button)

    def _on_rename_person(self) -> None:
        from PyQt6.QtWidgets import QInputDialog

        person = self.person_filter.currentData()
        people = self.session.people
        if person is None or people is None:
            QMessageBox.information(self, "Name a person", "Choose a person from the list first.")
            return
        name, ok = QInputDialog.getText(self, "Name this person", "Name:", text=people.names.get(person, ""))
        if ok:
            people.rename(person, name)
            people.save(self.edit_folder)
            self._refresh_people()

    def _on_forget_people(self) -> None:
        from src.ai_tools import person_clustering

        if self.edit_folder is None:
            return
        removed = person_clustering.PeopleIndex.forget(self.edit_folder)
        self.session.people = None
        self._refresh_people()
        self.status_label.setText(
            f"Removed {removed} file(s) of face data for this folder. Your photos were not touched.")

    # -- survey: several frames side by side, zoom locked together ------------- #

    def _toggle_survey(self) -> None:
        if self.view_stack.currentIndex() == 1:
            self.view_stack.setCurrentIndex(0)
            return
        chosen = [k for k in self._visible_keys() if k in self.selected][:SURVEY_MAX]
        if len(chosen) < 2:
            QMessageBox.information(self, "Survey", "Select 2 to 6 frames to compare them side by side.")
            return
        while self.survey_layout.count():
            item = self.survey_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        canvases = []
        columns = 3 if len(chosen) > 4 else 2
        for index, key in enumerate(chosen):
            canvas = ImageCanvas()
            preview = fast_preview.load_preview(key, max_side=2400)
            canvas.set_image(preview.image if preview is not None else None)
            canvas.caption = Path(key).name + ("   REJECTED" if key in self.rejected else "")
            canvas.pressed.connect(lambda *_a, k=key: self.session.set_current(k))
            self.survey_layout.addWidget(canvas, index // columns, index % columns)
            canvases.append(canvas)
        link_views(canvases)
        self.survey_canvases = canvases
        self.view_stack.setCurrentIndex(1)
        self.status_label.setText(
            f"Surveying {len(chosen)} frames. Zoom and pan move them together; "
            "double-click for 100%. Press S or Esc to return to the grid."
        )

    # -- face loupe: every face in the selection, at full size ----------------- #

    def _on_face_loupe(self) -> None:
        keys = [k for k in self._visible_keys() if k in self.selected] or (
            [self.session.current] if self.session.current else [])
        if not keys:
            return
        self.status_label.setText(f"Finding faces in {len(keys)} frame(s)…")

        def work():
            from src.ai_tools.face_analysis import FaceAnalyzer

            analyzer = FaceAnalyzer(download=False)
            if not analyzer.available:
                return None
            crops = []
            for key in keys[:24]:
                preview = fast_preview.load_preview(key, max_side=None)
                if preview is None:
                    continue
                bgr = cv2.cvtColor(preview.image, cv2.COLOR_RGB2BGR)
                for face in analyzer.detect_reduced(bgr, 800):
                    side = int(max(face.w, face.h) * 1.6)
                    cx, cy = face.x + face.w // 2, face.y + face.h // 2
                    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
                    crop = preview.image[y0:y0 + side, x0:x0 + side]
                    if crop.size:
                        crops.append((key, crop.copy()))
            return crops

        def done(crops):
            if crops is None:
                QMessageBox.information(self, "Face loupe", "The face model is not on this computer yet.")
                return
            while self.loupe_layout.count():
                item = self.loupe_layout.takeAt(0)
                if item.widget() is not None:
                    item.widget().deleteLater()
            for key, crop in crops:
                tile = QLabel()
                tile.setFixedSize(LOUPE_SIDE, LOUPE_SIDE + 18)
                shown = crop if max(crop.shape[:2]) <= LOUPE_SIDE else fast_preview.fit(crop, LOUPE_SIDE)
                tile.setPixmap(pil_to_pixmap(Image.fromarray(shown)))
                tile.setToolTip(f"{Path(key).name}  ({crop.shape[1]}×{crop.shape[0]} px at 100%)")
                tile.setStyleSheet("border: 1px solid #333; border-radius: 6px;")
                tile.mousePressEvent = lambda _e, k=key: self.session.set_current(k)
                self.loupe_layout.addWidget(tile)
            self.loupe_layout.addStretch()
            self.loupe_scroll.setVisible(bool(crops))
            self.status_label.setText(
                f"{len(crops)} face(s) found. Each is shown at full size, or shrunk to fit "
                "when larger than the tile; hover for its real size. F again to refresh."
                if crops else "No faces found in those frames."
            )

        self._run_task(work, done, self.cull_button)

    # --generated-body-start--
    def _build_photos_page(self) -> QWidget:
        """The single working pane: import, cull, reject, grade.

        These were three separate tabs (Library / AI Cull / Color-Filters),
        which forced the grid out of sight exactly when you needed to see
        what an action had done. Culling without seeing the frames, or
        grading a selection you cannot look at, is not a workflow. One grid
        stays on screen and every tool acts on it.
        """
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 18)
        layout.setSpacing(10)

        top = QHBoxLayout()
        top.addLayout(
            self._page_header("Cull", "Pick the keepers: rate, label, reject and compare, all on one grid")
        )
        top.addStretch()
        # Loading a folder lives in the workspace bar, shared by every workspace.
        layout.addLayout(top)

        # --- Row 1: selection and rejection ------------------------------ #
        sel_row = QHBoxLayout()
        sel_row.setSpacing(8)
        sel_row.addWidget(QLabel("Select:"))
        for text, slot in (
            ("All", self._on_select_all),
            ("None", self._on_select_none),
            ("Invert", self._on_select_invert),
        ):
            btn = QPushButton(text)
            btn.setObjectName("SecondaryButton")
            btn.clicked.connect(slot)
            sel_row.addWidget(btn)

        sel_row.addSpacing(18)

        self.reject_button = QPushButton("Reject Selected")
        self.reject_button.setObjectName("SecondaryButton")
        self.reject_button.setToolTip(
            "Mark the selected photos as rejected. They stop being copied "
            "into the -edit subfolder. Your original files are never deleted."
        )
        self.reject_button.clicked.connect(self._on_reject_selected)
        sel_row.addWidget(self.reject_button)

        self.restore_button = QPushButton("Keep Selected")
        self.restore_button.setObjectName("SecondaryButton")
        self.restore_button.setToolTip("Undo rejection on the selected photos.")
        self.restore_button.clicked.connect(self._on_restore_selected)
        sel_row.addWidget(self.restore_button)

        self.review_button = QPushButton("Review Rejected (0)")
        self.review_button.setObjectName("SecondaryButton")
        self.review_button.setToolTip(
            "Look through everything marked rejected before it is left out."
        )
        self.review_button.clicked.connect(self._on_review_rejects)
        sel_row.addWidget(self.review_button)

        sel_row.addStretch()

        sel_row.addWidget(QLabel("Show:"))
        self.view_filter = QComboBox()
        self.view_filter.addItems(
            ["All", "Keepers only", "AI picks (starred)", "Duplicates",
             "Rejected", "Blurry", "Motion blur", "Out of focus"]
        )
        self.person_filter = QComboBox()
        self.person_filter.addItem("Everyone", None)
        self.person_filter.setToolTip("Show only the photographs a person appears in")
        self.person_filter.setMinimumWidth(150)
        self.person_filter.currentIndexChanged.connect(self._apply_view_filter)
        self.view_filter.setToolTip("Filter which photos the grid shows.")
        self.view_filter.currentIndexChanged.connect(self._apply_view_filter)
        sel_row.addWidget(self.view_filter)
        layout.addLayout(sel_row)

        # --- Row 2: AI culling ------------------------------------------- #
        cull_row = QHBoxLayout()
        cull_row.setSpacing(8)
        cull_row.addWidget(QLabel("AI cull:"))

        self.cull_button = QPushButton("Run Smart Cull")
        self.cull_button.setObjectName("SecondaryButton")
        self.cull_button.clicked.connect(self._on_run_cull)
        cull_row.addWidget(self.cull_button)

        cull_row.addWidget(QLabel("keep best"))
        self.cull_target_input = QSpinBox()
        self.cull_target_input.setRange(1, 100000)
        self.cull_target_input.setValue(400)
        self.cull_target_input.setFixedWidth(90)
        self.cull_target_input.setToolTip(
            "How many keepers you want. The quota is spread across bursts so "
            "no part of the shoot gets dropped wholesale."
        )
        cull_row.addWidget(self.cull_target_input)

        self.cull_target_button = QPushButton("Select Top N")
        self.cull_target_button.setObjectName("SecondaryButton")
        self.cull_target_button.clicked.connect(self._on_cull_to_target)
        cull_row.addWidget(self.cull_target_button)

        self.cull_eyes_check = QCheckBox("Skip blinks")
        self.cull_eyes_check.setToolTip(
            "Detect faces and closed eyes, then prefer an open-eyed frame "
            "over a sharper one where everybody blinked. Slower, since every "
            "frame runs through the face model."
        )
        cull_row.addWidget(self.cull_eyes_check)

        self.reject_nonpicks_button = QPushButton("Reject Non-Picks")
        self.reject_nonpicks_button.setObjectName("SecondaryButton")
        self.reject_nonpicks_button.setToolTip(
            "Mark everything the cull did not pick as rejected, so only the "
            "keepers reach the edit folder. Reversible."
        )
        self.reject_nonpicks_button.clicked.connect(self._on_reject_nonpicks)
        cull_row.addWidget(self.reject_nonpicks_button)

        self.xmp_button = QPushButton("Write XMP")
        self.xmp_button.setObjectName("SecondaryButton")
        self.xmp_button.setToolTip(
            "Write star ratings and colour labels to .xmp files beside each "
            "image, so Lightroom, Capture One, Photo Mechanic or Bridge open "
            "with these picks already applied. Originals are not modified."
        )
        self.xmp_button.clicked.connect(self._on_write_xmp)
        cull_row.addWidget(self.xmp_button)

        self.xmp_reject_check = QCheckBox("Mark non-picks rejected in XMP")
        cull_row.addWidget(self.xmp_reject_check)
        cull_row.addStretch()
        layout.addLayout(cull_row)

        # --- Row 2b: blur triage ----------------------------------------- #
        blur_row = QHBoxLayout()
        blur_row.setSpacing(8)
        blur_row.addWidget(QLabel("Blur:"))

        self.find_blurry_button = QPushButton("Find Blurry")
        self.find_blurry_button.setObjectName("SecondaryButton")
        self.find_blurry_button.setToolTip(
            "Scan for out-of-focus and motion-blurred frames, then stage "
            "them in the rejected drawer for review. Nothing is deleted."
        )
        self.find_blurry_button.clicked.connect(self._on_find_blurry)
        blur_row.addWidget(self.find_blurry_button)

        blur_row.addWidget(QLabel("strictness"))
        self.blur_strictness = QComboBox()
        self.blur_strictness.addItems(["Aggressive", "Balanced", "Conservative"])
        self.blur_strictness.setCurrentText("Balanced")
        self.blur_strictness.setToolTip(
            "Aggressive catches mild softness (good for reclaiming storage). "
            "Conservative only flags frames that are clearly unusable."
        )
        blur_row.addWidget(self.blur_strictness)

        self.sharpen_button = QPushButton("Sharpen Recoverable (0)")
        self.sharpen_button.setObjectName("SecondaryButton")
        self.sharpen_button.setEnabled(False)
        self.sharpen_button.setToolTip(
            "Sharpen only the frames whose blur is mild enough to lift. "
            "Frames blurred past that point are not offered, because "
            "sharpening them adds halos rather than restoring detail."
        )
        self.sharpen_button.clicked.connect(self._on_sharpen_recoverable)
        blur_row.addWidget(self.sharpen_button)

        blur_row.addSpacing(18)
        blur_row.addWidget(QLabel("Straighten:"))
        self.auto_level_button = QPushButton("Auto-Level Selected")
        self.auto_level_button.setObjectName("SecondaryButton")
        self.auto_level_button.setToolTip(
            "Turn photos the right way up and level them, in one batch.\n\n"
            "Which way is up comes from face detection, falling back to the "
            "subject's body when no face is visible. How far off level comes "
            "from the straight lines in the scene, falling back to the line "
            "of the shoulders and hips when there are none."
        )
        self.auto_level_button.clicked.connect(self._on_auto_level)
        blur_row.addWidget(self.auto_level_button)

        self.level_body_check = QCheckBox("Use body")
        self.level_body_check.setChecked(pose_analysis.PoseAnalyzer.is_cached())
        self.level_body_check.setToolTip(
            "Read posture from the subject's shoulders and hips.\n\n"
            "This is what straightens a portrait shot against a plain wall, "
            "where there are no lines to measure, and what gets a "
            "back-turned subject the right way up.\n\n"
            "Needs a one-off model download."
        )
        self.level_body_check.toggled.connect(self._on_body_engine_toggled)
        blur_row.addWidget(self.level_body_check)

        self.posture_button = QPushButton("Check Posture")
        self.posture_button.setObjectName("SecondaryButton")
        self.posture_button.setToolTip(
            "Report what the selected photos' subjects are doing: how much "
            "of the body is in frame, whether the camera was rolled, and "
            "whether the subject is leaning. Changes nothing."
        )
        self.posture_button.clicked.connect(self._on_check_posture)
        blur_row.addWidget(self.posture_button)

        blur_row.addWidget(QLabel("corners"))
        self.level_corners = QComboBox()
        self.level_corners.addItems(["Crop", "Fill", "Leave black"])
        self.level_corners.setToolTip(
            "Straightening leaves wedge-shaped gaps at the corners. Crop cuts back "
            "to the largest clean rectangle. Fill paints them in from the "
            "surrounding picture at export, keeping the full frame."
        )
        blur_row.addWidget(self.level_corners)
        self.level_crop_check = QCheckBox("Crop corners")
        self.level_crop_check.setChecked(True)
        self.level_crop_check.setVisible(False)
        self.level_crop_check.setToolTip(
            "Rotating leaves wedge-shaped gaps at the corners. Ticked, the "
            "frame is cut back to the largest clean rectangle; unticked, the "
            "gaps are left black. There is no generative fill."
        )
        blur_row.addWidget(self.level_crop_check)
        blur_row.addStretch()
        layout.addLayout(blur_row)

        # --- Row 3: grading ---------------------------------------------- #
        color_row = QHBoxLayout()
        color_row.setSpacing(8)
        color_row.addWidget(QLabel("Grade:"))

        self.color_mode = QComboBox()
        self.color_mode.addItems(["Cinematic Filter", "Selective Color"])
        color_row.addWidget(self.color_mode)

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(ColorEngine.FILTERS)
        color_row.addWidget(self.filter_combo)

        self.color_combo = QComboBox()
        self.color_combo.addItems(ColorEngine.COLORS)
        color_row.addWidget(self.color_combo)

        self.apply_color_button = QPushButton("Apply Look to Selected")
        self.apply_color_button.setObjectName("ActionButton")
        self.apply_color_button.clicked.connect(self._on_apply_color)
        color_row.addWidget(self.apply_color_button)


        color_row.addSpacing(18)
        color_row.addWidget(QLabel("Label:"))
        self.label_combo = QComboBox()
        self.label_combo.setEditable(True)
        self.label_combo.addItems(
            ["Graduation", "Nature", "Portraits", "Travel", "Events", "Wildlife"]
        )
        self.label_combo.setMinimumWidth(140)
        color_row.addWidget(self.label_combo)

        apply_label_btn = QPushButton("Apply")
        apply_label_btn.setObjectName("SecondaryButton")
        apply_label_btn.clicked.connect(self._on_apply_label)
        color_row.addWidget(apply_label_btn)

        clear_label_btn = QPushButton("Clear Label")
        clear_label_btn.setObjectName("SecondaryButton")
        clear_label_btn.clicked.connect(self._on_clear_label)
        color_row.addWidget(clear_label_btn)

        color_row.addStretch()
        self.tech_info_check = QCheckBox("Show Tech Info")
        self.tech_info_check.toggled.connect(self._on_toggle_tech)
        color_row.addWidget(self.tech_info_check)
        layout.addLayout(color_row)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # --- The grid: always on screen ---------------------------------- #
        self.grid_scroll = QScrollArea()
        self.grid_scroll.setWidgetResizable(True)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setSpacing(12)
        self.grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.grid_scroll.setWidget(self.grid_host)

        # The grid, or several frames side by side for comparison.
        self.view_stack = QStackedWidget()
        self.view_stack.addWidget(self.grid_scroll)
        survey_host = QWidget()
        self.survey_layout = QGridLayout(survey_host)
        self.survey_layout.setSpacing(6)
        self.view_stack.addWidget(survey_host)
        self.survey_canvases: list = []
        layout.addWidget(self.view_stack, stretch=1)

        # Faces from the selection at full size, for checking focus and eyes.
        self.loupe_scroll = QScrollArea()
        self.loupe_scroll.setWidgetResizable(True)
        self.loupe_scroll.setFixedHeight(LOUPE_SIDE + 44)
        loupe_host = QWidget()
        self.loupe_layout = QHBoxLayout(loupe_host)
        self.loupe_layout.setSpacing(8)
        self.loupe_scroll.setWidget(loupe_host)
        self.loupe_scroll.setVisible(False)
        layout.addWidget(self.loupe_scroll)

        view_row = QHBoxLayout()
        self.survey_button = QPushButton("Survey (S)")
        self.survey_button.setObjectName("SecondaryButton")
        self.survey_button.clicked.connect(self._toggle_survey)
        self.loupe_button = QPushButton("Face Loupe (F)")
        self.loupe_button.setObjectName("SecondaryButton")
        self.loupe_button.clicked.connect(self._on_face_loupe)
        keys = QLabel("1–5 stars · 6–9 colours · P pick · X reject · U clear · arrows move · Enter develop")
        keys.setObjectName("PageSubtitle")
        view_row.addWidget(self.survey_button)
        view_row.addWidget(self.loupe_button)
        self.people_button = QPushButton("Group by Person")
        self.people_button.setObjectName("SecondaryButton")
        self.people_button.setToolTip(
            "Recognise faces across the folder and group the photos by who is in them. "
            "Everything stays on this computer; names are yours to give.")
        self.people_button.clicked.connect(self._on_group_people)
        view_row.addWidget(self.people_button)
        view_row.addWidget(self.person_filter)
        self.rename_person_button = QPushButton("Name…")
        self.rename_person_button.setObjectName("SecondaryButton")
        self.rename_person_button.clicked.connect(self._on_rename_person)
        view_row.addWidget(self.rename_person_button)
        self.forget_button = QPushButton("Forget Faces")
        self.forget_button.setObjectName("SecondaryButton")
        self.forget_button.setToolTip("Delete everything stored about faces for this folder")
        self.forget_button.clicked.connect(self._on_forget_people)
        view_row.addWidget(self.forget_button)
        view_row.addSpacing(12)
        view_row.addWidget(keys)
        view_row.addStretch()
        layout.addLayout(view_row)

        # Status messages go to the window's status bar (self.status_label).

        # Cull report, collapsed by default: useful detail, but it should not
        # compete with the photographs for space.
        self.cull_results = QLabel("Run a cull to see per-frame scores here.")
        self.cull_results.setObjectName("PageSubtitle")
        self.cull_results.setWordWrap(True)
        self.cull_results.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.cull_scroll = QScrollArea()
        self.cull_scroll.setWidgetResizable(True)
        self.cull_scroll.setMaximumHeight(170)
        self.cull_scroll.setVisible(False)
        cull_host = QWidget()
        cull_host_layout = QVBoxLayout(cull_host)
        cull_host_layout.addWidget(self.cull_results)
        cull_host_layout.addStretch()
        self.cull_scroll.setWidget(cull_host)
        layout.addWidget(self.cull_scroll)

        self.cull_detail_toggle = QCheckBox("Show cull report")
        self.cull_detail_toggle.toggled.connect(self.cull_scroll.setVisible)
        layout.addWidget(self.cull_detail_toggle)

        # The colour panel wrote its own status line; the shared status label
        # covers it now, but handlers still reference these names.
        self.color_status = self.status_label

        return page

    def _on_thumb_clicked(self, path: str) -> None:
        cell = self.cells.get(path)
        if cell is None:
            return
        self.setFocus()
        if path in self.selected:
            self.selected.discard(path)
            cell.set_selected(False)
        else:
            self.selected.add(path)
            cell.set_selected(True)
        self.session.set_current(path)
        self.session.notify(session_topics.SELECTION)
        self._refresh_status()

    def _pose_analyzer(self, download_ok: bool = False):
        """The pose analyzer, or None when the body engine is off or absent.

        Never downloads on its own: a levelling run must not stall partway
        through a batch on a model fetch nobody agreed to. The checkbox asks
        first, and passes ``download_ok`` once the user has said yes.
        """
        if not getattr(self, "level_body_check", None) or not self.level_body_check.isChecked():
            return None
        if not download_ok and not pose_analysis.PoseAnalyzer.is_cached():
            return None
        try:
            return pose_analysis.PoseAnalyzer(download=download_ok)
        except Exception as exc:
            logger.warning("Pose models unavailable: %s", exc)
            return None

    def _on_body_engine_toggled(self, checked: bool) -> None:
        """Offer the one-off model download the first time this is switched on."""
        if not checked or pose_analysis.PoseAnalyzer.is_cached():
            return
        size = pose_analysis.PoseAnalyzer.download_mb()
        answer = QMessageBox.question(
            self,
            "Download the pose models?",
            f"Reading posture needs two models, about {size:.0f} MB in total.\n\n"
            "They are downloaded once, checked against a known fingerprint, "
            "and kept on this computer. Nothing is uploaded, and your photos "
            "stay on this machine.\n\nDownload them now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self.level_body_check.setChecked(False)
            return

        def work():
            return pose_analysis.PoseAnalyzer(download=True).available

        def done(available):
            if available:
                self.status_label.setText(
                    "Pose models ready. Levelling can now use the subject's body."
                )
                return
            self.level_body_check.setChecked(False)
            QMessageBox.warning(
                self,
                "Download failed",
                "The pose models could not be fetched, so levelling will use "
                "scene lines and faces only. Check the connection and try again.",
            )

        self.status_label.setText(f"Downloading the pose models ({size:.0f} MB)…")
        self._run_task(work, done, self.posture_button)

    def _on_check_posture(self) -> None:
        """Report what the subjects are doing. Reads only; changes nothing."""
        targets = self._selected_paths() or list(self.image_paths)
        if not targets:
            QMessageBox.information(self, "No photos", "Load a folder first.")
            return
        if not self.level_body_check.isChecked():
            QMessageBox.information(
                self,
                "Body engine off",
                "Tick “Use body” first — reading posture needs the pose models.",
            )
            return
        analyzer = self._pose_analyzer()
        if analyzer is None or not analyzer.available:
            QMessageBox.information(
                self,
                "Pose models missing",
                "The pose models are not on this computer yet. Untick and "
                "re-tick “Use body” to download them.",
            )
            return

        loader = self.loader
        self.status_label.setText(f"Reading posture in {len(targets)} photo(s)…")

        def work():
            rows = []
            for path in targets:
                array = loader.load_array(path)
                bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
                rows.append((Path(path).name, analyzer.analyze(bgr)))
            return rows

        def done(rows):
            readable = [(n, r) for n, r in rows if r.count]
            rolled, leaning, level = [], [], 0
            for name, report in readable:
                pose = report.main
                if pose is None:
                    continue
                tilt = report.applicable_tilt
                if tilt is None:
                    if pose.asymmetry is not None:
                        leaning.append((name, pose))
                elif abs(tilt) >= orientation.MIN_CORRECTION_DEG:
                    rolled.append((name, tilt))
                else:
                    level += 1

            lines = [
                f"{len(readable)} of {len(rows)} photo(s) had a body clear enough to read.",
                "",
                f"  {level} already level",
                f"  {len(rolled)} look like a rolled camera (Auto-Level can fix these)",
                f"  {len(leaning)} have a leaning subject (left alone — that is "
                f"the pose, not the camera)",
            ]
            if rolled:
                lines += ["", "Most tilted:"]
                for name, tilt in sorted(rolled, key=lambda r: -abs(r[1]))[:8]:
                    lines.append(f"  {name}  {tilt:+.1f}°")
            if leaning:
                lines += ["", "Leaning subjects:"]
                for name, pose in leaning[:8]:
                    lines.append(
                        f"  {name}  shoulders and hips differ by {pose.asymmetry:.1f}°"
                    )
            shoulders_only = sum(
                1 for _, r in readable if r.main is not None and not r.main.has_hips
            )
            if shoulders_only:
                lines += [
                    "",
                    f"{shoulders_only} photo(s) showed shoulders but no hips. Those are "
                    "reported but never levelled automatically: with one line instead "
                    "of two the angle was out by up to 9° in testing.",
                ]

            QMessageBox.information(self, "Posture", "\n".join(lines))
            self.status_label.setText(
                f"Posture: {level} level, {len(rolled)} rolled, {len(leaning)} leaning "
                f"({len(rows) - len(readable)} with no readable body). Nothing changed."
            )

        self._run_task(work, done, self.posture_button)

    def _on_auto_level(self) -> None:
        """Straighten the selected photos by setting their recipes.

        Works on a batch, so a whole event can be normalised in one pass.
        Nothing is written: each frame's recipe records the turn and the
        straightening, and Export renders it.

        Two engines, because neither covers everything. Which way up a frame
        goes is decided by face detection, falling back to the direction from
        the subject's hips to their shoulders when no face is visible. How far
        from level is decided by the scene's straight lines, falling back to
        the line of the shoulders and hips when the frame has none.

        Head roll is still not used here; see src/ai_tools/orientation.py.
        """
        targets = self._drop_rejected(self._selected_paths(), "Auto-level")
        if not targets:
            QMessageBox.information(
                self, "No selection", "Select the photos you want straightened."
            )
            return
        if self.edit_folder is None:
            QMessageBox.information(self, "No folder", "Load a folder first.")
            return

        corners = self.level_corners.currentText()
        pose_analyzer = self._pose_analyzer()

        def work():
            from src.ai_tools.face_analysis import FaceAnalyzer

            analyzer = FaceAnalyzer()
            found = []
            for path in targets:
                rgb = self.loader.load_analysis(path, max_side=orientation.ANALYSIS_MAX_SIDE)
                if rgb is None:
                    continue
                result = orientation.detect(
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), analyzer=analyzer, pose_analyzer=pose_analyzer
                )
                found.append((path, result))
            return found

        def done(found):
            changed = [(path, r) for path, r in found if r.needs_rotation]
            for path, result in changed:
                recipe = self.session.recipe(path).copy()
                dp.set_rotation(recipe.geometry, result.total_angle)
                recipe.geometry.constrain_to_image = corners == "Crop"
                recipe.geometry.fill_blank = corners == "Fill"
                self.session.set_recipe(path, recipe)
            self.session.save_recipes()
            skipped = len(found) - len(changed)
            if not changed:
                self.status_label.setText(f"All {skipped} selected photo(s) were already upright.")
                return
            turns = sum(1 for _, r in changed if r.coarse)
            by_body = sum(1 for _, r in changed if r.method == "body")
            detail = f"{turns} needed a quarter turn; " if turns else ""
            body_note = f"{by_body} levelled from the subject's body; " if by_body else ""
            self.status_label.setText(
                f"Straightened {len(changed)} photo(s) ({detail}{body_note}{skipped} already upright). "
                f"Saved as edits; nothing is rendered until Export, and the originals are untouched."
            )

        self._run_task(work, done, self.auto_level_button)

    def _on_find_blurry(self) -> None:
        """Scan the folder for blurred frames and stage them for review.

        Blurry frames are marked rejected rather than deleted, so they land
        in the same review drawer as anything else rejected and can be put
        back one by one. Nothing is removed from disk at any point.
        """
        if not self._require_library():
            return
        strictness = self.blur_strictness.currentText().lower()
        paths = list(self.image_paths)
        self.status_label.setText(
            f"Checking {len(paths)} photo(s) for blur ({strictness})…"
        )

        def work():
            return blur_analysis.analyze_many(paths, strictness=strictness)

        def done(reports):
            self.blur_reports = reports
            blurry, recoverable = [], []
            for path, report in reports.items():
                cell = self.cells.get(path)
                if cell is not None:
                    cell.set_blur_state(report)
                if report.is_blurry:
                    blurry.append(Path(path))
                    if report.recoverable:
                        recoverable.append(path)

            if blurry:
                self._set_rejected(blurry, True)

            motion = sum(1 for r in reports.values() if r.verdict == blur_analysis.MOTION_BLUR)
            defocus = sum(1 for r in reports.values() if r.verdict == blur_analysis.OUT_OF_FOCUS)
            soft = sum(1 for r in reports.values() if r.verdict == blur_analysis.SOFT)

            self.status_label.setText(
                f"{len(blurry)} blurred of {len(paths)} "
                f"({motion} motion, {defocus} out of focus, {soft} slightly soft). "
                f"They are marked rejected, not deleted — open “Review Rejected” "
                f"to look through them."
                + (
                    f"  {len(recoverable)} may be rescued by sharpening."
                    if recoverable
                    else ""
                )
            )
            self._refresh_sharpen_button()

        self._run_task(work, done, self.find_blurry_button)

    def _refresh_sharpen_button(self) -> None:
        if not hasattr(self, "sharpen_button"):
            return
        count = sum(1 for r in self.blur_reports.values() if r.recoverable)
        self.sharpen_button.setText(f"Sharpen Recoverable ({count})")
        self.sharpen_button.setEnabled(count > 0)

    def _on_sharpen_recoverable(self) -> None:
        """Sharpen the frames whose blur is mild enough to be worth lifting.

        Only frames the analysis marked recoverable are offered, because
        sharpening something past that point adds halos to mush rather than
        bringing detail back -- the Laplacian score rises either way, which
        is exactly why the score alone does not decide this. The sharpening
        is a recipe setting, applied at export and adjustable in Develop.
        """
        if self.edit_folder is None:
            QMessageBox.information(self, "No folder", "Load a folder first.")
            return
        targets = [p for p, r in self.blur_reports.items() if r.recoverable]
        if not targets:
            QMessageBox.information(
                self,
                "Nothing to rescue",
                "No frames are in the window where sharpening helps.\n\n"
                "Frames blurred past that point cannot have detail restored, "
                "so no fix is offered for them.",
            )
            return
        for path in targets:
            recipe = self.session.recipe(path).copy()
            recipe.detail.sharpen_amount = max(recipe.detail.sharpen_amount, 80.0)
            recipe.detail.sharpen_radius = 1.2
            self.session.set_recipe(path, recipe)
        self.session.save_recipes()
        self.status_label.setText(
            f"Added sharpening to {len(targets)} frame(s)' edits. Adjust it in Develop; it is "
            f"applied at export. This lifts edge contrast; it does not reconstruct detail "
            f"that was never captured."
        )

    def _visible_paths(self) -> list[Path]:
        """Paths currently shown by the grid, honouring the view filter."""
        return [p for p in self.image_paths if self._passes_view_filter(str(p))]

    def _sync_selection_ui(self) -> None:
        for path, cell in self.cells.items():
            cell.set_selected(path in self.selected)
        self._refresh_status()

    def _on_select_all(self) -> None:
        """Select everything the grid is showing.

        Scoped to what is visible rather than the whole folder, so 'All'
        while filtered to Duplicates means those duplicates -- selecting
        hidden frames you cannot see would be a trap.
        """
        if not self.image_paths:
            return
        self.selected = {str(p) for p in self._visible_paths()}
        self._sync_selection_ui()

    def _on_select_none(self) -> None:
        self.selected.clear()
        self._sync_selection_ui()

    def _on_select_invert(self) -> None:
        visible = {str(p) for p in self._visible_paths()}
        self.selected = visible - self.selected
        self._sync_selection_ui()

    def _set_rejected(self, paths, rejected: bool) -> None:
        self.session.set_rejected(paths, rejected)
        self._refresh_status()

    def _refresh_reject_button(self) -> None:
        if hasattr(self, "review_button"):
            self.review_button.setText(f"Review Rejected ({len(self.rejected)})")

    def _on_reject_selected(self) -> None:
        targets = self._selected_paths()
        if not targets:
            QMessageBox.information(
                self, "Nothing selected", "Select the photos you want to reject first."
            )
            return
        self._set_rejected(targets, True)
        self.status_label.setText(
            f"{len(targets)} photo(s) rejected — they will not be copied into "
            f"the edit folder. Your original files are untouched."
        )

    def _on_restore_selected(self) -> None:
        targets = [p for p in self._selected_paths() if str(p) in self.rejected]
        if not targets:
            QMessageBox.information(
                self, "Nothing to restore", "None of the selected photos are rejected."
            )
            return
        self._set_rejected(targets, False)
        self.status_label.setText(f"{len(targets)} photo(s) kept.")

    def _on_reject_nonpicks(self) -> None:
        """Reject everything the last cull did not pick."""
        if not self.last_cull_frames:
            QMessageBox.information(
                self,
                "No cull yet",
                "Run Smart Cull or Select Top N first, so there are picks to keep.",
            )
            return
        losers = [f.path for f in self.last_cull_frames if not f.is_starred]
        if not losers:
            QMessageBox.information(self, "Nothing to reject", "Every frame was picked.")
            return
        self._set_rejected(losers, True)
        keepers = len(self.last_cull_frames) - len(losers)
        self.status_label.setText(
            f"Rejected {len(losers)} non-picks; {keepers} keeper(s) remain. "
            f"Nothing was deleted — review them before exporting."
        )

    def _on_review_rejects(self) -> None:
        if not self.rejected:
            QMessageBox.information(
                self,
                "Nothing rejected",
                "No photos are marked as rejected yet.\n\n"
                "Select photos and click “Reject Selected”, or run a cull and "
                "use “Reject Non-Picks”.",
            )
            return
        ordered = [str(p) for p in self.image_paths if str(p) in self.rejected]
        dialog = RejectsDialog(ordered, self.loader, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        still = dialog.still_rejected()
        restored = [p for p in ordered if p not in still]
        if restored:
            self._set_rejected(restored, False)
        self._set_rejected([p for p in ordered if p in still], True)
        self.status_label.setText(
            f"{len(still)} photo(s) stay rejected"
            + (f", {len(restored)} restored." if restored else ".")
            + " Originals are untouched either way."
        )

    def _passes_view_filter(self, path: str) -> bool:
        person = self.person_filter.currentData() if hasattr(self, "person_filter") else None
        if person is not None and self.session.people is not None:
            if path not in self.session.people.paths_of(person):
                return False
        mode = self.view_filter.currentText() if hasattr(self, "view_filter") else "All"
        is_rejected = path in self.rejected
        if mode == "All":
            return True
        if mode == "Keepers only":
            return not is_rejected
        if mode == "Rejected":
            return is_rejected
        cell = self.cells.get(path)
        if mode == "AI picks (starred)":
            return bool(cell and cell.starred)
        if mode == "Duplicates":
            return bool(cell and cell.duplicate)
        if mode == "Blurry":
            report = self.blur_reports.get(path)
            return bool(report and report.is_blurry)
        if mode == "Motion blur":
            report = self.blur_reports.get(path)
            return bool(report and report.verdict == blur_analysis.MOTION_BLUR)
        if mode == "Out of focus":
            report = self.blur_reports.get(path)
            return bool(report and report.verdict == blur_analysis.OUT_OF_FOCUS)
        return True

    def _apply_view_filter(self) -> None:
        """Show or hide cells to match the chosen filter."""
        if not hasattr(self, "view_filter"):
            return
        shown = 0
        for path, cell in self.cells.items():
            visible = self._passes_view_filter(path)
            cell.setVisible(visible)
            if visible:
                shown += 1
        self._refresh_status(shown)

    def _refresh_status(self, shown: int | None = None) -> None:
        if not self.image_paths:
            return
        if shown is None:
            shown = sum(1 for c in self.cells.values() if not c.isHidden())
        total = len(self.image_paths)
        parts = [f"{total} photo(s)"]
        if shown != total:
            parts.append(f"{shown} shown")
        parts.append(f"{len(self.selected)} selected")
        if self.rejected:
            parts.append(f"{len(self.rejected)} rejected")
        if self.edit_folder is not None:
            parts.append(f"edits → {self.edit_folder.name}/")
        self.status_label.setText("  ·  ".join(parts))

    def _on_load_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose an image folder")
        if folder:
            self.open_folder(folder)

    def _rebuild_grid(self) -> None:
        """Lay the grid out for the folder the session has just opened."""
        if self._thumb_worker is not None:
            self._thumb_worker.cancel()
            self._thumb_worker.wait()
        if hasattr(self, "view_filter"):
            self.view_filter.blockSignals(True)
            self.view_filter.setCurrentIndex(0)
            self.view_filter.blockSignals(False)
        self.view_stack.setCurrentIndex(0)
        self.loupe_scroll.setVisible(False)
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
            cell.opened.connect(self.shell.open_in_develop)
            cell.web_toggled.connect(self._on_web_toggle)
            self.grid.addWidget(cell, i // columns, i % columns)
            self.cells[str(path)] = cell
        self._refresh_marks()
        self._show_current()
        from src.ai_tools import person_clustering

        people = person_clustering.PeopleIndex.load(self.edit_folder)
        self.session.people = people if people.people else None
        self._refresh_people()

        self.progress.setVisible(True)
        self.progress.setMaximum(len(self.image_paths))
        self.progress.setValue(0)
        # Say plainly where edits will land. The folder is created here, not
        # on first export, so it is visible in Explorer straight away.
        self.status_label.setText(
            f"Loading {len(self.image_paths)} images…  "
            f"Edits will be saved into {self.current_folder.name}/{self.edit_folder.name}/ "
            f"— your originals are never modified."
        )

        self._thumb_worker = ThumbnailWorker(self.loader, list(self.image_paths), parent=self)
        self._thumb_worker.thumb_ready.connect(self._on_thumb_ready)
        self._thumb_worker.meta_ready.connect(self._on_meta_ready)
        self._thumb_worker.progress.connect(self._on_thumb_progress)
        self._thumb_worker.finished_all.connect(self._on_thumbs_done)
        self._thumb_worker.start()
        self.setFocus()

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
        self._refresh_status()

    def _on_apply_label(self) -> None:
        targets = self._selected_paths()
        if not targets:
            QMessageBox.information(
                self, "No selection", "Select one or more images first."
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
            self.web_selection_status.setText(f"{count} {noun} marked for web.")

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

        check_eyes = self.cull_eyes_check.isChecked()

        def work():
            return self.culler.cull_to_target(folder, target, check_eyes=check_eyes)

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
            checked = [f for f in frames if f.eyes_checked]
            if check_eyes and not checked:
                lines.insert(
                    0,
                    "Note: the face model could not be loaded, so blinks were "
                    "not checked. Selection fell back to sharpness only."
                    + "\n",
                )
            elif checked:
                blinks = [f for f in checked if f.eyes_closed]
                faces = sum(f.face_count for f in checked)
                lines.insert(
                    0,
                    f"Faces found in {sum(1 for f in checked if f.face_count):,} "
                    f"frame(s) ({faces:,} face(s) total). "
                    f"{len(blinks):,} frame(s) had closed eyes and were demoted."
                    + "\n",
                )
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

    def _on_apply_color(self) -> None:
        """Give the selected frames a look. A recipe setting, not a new file.

        A cinematic filter becomes the recipe's colour look. Selective colour
        is expressed with the HSL bands: every band but the chosen one is
        fully desaturated. Both stay adjustable in Develop.
        """
        targets = self._drop_rejected(self._selected_paths(), "Grade")
        if not targets:
            QMessageBox.information(
                self, "No selection", "Select one or more images first."
            )
            return
        if self.edit_folder is None:
            QMessageBox.information(self, "No folder", "Load a folder first.")
            return

        mode = self.color_mode.currentText()
        for path in targets:
            recipe = self.session.recipe(path).copy()
            if mode == "Selective Color":
                recipe = selective_colour_recipe(recipe, self.color_combo.currentText())
            else:
                recipe = pr.with_look(self.filter_combo.currentText(), recipe)
            self.session.set_recipe(path, recipe)
        self.session.save_recipes()
        self.color_status.setText(
            f"Applied to {len(targets)} photo(s) as an edit. Fine-tune it in Develop; "
            f"it is rendered at export and the originals are untouched."
        )


# Which HSL bands make up each colour the old selective-colour tool offered.
_SELECTIVE_BANDS = {
    "red": ("red",), "orange": ("orange",), "yellow": ("yellow",), "green": ("green",),
    "cyan": ("aqua",), "blue": ("blue",), "purple": ("purple",), "magenta": ("magenta",),
}


def selective_colour_recipe(recipe: pr.Recipe, colour: str) -> pr.Recipe:
    """Keep one colour and turn the rest of the frame grey."""
    keep = _SELECTIVE_BANDS.get(colour.lower(), ("red",))
    recipe = recipe.copy()
    recipe.hsl.saturation = [0.0 if band in keep else -100.0 for band in pr.HSL_BANDS]
    return recipe
