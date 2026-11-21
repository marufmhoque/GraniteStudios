"""The Develop workspace: one frame large, every develop setting beside it.

Keyboard, with the picture focused:

    left / right   previous / next frame       \\        before / after
    J              show clipped shadows and highlights
    Ctrl+Z / Y     undo / redo                 Ctrl+Shift+C / V   copy / paste settings
    0              fit to window               1          100 percent
    [ / ]          previous / next version     V          compare versions side by side
    Shift+V        compare with a different version
    Ctrl+'         duplicate: a new virtual copy of the version being edited

A rejected or removed photo is locked: the controls are disabled and a
banner offers to restore it (see ``src/ui/widgets/edit_lock.py``).

Every control edits the frame's recipe. The preview is rendered on a
background thread from a proxy of the frame, with the same pipeline the
export uses, so what is seen is what will be written. The recipe is saved to
its sidecar when the frame changes or the workspace is left.

A photograph can have several versions (see the version bar above the
picture). Develop edits whichever is active, and always writes back to the
version it opened, so switching never lands one version's settings on
another.

The History section lists every recorded change to the version being
edited, from any workspace, oldest first. Clicking a step puts the recipe
back as it was then; that is recorded as a new step, and Ctrl+Z undoes it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from datetime import datetime

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from src.engine import develop_pipeline as dp
from src.engine import edit_history as eh
from src.engine import layer_stack
from src.engine import parametric_recipe as pr
from src.engine import session as topics
from src.ui.widgets.canvas import ImageCanvas, link_views
from src.ui.widgets.controls import CurveEditor, Histogram, Section, SliderRow, clipping_overlay
from src.ui.widgets.edit_lock import EditLockBanner, apply_lock
from src.ui.widgets.filmstrip import Filmstrip
from src.ui.widgets.renderer import LatestOnlyRenderer
from src.ui.widgets.versions import VersionBar
from src.ui.workspaces.base import Workspace

logger = logging.getLogger(__name__)

PREVIEW_SIDE = 1600
# Two-tier preview. While a control is moving, a smaller draft is rendered
# (sized so it takes about DRAFT_BUDGET_MS); once it has rested for
# SETTLE_MS the full 1600 px preview follows. Drafts are only used when
# the full preview is slower than the budget.
DRAFT_BUDGET_MS = 50.0
DRAFT_MAX_SIDE = 1200
DRAFT_MIN_SIDE = 400
SETTLE_MS = 120
HISTORY_LIMIT = 100

# (attribute, label, low, high, step, decimals)
BASIC = [
    ("exposure", "Exposure", -5.0, 5.0, 0.05, 2),
    ("contrast", "Contrast", -100, 100, 1, 0),
    ("highlights", "Highlights", -100, 100, 1, 0),
    ("shadows", "Shadows", -100, 100, 1, 0),
    ("whites", "Whites", -100, 100, 1, 0),
    ("blacks", "Blacks", -100, 100, 1, 0),
    ("temperature", "Temperature", -100, 100, 1, 0),
    ("tint", "Tint", -100, 100, 1, 0),
    ("texture", "Texture", -100, 100, 1, 0),
    ("clarity", "Clarity", -100, 100, 1, 0),
    ("dehaze", "Dehaze", -100, 100, 1, 0),
    ("vibrance", "Vibrance", -100, 100, 1, 0),
    ("saturation", "Saturation", -100, 100, 1, 0),
]
ASPECTS = {"Original": None, "1:1": 1.0, "4:5": 4 / 5, "3:2": 3 / 2, "2:3": 2 / 3, "16:9": 16 / 9, "9:16": 9 / 16}


def centred_crop(aspect: float | None, frame_aspect: float) -> list | None:
    """The largest centred crop of the given width/height ratio."""
    if aspect is None:
        return None
    if aspect >= frame_aspect:
        h = frame_aspect / aspect
        return [0.0, (1.0 - h) / 2.0, 1.0, h]
    w = aspect / frame_aspect
    return [(1.0 - w) / 2.0, 0.0, w, 1.0]


class DevelopWorkspace(Workspace):
    title = "Develop"

    def __init__(self, shell) -> None:
        super().__init__(shell)
        self.key: str | None = None
        self.version = pr.MASTER            # the version of ``key`` being edited
        self._sibling: int | None = None    # the version shown beside it when comparing
        self._sibling_key: str | None = None
        self._comparing_versions = False
        self._dragging_deblur = False       # a de-blur slider is held: preview with Wiener
        self.source: dp.Source | None = None
        self.recipe = pr.Recipe()
        self._history: list[dict] = []
        self._future: list[dict] = []
        self._clipboard: dict | None = None
        self._show_clipping = False
        self._last_render: np.ndarray | None = None
        self._before_key: str | None = None
        self._layer_context: layer_stack.RenderContext | None = None
        self._loading = False
        self.renderer = LatestOnlyRenderer(self)
        self.renderer.ready.connect(self._on_rendered)
        self._render_ms: float | None = None     # the last full preview's render time
        self._draft_side = DRAFT_MAX_SIDE        # adapted so a draft stays near the budget
        self._drafts: dict[int, dp.Source] = {}  # draft proxies of the current source, by size
        self._settle = QTimer(self, singleShot=True, interval=SETTLE_MS)
        self._settle.timeout.connect(self._render)
        self.renderer.hold = self._settle.isActive
        self.renderer.show_progress = True       # the picture follows a slider that never pauses
        self.renderer.failed.connect(lambda message: self.status_label.setText(f"Preview failed: {message}"))
        self.loader_job = LatestOnlyRenderer(self)
        self.loader_job.ready.connect(self._on_source_loaded)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._build()
        # Ctrl+' is Cmd+' on macOS: Qt maps it. (Ctrl+D is Select None, in Cull.)
        QShortcut(QKeySequence("Ctrl+'"), self, activated=self.duplicate,
                  context=Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.session.subscribe(self.on_session)

    # -- layout -------------------------------------------------------------- #

    def _build(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        centre = QVBoxLayout()
        centre.setContentsMargins(12, 10, 8, 8)
        bar = QHBoxLayout()
        self.prev_button = self._button("◀", lambda: self._step(-1), "Previous frame (←)")
        self.next_button = self._button("▶", lambda: self._step(1), "Next frame (→)")
        self.frame_label = QLabel("No photo open")
        self.frame_label.setObjectName("PageTitle")
        bar.addWidget(self.prev_button)
        bar.addWidget(self.next_button)
        bar.addWidget(self.frame_label, stretch=1)
        self.compare_button = self._button("Before / After  \\", self._toggle_compare, "Compare with the unedited frame (\\)")
        self.compare_button.setCheckable(True)
        self.clip_button = self._button("Clipping  J", self._toggle_clipping, "Show lost shadows (blue) and highlights (red) (J)")
        self.clip_button.setCheckable(True)
        bar.addWidget(self.compare_button)
        self.versions_button = self._button("Versions  V", self._toggle_versions,
                                            "Show another version of this photo side by side (V; Shift+V picks which)")
        self.versions_button.setCheckable(True)
        bar.addWidget(self.versions_button)
        bar.addWidget(self.clip_button)
        bar.addWidget(self._button("Fit", lambda: self.canvas.fit(), "Fit the frame to the window (0)"))
        bar.addWidget(self._button("100%", lambda: self.canvas.actual_size(), "One image pixel per screen pixel (1)"))
        centre.addLayout(bar)
        self.version_bar = VersionBar(self.session, self._commit)
        centre.addWidget(self.version_bar)
        self.lock_banner = EditLockBanner()
        self.lock_banner.restore_requested.connect(self.restore_photo)
        centre.addWidget(self.lock_banner)
        views = QHBoxLayout()
        views.setSpacing(8)
        self.canvas = ImageCanvas()
        self.sibling_canvas = ImageCanvas()
        link_views([self.canvas, self.sibling_canvas])
        self.main_caption, self.sibling_caption = QLabel(), QLabel()
        for caption_label, view in ((self.main_caption, self.canvas), (self.sibling_caption, self.sibling_canvas)):
            column = QVBoxLayout()
            column.setSpacing(4)
            caption_label.setStyleSheet("color: #c9cad0; font-size: 12px; font-weight: 600;")
            caption_label.setVisible(False)
            column.addWidget(caption_label)
            column.addWidget(view, stretch=1)
            views.addLayout(column, stretch=1)
        self.sibling_canvas.setVisible(False)
        centre.addLayout(views, stretch=1)
        self.filmstrip = Filmstrip(self.session, self.shell.thumbnail_for)
        self.install_delete_photo(self.filmstrip)
        centre.addWidget(self.filmstrip)
        outer.addLayout(centre, stretch=1)

        panel_scroll = QScrollArea()
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setFixedWidth(372)
        panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel = QWidget()
        self.panel = QVBoxLayout(panel)
        self.panel.setContentsMargins(10, 10, 14, 10)
        self.panel.setSpacing(6)
        panel_scroll.setWidget(panel)
        outer.addWidget(panel_scroll)
        self.controls_container = panel_scroll       # everything that edits; disabled while locked

        self.histogram = Histogram()
        self.panel.addWidget(self.histogram)
        self.clip_readout = QLabel()
        self.clip_readout.setObjectName("PageSubtitle")
        self.panel.addWidget(self.clip_readout)

        actions = QGridLayout()
        actions.setSpacing(4)
        for index, (text, slot, tip) in enumerate((
            ("Auto", self._on_auto_tone, "Set exposure so the midtones sit at middle grey"),
            ("Reset", self._on_reset, "Return every setting to neutral"),
            ("Copy", self._on_copy, "Copy these settings (Ctrl+Shift+C)"),
            ("Paste", self._on_paste, "Paste copied settings onto this frame (Ctrl+Shift+V)"),
            ("Sync", self._on_sync, "Copy these settings to every selected frame"),
        )):
            button = self._button(text, slot, tip)
            button.setStyleSheet("padding: 6px 4px;")
            actions.addWidget(button, index // 3, index % 3)
        self.panel.addLayout(actions)

        self.history_section = Section("History", expanded=False)
        self.history_list = QListWidget()
        self.history_list.setObjectName("HistoryList")
        self.history_list.setMinimumHeight(160)
        self.history_list.setToolTip("Every change to this version, oldest first. "
                                     "Click a step to put the photo back as it was then.")
        self.history_list.itemClicked.connect(self._on_history_clicked)
        self.history_section.add(self.history_list)
        self.history_note = QLabel("No edits yet.")
        self.history_note.setObjectName("PageSubtitle")
        self.history_note.setWordWrap(True)
        self.history_section.add(self.history_note)
        self.panel.addWidget(self.history_section)

        self.sliders: dict[tuple[str, str], SliderRow] = {}
        basic = Section("Basic")
        for attr, label, low, high, step, decimals in BASIC:
            self._slider(basic, ("basic", attr), label, low, high, 0.0, step, decimals)
        self.panel.addWidget(basic)

        tone = Section("Tone curve", expanded=False)
        self.curve_channel = QComboBox()
        self.curve_channel.addItems(["master", "red", "green", "blue"])
        self.curve_channel.currentTextChanged.connect(lambda _c: self._load_curve())
        tone.add(self.curve_channel)
        self.curve = CurveEditor()
        self.curve.changed.connect(self._on_curve)
        self.curve.released.connect(self._commit)
        tone.add(self.curve)
        self.panel.addWidget(tone)

        hsl = Section("Colour (HSL)", expanded=False)
        self.hsl_property = QComboBox()
        self.hsl_property.addItems(["hue", "saturation", "luminance"])
        self.hsl_property.currentTextChanged.connect(lambda _p: self._load_hsl())
        hsl.add(self.hsl_property)
        self.hsl_sliders = []
        for index, band in enumerate(pr.HSL_BANDS):
            row = SliderRow(band.capitalize(), -100, 100, 0.0)
            row.changed.connect(lambda value, i=index: self._on_hsl(i, value))
            row.released.connect(self._commit)
            hsl.add(row)
            self.hsl_sliders.append(row)
        self.panel.addWidget(hsl)

        grade = Section("Colour grading", expanded=False)
        self.grade_zone = QComboBox()
        self.grade_zone.addItems(["shadows", "midtones", "highlights"])
        self.grade_zone.currentTextChanged.connect(lambda _z: self._load_grade())
        grade.add(self.grade_zone)
        self.grade_rows = {}
        for attr, label, low, high in (("hue", "Hue", 0, 360), ("saturation", "Strength", 0, 100),
                                       ("luminance", "Brightness", -100, 100)):
            row = SliderRow(label, low, high, 0.0)
            row.changed.connect(lambda value, a=attr: self._on_grade(a, value))
            row.released.connect(self._commit)
            grade.add(row)
            self.grade_rows[attr] = row
        self._slider(grade, ("grade", "balance"), "Balance", -100, 100, 0.0)
        self._slider(grade, ("grade", "blending"), "Blending", 0, 100, 50.0)
        self.panel.addWidget(grade)

        look = Section("Look and LUT", expanded=False)
        self.look_combo = QComboBox()
        self.look_combo.addItem("None", "")
        for name in pr.builtin_looks():
            self.look_combo.addItem(name.replace("_", " ").title(), name)
        self.look_combo.currentIndexChanged.connect(self._on_look)
        look.add(self.look_combo)
        self._slider(look, ("look", "strength"), "Look strength", 0, 100, 100.0)
        self.lut_label = QLabel("No LUT")
        self.lut_label.setObjectName("PageSubtitle")
        look.add_row(self._button("Load .cube…", self._on_lut, "Apply a 3D LUT"),
                     self._button("Clear", self._on_clear_lut, "Remove the LUT"))
        look.add(self.lut_label)
        self._slider(look, ("lut", "strength"), "LUT strength", 0, 100, 100.0)
        self.panel.addWidget(look)

        detail = Section("Detail", expanded=False)
        self._slider(detail, ("detail", "sharpen_amount"), "Sharpen", 0, 150, 0.0)
        self._slider(detail, ("detail", "sharpen_radius"), "Radius", 0.5, 3.0, 1.0, 0.1, 1)
        self._slider(detail, ("detail", "noise_reduction"), "Noise", 0, 100, 0.0)
        detail.add(self._build_deblur())
        self.panel.addWidget(detail)

        lens = Section("Lens", expanded=False)
        self.lens_profile = QCheckBox("Correct with the lens profile")
        self.lens_profile.toggled.connect(self._on_lens_profile)
        lens.add(self.lens_profile)
        self.lens_status = QLabel()
        self.lens_status.setObjectName("PageSubtitle")
        self.lens_status.setWordWrap(True)
        lens.add(self.lens_status)
        self._slider(lens, ("lens", "distortion"), "Distortion", -100, 100, 0.0)
        self._slider(lens, ("lens", "vignette"), "Vignette", -100, 100, 0.0)
        self.panel.addWidget(lens)

        geometry = Section("Crop and straighten", expanded=False)
        geometry.add_row(
            self._button("⟲ 90°", lambda: self._turn(-1), "Rotate a quarter turn anticlockwise"),
            self._button("⟳ 90°", lambda: self._turn(1), "Rotate a quarter turn clockwise"),
            self._button("Flip", self._flip, "Mirror left to right"),
        )
        self._slider(geometry, ("geometry", "rotation"), "Straighten", -45, 45, 0.0, 0.1, 1)
        self._slider(geometry, ("geometry", "keystone_vertical"), "Vertical", -100, 100, 0.0)
        self._slider(geometry, ("geometry", "keystone_horizontal"), "Horizontal", -100, 100, 0.0)
        self.corners_combo = QComboBox()
        self.corners_combo.addItems(["Crop", "Fill", "Leave black"])
        self.corners_combo.setToolTip("What happens to the corners straightening uncovers")
        self.corners_combo.currentTextChanged.connect(self._on_corners)
        self.aspect_combo = QComboBox()
        self.aspect_combo.addItems(list(ASPECTS))
        self.aspect_combo.currentTextChanged.connect(self._on_aspect)
        geometry.add_row(QLabel("Corners"), self.corners_combo)
        geometry.add_row(QLabel("Aspect"), self.aspect_combo)
        self.panel.addWidget(geometry)
        self.panel.addStretch()

    def _build_deblur(self) -> Section:
        """De-Blur: true deconvolution of motion smear or missed focus."""
        group = Section("De-Blur", expanded=False)
        self.deblur_enable = QCheckBox("Enable De-Blur")
        self.deblur_enable.setToolTip(
            "Undo motion smear or missed focus by deconvolution. This recovers detail the "
            "blur weakened; it cannot invent detail the blur erased.")
        self.deblur_enable.toggled.connect(self._on_deblur_enable)
        group.add(self.deblur_enable)
        self.deblur_auto = self._button("Auto-Detect from Analysis", self._on_deblur_auto,
                                        "Measure the smear's direction and length, or the focus blur's size")
        group.add(self.deblur_auto)
        self.deblur_mode = QComboBox()
        for label, data in (("Motion", "motion"), ("Defocus", "defocus"), ("AI Neural (Experimental)", "neural")):
            self.deblur_mode.addItem(label, data)
        # The longest label is wider than the panel allows; let the box
        # shrink to fit rather than push the panel into a sideways scroll.
        self.deblur_mode.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.deblur_mode.setMinimumContentsLength(12)
        self.deblur_mode.setToolTip(
            "Motion and Defocus deconvolve with the blur you describe below. AI Neural is an experimental "
            "trained model (92 MB download) that needs no settings but knows only motion blur. "
            "It is slow: about 2 minutes for the preview on a laptop, far longer for an export.")
        self.deblur_mode.currentIndexChanged.connect(self._on_deblur_mode)
        group.add_row(QLabel("Mode"), self.deblur_mode)
        self._slider(group, ("deblur", "angle"), "Angle", 0, 180, 0.0)
        self._slider(group, ("deblur", "length"), "Smear Length", 1.0, 30.0, 5.0, 0.5, 1)
        self._slider(group, ("deblur", "radius"), "Defocus Radius", 1.0, 20.0, 3.0, 0.5, 1)
        self._slider(group, ("deblur", "damping"), "Suppression", 0.0, 1.0, 0.05, 0.01, 2).setToolTip(
            "Artifact Suppression: hold back ringing and noise. Raise it on noisy, high-ISO frames.")
        for key in (("deblur", "angle"), ("deblur", "length"), ("deblur", "radius"), ("deblur", "damping")):
            row = self.sliders[key]
            row.released.disconnect()
            row.released.connect(self._on_deblur_released)
        self.deblur_area = QComboBox()
        for label, data in (("Whole frame", "full"), ("Subject only", "subject_only"), ("Faces only", "face_only")):
            self.deblur_area.addItem(label, data)
        self.deblur_area.setToolTip("Limit de-blurring to the subject or to faces, so background noise "
                                    "is not amplified")
        self.deblur_area.currentIndexChanged.connect(self._on_deblur_area)
        group.add_row(QLabel("Apply to"), self.deblur_area)
        self.deblur_status = QLabel("Lengths are in pixels of the full-size photo.")
        self.deblur_status.setObjectName("PageSubtitle")
        self.deblur_status.setWordWrap(True)
        group.add(self.deblur_status)
        return group

    def _button(self, text, slot, tip="") -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("SecondaryButton")
        button.setToolTip(tip)
        button.clicked.connect(slot)
        return button

    def _slider(self, section, key, label, low, high, default, step=1.0, decimals=0) -> SliderRow:
        row = SliderRow(label, low, high, default, step, decimals)
        row.changed.connect(lambda value, k=key: self._on_value(k, value))
        row.released.connect(self._commit)
        section.add(row)
        self.sliders[key] = row
        return row

    # -- the open frame -------------------------------------------------------- #

    def on_enter(self) -> None:
        self.filmstrip.refresh()
        if self.session.current != self.key or (
                self.key is not None and self.session.active_version(self.key) != self.version):
            self.load(self.session.current)
        else:
            self._refresh_history()
            self.update_edit_lock_state()
        self.setFocus()

    # -- the edit lock --------------------------------------------------------- #

    @property
    def locked(self) -> bool:
        return self.key is not None and not self.session.is_editable(self.key)

    def update_edit_lock_state(self) -> bool:
        """Disable editing, and show the banner, while the photo is rejected
        or removed. Returns True when it may be edited."""
        return apply_lock(self.session, self.key, self.lock_banner, [self.controls_container, self.version_bar])

    def restore_photo(self) -> None:
        """↺ Restore Photo: no longer rejected, back in the project; editable."""
        if self.key is None:
            return
        self.session.restore_photo(self.key)
        self.update_edit_lock_state()
        self._show_names()
        self.status_label.setText(f"Restored {Path(self.key).name}: it can be edited again.")

    def on_leave(self) -> None:
        self.session.save_recipes()

    def on_session(self, topic: str) -> None:
        if topic == topics.LIBRARY:
            self.key, self.source = None, None
            self._toggle_versions(False)
            self.canvas.set_image(None)
            self.filmstrip.rebuild()
        elif topic == topics.RECIPE:
            if self._is_shown():             # a batch elsewhere is caught up on entering
                self._refresh_history()
        elif topic in (topics.CURRENT, topics.MARKS, topics.PURGED):
            self.filmstrip.refresh()
            if topic == topics.CURRENT and self.shell.__dict__.get("_current_index") is not None and self.shell.current_workspace is self and self.session.current != self.key:
                self.load(self.session.current)
            elif topic != topics.CURRENT:
                self.update_edit_lock_state()
                self._show_names()
        elif topic == topics.VERSIONS and self.key is not None:
            if self.session.active_version(self.key) != self.version and self._is_shown():
                self.switch_version()
            elif self._comparing_versions and self.session.version_count(self.key) < 2:
                self._toggle_versions(False)
                self._show_names()
            else:
                if self._comparing_versions:
                    self._pick_sibling()
                    self._render()
                self._show_names()
            self._refresh_history()

    def _is_shown(self) -> bool:
        return self.shell.__dict__.get("_current_index") is not None and self.shell.current_workspace is self

    def switch_version(self) -> None:
        """Show the active version of the open frame, without decoding it again."""
        self.session.save_recipes()
        self.version = self.session.active_version(self.key)
        self.recipe = self.session.recipe(self.key, self.version).copy()
        self._history, self._future = [], []
        self._before_key, self._sibling_key, self._layer_context = None, None, None
        self._load_controls()
        self._show_names()
        self._refresh_history()
        if self._comparing_versions:
            self._pick_sibling()
        self._render()

    def _show_names(self) -> None:
        if self.key is None:
            return
        self.frame_label.setText(self.session.display_name(self.key, self.version)
                                 + ("   (rejected)" if self.key in self.rejected else ""))
        self.main_caption.setText(self.session.version_name(self.key, self.version) + "  (editing)")
        if self._sibling is not None and self._sibling in {v.id for v in self.session.versions(self.key)}:
            self.sibling_caption.setText(self.session.version_name(self.key, self._sibling))

    def load(self, key: str | None) -> None:
        """Open a frame. The decode runs in the background."""
        self.session.save_recipes()
        self.renderer.cancel()
        self.key = key
        self.version_bar.set_photo(key)
        self._sibling, self._sibling_key = None, None
        self._history, self._future = [], []
        self._before_key = None
        self._layer_context = None
        if key is None:
            self.source = None
            self.canvas.set_image(None)
            self.frame_label.setText("No photo open")
            self._refresh_history()
            self.update_edit_lock_state()
            return
        self.version = self.session.active_version(key)
        self.recipe = self.session.recipe(key, self.version).copy()
        self.update_edit_lock_state()
        self._load_controls()
        self._refresh_history()
        if self._comparing_versions:
            self._pick_sibling()
        self._show_names()
        self.canvas.caption = "Loading…"
        self._loading = True
        path = Path(key)
        self.loader_job.request(lambda: (key, self._open(path)))

    @staticmethod
    def _open(path: Path):
        # Half size is plenty for a 1600 px preview and twice as fast to develop.
        source = dp.open_source(path, half_size=True)
        return source.proxy(PREVIEW_SIDE) if source is not None else None

    def _on_source_loaded(self, result) -> None:
        key, source = result
        if key != self.key:
            return
        self._loading = False
        self.canvas.caption = ""
        if source is None:
            self.source = None
            self.canvas.set_image(None)
            self.canvas.caption = "This file could not be read."
            return
        self.source = source
        self._drafts.clear()
        self.lens_status.setText(
            "A profile for this camera and lens was found." if dp.lens_profile_available(key)
            else "No profile for this camera and lens; use the sliders."
        )
        if source.note:
            self.status_label.setText(f"{Path(key).name}: {source.note}.")
        self._render()

    # -- rendering ------------------------------------------------------------- #

    def _context(self, recipe: pr.Recipe | None = None):
        if not (recipe or self.recipe).layers:
            return None
        if self._layer_context is None:
            from src.ai_tools.segmentation import Segmenter

            self._layer_context = layer_stack.RenderContext(
                segmenter=Segmenter(download=False),
                assets=layer_stack.assets_folder(self.key, self.edit_folder) if self.edit_folder else None,
                metadata=self.metadata.get(self.key),
            )
        return self._layer_context

    def _wants_draft(self) -> bool:
        return (self.source is not None and self._render_ms is not None
                and self._render_ms > DRAFT_BUDGET_MS * 1.5
                and max(self.source.data.shape[:2]) > DRAFT_MIN_SIDE)

    def _draft_source(self) -> dp.Source:
        side = min(self._draft_side, max(self.source.data.shape[:2]))
        if side not in self._drafts:
            self._drafts.clear()
            self._drafts[side] = self.source.proxy(side)
        return self._drafts[side]

    def _render_draft(self) -> None:
        """The quick tier: the recipe on a smaller proxy, the picture only."""
        source, recipe, context = self._draft_source(), self.recipe.copy(), self._context()
        if self._dragging_deblur and recipe.deblur.enabled and recipe.deblur.engine == "parametric":
            recipe.deblur.engine = "wiener_fast"

        def job():
            import time

            started = time.perf_counter()
            image = dp.render(source, recipe, layers=context)
            return ("draft", image, (time.perf_counter() - started) * 1000.0)

        self.renderer.request(job)

    def _render(self) -> None:
        if self.source is None:
            return
        self._settle.stop()                 # a full render supersedes a pending one
        source, recipe, context = self.source, self.recipe.copy(), self._context()
        if self._dragging_deblur and recipe.deblur.enabled and recipe.deblur.engine == "parametric":
            # Richardson-Lucy takes seconds; the closed-form Wiener filter
            # follows the slider, and the full engine runs on release.
            recipe.deblur.engine = "wiener_fast"
        geometry_key = repr(pr.to_dict(recipe)["geometry"])
        want_before = self.canvas.comparing and geometry_key != self._before_key
        sibling, sibling_key, sibling_context = None, None, None
        if self._comparing_versions and self._sibling is not None:
            sibling = self.session.recipe(self.key, self._sibling).copy()
            sibling_key = f"{self._sibling}:{pr.dumps(sibling)}"
            if sibling_key == self._sibling_key:
                sibling = None              # already on screen
            else:
                sibling_context = self._context(sibling)

        def job():
            import time

            started = time.perf_counter()
            after = dp.render(source, recipe, layers=context)
            elapsed = (time.perf_counter() - started) * 1000.0
            before = None
            if want_before:
                neutral = pr.Recipe()
                neutral.geometry = recipe.geometry
                before = dp.render(source, neutral)
            other = dp.render(source, sibling, layers=sibling_context) if sibling is not None else None
            return after, before, geometry_key, other, sibling_key, elapsed

        self.renderer.request(job)

    def _show_render(self, after) -> None:
        self._last_render = after
        self.histogram.set_image(after)
        self.curve.set_histogram(self.histogram.luminance_counts())
        shadows, highlights = self.histogram.shadows_clipped, self.histogram.highlights_clipped
        self.clip_readout.setText(f"Clipped: {shadows * 100:.2f}% shadows · {highlights * 100:.2f}% highlights")
        self.canvas.set_image(clipping_overlay(after) if self._show_clipping else after)

    def _on_rendered(self, result) -> None:
        if isinstance(result[0], str):     # ("draft", image, ms); a full render starts with an array
            _tag, image, elapsed = result
            # Keep the draft near the budget: shrink it when it ran long,
            # grow it back when there was time to spare.
            scale = min(1.25, max(0.6, (DRAFT_BUDGET_MS / max(elapsed, 1.0)) ** 0.5))
            # In steps of 80 px, so the draft proxy is reused rather than remade each frame.
            side = int(round(self._draft_side * scale / 80.0)) * 80
            self._draft_side = min(DRAFT_MAX_SIDE, max(DRAFT_MIN_SIDE, side))
            self._show_render(image)
            return
        after, before, geometry_key, other, sibling_key, elapsed = result
        self._render_ms = elapsed
        if other is not None:
            self.sibling_canvas.set_image(other)
            self._sibling_key = sibling_key
        self._show_render(after)
        if before is not None:
            self.canvas.set_before(before)
            self._before_key = geometry_key

    # -- controls ↔ recipe ------------------------------------------------------ #

    def _section(self, name: str):
        return getattr(self.recipe, name)

    def _load_controls(self) -> None:
        for (section, attr), row in self.sliders.items():
            row.set_value(float(getattr(self._section(section), attr)))
        self._load_curve()
        self._load_hsl()
        self._load_grade()
        self.look_combo.blockSignals(True)
        self.look_combo.setCurrentIndex(max(0, self.look_combo.findData(self.recipe.look.name)))
        self.look_combo.blockSignals(False)
        self.lut_label.setText(Path(self.recipe.lut.path).name if self.recipe.lut.path else "No LUT")
        self.lens_profile.blockSignals(True)
        self.lens_profile.setChecked(self.recipe.lens.profile)
        self.lens_profile.blockSignals(False)
        self._load_deblur()
        g = self.recipe.geometry
        self.corners_combo.blockSignals(True)
        self.corners_combo.setCurrentText("Crop" if g.constrain_to_image else ("Fill" if g.fill_blank else "Leave black"))
        self.corners_combo.blockSignals(False)

    def _load_deblur(self) -> None:
        d = self.recipe.deblur
        for widget, value in ((self.deblur_enable, None), (self.deblur_mode, None), (self.deblur_area, None)):
            widget.blockSignals(True)
        self.deblur_enable.setChecked(d.enabled)
        self.deblur_mode.setCurrentIndex(self.deblur_mode.findData("neural" if d.engine == "neural" else d.blur_type))
        self.deblur_area.setCurrentIndex(max(0, self.deblur_area.findData(d.mask_mode)))
        for widget in (self.deblur_enable, self.deblur_mode, self.deblur_area):
            widget.blockSignals(False)
        self._show_deblur_controls()

    def _show_deblur_controls(self) -> None:
        """Only the sliders the chosen mode uses are shown."""
        mode = self.deblur_mode.currentData()
        self.sliders[("deblur", "angle")].setVisible(mode == "motion")
        self.sliders[("deblur", "length")].setVisible(mode == "motion")
        self.sliders[("deblur", "radius")].setVisible(mode == "defocus")
        self.sliders[("deblur", "damping")].setVisible(mode != "neural")

    def _load_curve(self) -> None:
        self.curve.set_points(getattr(self.recipe.curve, self.curve_channel.currentText()))

    def _load_hsl(self) -> None:
        values = getattr(self.recipe.hsl, self.hsl_property.currentText())
        for row, value in zip(self.hsl_sliders, values):
            row.set_value(float(value))

    def _load_grade(self) -> None:
        wheel = getattr(self.recipe.grade, self.grade_zone.currentText())
        for attr, row in self.grade_rows.items():
            row.set_value(float(getattr(wheel, attr)))

    def _changed(self) -> None:
        """A setting moved: show it now; it is committed on release.

        When the full preview is slow, a quick draft follows the hand and
        the full preview comes SETTLE_MS after the last movement (so on
        release, or when the slider rests)."""
        if self._wants_draft():
            self._render_draft()
            self._settle.start()
        else:
            self._render()

    def flush_edits(self) -> None:
        self._commit()

    def _commit(self) -> None:
        """A change is finished: record it for undo and hand it to the session."""
        if self.key is None:
            return
        if self.locked:                      # nothing is kept for a photo marked for removal
            self.recipe = self.session.recipe(self.key, self.version).copy()
            return
        snapshot = pr.to_dict(self.session.recipe(self.key, self.version))
        if snapshot != pr.to_dict(self.recipe):
            self._history.append(snapshot)
            del self._history[:-HISTORY_LIMIT]
            self._future.clear()
            self.session.set_recipe(self.key, self.recipe.copy(), self.version)

    def _on_value(self, key, value: float) -> None:
        section, attr = key
        self._dragging_deblur = section == "deblur"
        setattr(self._section(section), attr, pr.clamp(_range_name(section, attr), value)
                if _range_name(section, attr) else value)
        self._changed()

    # -- de-blur ------------------------------------------------------------- #

    def _on_deblur_released(self) -> None:
        self._dragging_deblur = False
        self._commit()
        if self.recipe.deblur.enabled:
            self.status_label.setText("De-blurring at full quality…")
            self._render()

    def _deblur_models(self) -> list:
        """Models the current de-blur settings need that may be missing."""
        from src.ai_tools import deblur, face_mesh
        from src.ai_tools.face_analysis import MODEL_SPEC as YUNET_SPEC
        from src.ai_tools.segmentation import Segmenter

        d = self.recipe.deblur
        specs = [deblur.NAFNET_SPEC] if d.engine == "neural" else []
        if d.mask_mode == "subject_only":
            specs += list(Segmenter.models_for("subject"))
        elif d.mask_mode == "face_only":
            specs += [YUNET_SPEC, face_mesh.MESH_SPEC]
        return specs

    def _apply_deblur(self, purpose: str) -> None:
        """Render once any model the settings need is here (asking first)."""
        def go() -> None:
            self._changed()
            self._commit()

        specs = self._deblur_models()
        if specs and self.recipe.deblur.enabled:
            self.shell.with_models(specs, purpose, go, self.deblur_auto)
        else:
            go()

    def _on_deblur_enable(self, checked: bool) -> None:
        self.recipe.deblur.enabled = checked
        self._apply_deblur("De-blurring")

    def _on_deblur_mode(self, _index: int) -> None:
        mode = self.deblur_mode.currentData()
        d = self.recipe.deblur
        if mode == "neural":
            d.engine = "neural"
        else:
            d.blur_type = mode
            d.engine = "wiener_fast" if d.engine == "wiener_fast" else "parametric"
        self._show_deblur_controls()
        if mode == "neural":
            self.deblur_status.setText(
                "AI Neural is experimental and slow: about 11 seconds per 512-pixel tile on a laptop CPU, around "
                "2 minutes for this preview, and longer again for a full-size export.")
        else:
            self.deblur_status.setText("Lengths are in pixels of the full-size photo.")
        self._apply_deblur("AI de-blurring" if mode == "neural" else "De-blurring")

    def _on_deblur_area(self, _index: int) -> None:
        self.recipe.deblur.mask_mode = self.deblur_area.currentData()
        self._apply_deblur("Limiting de-blur to the subject or faces")

    def _on_deblur_auto(self) -> None:
        """Fill in the blur from the frame itself: direction and length of a
        smear, or the size of a focus blur."""
        import cv2

        from src.ai_tools import blur_analysis, deblur

        if self.source is None:
            return
        encoded = np.round(dp.srgb_encode(self.source.to_linear()) * 255.0).astype(np.uint8)
        bgr = cv2.cvtColor(encoded, cv2.COLOR_RGB2BGR)
        report = blur_analysis.analyze(bgr)
        scale = self.source.scale or 1.0
        d = self.recipe.deblur
        # The verdict also weighs overall sharpness, so a frame the photographer
        # has already judged blurred may come back "soft". Asked to de-blur, only
        # the kind of blur matters: a directional loss of detail is a smear.
        directionality, angle = blur_analysis.directional_blur(bgr)
        motion = report.verdict == blur_analysis.MOTION_BLUR or (
            report.verdict == blur_analysis.SOFT and directionality >= blur_analysis.MOTION_DIRECTIONALITY)
        if motion:
            d.blur_type, d.angle = "motion", float(report.motion_angle if report.motion_angle is not None else angle)
            length = deblur.estimate_motion_length(encoded, d.angle)
            if length is not None:
                d.length = pr.clamp("deblur_length", length / scale)
                message = f"Motion blur at {d.angle:.0f}°, about {d.length:.1f} px long."
            else:
                message = (f"Motion blur at {d.angle:.0f}°; its length could not be measured, "
                           "so set Smear Length by eye.")
        elif report.verdict in (blur_analysis.OUT_OF_FOCUS, blur_analysis.SOFT):
            d.blur_type = "defocus"
            radius = deblur.estimate_defocus_radius(encoded)
            if radius is not None:
                d.radius = pr.clamp("deblur_radius", radius / scale)
                message = f"Out of focus, a blur about {d.radius:.1f} px in radius."
            else:
                message = "Out of focus; its size could not be measured, so set Defocus Radius by eye."
        else:
            self.deblur_status.setText(
                f"The analysis found no clear motion or focus blur ({report.blur_kind.lower()}); "
                "settings unchanged.")
            return
        if d.engine == "neural":
            d.engine = "parametric"
        d.enabled = True
        self.deblur_status.setText(message)
        self._load_controls()
        self._apply_deblur("De-blurring")

    def _on_curve(self, points: list) -> None:
        setattr(self.recipe.curve, self.curve_channel.currentText(), points)
        self._changed()

    def _on_hsl(self, index: int, value: float) -> None:
        getattr(self.recipe.hsl, self.hsl_property.currentText())[index] = value
        self._changed()

    def _on_grade(self, attr: str, value: float) -> None:
        setattr(getattr(self.recipe.grade, self.grade_zone.currentText()), attr, value)
        self._changed()

    def _on_look(self, _index: int) -> None:
        name = self.look_combo.currentData()
        strength = self.recipe.look.strength
        self.recipe.look = pr.builtin_looks()[name] if name else pr.Look()
        self.recipe.look.strength = strength if name else 100.0
        self._changed()
        self._commit()

    def _on_lut(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose a 3D LUT", "", "Cube LUT (*.cube)")
        if not path:
            return
        if dp.load_cube(path) is None:
            QMessageBox.warning(self, "LUT", "That file is not a 3D .cube LUT this app can read.")
            return
        self.recipe.lut = pr.Lut(path, self.recipe.lut.strength or 100.0)
        self.lut_label.setText(Path(path).name)
        self._changed()
        self._commit()

    def _on_clear_lut(self) -> None:
        self.recipe.lut = pr.Lut()
        self.lut_label.setText("No LUT")
        self._changed()
        self._commit()

    def _on_lens_profile(self, checked: bool) -> None:
        self.recipe.lens.profile = checked
        self._changed()
        self._commit()

    def _turn(self, clockwise: int) -> None:
        g = self.recipe.geometry
        g.quarter_turns = (g.quarter_turns + clockwise) % 4
        g.crop = None
        self._changed()
        self._commit()

    def _flip(self) -> None:
        self.recipe.geometry.flip_horizontal = not self.recipe.geometry.flip_horizontal
        self._changed()
        self._commit()

    def _on_corners(self, text: str) -> None:
        self.recipe.geometry.constrain_to_image = text == "Crop"
        self.recipe.geometry.fill_blank = text == "Fill"
        self._changed()
        self._commit()
        from src.ai_tools import retouch

        if text == "Fill" and not retouch.Inpainter.is_cached() and not getattr(self, "_asked_fill", False):
            # Offered once: the quick fill works without it, the neural one
            # invents texture where the quick one smears.
            self._asked_fill = True
            self.shell.with_models(
                [retouch.LAMA_SPEC], "Filling corners with invented texture (otherwise a quick smear fill is used)",
                lambda: self.status_label.setText("The neural fill will be used at export."),
                self.compare_button)

    def _on_aspect(self, text: str) -> None:
        if self.source is None:
            return
        w, h = self.source.size
        if self.recipe.geometry.quarter_turns % 2:
            w, h = h, w
        self.recipe.geometry.crop = centred_crop(ASPECTS[text], w / float(h))
        self._changed()
        self._commit()

    # -- whole-recipe actions --------------------------------------------------- #

    def _replace(self, recipe: pr.Recipe) -> None:
        if self.locked:
            return
        self.recipe = recipe
        self._load_controls()
        self._changed()
        self._commit()

    def _on_auto_tone(self) -> None:
        if self.source is None:
            return
        recipe = self.recipe.copy()
        recipe.basic.exposure = round(dp.auto_tone(self.source) - self.source.baseline, 2)
        self._replace(recipe)

    def _on_reset(self) -> None:
        recipe = pr.Recipe()
        recipe.layers, recipe.vectors = self.recipe.layers, self.recipe.vectors
        self._replace(recipe)

    def _on_copy(self) -> None:
        self._clipboard = pr.to_dict(self.recipe)
        self.status_label.setText("Develop settings copied. Layers and overlays are not included.")

    def _settings_from_clipboard(self, target: pr.Recipe) -> pr.Recipe:
        pasted = pr.from_dict(self._clipboard)
        # Framing and retouching belong to one photograph; tone and colour travel.
        pasted.geometry, pasted.layers, pasted.vectors = target.geometry, target.layers, target.vectors
        pasted.deblur = target.deblur
        return pasted

    def _on_paste(self) -> None:
        if self._clipboard is None:
            self.status_label.setText("Nothing copied yet.")
            return
        self._replace(self._settings_from_clipboard(self.recipe))

    def _on_sync(self) -> None:
        targets = [p for p in self._selected_paths() if str(p) != self.key]
        if not targets:
            QMessageBox.information(self, "Sync", "Select the frames in Cull to copy these settings to.")
            return
        self._commit()
        self._clipboard = pr.to_dict(self.recipe)
        synced = self.session.sync_settings(self.recipe, targets, label=f"Synced from {Path(self.key).name}")
        skipped = len(targets) - len(synced)
        self.session.save_all()
        log = getattr(type(self.shell), "log_action", None)
        if log is not None and synced:
            log(self.shell, "develop", f"Synced settings from {Path(self.key).name} to {len(synced):,} photos")
        self.status_label.setText(
            f"Settings copied to {len(synced)} frame(s). Their crops, layers and overlays were kept."
            + (f" {skipped} rejected frame(s) were skipped: they are locked until restored." if skipped else ""))

    # -- virtual copies and history ---------------------------------------------- #

    def duplicate(self):
        """Ctrl+': a new virtual copy of the version being edited, which
        Develop then shows."""
        if self.key is None:
            return None
        self._commit()
        entry = self.session.create_virtual_copy(self.key, self.version)
        self.status_label.setText(
            f"Created virtual copy {entry.variant_key}. It starts with these settings; "
            "no image data was duplicated.")
        return entry

    def _refresh_history(self) -> None:
        if not hasattr(self, "history_list"):
            return
        self.history_list.clear()
        steps = self.session.history(self.key, self.version).steps if self.key is not None else []
        current = pr.to_dict(self.session.recipe(self.key, self.version)) if self.key is not None else None
        for index, step in enumerate(steps):
            try:
                when = datetime.fromisoformat(step.time).astimezone().strftime("%d %b %H:%M")
            except ValueError:
                when = ""
            item = QListWidgetItem(f"{step.label}" + (f"   \u00b7 {when}" if when else ""))
            item.setData(Qt.ItemDataRole.UserRole, index)
            self.history_list.addItem(item)
        if steps:
            # The step the photo is at now, if it matches one.
            at = next((i for i in range(len(steps) - 1, -1, -1) if steps[i].recipe == current), None)
            if at is not None:
                self.history_list.setCurrentRow(at)
            self.history_list.scrollToBottom()
        self.history_note.setText(f"{len(steps)} step{'s' if len(steps) != 1 else ''}. Click one to go back to it."
                                  if steps else "No edits yet.")

    def _on_history_clicked(self, item) -> None:
        self.restore_step(item.data(Qt.ItemDataRole.UserRole))

    def restore_step(self, index: int) -> None:
        """Put the version back as it was at a history step (undoable)."""
        if self.key is None:
            return
        self._commit()
        before = pr.to_dict(self.recipe)
        restored = self.session.restore_history_step(self.key, self.version, index)
        if pr.to_dict(restored) != before:
            self._history.append(before)
            del self._history[:-HISTORY_LIMIT]
            self._future.clear()
        self.recipe = restored.copy()
        self._load_controls()
        self._changed()
        self.status_label.setText(f"Back to: {self.session.history(self.key, self.version).steps[index].label}")

    def undo(self) -> None:
        if self._history and not self.locked:
            current = pr.to_dict(self.recipe)
            self._future.append(current)
            previous = self._history.pop()
            self.recipe = pr.from_dict(previous)
            self.session.set_recipe(self.key, self.recipe.copy(), self.version,
                                    label="Undo: " + eh.describe_change(previous, current))
            self._load_controls()
            self._changed()

    def redo(self) -> None:
        if self._future and not self.locked:
            current = pr.to_dict(self.recipe)
            self._history.append(current)
            following = self._future.pop()
            self.recipe = pr.from_dict(following)
            self.session.set_recipe(self.key, self.recipe.copy(), self.version,
                                    label="Redo: " + eh.describe_change(current, following))
            self._load_controls()
            self._changed()

    # -- view ----------------------------------------------------------------- #

    def _toggle_compare(self) -> None:
        on = not self.canvas.comparing
        self.canvas.set_compare(on)
        self.compare_button.setChecked(on)
        if on:
            self._before_key = None
            self._render()

    def _pick_sibling(self, advance: bool = False) -> None:
        """Choose which other version to show beside the one being edited."""
        others = [v.id for v in self.session.versions(self.key)] if self.key else []
        others = [v for v in others if v != self.version]
        if not others:
            self._sibling = None
        elif advance and self._sibling in others:
            self._sibling = others[(others.index(self._sibling) + 1) % len(others)]
        elif self._sibling not in others:
            # The next version along, so Master compares with Copy 1 first.
            later = [v for v in others if v > self.version]
            self._sibling = (later or others)[0]
        self._sibling_key = None

    def _toggle_versions(self, on: bool | None = None) -> None:
        on = (not self._comparing_versions) if on is None else on
        if on and (self.key is None or self.session.version_count(self.key) < 2):
            self._comparing_versions = False
            self.versions_button.setChecked(False)
            self.status_label.setText("This photo has one version. Press + Version or + 3 Styles to branch another.")
            return
        self._comparing_versions = on
        self.versions_button.setChecked(on)
        self.sibling_canvas.setVisible(on)
        self.main_caption.setVisible(on)
        self.sibling_caption.setVisible(on)
        if on:
            self._pick_sibling()
            self._show_names()
            self._render()
        else:
            self._sibling, self._sibling_key = None, None
            self.sibling_canvas.set_image(None)

    def _next_sibling(self) -> None:
        if not self._comparing_versions:
            self._toggle_versions(True)
            return
        self._pick_sibling(advance=True)
        self._show_names()
        self._render()

    def _step_version(self, offset: int) -> None:
        if self.key is None or self.session.version_count(self.key) < 2:
            return
        self._commit()
        self.session.step_version(self.key, offset)

    def _toggle_clipping(self) -> None:
        self._show_clipping = not self._show_clipping
        self.clip_button.setChecked(self._show_clipping)
        if self._last_render is not None:
            self.canvas.set_image(clipping_overlay(self._last_render) if self._show_clipping else self._last_render)

    def _step(self, offset: int) -> None:
        """Left / Right: the next photo that can be edited."""
        self._commit()
        self.session.step(offset, self.session.editable_keys())

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        key, mods = event.key(), event.modifiers()
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        if key == Qt.Key.Key_Backslash:
            self._toggle_compare()
        elif key == Qt.Key.Key_V and shift and not ctrl:
            self._next_sibling()
        elif key == Qt.Key.Key_V and not ctrl:
            self._toggle_versions()
        elif key in (Qt.Key.Key_BracketLeft, Qt.Key.Key_BracketRight) and not ctrl:
            self._step_version(-1 if key == Qt.Key.Key_BracketLeft else 1)
        elif key == Qt.Key.Key_J:
            self._toggle_clipping()
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_Right) and not ctrl:
            self._step(-1 if key == Qt.Key.Key_Left else 1)
        elif ctrl and key == Qt.Key.Key_Z:
            self.undo()
        elif ctrl and key == Qt.Key.Key_Y:
            self.redo()
        elif ctrl and shift and key == Qt.Key.Key_C:
            self._on_copy()
        elif ctrl and shift and key == Qt.Key.Key_V:
            self._on_paste()
        elif key == Qt.Key.Key_0:
            self.canvas.fit()
        elif key == Qt.Key.Key_1:
            self.canvas.actual_size()
        else:
            return super().keyPressEvent(event)
        event.accept()


_RANGE_NAMES = {
    ("basic", attr): attr for attr, *_ in BASIC
} | {
    ("grade", "balance"): "grade_balance", ("grade", "blending"): "grade_blending",
    ("look", "strength"): "look_strength", ("lut", "strength"): "lut_strength",
    ("detail", "sharpen_amount"): "sharpen_amount", ("detail", "sharpen_radius"): "sharpen_radius",
    ("detail", "noise_reduction"): "noise_reduction", ("lens", "distortion"): "lens_distortion",
    ("lens", "vignette"): "lens_vignette", ("geometry", "rotation"): "rotation",
    ("deblur", "angle"): "deblur_angle", ("deblur", "length"): "deblur_length",
    ("deblur", "radius"): "deblur_radius", ("deblur", "damping"): "deblur_damping",
    ("geometry", "keystone_vertical"): "keystone", ("geometry", "keystone_horizontal"): "keystone",
}


def _range_name(section: str, attr: str) -> str | None:
    return _RANGE_NAMES.get((section, attr))
