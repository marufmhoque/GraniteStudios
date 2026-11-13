"""The Develop workspace: one frame large, every develop setting beside it.

Keyboard, with the picture focused:

    left / right   previous / next frame       \\        before / after
    J              show clipped shadows and highlights
    Ctrl+Z / Y     undo / redo                 Ctrl+Shift+C / V   copy / paste settings
    0              fit to window               1          100 percent

Every control edits the frame's recipe. The preview is rendered on a
background thread from a proxy of the frame, with the same pipeline the
export uses, so what is seen is what will be written. The recipe is saved to
its sidecar when the frame changes or the workspace is left.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from src.engine import develop_pipeline as dp
from src.engine import layer_stack
from src.engine import parametric_recipe as pr
from src.engine import session as topics
from src.ui.widgets.canvas import ImageCanvas
from src.ui.widgets.controls import CurveEditor, Histogram, Section, SliderRow, clipping_overlay
from src.ui.widgets.filmstrip import Filmstrip
from src.ui.widgets.renderer import LatestOnlyRenderer
from src.ui.workspaces.base import Workspace

logger = logging.getLogger(__name__)

PREVIEW_SIDE = 1600
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
        self.renderer.failed.connect(lambda message: self.status_label.setText(f"Preview failed: {message}"))
        self.loader_job = LatestOnlyRenderer(self)
        self.loader_job.ready.connect(self._on_source_loaded)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._build()
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
        bar.addWidget(self.clip_button)
        bar.addWidget(self._button("Fit", lambda: self.canvas.fit(), "Fit the frame to the window (0)"))
        bar.addWidget(self._button("100%", lambda: self.canvas.actual_size(), "One image pixel per screen pixel (1)"))
        centre.addLayout(bar)
        self.canvas = ImageCanvas()
        centre.addWidget(self.canvas, stretch=1)
        self.filmstrip = Filmstrip(self.session, self.shell.thumbnail_for)
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
        if self.session.current != self.key:
            self.load(self.session.current)
        self.setFocus()

    def on_leave(self) -> None:
        self.session.save_recipes()

    def on_session(self, topic: str) -> None:
        if topic == topics.LIBRARY:
            self.key, self.source = None, None
            self.canvas.set_image(None)
            self.filmstrip.rebuild()
        elif topic in (topics.CURRENT, topics.MARKS):
            self.filmstrip.refresh()
            if topic == topics.CURRENT and self.shell.__dict__.get("_current_index") is not None and self.shell.current_workspace is self and self.session.current != self.key:
                self.load(self.session.current)

    def load(self, key: str | None) -> None:
        """Open a frame. The decode runs in the background."""
        self.session.save_recipes()
        self.renderer.cancel()
        self.key = key
        self._history, self._future = [], []
        self._before_key = None
        self._layer_context = None
        if key is None:
            self.source = None
            self.canvas.set_image(None)
            self.frame_label.setText("No photo open")
            return
        self.recipe = self.session.recipe(key).copy()
        self._load_controls()
        self.frame_label.setText(Path(key).name + ("   (rejected)" if key in self.rejected else ""))
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
        self.lens_status.setText(
            "A profile for this camera and lens was found." if dp.lens_profile_available(key)
            else "No profile for this camera and lens; use the sliders."
        )
        if source.note:
            self.status_label.setText(f"{Path(key).name}: {source.note}.")
        self._render()

    # -- rendering ------------------------------------------------------------- #

    def _context(self):
        if not self.recipe.layers:
            return None
        if self._layer_context is None:
            from src.ai_tools.segmentation import Segmenter

            self._layer_context = layer_stack.RenderContext(
                segmenter=Segmenter(download=False),
                assets=layer_stack.assets_folder(self.key, self.edit_folder) if self.edit_folder else None,
                metadata=self.metadata.get(self.key),
            )
        return self._layer_context

    def _render(self) -> None:
        if self.source is None:
            return
        source, recipe, context = self.source, self.recipe.copy(), self._context()
        geometry_key = repr(pr.to_dict(recipe)["geometry"])
        want_before = self.canvas.comparing and geometry_key != self._before_key

        def job():
            after = dp.render(source, recipe, layers=context)
            before = None
            if want_before:
                neutral = pr.Recipe()
                neutral.geometry = recipe.geometry
                before = dp.render(source, neutral)
            return after, before, geometry_key

        self.renderer.request(job)

    def _on_rendered(self, result) -> None:
        after, before, geometry_key = result
        self._last_render = after
        self.histogram.set_image(after)
        self.curve.set_histogram(self.histogram.luminance_counts())
        shadows, highlights = self.histogram.shadows_clipped, self.histogram.highlights_clipped
        self.clip_readout.setText(f"Clipped: {shadows * 100:.2f}% shadows · {highlights * 100:.2f}% highlights")
        self.canvas.set_image(clipping_overlay(after) if self._show_clipping else after)
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
        g = self.recipe.geometry
        self.corners_combo.blockSignals(True)
        self.corners_combo.setCurrentText("Crop" if g.constrain_to_image else ("Fill" if g.fill_blank else "Leave black"))
        self.corners_combo.blockSignals(False)

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
        """A setting moved: show it now; it is committed on release."""
        self._render()

    def _commit(self) -> None:
        """A change is finished: record it for undo and hand it to the session."""
        if self.key is None:
            return
        snapshot = pr.to_dict(self.session.recipe(self.key))
        if snapshot != pr.to_dict(self.recipe):
            self._history.append(snapshot)
            del self._history[:-HISTORY_LIMIT]
            self._future.clear()
            self.session.set_recipe(self.key, self.recipe.copy())

    def _on_value(self, key, value: float) -> None:
        section, attr = key
        setattr(self._section(section), attr, pr.clamp(_range_name(section, attr), value)
                if _range_name(section, attr) else value)
        self._changed()

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
        for path in targets:
            self.session.set_recipe(path, self._settings_from_clipboard(self.session.recipe(path)))
        self.session.save_recipes()
        self.status_label.setText(
            f"Settings copied to {len(targets)} frame(s). Their crops, layers and overlays were kept.")

    def undo(self) -> None:
        if self._history:
            self._future.append(pr.to_dict(self.recipe))
            self.recipe = pr.from_dict(self._history.pop())
            self.session.set_recipe(self.key, self.recipe.copy())
            self._load_controls()
            self._changed()

    def redo(self) -> None:
        if self._future:
            self._history.append(pr.to_dict(self.recipe))
            self.recipe = pr.from_dict(self._future.pop())
            self.session.set_recipe(self.key, self.recipe.copy())
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

    def _toggle_clipping(self) -> None:
        self._show_clipping = not self._show_clipping
        self.clip_button.setChecked(self._show_clipping)
        if self._last_render is not None:
            self.canvas.set_image(clipping_overlay(self._last_render) if self._show_clipping else self._last_render)

    def _step(self, offset: int) -> None:
        self._commit()
        self.session.step(offset)

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        key, mods = event.key(), event.modifiers()
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        if key == Qt.Key.Key_Backslash:
            self._toggle_compare()
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
    ("geometry", "keystone_vertical"): "keystone", ("geometry", "keystone_horizontal"): "keystone",
}


def _range_name(section: str, attr: str) -> str | None:
    return _RANGE_NAMES.get((section, attr))
