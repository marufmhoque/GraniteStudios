"""The Retouch workspace: layers, masks, healing and portrait work.

Tools (left rail), used with the left mouse button; the middle button pans:

    Heal     click a blemish or a dust spot to remove it
    Clone    first click picks the source, later clicks stamp it
    Brush    paint the selected layer's mask (hold Alt to erase)
    Select   click an object to select it with the AI model

Every tool writes to a layer in the frame's recipe, never to pixels. Heal
spots and clone stamps are stored as positions, so they render at full
resolution on export. The frame is shown before cropping and straightening,
because that is the frame the layers are placed on.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPen
from PyQt6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QRadioButton, QScrollArea, QVBoxLayout, QWidget,
)

from src.engine import develop_pipeline as dp
from src.engine import layer_stack
from src.engine import parametric_recipe as pr
from src.engine import session as topics
from src.ui.widgets.canvas import ImageCanvas
from src.ui.widgets.controls import Section, SliderRow
from src.ui.widgets.filmstrip import Filmstrip
from src.ui.widgets.renderer import LatestOnlyRenderer
from src.ui.workspaces.base import Workspace

logger = logging.getLogger(__name__)

PREVIEW_SIDE = 1600
TOOLS = ("heal", "clone", "brush", "select")
AI_KINDS = ("subject", "background", "person", "skin", "teeth", "eyes", "iris", "lips", "hair", "sky")
ADJUST = (("exposure", "Exposure", -3.0, 3.0, 0.05, 2), ("contrast", "Contrast", -100, 100, 1, 0),
          ("saturation", "Saturation", -100, 100, 1, 0), ("temperature", "Temperature", -100, 100, 1, 0),
          ("clarity", "Clarity", -100, 100, 1, 0), ("shadows", "Shadows", -100, 100, 1, 0),
          ("highlights", "Highlights", -100, 100, 1, 0))


def describe(layer: layer_stack.Layer) -> str:
    if layer.name:
        return layer.name
    if layer.kind == "retouch":
        return layer.op.replace("_", " ").capitalize() or "Retouch"
    mask = layer.mask.params.get("kind") if layer.mask.kind == "ai" else layer.mask.kind
    return f"{layer.kind.capitalize()} ({mask})" if mask and mask != "none" else layer.kind.capitalize()


class RetouchWorkspace(Workspace):
    title = "Retouch"

    def __init__(self, shell) -> None:
        super().__init__(shell)
        self.key: str | None = None
        self.source: dp.Source | None = None
        self.recipe = pr.Recipe()
        self.base: np.ndarray | None = None        # developed, no layers, no framing
        self._base_key: str | None = None
        self._context: layer_stack.RenderContext | None = None
        self._segmenter = None
        self._clone_source: tuple[float, float] | None = None
        self._stroke: list | None = None
        self._cursor: tuple[float, float] | None = None
        self.renderer = LatestOnlyRenderer(self)
        self.renderer.ready.connect(self._on_rendered)
        self.renderer.failed.connect(lambda m: self.status_label.setText(f"Preview failed: {m}"))
        self.loader_job = LatestOnlyRenderer(self)
        self.loader_job.ready.connect(self._on_loaded)
        self._build()
        self.session.subscribe(self.on_session)

    # -- layout ----------------------------------------------------------------- #

    def _build(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        rail = QVBoxLayout()
        rail.setContentsMargins(10, 12, 6, 10)
        self.tool_group = QButtonGroup(self)
        for index, (tool, label, tip) in enumerate((
            ("heal", "Heal", "Click a blemish or dust spot to remove it"),
            ("clone", "Clone", "Click once to pick a source, then click to stamp it"),
            ("brush", "Brush", "Paint the selected layer's mask; Alt erases"),
            ("select", "Select", "Click an object to select it with the AI model"),
        )):
            button = QRadioButton(label)
            button.setToolTip(tip)
            button.setProperty("tool", tool)
            self.tool_group.addButton(button, index)
            rail.addWidget(button)
        self.tool_group.button(0).setChecked(True)
        self.tool_group.idToggled.connect(lambda *_: self._on_tool_changed())
        self.brush_size = SliderRow("Size", 0.2, 15.0, 2.0, 0.1, 1, label_width=58)
        self.brush_size.setToolTip("Brush size, in percent of the frame's long edge")
        self.brush_hardness = SliderRow("Hardness", 0, 100, 50, label_width=58)
        rail.addWidget(QLabel("Brush"))
        rail.addWidget(self.brush_size)
        rail.addWidget(self.brush_hardness)
        rail.addSpacing(12)
        rail.addWidget(QLabel("Portrait"))
        for label, op, tip in (("Smooth Skin", "skin_smooth", "Even out skin; pores are kept"),
                               ("Whiten Teeth", "teeth_whiten", "Take the yellow out of teeth"),
                               ("Brighten Eyes", "eye_clarity", "Add clarity to the irises")):
            rail.addWidget(self._button(label, lambda _c=False, o=op: self._add_retouch(o), tip))
        rail.addSpacing(12)
        rail.addWidget(QLabel("Adjust an area"))
        self.ai_kind = QComboBox()
        self.ai_kind.addItems(AI_KINDS)
        rail.addWidget(self.ai_kind)
        rail.addWidget(self._button("New Layer from Mask", self._add_ai_layer,
                                    "Create an adjustment layer limited to this kind of area"))
        rail.addWidget(self._button("New Brushed Layer", self._add_brush_layer,
                                    "An adjustment layer you paint where it applies"))
        rail.addWidget(self._button("New Gradient Layer", self._add_gradient_layer,
                                    "An adjustment that fades from top to bottom, e.g. for a sky"))
        rail.addStretch()
        rail_host = QWidget()
        rail_host.setLayout(rail)
        rail_host.setFixedWidth(210)
        outer.addWidget(rail_host)

        centre = QVBoxLayout()
        centre.setContentsMargins(6, 10, 6, 8)
        self.frame_label = QLabel("No photo open")
        self.frame_label.setObjectName("PageTitle")
        note = QLabel("Shown before cropping and straightening. Middle mouse button pans; wheel zooms.")
        note.setObjectName("PageSubtitle")
        centre.addWidget(self.frame_label)
        centre.addWidget(note)
        self.canvas = ImageCanvas()
        self.canvas.pan_with_left_button = False
        self.canvas.pressed.connect(self._on_press)
        self.canvas.dragged.connect(self._on_drag)
        self.canvas.released.connect(self._on_release)
        self.canvas.hovered.connect(self._on_hover)
        self.canvas.set_overlay(self._paint_overlay)
        centre.addWidget(self.canvas, stretch=1)
        self.filmstrip = Filmstrip(self.session, self.shell.thumbnail_for)
        centre.addWidget(self.filmstrip)
        outer.addLayout(centre, stretch=1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(330)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel_host = QWidget()
        panel = QVBoxLayout(panel_host)
        panel.setContentsMargins(8, 10, 12, 10)
        panel.addWidget(QLabel("Layers (top of the list is applied last)"))
        self.layer_list = QListWidget()
        self.layer_list.setMinimumHeight(180)
        self.layer_list.currentRowChanged.connect(lambda _r: self._load_layer_controls())
        self.layer_list.itemChanged.connect(self._on_item_toggled)
        panel.addWidget(self.layer_list)
        order = QHBoxLayout()
        order.addWidget(self._button("Up", lambda: self._move(1), "Apply later"))
        order.addWidget(self._button("Down", lambda: self._move(-1), "Apply earlier"))
        order.addWidget(self._button("Delete", self._delete, "Remove the layer"))
        panel.addLayout(order)
        self.show_mask = QCheckBox("Show mask")
        self.show_mask.toggled.connect(lambda _c: self.canvas.update())
        panel.addWidget(self.show_mask)

        props = Section("Layer")
        self.opacity = SliderRow("Opacity", 0, 100, 100)
        self.opacity.changed.connect(lambda v: self._set_layer("opacity", v))
        self.opacity.released.connect(self._commit)
        props.add(self.opacity)
        self.blend = QComboBox()
        self.blend.addItems(list(layer_stack.BLEND_MODES))
        self.blend.currentTextChanged.connect(lambda m: (self._set_layer("blend", m), self._commit()))
        props.add_row(QLabel("Blend"), self.blend)
        self.amount = SliderRow("Amount", 0, 100, 50)
        self.amount.changed.connect(lambda v: self._set_layer("amount", v))
        self.amount.released.connect(self._commit)
        props.add(self.amount)
        self.invert = QCheckBox("Invert mask")
        self.invert.toggled.connect(lambda c: (self._set_mask("invert", c), self._commit()))
        props.add(self.invert)
        self.feather = SliderRow("Feather", 0, 10, 0, 0.1, 1)
        self.feather.changed.connect(lambda v: self._set_mask("feather", v / 100.0))
        self.feather.released.connect(self._commit)
        props.add(self.feather)
        panel.addWidget(props)

        self.adjust_section = Section("Adjustment")
        self.adjust_rows: dict[str, SliderRow] = {}
        for attr, label, low, high, step, decimals in ADJUST:
            row = SliderRow(label, low, high, 0.0, step, decimals)
            row.changed.connect(lambda v, a=attr: self._set_adjust(a, v))
            row.released.connect(self._commit)
            self.adjust_section.add(row)
            self.adjust_rows[attr] = row
        panel.addWidget(self.adjust_section)
        self.notes = QLabel()
        self.notes.setObjectName("PageSubtitle")
        self.notes.setWordWrap(True)
        panel.addWidget(self.notes)
        panel.addStretch()
        scroll.setWidget(panel_host)
        outer.addWidget(scroll)

    def _button(self, text, slot, tip="") -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("SecondaryButton")
        button.setToolTip(tip)
        button.clicked.connect(slot)
        return button

    # -- the open frame ------------------------------------------------------------ #

    def on_enter(self) -> None:
        self.filmstrip.refresh()
        if self.session.current != self.key:
            self.load(self.session.current)
        else:
            # Develop may have changed the settings underneath the layers.
            self.recipe = self.session.recipe(self.key).copy() if self.key else self.recipe
            self._render(rebuild_base=True)

    def on_leave(self) -> None:
        self.session.save_recipes()

    def on_session(self, topic: str) -> None:
        if topic == topics.LIBRARY:
            self.key, self.source, self.base = None, None, None
            self.canvas.set_image(None)
            self.filmstrip.rebuild()
        elif topic in (topics.CURRENT, topics.MARKS):
            self.filmstrip.refresh()
            if topic == topics.CURRENT and self.shell.__dict__.get("_current_index") is not None and self.shell.current_workspace is self and self.session.current != self.key:
                self.load(self.session.current)

    def load(self, key: str | None) -> None:
        self.session.save_recipes()
        self.key, self.base, self._base_key = key, None, None
        self._context, self._clone_source = None, None
        if key is None:
            self.source = None
            self.canvas.set_image(None)
            return
        self.recipe = self.session.recipe(key).copy()
        self.frame_label.setText(Path(key).name)
        self._refresh_list()
        self.canvas.caption = "Loading…"
        path = Path(key)
        self.loader_job.request(lambda: (key, self._open(path)))

    @staticmethod
    def _open(path: Path):
        source = dp.open_source(path, half_size=True)
        return source.proxy(PREVIEW_SIDE) if source is not None else None

    def _on_loaded(self, result) -> None:
        key, source = result
        if key != self.key:
            return
        self.canvas.caption = "" if source is not None else "This file could not be read."
        self.source = source
        self._render(rebuild_base=True)

    # -- rendering ------------------------------------------------------------------- #

    def _layers(self) -> list[layer_stack.Layer]:
        return layer_stack.layers_from_recipe(self.recipe)

    def _ctx(self) -> layer_stack.RenderContext:
        if self._context is None:
            if self._segmenter is None:
                from src.ai_tools.segmentation import Segmenter

                self._segmenter = Segmenter(download=False)
            self._context = layer_stack.RenderContext(
                segmenter=self._segmenter,
                assets=layer_stack.assets_folder(self.key, self.edit_folder) if self.edit_folder else None,
                metadata=self.metadata.get(self.key),
            )
        self._context.notes.clear()
        return self._context

    def _render(self, rebuild_base: bool = False) -> None:
        if self.source is None:
            return
        develop = self.recipe.copy()
        develop.layers, develop.vectors = [], []
        base_key = repr(pr.to_dict(develop))
        if rebuild_base or base_key != self._base_key:
            self.base, self._base_key = None, base_key
        source, base, layers, ctx = self.source, self.base, self._layers(), self._ctx()

        def job():
            developed = base if base is not None else dp.render(source, develop, geometry=False)
            return developed, layer_stack.composite(developed, layers, ctx), list(ctx.notes)

        self.renderer.request(job)

    def _on_rendered(self, result) -> None:
        self.base, composed, notes = result
        self.canvas.set_image(composed)
        self.notes.setText("\n".join(notes))

    # -- layers ------------------------------------------------------------------------ #

    def _store(self, layers: list[layer_stack.Layer], select: int | None = None) -> None:
        self.recipe.layers = [layer_stack.layer_to_dict(layer) for layer in layers]
        self._refresh_list(select)
        self._render()

    def _commit(self) -> None:
        if self.key is not None:
            self.session.set_recipe(self.key, self.recipe.copy())

    def _refresh_list(self, select: int | None = None) -> None:
        layers = self._layers()
        current = self._selected_index() if select is None else select
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        for layer in reversed(layers):                 # top of the list = applied last
            item = QListWidgetItem(describe(layer))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked)
            self.layer_list.addItem(item)
        self.layer_list.blockSignals(False)
        if layers and current is not None:
            self.layer_list.setCurrentRow(len(layers) - 1 - min(max(current, 0), len(layers) - 1))
        self._load_layer_controls()

    def _selected_index(self) -> int | None:
        """Index into the stack (bottom first) of the highlighted layer."""
        row = self.layer_list.currentRow()
        count = self.layer_list.count()
        return None if row < 0 or count == 0 else count - 1 - row

    def _selected_layer(self) -> layer_stack.Layer | None:
        index = self._selected_index()
        layers = self._layers()
        return layers[index] if index is not None and index < len(layers) else None

    def _update_selected(self, change) -> None:
        index = self._selected_index()
        layers = self._layers()
        if index is None or index >= len(layers):
            return
        change(layers[index])
        self.recipe.layers[index] = layer_stack.layer_to_dict(layers[index])
        self._render()

    def _load_layer_controls(self) -> None:
        layer = self._selected_layer()
        for row in (self.opacity, self.amount, self.feather):
            row.setEnabled(layer is not None)
        if layer is None:
            self.adjust_section.setVisible(False)
            return
        self.opacity.set_value(layer.opacity)
        self.amount.set_value(layer.amount)
        self.amount.setVisible(layer.kind == "retouch" and layer.op not in ("heal", "clone"))
        self.feather.set_value(layer.mask.feather * 100.0)
        self.blend.blockSignals(True)
        self.blend.setCurrentText(layer.blend)
        self.blend.blockSignals(False)
        self.invert.blockSignals(True)
        self.invert.setChecked(layer.mask.invert)
        self.invert.blockSignals(False)
        self.adjust_section.setVisible(layer.kind == "adjustment")
        basic = layer.recipe.get("basic", {}) if isinstance(layer.recipe.get("basic"), dict) else {}
        for attr, row in self.adjust_rows.items():
            row.set_value(float(basic.get(attr, 0.0) or 0.0))
        self.canvas.update()

    def _set_layer(self, attr: str, value) -> None:
        self._update_selected(lambda layer: setattr(layer, attr, value))

    def _set_mask(self, attr: str, value) -> None:
        self._update_selected(lambda layer: setattr(layer.mask, attr, value))

    def _set_adjust(self, attr: str, value: float) -> None:
        def change(layer):
            basic = dict(layer.recipe.get("basic", {}))
            basic[attr] = value
            layer.recipe = {**layer.recipe, "basic": basic}
        self._update_selected(change)

    def _on_item_toggled(self, item: QListWidgetItem) -> None:
        row = self.layer_list.row(item)
        index = self.layer_list.count() - 1 - row
        layers = self._layers()
        if 0 <= index < len(layers):
            layers[index].visible = item.checkState() == Qt.CheckState.Checked
            self._store(layers, index)
            self._commit()

    def _add(self, layer: layer_stack.Layer) -> None:
        if self.key is None:
            QMessageBox.information(self, "Retouch", "Open a photo first: double-click one in Cull.")
            return
        layers = self._layers() + [layer]
        self._store(layers, len(layers) - 1)
        self._commit()

    def _add_retouch(self, op: str) -> None:
        self._add(layer_stack.Layer(kind="retouch", op=op, amount=50.0))

    def _add_ai_layer(self) -> None:
        kind = self.ai_kind.currentText()
        from src.ai_tools.segmentation import PARTIAL_KINDS, Segmenter

        if Segmenter.download_mb(kind) > 0:
            QMessageBox.information(
                self, "Model needed",
                f"The {kind} mask needs a model that is not on this computer yet "
                f"({Segmenter.download_mb(kind):.0f} MB). It is fetched the first time "
                "a feature that needs it is switched on.")
            return
        name = f"Adjust {kind}" + (" (approximate)" if kind in PARTIAL_KINDS else "")
        self._add(layer_stack.Layer(kind="adjustment", name=name, recipe={"basic": {}},
                                    mask=layer_stack.Mask(kind="ai", params={"kind": kind}, feather=0.003)))

    def _add_brush_layer(self) -> None:
        self._add(layer_stack.Layer(kind="adjustment", name="Brushed adjustment", recipe={"basic": {}},
                                    mask=layer_stack.Mask(kind="brush", params={"strokes": []})))
        self.tool_group.button(TOOLS.index("brush")).setChecked(True)

    def _add_gradient_layer(self) -> None:
        self._add(layer_stack.Layer(kind="adjustment", name="Graduated adjustment", recipe={"basic": {}},
                                    mask=layer_stack.Mask(kind="gradient", params={"start": [0.5, 0.55],
                                                                                  "end": [0.5, 0.0]})))

    def _move(self, direction: int) -> None:
        index = self._selected_index()
        layers = self._layers()
        target = None if index is None else index + direction
        if target is None or not 0 <= target < len(layers):
            return
        layers[index], layers[target] = layers[target], layers[index]
        self._store(layers, target)
        self._commit()

    def _delete(self) -> None:
        index = self._selected_index()
        layers = self._layers()
        if index is None or index >= len(layers):
            return
        del layers[index]
        self._store(layers, max(0, index - 1))
        self._commit()

    # -- tools --------------------------------------------------------------------------- #

    def _tool(self) -> str:
        return self.tool_group.checkedButton().property("tool")

    def _on_tool_changed(self) -> None:
        self._clone_source = None
        self.canvas.update()

    def _radius(self) -> float:
        return self.brush_size.value() / 100.0

    def _layer_of(self, op: str) -> tuple[list, int]:
        """The heal or clone layer to add to, creating it if needed."""
        layers = self._layers()
        for index in range(len(layers) - 1, -1, -1):
            if layers[index].kind == "retouch" and layers[index].op == op:
                return layers, index
        layers.append(layer_stack.Layer(kind="retouch", op=op, name="Healing" if op == "heal" else "Clone"))
        return layers, len(layers) - 1

    def _on_press(self, x: float, y: float, button) -> None:
        if self.key is None or button != Qt.MouseButton.LeftButton or not (0 <= x <= 1 and 0 <= y <= 1):
            return
        tool = self._tool()
        if tool == "heal":
            layers, index = self._layer_of("heal")
            spots = list(layers[index].params.get("spots", []))
            spots.append([round(x, 5), round(y, 5), round(self._radius() / 2.0, 5)])
            layers[index].params = {**layers[index].params, "spots": spots}
            self._store(layers, index)
            self._commit()
        elif tool == "clone":
            if self._clone_source is None:
                self._clone_source = (x, y)
                self.status_label.setText("Clone source set. Now click where it should go.")
                self.canvas.update()
                return
            layers, index = self._layer_of("clone")
            stamps = list(layers[index].params.get("stamps", []))
            stamps.append({"source": [round(self._clone_source[0], 5), round(self._clone_source[1], 5)],
                           "target": [round(x, 5), round(y, 5)], "radius": round(self._radius() / 2.0, 5),
                           "hardness": self.brush_hardness.value() / 100.0})
            layers[index].params = {**layers[index].params, "stamps": stamps}
            self._store(layers, index)
            self._commit()
        elif tool == "brush":
            layer = self._selected_layer()
            if layer is None or layer.mask.kind != "brush":
                self.status_label.setText("Select a brushed layer first, or make one with “New Brushed Layer”.")
                return
            erase = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.AltModifier)
            self._stroke = [{"points": [[round(x, 5), round(y, 5)]], "radius": round(self._radius() / 2.0, 5),
                             "hardness": self.brush_hardness.value() / 100.0, "erase": erase}]
        elif tool == "select":
            self._click_select(x, y)

    def _on_drag(self, x: float, y: float) -> None:
        self._cursor = (x, y)
        if self._stroke is not None:
            self._stroke[0]["points"].append([round(x, 5), round(y, 5)])
        self.canvas.update()

    def _on_release(self, _x: float, _y: float) -> None:
        if self._stroke is None:
            return
        stroke, self._stroke = self._stroke[0], None

        def change(layer):
            strokes = list(layer.mask.params.get("strokes", []))
            strokes.append(stroke)
            layer.mask.params = {**layer.mask.params, "strokes": strokes}
        self._update_selected(change)
        self._commit()

    def _on_hover(self, x: float, y: float) -> None:
        self._cursor = (x, y)
        self.canvas.update()

    def _click_select(self, x: float, y: float) -> None:
        from src.ai_tools import model_store
        from src.ai_tools.segmentation import SAM_SPEC

        if not model_store.is_cached(SAM_SPEC):
            QMessageBox.information(
                self, "Model needed",
                f"Click-to-select needs a {SAM_SPEC.size / 1e6:.0f} MB model that is not on this computer yet.")
            return
        layer = self._selected_layer()
        if layer is not None and layer.mask.kind == "ai" and layer.mask.params.get("kind") == "click":
            def change(l):
                points = list(l.mask.params.get("points", []))[:5] + [[round(x, 5), round(y, 5)]]
                l.mask.params = {**l.mask.params, "points": points}
            self._update_selected(change)
            self._commit()
        else:
            self._add(layer_stack.Layer(kind="adjustment", name="Selected object", recipe={"basic": {}},
                                        mask=layer_stack.Mask(kind="ai", params={"kind": "click",
                                                                                 "points": [[x, y]]})))
        self.status_label.setText("Selecting… Click again to add to the selection.")

    # -- overlay ---------------------------------------------------------------------- #

    def _paint_overlay(self, painter, rect) -> None:
        def at(x, y):
            return QPointF(rect.x() + x * rect.width(), rect.y() + y * rect.height())

        long_edge = max(rect.width(), rect.height())
        layer = self._selected_layer()
        if self.show_mask.isChecked() and layer is not None and self.base is not None:
            try:
                mask = layer_stack.render_mask(layer.mask, self.base.astype(np.float32) / 255.0, self._ctx())
            except Exception:
                mask = None
            if mask is not None:
                from PyQt6.QtGui import QImage

                tint = np.zeros(mask.shape + (4,), np.uint8)
                tint[..., 0] = 255
                tint[..., 3] = (mask * 140).astype(np.uint8)
                image = QImage(tint.data, tint.shape[1], tint.shape[0], tint.shape[1] * 4,
                               QImage.Format.Format_RGBA8888).copy()
                painter.drawImage(rect, image)
        if layer is not None and layer.op == "heal":
            painter.setPen(QPen(QColor(255, 255, 255, 160), 1))
            for x, y, r in layer.params.get("spots", []):
                painter.drawEllipse(at(x, y), r * long_edge, r * long_edge)
        if self._clone_source is not None:
            painter.setPen(QPen(QColor("#4fc3f7"), 1.5))
            centre = at(*self._clone_source)
            painter.drawLine(centre + QPointF(-8, 0), centre + QPointF(8, 0))
            painter.drawLine(centre + QPointF(0, -8), centre + QPointF(0, 8))
        if self._cursor is not None and self._tool() in ("heal", "clone", "brush"):
            radius = self._radius() / 2.0 * long_edge
            painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
            painter.drawEllipse(at(*self._cursor), radius, radius)
