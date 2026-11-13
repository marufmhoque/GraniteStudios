"""The Design workspace: text, shapes and watermarks over the finished frame.

Drag an item to move it. With "Snap" on, positions jump to the grid; edges
and centres also catch on other items and on the middle and edges of the
frame, and a guide line shows what they caught on. Arrow keys nudge the
selected item, Delete removes it.

Items are stored in the frame's recipe and drawn over the final, cropped
frame at export, so they land in the corner of what is delivered. The
design can also be written out on its own as SVG, or with the photograph as
a print-ready PDF.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPen
from PyQt6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from src.engine import develop_pipeline as dp
from src.engine import parametric_recipe as pr
from src.engine import session as topics
from src.graphics import vector_canvas as vc
from src.ui.widgets.canvas import ImageCanvas
from src.ui.widgets.controls import Section, SliderRow
from src.ui.widgets.filmstrip import Filmstrip
from src.ui.widgets.renderer import LatestOnlyRenderer
from src.ui.workspaces.base import Workspace

logger = logging.getLogger(__name__)

PREVIEW_SIDE = 1600
NUDGE = 0.002


def item_box(item: vc.Item, size: tuple[int, int], metadata=None) -> tuple[float, float, float, float]:
    """An item's bounding box as ``(x, y, w, h)`` fractions of the canvas."""
    w, h = size
    if item.kind == "text":
        layout = vc.layout_text(item, size, metadata)
        fx, fy = vc._ANCHOR_FACTORS[item.anchor]
        left = layout.origin[0] - fx * layout.width
        top = layout.origin[1] - fy * layout.height
        return left / w, top / h, layout.width / w, layout.height / h
    if item.kind == "path" and item.commands:
        xs = [v for c in item.commands for v in c[1::2]]
        ys = [v for c in item.commands for v in c[2::2]]
        if xs and ys:
            return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
    return item.x, item.y, item.w, item.h


def move_item(item: vc.Item, dx: float, dy: float) -> None:
    """Shift an item, and everything inside it, by a fraction of the canvas."""
    item.x += dx
    item.y += dy
    if item.kind == "path":
        item.commands = [
            [c[0]] + [v + (dx if i % 2 == 0 else dy) for i, v in enumerate(c[1:])] for c in item.commands
        ]
    for child in item.children:
        move_item(child, dx, dy)


class DesignWorkspace(Workspace):
    title = "Design"

    def __init__(self, shell) -> None:
        super().__init__(shell)
        self.key: str | None = None
        self.source: dp.Source | None = None
        self.items: list[vc.Item] = []
        self.size = (1600, 1067)
        self._drag: tuple[float, float, float, float] | None = None  # grab x, y, item x, item y
        self._guides: list = []
        self.renderer = LatestOnlyRenderer(self)
        self.renderer.ready.connect(self._on_rendered)
        self.loader_job = LatestOnlyRenderer(self)
        self.loader_job.ready.connect(self._on_loaded)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._build()
        self.session.subscribe(self.on_session)

    # -- layout ------------------------------------------------------------------ #

    def _build(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        left = QVBoxLayout()
        left.setContentsMargins(10, 12, 6, 10)
        left.addWidget(QLabel("Add"))
        for label, kind in (("Text", "text"), ("Rectangle", "rect"), ("Ellipse", "ellipse"), ("Star", "star")):
            left.addWidget(self._button(label, lambda _c=False, k=kind: self._add_item(k)))
        left.addSpacing(10)
        left.addWidget(QLabel("Watermarks"))
        self.signature_name = QLineEdit()
        self.signature_name.setPlaceholderText("Your name")
        left.addWidget(self.signature_name)
        for label, kind in (("Signature", "signature"), ("Seal", "seal"), ("Camera details", "exif_strip")):
            left.addWidget(self._button(label, lambda _c=False, k=kind: self._add_template(k)))
        left.addSpacing(10)
        left.addWidget(QLabel("Items (top drawn last)"))
        self.item_list = QListWidget()
        self.item_list.currentRowChanged.connect(lambda _r: self._load_properties())
        left.addWidget(self.item_list, stretch=1)
        row = QHBoxLayout()
        for text, slot, tip in (("▲", lambda: self._reorder(1), "Draw later (on top)"),
                                ("▼", lambda: self._reorder(-1), "Draw earlier (underneath)"),
                                ("✕", self._delete, "Delete (Del)")):
            button = self._button(text, slot, tip)
            button.setStyleSheet("padding: 6px 4px;")
            row.addWidget(button)
        left.addLayout(row)
        left_host = QWidget()
        left_host.setLayout(left)
        left_host.setFixedWidth(200)
        outer.addWidget(left_host)

        centre = QVBoxLayout()
        centre.setContentsMargins(6, 10, 6, 8)
        bar = QHBoxLayout()
        self.frame_label = QLabel("No photo open")
        self.frame_label.setObjectName("PageTitle")
        bar.addWidget(self.frame_label, stretch=1)
        self.snap = QCheckBox("Snap")
        self.snap.setChecked(True)
        self.grid_size = QComboBox()
        self.grid_size.addItems(["2%", "5%", "10%"])
        self.grid_size.setCurrentText("5%")
        self.show_grid = QCheckBox("Grid")
        self.show_grid.toggled.connect(lambda _c: self.canvas.update())
        for widget in (self.snap, self.grid_size, self.show_grid):
            bar.addWidget(widget)
        bar.addWidget(self._button("Export SVG…", self._on_export_svg, "The design alone, as vector SVG"))
        bar.addWidget(self._button("Export PDF…", self._on_export_pdf, "The photo with the design, as a PDF page"))
        centre.addLayout(bar)
        self.canvas = ImageCanvas()
        self.canvas.pan_with_left_button = False
        self.canvas.set_overlay(self._paint_overlay)
        self.canvas.pressed.connect(self._on_press)
        self.canvas.dragged.connect(self._on_drag)
        self.canvas.released.connect(self._on_release)
        centre.addWidget(self.canvas, stretch=1)
        self.filmstrip = Filmstrip(self.session, self.shell.thumbnail_for)
        centre.addWidget(self.filmstrip)
        outer.addLayout(centre, stretch=1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(320)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        host = QWidget()
        panel = QVBoxLayout(host)
        panel.setContentsMargins(8, 10, 12, 10)
        text = Section("Text")
        self.text_edit = QPlainTextEdit()
        self.text_edit.setFixedHeight(70)
        self.text_edit.setPlaceholderText("Text. Tokens such as {camera}, {lens}, {settings}, {date} are filled in.")
        self.text_edit.textChanged.connect(lambda: self._set("text", self.text_edit.toPlainText()))
        text.add(self.text_edit)
        self.font_combo = QComboBox()
        self.font_combo.addItems(vc.available_fonts())
        self.font_combo.currentTextChanged.connect(lambda f: self._set("font_family", f))
        text.add(self.font_combo)
        self.size_row = self._slider(text, "font_size", "Size", 0.5, 30, 0.1, 1, scale=100.0)
        self.bold = QCheckBox("Bold")
        self.bold.toggled.connect(lambda c: self._set("bold", c))
        self.italic = QCheckBox("Italic")
        self.italic.toggled.connect(lambda c: self._set("italic", c))
        self.kerning = QCheckBox("Kerning")
        self.kerning.toggled.connect(lambda c: self._set("kerning", c))
        text.add_row(self.bold, self.italic, self.kerning)
        self._slider(text, "letter_spacing", "Tracking", 50, 300, 1, 0)
        self._slider(text, "line_spacing", "Leading", 0.5, 3.0, 0.05, 2)
        self.align = QComboBox()
        self.align.addItems(list(vc.ALIGNMENTS))
        self.align.currentTextChanged.connect(lambda a: self._set("align", a))
        self.anchor = QComboBox()
        self.anchor.addItems(list(vc.ANCHORS))
        self.anchor.currentTextChanged.connect(lambda a: self._set("anchor", a))
        text.add_row(QLabel("Align"), self.align)
        text.add_row(QLabel("Anchor"), self.anchor)
        panel.addWidget(text)

        look = Section("Appearance")
        self.fill_button = self._button("Fill colour", lambda: self._pick_colour("fill"))
        self.stroke_button = self._button("Outline colour", lambda: self._pick_colour("stroke"))
        self.no_fill = self._button("No fill", lambda: self._set("fill", None))
        self.no_stroke = self._button("No outline", lambda: self._set("stroke", None))
        look.add_row(self.fill_button, self.no_fill)
        look.add_row(self.stroke_button, self.no_stroke)
        self._slider(look, "stroke_width", "Outline", 0, 3, 0.05, 2, scale=100.0)
        self._slider(look, "opacity", "Opacity", 0, 100, 1, 0)
        self._slider(look, "rotation", "Rotation", -180, 180, 1, 0)
        self._slider(look, "corner_radius", "Corners", 0, 50, 1, 0, scale=100.0)
        self.shadow = QCheckBox("Drop shadow")
        self.shadow.toggled.connect(lambda c: self._set("shadow", vc.Shadow() if c else None))
        look.add(self.shadow)
        panel.addWidget(look)
        panel.addStretch()
        scroll.setWidget(host)
        outer.addWidget(scroll)

    def _button(self, text, slot, tip="") -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("SecondaryButton")
        button.setToolTip(tip)
        button.clicked.connect(slot)
        return button

    def _slider(self, section, attr, label, low, high, step, decimals, scale=1.0) -> SliderRow:
        row = SliderRow(label, low, high, low if low > 0 else 0.0, step, decimals)
        row.changed.connect(lambda value, a=attr, s=scale: self._set(a, value / s))
        row.released.connect(self._commit)
        section.add(row)
        setattr(self, f"_row_{attr}", (row, scale))
        return row

    # -- the open frame ------------------------------------------------------------- #

    def on_enter(self) -> None:
        self.filmstrip.refresh()
        if self.session.current != self.key:
            self.load(self.session.current)
        else:
            self._render_base()
        self.setFocus()

    def on_leave(self) -> None:
        self._commit()
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
        self._commit()
        self.key = key
        if key is None:
            self.items, self.source = [], None
            self.canvas.set_image(None)
            self._refresh_list()
            return
        self.items = vc.items_from(self.session.recipe(key).vectors)
        self.frame_label.setText(Path(key).name)
        self._refresh_list()
        path = Path(key)
        self.loader_job.request(lambda: (key, self._open(path)))

    @staticmethod
    def _open(path: Path):
        source = dp.open_source(path, half_size=True)
        return source.proxy(PREVIEW_SIDE) if source is not None else None

    def _on_loaded(self, result) -> None:
        key, source = result
        if key == self.key:
            self.source = source
            self._render_base()

    def _render_base(self) -> None:
        """The finished frame without the design, which is drawn live on top."""
        if self.source is None or self.key is None:
            return
        recipe = self.session.recipe(self.key).copy()
        recipe.vectors = []
        source = self.source
        self.renderer.request(lambda: dp.render(source, recipe))

    def _on_rendered(self, image) -> None:
        self.size = (image.shape[1], image.shape[0])
        self.canvas.set_image(image)

    # -- items ----------------------------------------------------------------------- #

    def _metadata(self):
        return self.metadata.get(self.key) if self.key else None

    def _commit(self) -> None:
        if self.key is None:
            return
        recipe = self.session.recipe(self.key)
        stored = [vc.item_to_dict(item) for item in self.items]
        if recipe.vectors != stored:
            updated = recipe.copy()
            updated.vectors = stored
            self.session.set_recipe(self.key, updated)

    def _refresh_list(self, select: int | None = None) -> None:
        self.item_list.blockSignals(True)
        self.item_list.clear()
        for item in reversed(self.items):
            label = item.name or (item.text.splitlines()[0][:24] if item.kind == "text" and item.text else item.kind)
            self.item_list.addItem(label)
        self.item_list.blockSignals(False)
        if select is not None and self.items:
            self.item_list.setCurrentRow(len(self.items) - 1 - select)
        self._load_properties()
        self.canvas.update()

    def _index(self) -> int | None:
        row = self.item_list.currentRow()
        return None if row < 0 or not self.items else len(self.items) - 1 - row

    def _current(self) -> vc.Item | None:
        index = self._index()
        return self.items[index] if index is not None and index < len(self.items) else None

    def _add(self, items: list[vc.Item]) -> None:
        if self.key is None:
            QMessageBox.information(self, "Design", "Open a photo first: double-click one in Cull.")
            return
        self.items.extend(items)
        self._refresh_list(len(self.items) - 1)
        self._commit()

    def _add_item(self, kind: str) -> None:
        if kind == "text":
            item = vc.Item(kind="text", text="Your text", x=0.5, y=0.5, anchor="center", align="center",
                           font_size=0.06, fill="#ffffffff", shadow=vc.Shadow())
        else:
            item = vc.Item(kind=kind, x=0.4, y=0.4, w=0.2, h=0.2, fill="#ffffffcc")
        self._add([item])

    def _add_template(self, kind: str) -> None:
        name = self.signature_name.text().strip()
        if kind in ("signature", "seal") and not name:
            QMessageBox.information(self, "Watermark", "Type the name to use first.")
            return
        items = {"signature": lambda: vc.watermark_signature(name),
                 "seal": lambda: vc.watermark_seal(name.upper()),
                 "exif_strip": vc.watermark_exif_strip}[kind]()
        self._add(items)

    def _reorder(self, direction: int) -> None:
        index = self._index()
        target = None if index is None else index + direction
        if target is None or not 0 <= target < len(self.items):
            return
        self.items[index], self.items[target] = self.items[target], self.items[index]
        self._refresh_list(target)
        self._commit()

    def _delete(self) -> None:
        index = self._index()
        if index is None:
            return
        del self.items[index]
        self._refresh_list(max(0, index - 1) if self.items else None)
        self._commit()

    def _set(self, attr: str, value) -> None:
        item = self._current()
        if item is None or getattr(item, attr, None) == value:
            return
        setattr(item, attr, value)
        if attr == "text":
            self._refresh_label()
        self.canvas.update()
        if attr not in ("text",):
            self._commit()

    def _refresh_label(self) -> None:
        item, row = self._current(), self.item_list.currentRow()
        if item is not None and row >= 0 and item.kind == "text":
            self.item_list.item(row).setText(item.name or (item.text.splitlines()[0][:24] if item.text else "text"))

    def _pick_colour(self, attr: str) -> None:
        item = self._current()
        if item is None:
            return
        current = getattr(item, attr) or "#ffffffff"
        start = QColor(current[:7])
        start.setAlpha(int(current[7:9], 16) if len(current) == 9 else 255)
        chosen = QColorDialog.getColor(start, self, "Choose a colour", QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if chosen.isValid():
            self._set(attr, f"#{chosen.red():02x}{chosen.green():02x}{chosen.blue():02x}{chosen.alpha():02x}")

    def _load_properties(self) -> None:
        item = self._current()
        widgets = (self.text_edit, self.font_combo, self.bold, self.italic, self.kerning, self.align,
                   self.anchor, self.shadow)
        for widget in widgets:
            widget.blockSignals(True)
        try:
            if item is None:
                return
            if self.text_edit.toPlainText() != item.text:
                self.text_edit.setPlainText(item.text)
            self.font_combo.setCurrentText(item.font_family)
            self.bold.setChecked(item.bold)
            self.italic.setChecked(item.italic)
            self.kerning.setChecked(item.kerning)
            self.align.setCurrentText(item.align)
            self.anchor.setCurrentText(item.anchor)
            self.shadow.setChecked(item.shadow is not None)
            for attr in ("font_size", "letter_spacing", "line_spacing", "stroke_width", "opacity",
                         "rotation", "corner_radius"):
                row, scale = getattr(self, f"_row_{attr}")
                row.set_value(float(getattr(item, attr)) * scale)
        finally:
            for widget in widgets:
                widget.blockSignals(False)

    # -- dragging, snapping and guides ------------------------------------------------- #

    def _hit(self, x: float, y: float) -> int | None:
        for index in range(len(self.items) - 1, -1, -1):
            bx, by, bw, bh = item_box(self.items[index], self.size, self._metadata())
            pad = 0.01
            if bx - pad <= x <= bx + bw + pad and by - pad <= y <= by + bh + pad:
                return index
        return None

    def _grid(self) -> float:
        return float(self.grid_size.currentText().rstrip("%")) / 100.0

    def _on_press(self, x: float, y: float, button) -> None:
        if button != Qt.MouseButton.LeftButton:
            return
        self.setFocus()
        index = self._hit(x, y)
        if index is None:
            self.item_list.setCurrentRow(-1)
            self._drag = None
            return
        self.item_list.setCurrentRow(len(self.items) - 1 - index)
        item = self.items[index]
        self._drag = (x, y, item.x, item.y)

    def _on_drag(self, x: float, y: float) -> None:
        item = self._current()
        if item is None or self._drag is None:
            return
        grab_x, grab_y, start_x, start_y = self._drag
        target_x, target_y = start_x + (x - grab_x), start_y + (y - grab_y)
        self._guides = []
        if self.snap.isChecked():
            grid = self._grid()
            target_x = vc.snap_to_grid(target_x, grid, grid * 0.25)
            target_y = vc.snap_to_grid(target_y, grid, grid * 0.25)
        move_item(item, target_x - item.x, target_y - item.y)
        if self.snap.isChecked():
            others = [item_box(o, self.size, self._metadata()) for o in self.items if o is not item]
            dx, dy, self._guides = vc.alignment_guides(item_box(item, self.size, self._metadata()), others, 0.008)
            move_item(item, dx, dy)
        self.canvas.update()

    def _on_release(self, _x: float, _y: float) -> None:
        if self._drag is not None:
            self._drag, self._guides = None, []
            self._commit()
            self.canvas.update()

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        item = self._current()
        moves = {Qt.Key.Key_Left: (-NUDGE, 0), Qt.Key.Key_Right: (NUDGE, 0),
                 Qt.Key.Key_Up: (0, -NUDGE), Qt.Key.Key_Down: (0, NUDGE)}
        if item is not None and event.key() in moves:
            step = 5 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
            dx, dy = moves[event.key()]
            move_item(item, dx * step, dy * step)
            self.canvas.update()
            self._commit()
        elif item is not None and event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self._delete()
        else:
            return super().keyPressEvent(event)
        event.accept()

    def _paint_overlay(self, painter, rect: QRectF) -> None:
        if not self.canvas.has_image:
            return
        painter.save()
        painter.translate(rect.topLeft())
        painter.scale(rect.width() / self.size[0], rect.height() / self.size[1])
        vc.paint(painter, self.items, self.size, self._metadata())
        painter.restore()

        def at(x, y):
            return QPointF(rect.x() + x * rect.width(), rect.y() + y * rect.height())

        if self.show_grid.isChecked():
            painter.setPen(QPen(QColor(255, 255, 255, 40), 1))
            step = self._grid()
            value = step
            while value < 1.0 - 1e-6:
                painter.drawLine(at(value, 0), at(value, 1))
                painter.drawLine(at(0, value), at(1, value))
                value += step
        item = self._current()
        if item is not None:
            bx, by, bw, bh = item_box(item, self.size, self._metadata())
            painter.setPen(QPen(QColor("#4fc3f7"), 1, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(at(bx, by), at(bx + bw, by + bh)))
        painter.setPen(QPen(QColor("#ff4081"), 1))
        for axis, value in self._guides:
            if axis == "v":
                painter.drawLine(at(value, 0), at(value, 1))
            else:
                painter.drawLine(at(0, value), at(1, value))

    # -- export --------------------------------------------------------------------- #

    def _full_size(self) -> tuple[int, int]:
        """Pixel size of the delivered frame, for SVG units and PDF pages."""
        if self.key is None:
            return self.size
        source = dp.open_source(self.key)
        if source is None:
            return self.size
        recipe = self.session.recipe(self.key)
        probe = dp.apply_geometry(source.data[::8, ::8], recipe.geometry)
        return probe.shape[1] * 8, probe.shape[0] * 8

    def _on_export_svg(self) -> None:
        if not self.items:
            QMessageBox.information(self, "Design", "There is nothing to export yet.")
            return
        default = str(self.edit_folder / f"{Path(self.key).stem}_design.svg") if self.edit_folder else "design.svg"
        path, _ = QFileDialog.getSaveFileName(self, "Save the design as SVG", default, "SVG (*.svg)")
        if path:
            vc.export_svg(self.items, path, self._full_size(), self._metadata())
            self.status_label.setText(f"Design saved as {Path(path).name}.")

    def _on_export_pdf(self) -> None:
        if self.key is None:
            return
        default = str(self.edit_folder / f"{Path(self.key).stem}_print.pdf") if self.edit_folder else "print.pdf"
        path, _ = QFileDialog.getSaveFileName(self, "Save a print PDF", default, "PDF (*.pdf)")
        if not path:
            return
        self._commit()
        recipe = self.session.recipe(self.key).copy()
        items, metadata = copy.deepcopy(self.items), self._metadata()
        recipe.vectors = []
        key = self.key

        def work():
            source = dp.open_source(key)
            photo = dp.render(source, recipe)
            h, w = photo.shape[:2]
            long_inches = 10.0      # a 10-inch long edge; the photo's own aspect ratio
            size = (long_inches, long_inches * h / w) if w >= h else (long_inches * w / h, long_inches)
            return vc.export_pdf(items, path, size, metadata, dpi=300, background=photo)

        def done(result):
            self.status_label.setText(f"Print PDF saved as {Path(result).name} (10 inches on the long edge).")

        self._run_task(work, done, self.fill_button)
