"""Small controls the editing panels are built from."""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget


class SliderRow(QWidget):
    """A labelled slider with its value shown. Double-click the name to reset."""

    changed = pyqtSignal(float)
    released = pyqtSignal()

    def __init__(self, label: str, low: float, high: float, default: float = 0.0,
                 step: float = 1.0, decimals: int = 0, parent=None, label_width: int = 96) -> None:
        super().__init__(parent)
        self._low, self._step, self._default, self._decimals = low, step, default, decimals
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.name = QLabel(label)
        self.name.setFixedWidth(label_width)
        self.name.setToolTip("Double-click to reset")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, int(round((high - low) / step)))
        self.readout = QLabel()
        self.readout.setFixedWidth(42)
        self.readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.name)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.readout)
        self.slider.valueChanged.connect(self._moved)
        self.slider.sliderReleased.connect(self.released.emit)
        self.set_value(default)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.position().x() <= self.name.width() + 8:
            self.set_value(self._default, announce=True)
            self.released.emit()

    def value(self) -> float:
        return round(self._low + self.slider.value() * self._step, 6)

    def set_value(self, value: float, announce: bool = False) -> None:
        self.slider.blockSignals(not announce)
        self.slider.setValue(int(round((value - self._low) / self._step)))
        self.slider.blockSignals(False)
        self._show()

    def _show(self) -> None:
        value = self.value()
        text = f"{value:+.{self._decimals}f}" if self._low < 0 else f"{value:.{self._decimals}f}"
        self.readout.setText(text)

    def _moved(self, _raw: int) -> None:
        self._show()
        self.changed.emit(self.value())


class Section(QWidget):
    """A titled group of controls that folds away when its header is clicked."""

    def __init__(self, title: str, expanded: bool = True, parent=None) -> None:
        super().__init__(parent)
        self._title = title
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        self.header = QPushButton()
        self.header.setObjectName("SectionHeader")
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.toggled.connect(self._toggle)
        self.body = QWidget()
        self.content = QVBoxLayout(self.body)
        self.content.setContentsMargins(6, 2, 2, 8)
        self.content.setSpacing(5)
        outer.addWidget(self.header)
        outer.addWidget(self.body)
        self._toggle(expanded)

    def _toggle(self, open_: bool) -> None:
        self.header.setText(("▾  " if open_ else "▸  ") + self._title)
        self.body.setVisible(open_)

    def add(self, widget: QWidget) -> QWidget:
        self.content.addWidget(widget)
        return widget

    def add_row(self, *widgets) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        for widget in widgets:
            row.addWidget(widget)
        self.content.addLayout(row)
        return row


class CurveEditor(QWidget):
    """A tone curve on the unit square.

    Click to add a point, drag to move it, right-click a point to remove
    it. The two end points can be moved up and down but not removed.
    """

    changed = pyqtSignal(list)
    released = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(190)
        self.setMinimumWidth(190)
        self._points: list[list[float]] = [[0.0, 0.0], [1.0, 1.0]]
        self._active: int | None = None
        self._histogram: np.ndarray | None = None

    def points(self) -> list:
        return [list(p) for p in self._points]

    def set_points(self, points: list) -> None:
        self._points = [list(p) for p in points] if len(points) >= 2 else [[0.0, 0.0], [1.0, 1.0]]
        self.update()

    def set_histogram(self, luminance_counts: np.ndarray | None) -> None:
        self._histogram = luminance_counts
        self.update()

    def _area(self) -> QRectF:
        return QRectF(8, 8, self.width() - 16, self.height() - 16)

    def _to_widget(self, x: float, y: float) -> QPointF:
        area = self._area()
        return QPointF(area.x() + x * area.width(), area.bottom() - y * area.height())

    def _from_widget(self, point) -> tuple[float, float]:
        area = self._area()
        x = (point.x() - area.x()) / max(area.width(), 1.0)
        y = (area.bottom() - point.y()) / max(area.height(), 1.0)
        return min(1.0, max(0.0, x)), min(1.0, max(0.0, y))

    def _nearest(self, point) -> int | None:
        best, best_distance = None, 12.0
        for index, (x, y) in enumerate(self._points):
            spot = self._to_widget(x, y)
            distance = abs(spot.x() - point.x()) + abs(spot.y() - point.y())
            if distance < best_distance:
                best, best_distance = index, distance
        return best

    def mousePressEvent(self, event) -> None:  # noqa: N802
        index = self._nearest(event.position())
        if event.button() == Qt.MouseButton.RightButton:
            if index is not None and 0 < index < len(self._points) - 1:
                del self._points[index]
                self._announce(final=True)
            return
        if index is None:
            x, y = self._from_widget(event.position())
            self._points.append([x, y])
            self._points.sort(key=lambda p: p[0])
            index = next(i for i, p in enumerate(self._points) if p[0] == x and p[1] == y)
        self._active = index
        self._announce()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._active is None:
            return
        x, y = self._from_widget(event.position())
        index = self._active
        if index == 0:
            x = 0.0
        elif index == len(self._points) - 1:
            x = 1.0
        else:       # a point may not pass its neighbours
            x = min(self._points[index + 1][0] - 0.01, max(self._points[index - 1][0] + 0.01, x))
        self._points[index] = [x, y]
        self._announce()

    def mouseReleaseEvent(self, _event) -> None:  # noqa: N802
        if self._active is not None:
            self._active = None
            self.released.emit()

    def _announce(self, final: bool = False) -> None:
        self.update()
        self.changed.emit(self.points())
        if final:
            self.released.emit()

    def paintEvent(self, _event) -> None:  # noqa: N802
        from src.engine.develop_pipeline import curve_lut

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        area = self._area()
        painter.fillRect(self.rect(), QColor("#161616"))
        painter.setPen(QPen(QColor("#2c2c2c"), 1))
        for step in range(5):
            x = area.x() + area.width() * step / 4.0
            y = area.y() + area.height() * step / 4.0
            painter.drawLine(QPointF(x, area.top()), QPointF(x, area.bottom()))
            painter.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
        if self._histogram is not None and self._histogram.max() > 0:
            counts = self._histogram / self._histogram.max()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255, 34))
            width = area.width() / len(counts)
            for index, count in enumerate(counts):
                height = float(count) * area.height()
                painter.drawRect(QRectF(area.x() + index * width, area.bottom() - height, width + 0.5, height))
        painter.setPen(QPen(QColor("#3a3a3a"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(self._to_widget(0, 0), self._to_widget(1, 1))

        lut = curve_lut(self._points, 128)
        path = QPainterPath()
        for index, value in enumerate(lut):
            spot = self._to_widget(index / (len(lut) - 1), float(value))
            path.lineTo(spot) if index else path.moveTo(spot)
        painter.setPen(QPen(QColor("#f0f0f0"), 1.6))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        painter.setBrush(QColor("#E31B23"))
        painter.setPen(QPen(QColor("#ffffff"), 1))
        for x, y in self._points:
            painter.drawEllipse(self._to_widget(x, y), 4, 4)


class Histogram(QWidget):
    """An RGB histogram with clipping indicators in the top corners."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(110)
        self._counts: np.ndarray | None = None
        self.shadows_clipped = 0.0
        self.highlights_clipped = 0.0

    def set_image(self, rgb: np.ndarray | None) -> None:
        if rgb is None:
            self._counts = None
            self.update()
            return
        small = rgb if max(rgb.shape[:2]) <= 480 else cv2.resize(
            rgb, None, fx=480.0 / max(rgb.shape[:2]), fy=480.0 / max(rgb.shape[:2]), interpolation=cv2.INTER_AREA)
        self._counts = np.stack([np.bincount(small[:, :, c].reshape(-1), minlength=256) for c in range(3)])
        self.shadows_clipped, self.highlights_clipped = clipping_fractions(small)
        self.update()

    def luminance_counts(self) -> np.ndarray | None:
        if self._counts is None:
            return None
        grey = self._counts.sum(axis=0).astype(np.float64)
        return grey.reshape(64, 4).sum(axis=1)

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#161616"))
        if self._counts is None:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        # The end bins hold clipped pixels and would flatten everything else.
        peak = float(self._counts[:, 2:254].max()) or 1.0
        width, height = self.width(), self.height() - 4
        for channel, colour in enumerate(("#c0392b", "#27ae60", "#2980b9")):
            path = QPainterPath()
            path.moveTo(0, self.height())
            for value in range(256):
                level = min(1.0, float(self._counts[channel, value]) / peak)
                path.lineTo(value / 255.0 * width, self.height() - level * height)
            path.lineTo(width, self.height())
            fill = QColor(colour)
            fill.setAlpha(200)
            painter.fillPath(path, fill)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        for clipped, x, colour in ((self.shadows_clipped, 4, "#3498db"),
                                   (self.highlights_clipped, width - 16, "#e74c3c")):
            painter.setBrush(QColor(colour) if clipped > 0.0005 else QColor("#3a3a3a"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(QRectF(x, 4, 12, 12))


def clipping_fractions(rgb: np.ndarray) -> tuple[float, float]:
    """Fractions of pixels crushed to black and blown to white."""
    total = float(rgb.shape[0] * rgb.shape[1]) or 1.0
    return float((rgb.max(axis=2) <= 1).sum() / total), float((rgb.max(axis=2) >= 254).sum() / total)


def clipping_overlay(rgb: np.ndarray) -> np.ndarray:
    """The image with lost detail painted over: blue for crushed shadows,
    red for blown highlights. A pixel counts as blown if any channel is."""
    out = rgb.copy()
    out[rgb.max(axis=2) <= 1] = (40, 110, 255)
    out[rgb.max(axis=2) >= 254] = (255, 40, 40)
    return out
