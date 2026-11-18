"""An image view with zoom, pan, a before/after split and drawing hooks.

Every workspace that shows a photograph large uses this one widget, so
zooming and panning behave the same everywhere and several canvases can be
locked together for side-by-side comparison.

Positions are exchanged as fractions of the image (0..1 across and down),
never widget pixels, so a click means the same place whatever the zoom.

In :data:`CANVAS_MODE_COLOR_SPLASH_BRUSH` the system cursor is replaced by
a circle the size of the brush with a crosshair at its centre (red, with a
minus, while Alt erases), and :meth:`ImageCanvas.update_region` repaints just
the part of the picture a dab changed, so painting keeps up with the mouse.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen
from PyQt6.QtWidgets import QApplication, QWidget

MIN_ZOOM, MAX_ZOOM = 0.02, 16.0
CANVAS_MODE_NORMAL = "normal"
CANVAS_MODE_COLOR_SPLASH_BRUSH = "color_splash_brush"


def to_qimage(rgb: np.ndarray) -> QImage:
    """Wrap an RGB uint8 array as a QImage that owns its own copy."""
    rgb = np.ascontiguousarray(rgb[:, :, :3])
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()


class ImageCanvas(QWidget):
    """Shows one image. ``fit`` mode follows the widget size until the user
    zooms, after which the view is theirs until :meth:`fit` is called."""

    view_changed = pyqtSignal(float, float, float)     # zoom, centre x, centre y (image fractions)
    pressed = pyqtSignal(float, float, object)         # image fractions, mouse button
    dragged = pyqtSignal(float, float)
    released = pyqtSignal(float, float)
    hovered = pyqtSignal(float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(160, 120)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._image: QImage | None = None
        self._before: QImage | None = None
        self._split = 0.5
        self._compare = False
        self._fit = True
        self._zoom = 1.0
        self._centre = QPointF(0.5, 0.5)
        self._panning = False
        self._last = QPointF()
        self._drawing = False
        self._overlay = None            # callable(painter, image_rect)
        self._silent = False
        self.pan_with_left_button = True
        self.caption = ""
        self.mode = CANVAS_MODE_NORMAL
        self.brush_radius = 40.0        # image pixels, for the brush cursor
        self.brush_erase = False        # erase mode chosen (Alt erases too)
        self._mouse: QPointF | None = None

    # -- the brush mode ------------------------------------------------------ #

    def set_mode(self, mode: str) -> None:
        """Normal, or the Color Splash brush (its own cursor)."""
        self.mode = mode
        if mode == CANVAS_MODE_COLOR_SPLASH_BRUSH:
            self.setCursor(Qt.CursorShape.BlankCursor)
        else:
            self.unsetCursor()
        self.update()

    def set_brush(self, radius: float | None = None, erase: bool | None = None) -> None:
        if radius is not None:
            self.brush_radius = float(radius)
        if erase is not None:
            self.brush_erase = bool(erase)
        self.update()

    def erasing(self) -> bool:
        return self.brush_erase or bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.AltModifier)

    def update_region(self, patch: np.ndarray, x: int, y: int) -> None:
        """Put an RGB patch into the picture at image pixel ``(x, y)`` and
        repaint only that part of the screen."""
        if self._image is None or patch.size == 0:
            return
        painter = QPainter(self._image)
        painter.drawImage(int(x), int(y), to_qimage(patch))
        painter.end()
        rect = self.image_rect()
        zoom = rect.width() / max(self._image.width(), 1)
        dirty = QRectF(rect.x() + x * zoom, rect.y() + y * zoom, patch.shape[1] * zoom, patch.shape[0] * zoom)
        self.update(dirty.adjusted(-2, -2, 2, 2).toAlignedRect())

    def _paint_brush_cursor(self, painter: QPainter) -> None:
        if self._mouse is None or self._image is None:
            return
        rect = self.image_rect()
        radius = self.brush_radius * rect.width() / max(self._image.width(), 1)
        erase = self.erasing()
        colour = QColor("#ff5a5f") if erase else QColor("#ffffff")
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(QColor(0, 0, 0, 160), 3))
        painter.drawEllipse(self._mouse, radius, radius)
        painter.setPen(QPen(colour, 1.5))
        painter.drawEllipse(self._mouse, radius, radius)
        cross = 6.0
        painter.drawLine(self._mouse + QPointF(-cross, 0), self._mouse + QPointF(cross, 0))
        if not erase:
            painter.drawLine(self._mouse + QPointF(0, -cross), self._mouse + QPointF(0, cross))
        else:
            painter.drawText(self._mouse + QPointF(radius * 0.7 + 4, -radius * 0.7 - 4), "\u2212")
        painter.restore()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._mouse = None
        if self.mode == CANVAS_MODE_COLOR_SPLASH_BRUSH:
            self.update()
        super().leaveEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Alt and self.mode == CANVAS_MODE_COLOR_SPLASH_BRUSH:
            self.update()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Alt and self.mode == CANVAS_MODE_COLOR_SPLASH_BRUSH:
            self.update()
        super().keyReleaseEvent(event)

    # -- content ----------------------------------------------------------- #

    def set_image(self, rgb: np.ndarray | None) -> None:
        had = self._image.size() if self._image is not None else None
        self._image = to_qimage(rgb) if rgb is not None else None
        if self._image is not None and had != self._image.size() and self._fit:
            self._centre = QPointF(0.5, 0.5)
        self.update()

    def set_before(self, rgb: np.ndarray | None) -> None:
        """The unedited version, shown left of the split when comparing."""
        self._before = to_qimage(rgb) if rgb is not None else None
        self.update()

    def set_compare(self, on: bool) -> None:
        self._compare = bool(on)
        self.update()

    @property
    def comparing(self) -> bool:
        return self._compare

    def set_overlay(self, painter_fn) -> None:
        """``painter_fn(painter, image_rect)`` draws over the image."""
        self._overlay = painter_fn
        self.update()

    @property
    def has_image(self) -> bool:
        return self._image is not None

    # -- geometry ---------------------------------------------------------- #

    def _fit_zoom(self) -> float:
        if self._image is None or self._image.width() == 0:
            return 1.0
        return min(self.width() / self._image.width(), self.height() / self._image.height())

    @property
    def zoom(self) -> float:
        return self._fit_zoom() if self._fit else self._zoom

    def image_rect(self) -> QRectF:
        """Where the image is drawn, in widget coordinates."""
        if self._image is None:
            return QRectF()
        zoom = self.zoom
        w, h = self._image.width() * zoom, self._image.height() * zoom
        centre = QPointF(0.5, 0.5) if self._fit else self._centre
        return QRectF(self.width() / 2.0 - centre.x() * w, self.height() / 2.0 - centre.y() * h, w, h)

    def to_image(self, point) -> tuple[float, float]:
        rect = self.image_rect()
        if rect.width() <= 0:
            return 0.0, 0.0
        return (point.x() - rect.x()) / rect.width(), (point.y() - rect.y()) / rect.height()

    def to_widget(self, x: float, y: float) -> QPointF:
        rect = self.image_rect()
        return QPointF(rect.x() + x * rect.width(), rect.y() + y * rect.height())

    # -- view control ------------------------------------------------------- #

    def fit(self) -> None:
        self._fit = True
        self._centre = QPointF(0.5, 0.5)
        self._emit_view()
        self.update()

    def set_zoom(self, zoom: float, anchor: tuple[float, float] | None = None) -> None:
        """Zoom to ``zoom`` (1.0 is one image pixel per screen pixel)."""
        self._zoom = float(min(MAX_ZOOM, max(MIN_ZOOM, zoom)))
        self._fit = False
        if anchor is not None:
            self._centre = QPointF(*anchor)
        self._clamp()
        self._emit_view()
        self.update()

    def actual_size(self, anchor: tuple[float, float] | None = None) -> None:
        self.set_zoom(1.0, anchor)

    def set_view(self, zoom: float, cx: float, cy: float) -> None:
        """Adopt another canvas's view without announcing it back."""
        self._silent = True
        self._fit = False
        self._zoom = float(min(MAX_ZOOM, max(MIN_ZOOM, zoom)))
        self._centre = QPointF(cx, cy)
        self._clamp()
        self._silent = False
        self.update()

    def view(self) -> tuple[float, float, float]:
        centre = QPointF(0.5, 0.5) if self._fit else self._centre
        return self.zoom, centre.x(), centre.y()

    def _clamp(self) -> None:
        self._centre = QPointF(min(1.0, max(0.0, self._centre.x())), min(1.0, max(0.0, self._centre.y())))

    def _emit_view(self) -> None:
        if not self._silent:
            self.view_changed.emit(*self.view())

    # -- painting ------------------------------------------------------------ #

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0d0d0d"))
        if self._image is None:
            painter.setPen(QColor("#6a6a6e"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.caption or "No photo open")
            return
        rect = self.image_rect()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom < 1.0)
        painter.drawImage(rect, self._image)

        if self._compare and self._before is not None:
            cut = rect.x() + self._split * rect.width()
            painter.save()
            painter.setClipRect(QRectF(rect.x(), rect.y(), cut - rect.x(), rect.height()))
            painter.drawImage(rect, self._before)
            painter.restore()
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.drawLine(QPointF(cut, rect.top()), QPointF(cut, rect.bottom()))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(QPointF(rect.x() + 10, rect.y() + 20), "Before")
            painter.drawText(QPointF(cut + 10, rect.y() + 20), "After")

        if self._overlay is not None:
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            self._overlay(painter, rect)
            painter.restore()

        if self.caption:
            painter.setPen(QColor("#d0d0d4"))
            painter.drawText(QPointF(10, self.height() - 10), self.caption)

        if self.mode == CANVAS_MODE_COLOR_SPLASH_BRUSH:
            self._paint_brush_cursor(painter)

    # -- input ---------------------------------------------------------------- #

    def wheelEvent(self, event) -> None:  # noqa: N802
        if self._image is None:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        under = self.to_image(event.position())
        new_zoom = min(MAX_ZOOM, max(MIN_ZOOM, self.zoom * factor))
        # Keep the point under the cursor where it is.
        w, h = self._image.width() * new_zoom, self._image.height() * new_zoom
        cx = under[0] - (event.position().x() - self.width() / 2.0) / max(w, 1.0)
        cy = under[1] - (event.position().y() - self.height() / 2.0) / max(h, 1.0)
        self.set_zoom(new_zoom, (cx, cy))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._image is None:
            return
        near_split = False
        if self._compare:
            rect = self.image_rect()
            near_split = abs(event.position().x() - (rect.x() + self._split * rect.width())) < 8
        middle = event.button() == Qt.MouseButton.MiddleButton
        left_pan = event.button() == Qt.MouseButton.LeftButton and self.pan_with_left_button
        if near_split:
            self._drawing = "split"
        elif middle or left_pan:
            self._panning = True
            self._last = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            self._drawing = True
            self.pressed.emit(*self.to_image(event.position()), event.button())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._image is None:
            return
        if self.mode == CANVAS_MODE_COLOR_SPLASH_BRUSH:
            old, self._mouse = self._mouse, event.position()
            reach = self.brush_radius * self.zoom + 12
            for point in (old, self._mouse):
                if point is not None:
                    self.update(QRectF(point.x() - reach, point.y() - reach, 2 * reach, 2 * reach).toAlignedRect())
        if self._drawing == "split":
            rect = self.image_rect()
            self._split = min(1.0, max(0.0, (event.position().x() - rect.x()) / max(rect.width(), 1.0)))
            self.update()
        elif self._panning:
            delta = event.position() - self._last
            self._last = event.position()
            rect = self.image_rect()
            if self._fit:
                self._zoom, self._fit = self._fit_zoom(), False
            self._centre = QPointF(self._centre.x() - delta.x() / max(rect.width(), 1.0),
                                   self._centre.y() - delta.y() / max(rect.height(), 1.0))
            self._clamp()
            self._emit_view()
            self.update()
        elif self._drawing:
            self.dragged.emit(*self.to_image(event.position()))
        else:
            self.hovered.emit(*self.to_image(event.position()))

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drawing is True:
            self.released.emit(*self.to_image(event.position()))
        self._drawing = False
        if self._panning:
            self._panning = False
            self.unsetCursor()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if self._image is None:
            return
        if self._fit:
            self.actual_size(self.to_image(event.position()))
        else:
            self.fit()


def link_views(canvases: list[ImageCanvas]) -> None:
    """Lock several canvases together: zoom or pan one and the rest follow."""
    for source in canvases:
        for target in canvases:
            if target is not source:
                source.view_changed.connect(target.set_view)
