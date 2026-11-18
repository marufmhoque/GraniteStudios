"""An edge-aware brush: paint inside an object and the stroke stops at its edges.

Used by the Color Splash Brush in Retouch, where painting reveals colour on
a black-and-white photo. Each dab works on the circle under the brush:

1. The colour under the brush centre is the reference, in CIE-Lab, where
   distance is close to how different two colours look (Delta E, CIE76).
2. Edges are where the colour changes fast: the Lab gradient (Sobel, in
   Delta E per pixel). Pixels on a strong edge are walls.
3. A candidate is a pixel in the circle, close in colour to the reference
   and not a wall. Only the candidates *connected to the centre* are
   painted, so the stroke cannot jump a wall onto a similar colour beyond
   it. The painted region's border is softened by about a pixel.

Edge sensitivity (0.1 .. 1.0) sets how strict both tests are: high stops at
faint edges and accepts only close colours; low paints through texture and
stops only at strong edges. With snapping off the dab is a plain soft
circle. Erasing uses the same region and sets it back to 0.

Everything is NumPy and OpenCV on the region under the brush, so a dab
costs a few milliseconds; the whole image's Lab and gradient are worked out
once per image and reused for every dab.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MIN_RADIUS, MAX_RADIUS = 2, 600
SENSITIVITY_LEVELS = {"Low": 0.3, "Medium": 0.6, "High": 0.9}


def tolerances(edge_sensitivity: float) -> tuple[float, float]:
    """``(colour tolerance in Delta E, edge threshold in Delta E per pixel)``
    for a sensitivity between 0.1 (loose) and 1.0 (strict)."""
    s = float(min(1.0, max(0.1, edge_sensitivity)))
    colour = 15.0 + (1.0 - s) * 45.0          # 15 (strict) .. 55 (loose)
    edge = 7.0 + (1.0 - s) * 23.0             # 7 .. 28
    return colour, edge


def to_lab(rgb: np.ndarray) -> np.ndarray:
    """Float CIE-Lab (L 0..100) of an RGB image, uint8 or float 0..1."""
    image = rgb.astype(np.float32)
    if rgb.dtype == np.uint8:
        image /= 255.0
    return cv2.cvtColor(np.clip(image, 0.0, 1.0), cv2.COLOR_RGB2Lab)


def lab_gradient(lab: np.ndarray) -> np.ndarray:
    """How fast the colour changes at each pixel, in Delta E per pixel."""
    total = np.zeros(lab.shape[:2], np.float32)
    for channel in range(3):
        gx = cv2.Sobel(lab[..., channel], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(lab[..., channel], cv2.CV_32F, 0, 1, ksize=3)
        total += gx * gx + gy * gy
    return np.sqrt(total) / 4.0              # a 3x3 Sobel answers 4 to a one-unit step


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def empty(self) -> bool:
        return self.w <= 0 or self.h <= 0

    def united(self, other: "Rect") -> "Rect":
        if self.empty:
            return other
        if other.empty:
            return self
        x0, y0 = min(self.x, other.x), min(self.y, other.y)
        x1, y1 = max(self.x + self.w, other.x + other.w), max(self.y + self.h, other.y + other.h)
        return Rect(x0, y0, x1 - x0, y1 - y0)


class SmartBrushEngine:
    """Paints dabs into an 8-bit mask, snapping to the edges of an image."""

    def __init__(self) -> None:
        self._for: tuple | None = None
        self._lab: np.ndarray | None = None
        self._gradient: np.ndarray | None = None

    def prepare(self, image_rgb: np.ndarray) -> None:
        """Work out the image's Lab and edges (once per image)."""
        key = (id(image_rgb), image_rgb.shape, image_rgb.__array_interface__["data"][0])
        if self._for != key:
            self._lab = to_lab(image_rgb)
            self._gradient = lab_gradient(self._lab)
            self._for = key

    def region(self, image_rgb: np.ndarray, center_point, radius: int, edge_sensitivity: float = 0.6,
               snapping: bool = True) -> tuple[np.ndarray, Rect]:
        """The dab: a float 0..1 weight over the circle's bounding box, and
        that box (clipped to the image)."""
        h, w = image_rgb.shape[:2]
        cx, cy = int(round(center_point[0])), int(round(center_point[1]))
        r = int(min(MAX_RADIUS, max(MIN_RADIUS, radius)))
        x0, y0 = max(0, cx - r), max(0, cy - r)
        x1, y1 = min(w, cx + r + 1), min(h, cy + r + 1)
        rect = Rect(x0, y0, x1 - x0, y1 - y0)
        if rect.empty or not (0 <= cx < w and 0 <= cy < h):
            return np.zeros((0, 0), np.float32), Rect(0, 0, 0, 0)
        ys, xs = np.mgrid[y0:y1, x0:x1]
        distance = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).astype(np.float32)
        circle = distance <= r
        if not snapping:
            soft = np.clip((r - distance) / max(1.0, r * 0.15), 0.0, 1.0)
            return soft.astype(np.float32), rect

        self.prepare(image_rgb)
        lab = self._lab[y0:y1, x0:x1]
        gradient = self._gradient[y0:y1, x0:x1]
        lx, ly = cx - x0, cy - y0
        # The median of the 3x3 under the centre: robust to noise, and on an
        # edge it is one side's colour, never a mix that matches neither.
        reference = np.median(lab[max(0, ly - 1):ly + 2, max(0, lx - 1):lx + 2].reshape(-1, 3), axis=0)
        delta_e = np.sqrt(((lab - reference) ** 2).sum(axis=-1))
        colour_tol, edge_tol = tolerances(edge_sensitivity)
        candidate = circle & (delta_e <= colour_tol) & (gradient <= edge_tol)
        if not candidate.any():
            return np.zeros(circle.shape, np.float32), rect
        count, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=4)
        seed = labels[ly, lx]
        if seed == 0:                       # the centre is on an edge: start from the nearest candidate
            near = np.argwhere(candidate)
            iy, ix = near[np.argmin(((near - (ly, lx)) ** 2).sum(axis=1))]
            seed = labels[iy, ix]
        inside = (labels == seed).astype(np.uint8)
        # The pixels on the edge itself are walls, but the ones that are the
        # object's colour belong to it: grow two pixels into them, never
        # into the other side's colour, so no grey seam is left at the edge.
        grown = cv2.dilate(inside, np.ones((3, 3), np.uint8), iterations=2)
        inside = (inside > 0) | ((grown > 0) & (delta_e <= colour_tol * 0.5))
        return (inside & circle).astype(np.float32), rect

    def paint_stroke(self, current_image_rgb: np.ndarray, mask: np.ndarray, center_point, radius: int,
                     edge_sensitivity: float = 0.6, is_erase: bool = False,
                     snapping: bool = True) -> tuple[np.ndarray, Rect]:
        """One dab into ``mask`` (uint8, the image's height and width), in
        place: 255 reveals colour, 0 is black and white. Returns the mask and
        the rectangle that changed, for a fast repaint."""
        if mask.shape[:2] != current_image_rgb.shape[:2]:
            raise ValueError("the mask must be the size of the image")
        weight, rect = self.region(current_image_rgb, center_point, radius, edge_sensitivity, snapping)
        if rect.empty:
            return mask, rect
        patch = mask[rect.y:rect.y + rect.h, rect.x:rect.x + rect.w]
        amount = np.round(weight * 255.0).astype(np.uint8)
        if is_erase:
            patch[...] = np.minimum(patch, 255 - amount)
        else:
            patch[...] = np.maximum(patch, amount)
        return mask, rect

    def paint_line(self, current_image_rgb: np.ndarray, mask: np.ndarray, start, end, radius: int,
                   edge_sensitivity: float = 0.6, is_erase: bool = False,
                   snapping: bool = True) -> tuple[np.ndarray, Rect]:
        """Dabs along a drag from ``start`` to ``end``, a third of the radius
        apart, so a fast drag leaves no gaps."""
        (sx, sy), (ex, ey) = start, end
        steps = max(1, int(np.hypot(ex - sx, ey - sy) / max(1.0, radius / 3.0)))
        changed = Rect(0, 0, 0, 0)
        for i in range(0, steps + 1):
            t = i / steps
            _, rect = self.paint_stroke(current_image_rgb, mask, (sx + (ex - sx) * t, sy + (ey - sy) * t),
                                        radius, edge_sensitivity, is_erase, snapping)
            changed = changed.united(rect)
        return mask, changed


def reveal_preview(colour: np.ndarray, mask: np.ndarray, grey: np.ndarray | None = None) -> np.ndarray:
    """The colour image where ``mask`` reveals it, black and white elsewhere
    (uint8 RGB in, uint8 RGB out)."""
    if grey is None:
        grey = desaturate(colour)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    return np.round(grey * (1.0 - alpha) + colour * alpha).astype(np.uint8)


def desaturate(rgb: np.ndarray) -> np.ndarray:
    """Black and white with the photo's own brightness (Rec. 709 weights)."""
    luma = rgb.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return np.repeat(np.round(luma)[..., None], 3, axis=-1).astype(np.uint8)
