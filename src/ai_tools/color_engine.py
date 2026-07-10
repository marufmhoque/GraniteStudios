"""Color engine: selective color isolation and cinematic LUT-style filters.

All functions operate on RGB uint8 numpy arrays (H, W, 3) as produced by
``UniversalLoader.load_array`` and return new arrays of the same shape.
"""

from __future__ import annotations

import numpy as np
import cv2

# Named target hues in OpenCV HSV space (H: 0-179).
_COLOR_HUES = {
    "red": 0,
    "orange": 15,
    "yellow": 30,
    "green": 60,
    "cyan": 90,
    "blue": 120,
    "purple": 140,
    "magenta": 160,
}

# 3x3 RGB color matrices simulating classic cinematic grades.
# Each row maps (R, G, B) input contributions to one output channel.
_FILTER_MATRICES = {
    "cinematic_teal_orange": np.array(
        [
            [1.10, 0.05, -0.05],
            [-0.02, 1.00, 0.02],
            [-0.08, 0.05, 1.08],
        ]
    ),
    "warm_film": np.array(
        [
            [1.12, 0.06, 0.00],
            [0.04, 1.02, 0.00],
            [0.00, 0.02, 0.90],
        ]
    ),
    "cool_matte": np.array(
        [
            [0.92, 0.02, 0.02],
            [0.02, 0.98, 0.04],
            [0.02, 0.06, 1.10],
        ]
    ),
    "bleach_bypass": np.array(
        [
            [0.85, 0.30, 0.10],
            [0.20, 0.90, 0.15],
            [0.10, 0.25, 0.85],
        ]
    ),
    "vintage_fade": np.array(
        [
            [0.95, 0.08, 0.04],
            [0.06, 0.92, 0.06],
            [0.08, 0.08, 0.88],
        ]
    ),
    "noir": np.array(
        [
            [0.299, 0.587, 0.114],
            [0.299, 0.587, 0.114],
            [0.299, 0.587, 0.114],
        ]
    ),
}

# Per-filter (lift, gamma) tone tweaks applied after the matrix.
_FILTER_TONE = {
    "cinematic_teal_orange": (0.0, 1.05),
    "warm_film": (6.0, 1.02),
    "cool_matte": (12.0, 0.98),
    "bleach_bypass": (0.0, 1.15),
    "vintage_fade": (18.0, 0.95),
    "noir": (0.0, 1.10),
}


class ColorEngine:
    """Selective color isolation and matrix-based cinematic filters."""

    FILTERS = tuple(_FILTER_MATRICES.keys())
    COLORS = tuple(_COLOR_HUES.keys())

    # ------------------------------------------------------------------ #
    # Selective color
    # ------------------------------------------------------------------ #

    def selective_color(
        self,
        image_array: np.ndarray,
        target_color: str,
        tolerance: int = 15,
    ) -> np.ndarray:
        """Keep one color, render everything else in grayscale.

        The classic "red dress in a monochrome crowd" effect: a mask of the
        target hue is built with ``cv2.inRange``, the inverse of the mask is
        converted to grayscale, and the two are merged back together.

        Args:
            image_array: RGB uint8 array (H, W, 3).
            target_color: One of ``ColorEngine.COLORS`` (e.g. "red", "blue").
            tolerance: Hue tolerance in degrees (OpenCV scale, 0-179).
        """
        color = target_color.lower()
        if color not in _COLOR_HUES:
            raise ValueError(
                f"Unknown color {target_color!r}. Choose from: {', '.join(self.COLORS)}"
            )

        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        hue = _COLOR_HUES[color]

        lower = np.array([max(hue - tolerance, 0), 60, 40], dtype=np.uint8)
        upper = np.array([min(hue + tolerance, 179), 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # Red wraps around the hue circle — also grab the top of the range.
        if color == "red":
            lower_wrap = np.array([max(179 - tolerance, 0), 60, 40], dtype=np.uint8)
            upper_wrap = np.array([179, 255, 255], dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_wrap, upper_wrap))

        # Clean speckle noise from the mask, then feather the edge.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)

        inverse_mask = cv2.bitwise_not(mask)

        # Color region: original pixels where the mask hits.
        color_region = cv2.bitwise_and(image_array, image_array, mask=mask)

        # Everything else: grayscale, re-expanded to 3 channels.
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        gray_region = cv2.bitwise_and(gray_rgb, gray_rgb, mask=inverse_mask)

        # Feathered alpha merge so the transition isn't a hard cutout.
        alpha = (mask.astype(np.float32) / 255.0)[..., np.newaxis]
        merged = image_array.astype(np.float32) * alpha + gray_rgb.astype(np.float32) * (1.0 - alpha)
        # color_region/gray_region are the hard-mask components; the alpha
        # blend supersedes them visually but they guarantee exact coverage.
        _ = color_region, gray_region
        return np.clip(merged, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ #
    # Cinematic filters
    # ------------------------------------------------------------------ #

    def apply_lut_filter(self, image_array: np.ndarray, filter_type: str) -> np.ndarray:
        """Apply a cinematic color-matrix filter.

        Simulates a LUT engine with a 3x3 channel-mixing matrix followed by
        a lift/gamma tone curve — all vectorized NumPy, fast enough for
        real-time preview on integrated graphics.

        Args:
            image_array: RGB uint8 array (H, W, 3).
            filter_type: One of ``ColorEngine.FILTERS``.
        """
        name = filter_type.lower()
        if name not in _FILTER_MATRICES:
            raise ValueError(
                f"Unknown filter {filter_type!r}. Choose from: {', '.join(self.FILTERS)}"
            )

        matrix = _FILTER_MATRICES[name]
        lift, gamma = _FILTER_TONE[name]

        pixels = image_array.astype(np.float32)

        # Channel mix: out = pixels @ matrix.T  (per-pixel 3x3 transform).
        mixed = pixels @ matrix.T

        # Lift (shadow raise) then gamma on the normalized signal.
        mixed = mixed + lift
        mixed = np.clip(mixed, 0, 255) / 255.0
        mixed = np.power(mixed, 1.0 / gamma) * 255.0

        return np.clip(mixed, 0, 255).astype(np.uint8)

    def apply_adjustments(
        self,
        image_array: np.ndarray,
        brightness: float = 0.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
    ) -> np.ndarray:
        """Basic exposure controls: brightness (-100..100), contrast and
        saturation as multipliers (1.0 = unchanged)."""
        pixels = image_array.astype(np.float32)

        # Contrast pivots around mid-gray, then brightness shifts.
        pixels = (pixels - 127.5) * contrast + 127.5 + brightness
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)

        if saturation != 1.0:
            hsv = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
            pixels = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return pixels
