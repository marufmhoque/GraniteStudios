"""Portrait retouching and inpainting.

Every function here takes pixels and returns new pixels; none touches a
file. In the app they run as layers over the developed image, so each can
be turned down, masked or removed later.

Skin smoothing by frequency separation
--------------------------------------
Blurring skin makes it look like plastic because it removes two different
things at once: blotchiness, which people want gone, and pores, which they
do not. Splitting the image by the size of its detail lets them be treated
separately:

* **fine** -- pores and fine hair, a few pixels across;
* **middle** -- blemishes, blotches and uneven patches;
* **coarse** -- the shape of the face and its lighting.

Smoothing here turns down the *middle* band only. The fine band is passed
through untouched, which is the whole point and is what the tests check.
Band sizes follow the distance between the eyes, so the result does not
depend on how large the face is in the frame.

Inpainting
----------
Two engines. Small spots use OpenCV's fast-marching fill, which is instant
and good at continuing smooth skin or sky across a blemish. Larger holes
use LaMa, a neural network that can invent plausible texture; it is slow on
a processor (about 13 s per tile on the reference laptop) and works at a
fixed 512 pixels, so a hole is filled inside a window around it and only
the missing pixels are taken from the result.

What is unmeasured: how convincing LaMa's fill is on real photographs. The
tests establish that it runs, fills only what it is asked to, and produces
colours consistent with the surroundings. Whether a filled corner passes
inspection in a print is a judgement nobody has made here yet.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.ai_tools import model_store
from src.ai_tools.model_store import ModelSpec

logger = logging.getLogger(__name__)

LAMA_SPEC = ModelSpec(
    filename="inpainting_lama_2025jan.onnx",
    sha256="7df918ac3921d3daf0aae1d219776cf0dc4e4935f035af81841b40adcf74fdf2",
    size=92591623,
    zoo_dir="inpainting_lama",
    label="the inpainting model",
)
LAMA_SIDE = 512

# Band sizes as a fraction of the distance between the eyes.
FINE_SIGMA = 0.020
COARSE_SIGMA = 0.130

SMALL_HOLE_PX = 24          # holes narrower than this use the fast fill


# --------------------------------------------------------------------- #
# Frequency separation
# --------------------------------------------------------------------- #

def _as_float(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    return image.astype(np.float32, copy=False)


def _like(result: np.ndarray, original: np.ndarray) -> np.ndarray:
    result = np.clip(result, 0.0, 1.0)
    if original.dtype == np.uint8:
        return np.round(result * 255.0).astype(np.uint8)
    if original.dtype == np.uint16:
        return np.round(result * 65535.0).astype(np.uint16)
    return result.astype(np.float32)


def split_bands(image: np.ndarray, fine_sigma: float, coarse_sigma: float):
    """Split a float image into ``(fine, middle, coarse)``.

    The three add back to the original exactly:
    ``fine + middle + coarse == image``.
    """
    image = _as_float(image)
    soft = cv2.GaussianBlur(image, (0, 0), max(0.5, fine_sigma))
    softer = cv2.GaussianBlur(image, (0, 0), max(fine_sigma * 1.5, coarse_sigma))
    return image - soft, soft - softer, softer


def smooth_skin(
    image: np.ndarray, mask: np.ndarray, amount: float, eye_distance: float,
) -> np.ndarray:
    """Even out skin without removing its texture.

    ``amount`` runs 0..1. ``eye_distance`` is the distance between the outer
    eye corners in pixels, which sets the band sizes. ``mask`` is the skin
    mask. Returns an image of the same dtype.
    """
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0 or mask is None or not np.any(mask > 0.01):
        return image.copy()
    fine, middle, coarse = split_bands(image, FINE_SIGMA * eye_distance, COARSE_SIGMA * eye_distance)
    weight = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), max(1.0, 0.03 * eye_distance))
    # Up to 85 percent of the middle band goes; never all of it, or the
    # face loses its modelling.
    keep = 1.0 - 0.85 * amount * weight[:, :, None]
    return _like(fine + middle * keep + coarse, image)


def band_energy(image: np.ndarray, mask: np.ndarray, eye_distance: float) -> tuple[float, float]:
    """``(fine, middle)`` detail energy inside a mask. For tests and tuning."""
    fine, middle, _ = split_bands(image, FINE_SIGMA * eye_distance, COARSE_SIGMA * eye_distance)
    inside = mask > 0.5
    if not inside.any():
        return 0.0, 0.0
    return float(np.mean(fine[inside] ** 2)), float(np.mean(middle[inside] ** 2))


# --------------------------------------------------------------------- #
# Teeth and eyes
# --------------------------------------------------------------------- #

def whiten_teeth(image: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    """Take the yellow out of teeth and lift them slightly.

    Mostly a desaturation: teeth read as stained because of colour, not
    because they are dark. The brightening is deliberately small, since
    glowing teeth are the usual tell of a retouched portrait.
    """
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0 or mask is None or not np.any(mask > 0.01):
        return image.copy()
    rgb = _as_float(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0) * amount
    hsv[:, :, 1] *= 1.0 - 0.70 * soft
    hsv[:, :, 2] = hsv[:, :, 2] + 0.12 * soft * (1.0 - hsv[:, :, 2])
    return _like(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), image)


def clarify_eyes(
    image: np.ndarray, iris_mask: np.ndarray, eye_mask: np.ndarray | None, amount: float,
) -> np.ndarray:
    """Add contrast to the irises and brighten the whites a little.

    ``iris_mask`` selects the irises; ``eye_mask`` the whole opening between
    the lids, from which the whites are taken as "eye but not iris".
    """
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0 or iris_mask is None or not np.any(iris_mask > 0.01):
        return image.copy()
    rgb = _as_float(image)
    iris = cv2.GaussianBlur(iris_mask.astype(np.float32), (0, 0), 0.8)[:, :, None]

    # Local contrast in the iris: detail against its own blurred version.
    radius = max(1.0, float(np.sqrt(max(1.0, iris_mask.sum())) * 0.25))
    detail = rgb - cv2.GaussianBlur(rgb, (0, 0), radius)
    out = rgb + 0.9 * amount * iris * detail + 0.04 * amount * iris

    if eye_mask is not None:
        whites = np.clip(eye_mask.astype(np.float32) - iris_mask.astype(np.float32), 0.0, 1.0)
        whites = cv2.GaussianBlur(whites, (0, 0), 0.8)[:, :, None] * amount
        grey = out.mean(axis=2, keepdims=True)
        out = out + 0.35 * whites * (grey - out)            # less red in the whites
        out = out + 0.06 * whites * (1.0 - out)
    return _like(out, image)


# --------------------------------------------------------------------- #
# Inpainting
# --------------------------------------------------------------------- #

class Inpainter:
    """Fills holes. ``available`` says whether the neural engine is loaded;
    the fast fill needs no model and always works."""

    def __init__(self, download: bool = False) -> None:
        self._net = None
        self._load_error: str | None = None
        path = model_store.ensure_model(LAMA_SPEC, download=download)
        if path is None:
            self._load_error = "inpainting model unavailable"
            return
        try:
            self._net = cv2.dnn.readNetFromONNX(str(path))
        except Exception as exc:
            self._load_error = f"inpainting model failed to load: {exc}"
            logger.warning(self._load_error)

    @property
    def available(self) -> bool:
        return self._net is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @staticmethod
    def is_cached() -> bool:
        return model_store.is_cached(LAMA_SPEC)

    @staticmethod
    def download_mb() -> float:
        return model_store.total_download_mb([LAMA_SPEC])

    # -- engines ----------------------------------------------------------- #

    @staticmethod
    def fast_fill(rgb: np.ndarray, hole: np.ndarray, radius: int = 5) -> np.ndarray:
        """Fast-marching fill. Good for spots; smears across large holes."""
        return cv2.inpaint(rgb, (hole > 0).astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)

    def _lama_tile(self, rgb: np.ndarray, hole: np.ndarray) -> np.ndarray | None:
        """Run the network on one window. Returns the window, filled."""
        h, w = rgb.shape[:2]
        # The network was exported to take pixels in OpenCV's own channel
        # order, 0..255 scaled to 0..1, and to return 0..255.
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        image_blob = cv2.dnn.blobFromImage(bgr, 1.0 / 255.0, (LAMA_SIDE, LAMA_SIDE), (0, 0, 0), False, False)
        mask_blob = cv2.dnn.blobFromImage((hole > 0).astype(np.uint8) * 255, 1.0, (LAMA_SIDE, LAMA_SIDE), (0,), False, False)
        mask_blob = (mask_blob > 0).astype(np.float32)
        try:
            self._net.setInput(image_blob, "image")
            self._net.setInput(mask_blob, "mask")
            output = np.asarray(self._net.forward(), dtype=np.float32)[0]
        except Exception as exc:
            logger.warning("Inpainting inference failed: %s", exc)
            return None
        filled = np.clip(np.transpose(output, (1, 2, 0)), 0, 255).astype(np.uint8)
        filled = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC if w > LAMA_SIDE else cv2.INTER_AREA)
        return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)

    # -- the public fill --------------------------------------------------- #

    def fill(self, rgb: np.ndarray, hole: np.ndarray, engine: str = "auto") -> tuple[np.ndarray, str]:
        """Fill the pixels where ``hole`` is non-zero.

        ``rgb`` is uint8. ``engine`` is ``"auto"``, ``"fast"`` or
        ``"neural"``. Returns the filled image and the engine that was used
        (which may be ``"fast"`` even when ``"neural"`` was asked for, if
        the model is not available). Pixels outside the hole are returned
        exactly as they came in.
        """
        hole = (np.asarray(hole) > 0).astype(np.uint8)
        if rgb.dtype != np.uint8:
            raise TypeError("fill expects an 8-bit RGB image")
        if not hole.any():
            return rgb.copy(), "none"

        out = rgb.copy()
        used: set[str] = set()
        count, labels, stats, _ = cv2.connectedComponentsWithStats(hole, connectivity=8)
        for label in range(1, count):
            x, y, w, h, _area = (int(v) for v in stats[label])
            region = labels == label
            thin = min(w, h) < SMALL_HOLE_PX
            neural = self.available and (engine == "neural" or (engine == "auto" and not thin))
            if neural:
                window = _context_window(rgb.shape, (x, y, w, h))
                ys, xs = window
                tile = self._lama_tile(out[ys, xs], region[ys, xs])
                if tile is not None:
                    # Only the missing pixels are taken. What was in the
                    # hole before is not picture, so it is not blended in.
                    out[ys, xs] = np.where(region[ys, xs][:, :, None], tile, out[ys, xs])
                    used.add("neural")
                    continue
            filled = self.fast_fill(out, region.astype(np.uint8), radius=max(3, min(w, h) // 3))
            out[region] = filled[region]
            used.add("fast")
        return out, "+".join(sorted(used))


def _context_window(shape, box, expand: float = 2.2, minimum: int = 256) -> tuple[slice, slice]:
    """A square window around a hole, with room for surrounding context."""
    height, width = shape[:2]
    x, y, w, h = box
    side = int(min(max(height, width), max(minimum, expand * max(w, h))))
    cx, cy = x + w / 2.0, y + h / 2.0
    x0 = int(np.clip(cx - side / 2.0, 0, max(0, width - side)))
    y0 = int(np.clip(cy - side / 2.0, 0, max(0, height - side)))
    return slice(y0, min(height, y0 + side)), slice(x0, min(width, x0 + side))


def heal_spots(rgb: np.ndarray, spots, inpainter: Inpainter | None = None) -> np.ndarray:
    """Remove blemishes or sensor dust. ``spots`` is a list of ``(x, y, radius)``."""
    hole = np.zeros(rgb.shape[:2], np.uint8)
    for x, y, radius in spots:
        cv2.circle(hole, (int(round(x)), int(round(y))), max(1, int(round(radius))), 1, -1)
    if not hole.any():
        return rgb.copy()
    engine = inpainter or _FastOnly()
    return engine.fill(rgb, hole)[0]


class _FastOnly:
    """Stands in for :class:`Inpainter` when only the fast fill is wanted."""

    available = False

    @staticmethod
    def fill(rgb: np.ndarray, hole: np.ndarray, engine: str = "fast") -> tuple[np.ndarray, str]:
        return Inpainter.fast_fill(rgb, hole), "fast"


def blank_corners(rgb: np.ndarray) -> np.ndarray:
    """The blank wedges an uncropped rotation leaves, as a 0/1 mask.

    Only blank areas touching the frame edge count, so a genuinely black
    object inside the picture is not mistaken for one.
    """
    dark = (rgb.max(axis=2) <= 8).astype(np.uint8)
    count, labels = cv2.connectedComponents(dark)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    mask = np.isin(labels, border[border != 0]).astype(np.uint8)
    if mask.any():
        # A few pixels wider: the wedge's edge is blended with the picture,
        # so its outermost rows are dark without being blank.
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
    return mask


def fill_corners(rgb: np.ndarray, inpainter: Inpainter | None = None) -> tuple[np.ndarray, str]:
    """Fill the wedges left by straightening, instead of cropping them off."""
    hole = blank_corners(rgb)
    if not hole.any():
        return rgb.copy(), "none"
    return (inpainter or _FastOnly()).fill(rgb, hole, engine="auto")
