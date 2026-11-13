"""Face detection and closed-eye (blink) detection.

Uses YuNet, the small DNN face detector that ships with OpenCV's model zoo.
It runs on CPU in a few milliseconds per frame, which matters when a cull
pass has thousands of frames to get through.

Why not Haar cascades: OpenCV 5 wheels no longer ship the cascade XML files
(``cv2.CascadeClassifier`` is absent entirely in the headless build this app
depends on), so the classic detector is not an option here.

Why not Eye Aspect Ratio: EAR is the usual way to score a blink, but it needs
six contour points per eye. YuNet returns five landmarks — two of which are
eye *centres*, with no contour — so EAR cannot be computed from it. Instead
this module crops a patch around each eye centre and measures how *tall* the
dark region inside it is:

  * an open eye shows the iris and pupil — a roughly round dark blob that
    fills much of the patch vertically;
  * a closed eye shows only the lash line — a thin, wide, horizontal streak.

The ratio of dark-region height to patch height therefore separates the two
without needing a second model. It is a heuristic, not a trained classifier:
see ``EYE_OPEN_THRESHOLD`` and read ``openness`` as a continuous score that
you may want to tune against your own images.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.ai_tools import model_store
from src.ai_tools.model_store import ModelSpec

logger = logging.getLogger(__name__)

# --- Model -------------------------------------------------------------- #

# Pinned so a corrupted or substituted download is rejected rather than fed
# to the DNN loader. Both values come from the git-lfs pointer that
# opencv_zoo commits for this file. Fetching and caching live in
# ``model_store``, which the pose models share.
MODEL_SPEC = ModelSpec(
    filename="face_detection_yunet_2023mar.onnx",
    sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    size=232589,
    zoo_dir="face_detection_yunet",
    label="the YuNet face model",
)

# Kept as module-level names because callers and tests reference them.
MODEL_FILENAME = MODEL_SPEC.filename
MODEL_SHA256 = MODEL_SPEC.sha256
MODEL_SIZE = MODEL_SPEC.size
MODEL_DIR = model_store.MODEL_DIR


# --- Tuning ------------------------------------------------------------- #

# Eye patch size as a fraction of interocular distance. Wide enough to take
# in the whole eye opening, shallow enough to exclude brow and cheek, whose
# shadows would otherwise read as "dark region".
EYE_PATCH_W = 0.44
EYE_PATCH_H = 0.30

# The patch is tiny at normal face sizes, so it is upscaled before measuring
# to keep the column statistics stable.
_PATCH_SIZE = (64, 44)  # (w, h)

# Below this openness score an eye is reported closed. Calibrated so a
# clearly open eye lands well above it and a covered eye well below; a
# half-blink sits in between, which is why openness is also exposed raw.
EYE_OPEN_THRESHOLD = 0.42

DEFAULT_SCORE_THRESHOLD = 0.6
DEFAULT_NMS_THRESHOLD = 0.3
DEFAULT_TOP_K = 5000


@dataclass
class EyeState:
    """Openness measurement for a single eye."""

    openness: float
    is_open: bool

    @property
    def is_closed(self) -> bool:
        return not self.is_open


@dataclass
class Face:
    """One detected face, with optional per-eye openness."""

    x: int
    y: int
    w: int
    h: int
    confidence: float
    right_eye: tuple[float, float]
    left_eye: tuple[float, float]
    right_eye_state: EyeState | None = None
    left_eye_state: EyeState | None = None

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    @property
    def interocular(self) -> float:
        return float(
            np.hypot(
                self.left_eye[0] - self.right_eye[0],
                self.left_eye[1] - self.right_eye[1],
            )
        )

    @property
    def eyes_measured(self) -> bool:
        return self.right_eye_state is not None and self.left_eye_state is not None

    @property
    def eyes_closed(self) -> bool:
        """True when both eyes read as closed.

        Deliberately requires *both*: one eye below threshold is far more
        often a highlight, a hair strand or a head turn hiding one eye than
        an actual blink, and flagging those would make the cull noisy.
        """
        if not self.eyes_measured:
            return False
        return self.right_eye_state.is_closed and self.left_eye_state.is_closed

    @property
    def min_openness(self) -> float | None:
        if not self.eyes_measured:
            return None
        return min(self.right_eye_state.openness, self.left_eye_state.openness)


@dataclass
class FaceReport:
    """Result of analysing one image."""

    faces: list[Face]
    model_available: bool = True
    error: str | None = None

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def any_eyes_closed(self) -> bool:
        return any(f.eyes_closed for f in self.faces)

    @property
    def closed_eye_faces(self) -> list[Face]:
        return [f for f in self.faces if f.eyes_closed]


# --- Model acquisition -------------------------------------------------- #


def model_path() -> Path:
    return MODEL_SPEC.path


def ensure_model(download: bool = True, timeout: int = 180) -> Path | None:
    """Return a verified model path, downloading it once if needed.

    Returns None when the model is absent and cannot be fetched -- callers
    must treat face analysis as unavailable rather than failing, the same
    way the rest of the app degrades when an optional engine is missing.
    """
    return model_store.ensure_model(MODEL_SPEC, download=download, timeout=timeout)


# --- Eye openness ------------------------------------------------------- #


def _eye_patch(gray: np.ndarray, centre: tuple[float, float], interocular: float) -> np.ndarray | None:
    """Crop and normalise the region around one eye centre."""
    half_w = max(3.0, interocular * EYE_PATCH_W / 2.0)
    half_h = max(2.0, interocular * EYE_PATCH_H / 2.0)
    cx, cy = centre
    x0, x1 = int(round(cx - half_w)), int(round(cx + half_w))
    y0, y1 = int(round(cy - half_h)), int(round(cy + half_h))

    h, w = gray.shape[:2]
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 - x0 < 4 or y1 - y0 < 3:
        return None

    patch = gray[y0:y1, x0:x1]
    patch = cv2.resize(patch, _PATCH_SIZE, interpolation=cv2.INTER_CUBIC)
    # Local contrast normalisation, so exposure and skin tone do not decide
    # what counts as "dark".
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(patch)


def eye_openness(gray: np.ndarray, centre: tuple[float, float], interocular: float) -> float | None:
    """Score how open one eye is, from 0 (shut) to about 1 (wide open).

    Measures the vertical extent of the dark region in the eye patch: the
    iris of an open eye is a tall blob, whereas a closed lid leaves only a
    flat lash line.
    """
    patch = _eye_patch(gray, centre, interocular)
    if patch is None:
        return None

    patch = cv2.GaussianBlur(patch, (3, 3), 0)
    # Otsu picks the dark/light split per patch, which copes with very
    # different lighting between frames.
    _, dark = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ph = patch.shape[0]
    column_counts = (dark > 0).sum(axis=0).astype(np.float32)
    if not column_counts.any():
        return 0.0

    # A high percentile rather than the max: one stray dark column (a lash,
    # a frame edge) should not read as a wide-open eye.
    tall = float(np.percentile(column_counts, 80))
    openness = tall / float(ph)

    # Clamp: a patch that is almost entirely dark is hair or deep shadow,
    # not an eye, so it should not score as maximally open.
    area_fraction = float((dark > 0).mean())
    if area_fraction > 0.85:
        return 0.0
    return float(min(1.0, max(0.0, openness)))


# --- Detector ----------------------------------------------------------- #


class FaceAnalyzer:
    """Detects faces and measures eye openness.

    The detector is created lazily and reused; ``available`` says whether
    the model could be loaded at all, so callers can skip face-dependent
    features instead of handling exceptions.
    """

    def __init__(
        self,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        download: bool = True,
        eye_open_threshold: float = EYE_OPEN_THRESHOLD,
    ) -> None:
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.eye_open_threshold = eye_open_threshold
        self._detector = None
        self._input_size: tuple[int, int] | None = None
        self._model = ensure_model(download=download)
        self._load_error: str | None = None if self._model else "YuNet model unavailable"

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _detector_for(self, width: int, height: int):
        if self._model is None:
            return None
        if self._detector is None:
            try:
                self._detector = cv2.FaceDetectorYN.create(
                    str(self._model),
                    "",
                    (width, height),
                    self.score_threshold,
                    self.nms_threshold,
                    self.top_k,
                )
            except Exception as exc:
                self._load_error = f"Could not create YuNet detector: {exc}"
                logger.warning(self._load_error)
                self._model = None
                return None
            self._input_size = (width, height)
        elif self._input_size != (width, height):
            self._detector.setInputSize((width, height))
            self._input_size = (width, height)
        return self._detector

    def detect(self, image: np.ndarray, measure_eyes: bool = True) -> list[Face]:
        """Detect faces in a BGR image, optionally scoring eye openness."""
        if image is None or image.size == 0:
            return []
        if image.ndim == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            bgr = image

        h, w = bgr.shape[:2]
        detector = self._detector_for(w, h)
        if detector is None:
            return []

        try:
            _, raw = detector.detect(bgr)
        except Exception as exc:
            logger.warning("Face detection failed: %s", exc)
            return []
        if raw is None:
            return []

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if measure_eyes else None
        faces: list[Face] = []
        for row in raw:
            face = Face(
                x=int(row[0]),
                y=int(row[1]),
                w=int(row[2]),
                h=int(row[3]),
                confidence=float(row[14]),
                right_eye=(float(row[4]), float(row[5])),
                left_eye=(float(row[6]), float(row[7])),
            )
            if measure_eyes and gray is not None:
                span = face.interocular
                if span >= 8.0:  # below this the eye patch is too small to judge
                    for attr, centre in (
                        ("right_eye_state", face.right_eye),
                        ("left_eye_state", face.left_eye),
                    ):
                        score = eye_openness(gray, centre, span)
                        if score is not None:
                            setattr(
                                face,
                                attr,
                                EyeState(
                                    openness=score,
                                    is_open=score >= self.eye_open_threshold,
                                ),
                            )
            faces.append(face)
        return faces

    def analyze(self, image: np.ndarray, measure_eyes: bool = True) -> FaceReport:
        """Analyse an in-memory BGR image."""
        if not self.available:
            return FaceReport(faces=[], model_available=False, error=self._load_error)
        return FaceReport(faces=self.detect(image, measure_eyes=measure_eyes))

    def analyze_path(self, path: str | Path, max_side: int = 1600) -> FaceReport:
        """Analyse an image file.

        Large frames are downscaled first: detection accuracy is unaffected
        at these face sizes and it keeps a thousand-frame pass tractable.
        Landmarks are mapped back to original-image coordinates.
        """
        if not self.available:
            return FaceReport(faces=[], model_available=False, error=self._load_error)

        path = Path(path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            # RAW and HEIC need the project loader rather than cv2.imread.
            try:
                from src.engine.universal_loader import UniversalLoader

                rgb = UniversalLoader().load_array(path)
                image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception as exc:
                return FaceReport(faces=[], error=f"Could not read {path.name}: {exc}")

        h, w = image.shape[:2]
        scale = 1.0
        if max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        faces = self.detect(image)
        if scale != 1.0 and faces:
            inv = 1.0 / scale
            for f in faces:
                f.x, f.y = int(f.x * inv), int(f.y * inv)
                f.w, f.h = int(f.w * inv), int(f.h * inv)
                f.right_eye = (f.right_eye[0] * inv, f.right_eye[1] * inv)
                f.left_eye = (f.left_eye[0] * inv, f.left_eye[1] * inv)
        return FaceReport(faces=faces)
