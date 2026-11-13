"""Facial landmarks, and the measurements taken from them.

A 478-point face mesh runs on each face the detector finds. From the mesh:

* **Eye aspect ratio (EAR).** The standard six-point measure,

      EAR = (|p2 - p6| + |p3 - p5|) / (2 |p1 - p4|)

  the eye's height over its width. It replaces the earlier approach of
  judging an eye by how dark the patch around it is, which depends on iris
  colour and lighting. EAR is a ratio of distances on the face itself, so it
  does not.
* **Mouth opening**, by the same construction on the inner lips.
* **Head roll**, from the outer eye corners. The detector's own landmarks
  could not give this (see orientation.py: a 24 degree roll read as 5).

and from a small second network that reads the mesh, **expression
coefficients** (smile, jaw open, and so on).

Which model, and why
--------------------
Two meshes were tried and the difference decided the design:

* The original 468-point model fits the face well but barely moves its
  eyelids. On four reference faces, EAR read 0.22-0.32 with eyes open and
  0.17-0.26 with them shut -- overlapping ranges, useless as a blink test.
* The 478-point model used here read 0.155-0.282 open and 0.064-0.119 shut
  on the same faces: every case separated.

The same bundle's trained "eye blink" coefficient was also tried and did
*worse* than plain EAR on those faces (open 0.09-0.58, shut 0.42-0.72), so
EAR decides blinks and the coefficient is reported but not used for it.
For smiles the coefficients beat a separate expression classifier, which
labelled an open mouth mid-shout as "happy" at 0.91.

Scope of the claims
-------------------
The "shut" faces above are the open ones with the eyes painted over, because
no labelled photographs of real blinks were available. That is enough to
show the measurement responds and to reject the model that does not; it is
not validation on real closed eyes. Four faces is also four faces: nothing
here is established across skin tones, ages, eyewear or lighting. A
downward gaze reads low (0.155 in one reference frame) and can sit close to
the threshold.
"""

from __future__ import annotations

import io
import logging
import math
import threading
import zipfile
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.ai_tools import model_store
from src.ai_tools.model_store import ModelSpec

logger = logging.getLogger(__name__)

# One archive holding the mesh and the expression network. Pinned as a
# whole, then unpacked in memory, so no unverified file ever reaches disk.
MESH_SPEC = ModelSpec(
    filename="face_landmarker_v1_float16.task",
    sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    size=3758596,
    zoo_dir="",
    label="the face-mesh model",
    url=(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    ),
)
MODEL_SPECS = (MESH_SPEC,)
_MESH_MEMBER = "face_landmarks_detector.tflite"
_EXPRESSION_MEMBER = "face_blendshapes.tflite"

MESH_INPUT = 256
MESH_POINTS = 478
ROI_SCALE = 1.5                 # the mesh expects the face with a margin
REFINE_ROLL_ABOVE_DEG = 8.0     # re-run upright if the head is turned this far
MIN_FACE_SIDE = 24              # below this the eyelids are not resolved

# Six points per eye, ordered p1..p6 as in the EAR formula: p1 and p4 are the
# corners, p2/p3 the upper lid, p6/p5 the lower lid beneath them.
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)
RIGHT_EYE_OUTER, LEFT_EYE_OUTER = 33, 263

# Closed outlines, for masks and crops.
RIGHT_EYE_RING = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
LEFT_EYE_RING = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
INNER_LIPS = (78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191)
OUTER_LIPS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)

# Inner lips: corners, then upper and lower at three stations.
MOUTH_CORNERS = (78, 308)
MOUTH_UPPER = (81, 13, 311)
MOUTH_LOWER = (178, 14, 402)
LIP_CORNERS_OUTER = (61, 291)
LIP_TOP, LIP_BOTTOM = 0, 17

# Thresholds, set against the reference faces described above.
EAR_CLOSED_BELOW = 0.135
EAR_OPEN_ABOVE = 0.20
MOUTH_OPEN_ABOVE = 0.35
JAW_OPEN_ABOVE = 0.30
SMILE_ABOVE = 0.45

# The mesh points the expression network reads, in the order it expects.
_EXPRESSION_POINTS = (
    0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65,
    66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136,
    144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 168,
    172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246, 249, 251, 263, 267, 269, 270, 276,
    282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318,
    321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380,
    381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466,
    468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
)
EXPRESSION_NAMES = (
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft",
    "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneer", "tongueOut",
)


# --------------------------------------------------------------------- #
# Geometry: pure functions, exact, unit-tested on constructed points
# --------------------------------------------------------------------- #

def eye_aspect_ratio(points: np.ndarray) -> float:
    """EAR for six points ordered p1..p6. ``points`` is 6 x 2 (or 6 x 3)."""
    p = np.asarray(points, dtype=np.float64)[:, :2]
    width = np.linalg.norm(p[0] - p[3])
    if width < 1e-9:
        return 0.0
    return float((np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2.0 * width))


def ear_pair(landmarks: np.ndarray) -> tuple[float, float]:
    """``(right, left)`` EAR from a full mesh."""
    return (
        eye_aspect_ratio(landmarks[list(RIGHT_EYE)]),
        eye_aspect_ratio(landmarks[list(LEFT_EYE)]),
    )


def mouth_aspect_ratio(landmarks: np.ndarray) -> float:
    """Inner-lip opening over mouth width. About 0 closed, 0.5+ wide open."""
    p = np.asarray(landmarks, dtype=np.float64)[:, :2]
    width = np.linalg.norm(p[MOUTH_CORNERS[0]] - p[MOUTH_CORNERS[1]])
    if width < 1e-9:
        return 0.0
    gaps = [np.linalg.norm(p[u] - p[l]) for u, l in zip(MOUTH_UPPER, MOUTH_LOWER)]
    return float(np.mean(gaps) / width)


def head_roll(landmarks: np.ndarray) -> float:
    """Degrees the eye line is turned from horizontal. Positive = clockwise
    on screen (the subject's left eye lower than their right)."""
    p = np.asarray(landmarks, dtype=np.float64)
    dx, dy = p[LEFT_EYE_OUTER, :2] - p[RIGHT_EYE_OUTER, :2]
    return float(math.degrees(math.atan2(dy, dx)))


def smile_geometry(landmarks: np.ndarray) -> float:
    """A 0..1 smile estimate from lip shape alone.

    Used only when the expression network is unavailable. A smile widens
    the mouth relative to the eyes and lifts its corners above the midline
    of the lips; both are folded into one score. It is a heuristic and
    cannot tell a smile from a grimace.
    """
    p = np.asarray(landmarks, dtype=np.float64)[:, :2]
    eye_span = np.linalg.norm(p[RIGHT_EYE_OUTER] - p[LEFT_EYE_OUTER])
    if eye_span < 1e-9:
        return 0.0
    left, right = p[LIP_CORNERS_OUTER[0]], p[LIP_CORNERS_OUTER[1]]
    width = np.linalg.norm(left - right) / eye_span
    # Lip midline in the face's own frame, so head roll does not read as lift.
    axis = (p[LEFT_EYE_OUTER] - p[RIGHT_EYE_OUTER]) / eye_span
    down = np.array([-axis[1], axis[0]])
    mid = (p[LIP_TOP] + p[LIP_BOTTOM]) / 2.0
    lift = -np.mean([np.dot(left - mid, down), np.dot(right - mid, down)]) / eye_span
    widen = np.clip((width - 0.50) / 0.20, 0.0, 1.0)
    raise_ = np.clip((lift + 0.01) / 0.06, 0.0, 1.0)
    return float(0.5 * widen + 0.5 * raise_)


# --------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------- #

@dataclass
class FaceMeshResult:
    """Everything measured on one face."""

    landmarks: np.ndarray          # 478 x 3, image pixels (z is relative depth)
    ear_right: float
    ear_left: float
    mouth_open: float              # inner-lip aspect ratio
    roll_deg: float
    smile: float | None = None     # 0..1, None when it could not be measured
    jaw_open: float | None = None  # 0..1, from the expression network
    smile_source: str = "none"     # "expression", "geometry" or "none"
    expressions: dict[str, float] = field(default_factory=dict)

    @property
    def ear(self) -> float:
        """The more open of the two eyes.

        A blink closes both. One low reading is far more often a head turn
        hiding an eye, hair, or a wink, so the frame is judged on the eye
        that can be seen best -- the same rule the darkness heuristic used.
        """
        return max(self.ear_right, self.ear_left)

    @property
    def eyes_closed(self) -> bool:
        return self.ear < EAR_CLOSED_BELOW

    @property
    def eyes_open(self) -> bool:
        return self.ear >= EAR_OPEN_ABOVE

    @property
    def eye_openness(self) -> float:
        """EAR mapped to 0..1 between the closed and open thresholds."""
        span = EAR_OPEN_ABOVE - EAR_CLOSED_BELOW
        return float(np.clip((self.ear - EAR_CLOSED_BELOW) / span, 0.0, 1.0))

    @property
    def is_smiling(self) -> bool:
        return self.smile is not None and self.smile >= SMILE_ABOVE

    @property
    def mouth_is_open(self) -> bool:
        """Open as in mid-word, not as in a broad smile showing teeth."""
        if self.jaw_open is not None:
            return self.jaw_open >= JAW_OPEN_ABOVE
        return self.mouth_open >= MOUTH_OPEN_ABOVE and not self.is_smiling

    def eye_box(self, pad: float = 0.6) -> tuple[int, int, int, int]:
        """``(x, y, w, h)`` around both eyes, for focus measurement."""
        pts = self.landmarks[list(RIGHT_EYE_RING + LEFT_EYE_RING), :2]
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        grow_x, grow_y = pad * 0.25 * (x1 - x0), pad * (x1 - x0) * 0.35
        return (int(x0 - grow_x), int(y0 - grow_y), int((x1 - x0) + 2 * grow_x), int((y1 - y0) + 2 * grow_y))

    def polygon(self, indices) -> np.ndarray:
        """Image-space polygon through the given mesh points (N x 2, int32)."""
        return np.round(self.landmarks[list(indices), :2]).astype(np.int32)


# --------------------------------------------------------------------- #
# The models
# --------------------------------------------------------------------- #

def _roi_matrix(centre: tuple[float, float], side: float, roll_deg: float) -> np.ndarray:
    """Affine map from the image to the mesh's upright square input."""
    matrix = cv2.getRotationMatrix2D(centre, roll_deg, MESH_INPUT / side)
    matrix[0, 2] += MESH_INPUT / 2.0 - centre[0]
    matrix[1, 2] += MESH_INPUT / 2.0 - centre[1]
    return matrix


def _read_bundle(path) -> tuple[bytes, bytes | None] | None:
    """The mesh and expression networks from the verified archive."""
    try:
        with zipfile.ZipFile(path) as bundle:
            mesh = bundle.read(_MESH_MEMBER)
            try:
                expression = bundle.read(_EXPRESSION_MEMBER)
            except KeyError:
                expression = None
            return mesh, expression
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        logger.warning("Face-mesh archive could not be read: %s", exc)
        return None


def _net_from_bytes(data: bytes):
    return cv2.dnn.readNetFromTFLite(np.frombuffer(data, dtype=np.uint8))


class FaceMesh:
    """Runs the mesh and the expression network on detected faces.

    ``available`` is False when the model is not on disk and could not be
    fetched; callers then fall back to the older eye heuristic.

    Not safe to share between threads: use :class:`ThreadLocalFaceMesh`.
    """

    def __init__(self, download: bool = True) -> None:
        self._mesh = None
        self._expression = None
        self._load_error: str | None = None

        path = model_store.ensure_model(MESH_SPEC, download=download)
        if path is None:
            self._load_error = "face-mesh model unavailable"
            return
        parts = _read_bundle(path)
        if parts is None:
            self._load_error = "face-mesh archive unreadable"
            return
        try:
            self._mesh = _net_from_bytes(parts[0])
        except Exception as exc:
            self._load_error = f"face-mesh model failed to load: {exc}"
            logger.warning(self._load_error)
            return
        if parts[1] is not None:
            try:
                self._expression = _net_from_bytes(parts[1])
            except Exception as exc:
                logger.warning("Expression network failed to load: %s", exc)

    # -- availability ---------------------------------------------------- #

    @property
    def available(self) -> bool:
        return self._mesh is not None

    @property
    def has_expression(self) -> bool:
        return self._expression is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @staticmethod
    def is_cached() -> bool:
        return model_store.is_cached(MESH_SPEC)

    @staticmethod
    def download_mb() -> float:
        return model_store.total_download_mb(MODEL_SPECS)

    # -- mesh -------------------------------------------------------------- #

    def _run_mesh(self, bgr: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Returns ``(landmarks in image space, landmarks in crop space)``."""
        crop = cv2.warpAffine(
            bgr, matrix, (MESH_INPUT, MESH_INPUT), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )
        blob = cv2.dnn.blobFromImage(crop, 1.0 / 255.0, (MESH_INPUT, MESH_INPUT), swapRB=True)
        try:
            self._mesh.setInput(blob)
            raw = np.asarray(self._mesh.forward("Identity"), dtype=np.float32).reshape(-1)
        except Exception as exc:
            logger.warning("Face mesh inference failed: %s", exc)
            return None
        if raw.size != MESH_POINTS * 3:
            return None
        local = raw.reshape(MESH_POINTS, 3)

        inverse = cv2.invertAffineTransform(matrix)
        xy = local[:, :2] @ inverse[:, :2].T + inverse[:, 2]
        depth = local[:, 2:3] * float(np.linalg.norm(inverse[:, 0]))
        return np.hstack([xy, depth]).astype(np.float32), local

    @staticmethod
    def _plausible(local: np.ndarray) -> bool:
        """Reject a mesh that did not land on a face.

        The network always returns 478 points, face or not. A real fit keeps
        the eyes inside the crop and a sensible distance apart.
        """
        eyes = local[[RIGHT_EYE_OUTER, LEFT_EYE_OUTER], :2]
        if eyes.min() < 0 or eyes.max() > MESH_INPUT:
            return False
        span = float(np.linalg.norm(eyes[0] - eyes[1]))
        return 0.18 * MESH_INPUT <= span <= 0.75 * MESH_INPUT

    def analyze_face(self, bgr: np.ndarray, face) -> FaceMeshResult | None:
        """Measure one detected face. ``face`` is a ``face_analysis.Face``."""
        if not self.available or bgr is None or bgr.size == 0:
            return None
        centre = (face.x + face.w / 2.0, face.y + face.h / 2.0)
        side = ROI_SCALE * max(face.w, face.h)
        if max(face.w, face.h) < MIN_FACE_SIDE:
            return None

        found = self._run_mesh(bgr, _roi_matrix(centre, side, 0.0))
        if found is None:
            return None
        landmarks, local = found

        # The mesh is trained on roughly upright crops. If the head is
        # rolled, turn the crop by the roll just measured and go again.
        roll = head_roll(landmarks)
        if abs(roll) > REFINE_ROLL_ABOVE_DEG:
            again = self._run_mesh(bgr, _roi_matrix(centre, side, roll))
            if again is not None and self._plausible(again[1]):
                landmarks, local = again
                roll = head_roll(landmarks)

        if not self._plausible(local):
            return None

        right, left = ear_pair(landmarks)
        result = FaceMeshResult(
            landmarks=landmarks,
            ear_right=right,
            ear_left=left,
            mouth_open=mouth_aspect_ratio(landmarks),
            roll_deg=roll,
        )
        self._measure_expression(local, result)
        return result

    def analyze(self, bgr: np.ndarray, faces) -> list[FaceMeshResult | None]:
        """One result per face, in order; None where the mesh did not land."""
        return [self.analyze_face(bgr, face) for face in faces]

    # -- expression -------------------------------------------------------- #

    def _measure_expression(self, local: np.ndarray, result: FaceMeshResult) -> None:
        if self._expression is not None:
            try:
                # The network reads the upright crop's coordinates; it is
                # indifferent to their scale (checked: x3 changes nothing).
                points = local[list(_EXPRESSION_POINTS), :2].astype(np.float32)[None]
                self._expression.setInput(points)
                scores = np.asarray(self._expression.forward(), dtype=np.float64).reshape(-1)
                if scores.size == len(EXPRESSION_NAMES):
                    named = dict(zip(EXPRESSION_NAMES, (float(s) for s in scores)))
                    result.expressions = named
                    result.smile = (named["mouthSmileLeft"] + named["mouthSmileRight"]) / 2.0
                    result.jaw_open = named["jawOpen"]
                    result.smile_source = "expression"
                    return
            except Exception as exc:
                logger.debug("Expression inference failed: %s", exc)
        result.smile = smile_geometry(result.landmarks)
        result.smile_source = "geometry"


class ThreadLocalFaceMesh:
    """One :class:`FaceMesh` per thread, created on first use.

    An OpenCV network keeps per-call state, so a single instance cannot be
    used from a pool. Construction here checks availability once, on the
    calling thread, so a first-run download does not race.
    """

    def __init__(self, download: bool = True) -> None:
        self._local = threading.local()
        first = FaceMesh(download=download)
        self.available = first.available
        self.has_expression = first.has_expression
        self._local.mesh = first

    def get(self) -> FaceMesh:
        mesh = getattr(self._local, "mesh", None)
        if mesh is None:
            mesh = self._local.mesh = FaceMesh(download=False)
        return mesh

    def analyze_face(self, bgr: np.ndarray, face) -> FaceMeshResult | None:
        return self.get().analyze_face(bgr, face)
