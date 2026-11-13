"""Skeletal pose: where a subject's shoulders, spine and hips actually are.

This is the model-backed half of auto-levelling. ``orientation`` can measure
the room from its straight lines, but a portrait shot against a plain wall has
no lines to measure, and a subject standing in a crooked doorway has the wrong
ones. The body itself is the better reference there.

Two models are used, because BlazePose is a two-stage pipeline: a person
detector finds roughly where a body is and returns four coarse keypoints, then
the landmark model crops around them, rotates the crop upright and regresses
33 body landmarks. The landmarks are mapped back into the original frame, so
unlike the face detector's landmarks they do track the body through rotation
-- which is what makes anatomy-aware levelling possible at all.

**Camera roll versus a leaning subject.** A tilted shoulder line on its own is
ambiguous: it happens when the camera was rolled, and it happens when the
subject shifted their weight. The two are told apart by the hips. Camera roll
rotates the whole body by the same amount, so the shoulder line and the hip
line tilt together; a posed lean moves the shoulders against the hips. So a
correction is only offered when both agree, and the spread between them is
reported as ``asymmetry`` so a caller can see how it was decided.

**What this does not do.** It reports posture, it does not judge it -- there is
no "good pose" score here, because what counts as good depends on what the
photograph is for. It also needs a reasonably visible body: a head-and-
shoulders crop gives no hips, and the module says so via ``has_hips`` rather
than guessing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.ai_tools import model_store
from src.ai_tools.model_store import ModelSpec

logger = logging.getLogger(__name__)

# --- Models ------------------------------------------------------------- #

PERSON_SPEC = ModelSpec(
    filename="person_detection_mediapipe_2023mar.onnx",
    sha256="47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f",
    size=11990159,
    zoo_dir="person_detection_mediapipe",
    label="the person detector",
)

POSE_SPEC = ModelSpec(
    filename="pose_estimation_mediapipe_2023mar.onnx",
    sha256="9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f",
    size=5557238,
    zoo_dir="pose_estimation_mediapipe",
    label="the pose model",
)

MODEL_SPECS = (PERSON_SPEC, POSE_SPEC)

# --- Landmark indices (BlazePose's 33-point topology) ------------------- #

NOSE = 0
LEFT_EYE, RIGHT_EYE = 2, 5
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

# --- Tuning ------------------------------------------------------------- #

PERSON_SCORE_THRESHOLD = 0.5
PERSON_NMS_THRESHOLD = 0.3
POSE_CONF_THRESHOLD = 0.5

# A landmark below this visibility is treated as not seen. The model reports
# visibility as "in frame and not occluded", which is exactly the question
# that matters before levelling a frame by someone's hips.
MIN_VISIBILITY = 0.55

# Shoulder line and hip line must agree within this many degrees before their
# tilt is read as camera roll. Wider than this and the subject is leaning, so
# there is no single body angle to level to.
# Measured rather than guessed: a subject standing square still reads a
# shoulder/hip spread of roughly 4 to 7 degrees frame to frame, because the
# two lines are regressed independently. A tighter gate than that refuses
# ordinary upright photographs. A genuine lean is far wider -- the worst case
# in testing, a body half out of frame, spread 57 degrees and is refused.
MAX_ASYMMETRY_DEG = 12.0

# Past this the body reading is almost certainly a misdetection rather than a
# genuinely rolled camera.
MAX_BODY_TILT_DEG = 30.0

# Only level a frame automatically when the reading is at least this good.
#
# The floor exists to exclude shoulders-only readings, and it is set from
# measurement, not taste. Correcting a frame by the body angle and then
# re-measuring what is left over:
#
#   shoulders and hips visible   mean residual 0.53 deg, worst 1.33 (13 frames)
#   shoulders only               mean residual 2.90 deg, worst 9.54 (13 frames)
#
# A ten-degree error is worse than the tilt most people would have left
# alone, so shoulders-only is reported and never applied. By construction a
# shoulders-only reading scores at most 0.45, which is below this floor.
MIN_APPLY_CONFIDENCE = 0.55

# Frames are analysed at this size. The models take fixed 224 and 256 pixel
# inputs anyway, so feeding them a 50MP original only costs resize time.
ANALYSIS_MAX_SIDE = 1280

_PERSON_INPUT = 224
_POSE_INPUT = 256

# The landmark model's crop is grown by this much around the detector's
# full-body span, matching the upstream pipeline it was trained with.
_BOX_ENLARGE = 1.0


def _fold(angle: float) -> float:
    """Fold an angle onto "how far from square", in -45..45 degrees."""
    return ((angle + 45.0) % 90.0) - 45.0


def _line_deviation(start, end) -> float:
    """Tilt of a line, in the sign convention ``rotate_image`` takes.

    Same formula as ``orientation.line_tilt`` uses for scene edges, so a body
    tilt and a scene tilt compose directly and mean the same thing.
    """
    return _fold(float(np.degrees(np.arctan2(end[1] - start[1], end[0] - start[0]))))


@dataclass
class Pose:
    """One detected body in a frame.

    ``landmarks`` is a 39x5 array of ``[x, y, z, visibility, presence]`` in
    original-image pixel coordinates; the first 33 rows are the body topology
    the index constants above refer to and the rest are auxiliary points the
    model uses internally.
    """

    landmarks: np.ndarray
    confidence: float
    box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    world: np.ndarray | None = None

    def point(self, index: int) -> tuple[float, float]:
        return float(self.landmarks[index, 0]), float(self.landmarks[index, 1])

    def visibility(self, index: int) -> float:
        return float(self.landmarks[index, 3])

    def seen(self, *indices: int) -> bool:
        return all(self.visibility(i) >= MIN_VISIBILITY for i in indices)

    def midpoint(self, a: int, b: int) -> tuple[float, float]:
        pa, pb = self.point(a), self.point(b)
        return (pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0

    # -- what is actually visible ---------------------------------------- #

    @property
    def has_shoulders(self) -> bool:
        return self.seen(LEFT_SHOULDER, RIGHT_SHOULDER)

    @property
    def has_hips(self) -> bool:
        return self.seen(LEFT_HIP, RIGHT_HIP)

    @property
    def has_legs(self) -> bool:
        return self.seen(LEFT_KNEE, RIGHT_KNEE)

    @property
    def framing(self) -> str:
        """Rough description of how much of the body is in frame."""
        if self.has_legs:
            return "full body"
        if self.has_hips:
            return "three-quarter"
        if self.has_shoulders:
            return "head and shoulders"
        return "partial"

    # -- posture --------------------------------------------------------- #

    @property
    def shoulder_tilt(self) -> float | None:
        """Degrees to rotate the frame to bring the shoulder line level."""
        if not self.has_shoulders:
            return None
        return _line_deviation(self.point(RIGHT_SHOULDER), self.point(LEFT_SHOULDER))

    @property
    def hip_tilt(self) -> float | None:
        """Degrees to rotate the frame to bring the hip line level."""
        if not self.has_hips:
            return None
        return _line_deviation(self.point(RIGHT_HIP), self.point(LEFT_HIP))

    @property
    def spine_tilt(self) -> float | None:
        """Lean of the shoulder-to-hip axis away from vertical.

        Positive means the upper body leans towards the right of the frame.
        This is a property of the *subject*, not the camera, so it is reported
        but never used to level a frame.
        """
        if not (self.has_shoulders and self.has_hips):
            return None
        sx, sy = self.midpoint(LEFT_SHOULDER, RIGHT_SHOULDER)
        hx, hy = self.midpoint(LEFT_HIP, RIGHT_HIP)
        # Measured from vertical, with y increasing downwards.
        return float(np.degrees(np.arctan2(sx - hx, hy - sy)))

    @property
    def asymmetry(self) -> float | None:
        """How far the shoulder line and the hip line disagree, in degrees."""
        shoulders, hips = self.shoulder_tilt, self.hip_tilt
        if shoulders is None or hips is None:
            return None
        return abs(shoulders - hips)

    @property
    def body_tilt(self) -> float | None:
        """Camera roll read off the body, or None when it cannot be read.

        Returns a correction only when the evidence supports one: both
        shoulders and hips visible and tilted together. Shoulders alone are
        reported too, but with lower confidence, because a shrug or a turned
        torso can fake a few degrees.
        """
        shoulders, hips = self.shoulder_tilt, self.hip_tilt
        if shoulders is not None and hips is not None:
            if abs(shoulders - hips) > MAX_ASYMMETRY_DEG:
                return None
            return (shoulders + hips) / 2.0
        return shoulders

    @property
    def body_tilt_confidence(self) -> float:
        """How much to trust ``body_tilt``, from 0 to 1."""
        tilt = self.body_tilt
        if tilt is None or abs(tilt) > MAX_BODY_TILT_DEG:
            return 0.0
        spread = self.asymmetry
        if spread is None:
            # Shoulders only: usable, but a torso turn can imitate a roll.
            return float(np.clip(0.45 * self.confidence, 0.0, 1.0))
        # Full agreement is the strong case; confidence falls off as the
        # shoulder and hip readings diverge.
        agreement = 1.0 - min(1.0, spread / MAX_ASYMMETRY_DEG)
        return float(np.clip(0.55 + 0.45 * agreement, 0.0, 1.0) * self.confidence)

    @property
    def spine_axis(self) -> float | None:
        """Direction the body points, in degrees, as a correction to apply.

        Zero means the subject's head is towards the top of the frame. The
        value is the rotation that would put them that way up again, so it
        composes with everything else here.
        """
        if not (self.has_shoulders and self.has_hips):
            return None
        sx, sy = self.midpoint(LEFT_SHOULDER, RIGHT_SHOULDER)
        hx, hy = self.midpoint(LEFT_HIP, RIGHT_HIP)
        return float(np.degrees(np.arctan2(sx - hx, hy - sy)))

    @property
    def upright_quarter_turn(self) -> int | None:
        """Which quarter turn would stand this body up: 0, 90, 180 or 270.

        This is the one orientation signal that survives a subject with their
        back to the camera, which no face-based method can do. It is read off
        the geometry rather than swept for, which is why it works where
        sweeping detector confidence does not: both this model and the face
        detector are deliberately tolerant of in-plane rotation, so their
        confidence barely moves as a frame is turned, but the landmarks
        themselves move with the body.
        """
        axis = self.spine_axis
        if axis is None:
            return None
        return int(round(axis / 90.0)) * 90 % 360

    @property
    def posture_summary(self) -> str:
        bits = [self.framing]
        if self.shoulder_tilt is not None:
            bits.append(f"shoulders {self.shoulder_tilt:+.1f}°")
        if self.hip_tilt is not None:
            bits.append(f"hips {self.hip_tilt:+.1f}°")
        if self.spine_tilt is not None:
            bits.append(f"spine {self.spine_tilt:+.1f}° from vertical")
        tilt = self.body_tilt
        if tilt is not None:
            bits.append(
                f"body roll {tilt:+.1f}° (confidence {self.body_tilt_confidence:.2f})"
            )
        elif self.asymmetry is not None:
            bits.append(
                f"leaning — shoulders and hips differ by {self.asymmetry:.1f}°"
            )
        return " · ".join(bits)


@dataclass
class PoseReport:
    """Result of analysing one frame."""

    poses: list[Pose] = field(default_factory=list)
    model_available: bool = True
    error: str | None = None

    @property
    def count(self) -> int:
        return len(self.poses)

    @property
    def main(self) -> Pose | None:
        """The pose a levelling decision should follow.

        The largest confident body: in a group shot the person nearest the
        camera is the subject, and a bystander at the edge of frame should not
        decide how the photograph is rotated.
        """
        if not self.poses:
            return None
        return max(
            self.poses,
            key=lambda p: (p.box[2] - p.box[0])
            * (p.box[3] - p.box[1])
            * max(p.confidence, 1e-3),
        )

    @property
    def body_tilt(self) -> tuple[float | None, float]:
        """``(tilt, confidence)`` from the main subject."""
        pose = self.main
        if pose is None:
            return None, 0.0
        return pose.body_tilt, pose.body_tilt_confidence

    @property
    def applicable_tilt(self) -> float | None:
        """The body tilt only when it is good enough to act on."""
        tilt, confidence = self.body_tilt
        if tilt is None or confidence < MIN_APPLY_CONFIDENCE:
            return None
        return tilt

    @property
    def quarter_turn(self) -> int | None:
        """Quarter turn that would stand the main subject up, if readable."""
        pose = self.main
        return None if pose is None else pose.upright_quarter_turn


# --- Person detector ---------------------------------------------------- #


def _person_anchors() -> np.ndarray:
    """SSD anchor centres for the 224x224 person detector.

    Three feature maps (28x28, 14x14 and 7x7) with 2, 2 and 6 anchors per
    cell, 2254 in total. Generated rather than tabulated -- the upstream demo
    ships the same numbers as a 2200-line literal, and this reproduces it
    exactly (verified element-wise against that table).
    """
    rows: list[tuple[float, float]] = []
    for grid, per_cell in ((28, 2), (14, 2), (7, 6)):
        for y in range(grid):
            for x in range(grid):
                centre = ((x + 0.5) / grid, (y + 0.5) / grid)
                rows.extend([centre] * per_cell)
    return np.array(rows, dtype=np.float32)


def _outputs_by_width(net, blob) -> dict[int, np.ndarray]:
    """Run a net and key its outputs by their last dimension.

    The ONNX files expose several outputs, and relying on their declared
    order means a graph re-export can silently swap scores for boxes. Their
    trailing dimensions are distinct, so keying on those is stable.
    """
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    if isinstance(outputs, np.ndarray):
        outputs = [outputs]
    keyed: dict[int, np.ndarray] = {}
    for out in outputs:
        keyed.setdefault(int(out.shape[-1]), out)
    return keyed


class _PersonDetector:
    """BlazePose's first stage: roughly where a body is."""

    def __init__(self, model: Path) -> None:
        self._net = cv2.dnn.readNet(str(model))
        self._anchors = _person_anchors()

    def detect(self, bgr: np.ndarray) -> np.ndarray:
        """Return rows of ``[x1, y1, x2, y2, 8 keypoint coords, score]``."""
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - 0.5) * 2.0  # the detector was trained on [-1, 1]

        ratio = min(_PERSON_INPUT / h, _PERSON_INPUT / w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        resized = cv2.resize(rgb, (new_w, new_h))
        pad_w, pad_h = _PERSON_INPUT - new_w, _PERSON_INPUT - new_h
        left, top = pad_w // 2, pad_h // 2
        padded = cv2.copyMakeBorder(
            resized,
            top,
            pad_h - top,
            left,
            pad_w - left,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        blob = np.transpose(padded, (2, 0, 1))[np.newaxis, ...]
        # Pad offsets are needed in original-image pixels, not input pixels.
        pad_bias = np.array([left / ratio, top / ratio], dtype=np.float32)

        keyed = _outputs_by_width(self._net, blob)
        if 12 not in keyed or 1 not in keyed:
            logger.warning("Unexpected person-detector outputs: %s", sorted(keyed))
            return np.empty((0, 13), dtype=np.float32)
        deltas = keyed[12][0]
        scores = keyed[1][0, :, 0].astype(np.float64)

        scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -100, 100)))
        scale = float(max(w, h))

        centre_delta = deltas[:, :2] / _PERSON_INPUT
        size_delta = deltas[:, 2:4] / _PERSON_INPUT
        xy1 = (centre_delta - size_delta / 2 + self._anchors) * scale
        xy2 = (centre_delta + size_delta / 2 + self._anchors) * scale
        boxes = np.concatenate([xy1, xy2], axis=1)
        boxes -= np.tile(pad_bias, 2)

        # NMSBoxes takes x, y, w, h -- feeding it corners (as the upstream
        # demo does) makes it suppress on the wrong geometry.
        wh = np.column_stack(
            [
                boxes[:, 0],
                boxes[:, 1],
                boxes[:, 2] - boxes[:, 0],
                boxes[:, 3] - boxes[:, 1],
            ]
        )
        keep = cv2.dnn.NMSBoxes(
            wh.tolist(),
            scores.astype(np.float32).tolist(),
            PERSON_SCORE_THRESHOLD,
            PERSON_NMS_THRESHOLD,
        )
        keep = np.asarray(keep).reshape(-1).astype(int)
        if keep.size == 0:
            return np.empty((0, 13), dtype=np.float32)

        keypoints = deltas[keep][:, 4:12].reshape(-1, 4, 2) / _PERSON_INPUT
        keypoints += self._anchors[keep][:, np.newaxis, :]
        keypoints = keypoints * scale - pad_bias

        return np.c_[
            boxes[keep].reshape(-1, 4),
            keypoints.reshape(-1, 8),
            scores[keep].reshape(-1, 1),
        ].astype(np.float32)


class _LandmarkModel:
    """BlazePose's second stage: 33 landmarks inside a person crop."""

    def __init__(self, model: Path) -> None:
        self._net = cv2.dnn.readNet(str(model))

    def estimate(self, bgr: np.ndarray, person: np.ndarray) -> Pose | None:
        prepared = self._prepare(bgr, person)
        if prepared is None:
            return None
        blob, crop_wh, angle, rotation, pad_bias = prepared

        keyed = _outputs_by_width(self._net, blob)
        if 195 not in keyed or 1 not in keyed:
            logger.warning("Unexpected pose-model outputs: %s", sorted(keyed))
            return None
        confidence = float(keyed[1].reshape(-1)[0])
        if confidence < POSE_CONF_THRESHOLD:
            return None

        landmarks = keyed[195][0].reshape(-1, 5).astype(np.float64)
        # Visibility and presence come out as logits.
        landmarks[:, 3:] = 1.0 / (1.0 + np.exp(-np.clip(landmarks[:, 3:], -100, 100)))

        world = None
        if 117 in keyed:
            world = keyed[117][0].reshape(-1, 3).astype(np.float64).copy()

        # Undo the letterbox scale, then the crop rotation, then the crop
        # offset, which puts the landmarks back in original-frame pixels.
        scale = np.asarray(crop_wh, dtype=np.float64) / _POSE_INPUT
        landmarks[:, :2] = (landmarks[:, :2] - _POSE_INPUT / 2.0) * scale

        coord_rotation = cv2.getRotationMatrix2D((0.0, 0.0), angle, 1.0)
        rotated = landmarks[:, :2] @ coord_rotation[:, :2]
        if world is not None:
            world[:, :2] = world[:, :2] @ coord_rotation[:, :2]

        rotation_part = np.array(
            [[rotation[0][0], rotation[1][0]], [rotation[0][1], rotation[1][1]]]
        )
        translation = np.array([rotation[0][2], rotation[1][2]])
        inverse = np.c_[
            rotation_part,
            [-rotation_part[0] @ translation, -rotation_part[1] @ translation],
        ]
        centre = np.array([crop_wh[0] / 2.0, crop_wh[1] / 2.0, 1.0])
        original_centre = np.array([centre @ inverse[0], centre @ inverse[1]])
        landmarks[:, :2] = rotated + original_centre + pad_bias

        body = landmarks[:33, :2]
        box = (
            float(body[:, 0].min()),
            float(body[:, 1].min()),
            float(body[:, 0].max()),
            float(body[:, 1].max()),
        )
        return Pose(landmarks=landmarks, confidence=confidence, box=box, world=world)

    def _prepare(self, bgr: np.ndarray, person: np.ndarray):
        """Crop a square around the body, rotate it upright, and letterbox it.

        The detector's keypoints are the mid-hip and a full-body point; the
        vector between them gives both the extent of the crop and the angle
        the landmark model expects its input to be rotated to.
        """
        keypoints = np.array(person[4:12], dtype=np.float64).reshape(-1, 2)
        mid_hip = keypoints[0].copy()
        full_body = keypoints[1].copy()

        span = float(np.linalg.norm(mid_hip - full_body))
        if span < 4.0:
            return None
        half = span * _BOX_ENLARGE
        full_box = np.array([mid_hip - half, mid_hip + half])

        h, w = bgr.shape[:2]
        clipped = full_box.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0, w)
        clipped[:, 1] = np.clip(clipped[:, 1], 0, h)
        x0, y0 = clipped[0].astype(int)
        x1, y1 = clipped[1].astype(int)
        if x1 - x0 < 8 or y1 - y0 < 8:
            # The body is (almost) outside the frame; nothing to measure.
            return None

        crop = bgr[y0:y1, x0:x1]
        left, top = (clipped[0] - full_box[0]).astype(int)
        right, bottom = (full_box[1] - clipped[1]).astype(int)
        crop = cv2.copyMakeBorder(
            crop,
            max(0, int(top)),
            max(0, int(bottom)),
            max(0, int(left)),
            max(0, int(right)),
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        if crop.size == 0:
            return None
        pad_bias = np.array(
            [x0 - max(0, int(left)), y0 - max(0, int(top))], dtype=np.float64
        )

        hip_local = mid_hip - pad_bias
        body_local = full_body - pad_bias
        radians = np.pi / 2 - np.arctan2(
            -(body_local[1] - hip_local[1]), body_local[0] - hip_local[0]
        )
        radians -= 2 * np.pi * np.floor((radians + np.pi) / (2 * np.pi))
        angle = float(np.rad2deg(radians))

        rotation = cv2.getRotationMatrix2D(
            (float(hip_local[0]), float(hip_local[1])), angle, 1.0
        )
        rotated = cv2.warpAffine(crop, rotation, (crop.shape[1], crop.shape[0]))

        blob = cv2.resize(rotated, (_POSE_INPUT, _POSE_INPUT), interpolation=cv2.INTER_AREA)
        blob = cv2.cvtColor(blob.astype(np.float32), cv2.COLOR_BGR2RGB) / 255.0
        return (
            blob[np.newaxis, ...],
            (crop.shape[1], crop.shape[0]),
            angle,
            rotation,
            pad_bias,
        )


# --- Public analyzer ---------------------------------------------------- #


class PoseAnalyzer:
    """Detects bodies and reports their posture.

    Both nets are created lazily on first use and then reused. ``available``
    says whether the models could be fetched at all, so callers skip
    pose-dependent features rather than handling exceptions.
    """

    def __init__(self, download: bool = True, max_people: int = 6) -> None:
        self.max_people = max_people
        self._person: _PersonDetector | None = None
        self._landmarks: _LandmarkModel | None = None
        self._paths = model_store.ensure_models(MODEL_SPECS, download=download)
        self._load_error: str | None = None if self._paths else "Pose models unavailable"

    @property
    def available(self) -> bool:
        return self._paths is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @staticmethod
    def download_mb() -> float:
        """Megabytes still to fetch before pose features can run."""
        return model_store.total_download_mb(MODEL_SPECS)

    @staticmethod
    def is_cached() -> bool:
        return all(model_store.is_cached(s) for s in MODEL_SPECS)

    def _nets(self):
        if self._paths is None:
            return None
        if self._person is None or self._landmarks is None:
            try:
                self._person = _PersonDetector(self._paths[PERSON_SPEC.filename])
                self._landmarks = _LandmarkModel(self._paths[POSE_SPEC.filename])
            except Exception as exc:
                self._load_error = f"Could not load the pose models: {exc}"
                logger.warning(self._load_error)
                self._paths = None
                return None
        return self._person, self._landmarks

    def analyze(self, image: np.ndarray) -> PoseReport:
        """Find and measure the bodies in a BGR frame."""
        if not self.available:
            return PoseReport(model_available=False, error=self._load_error)
        if image is None or image.size == 0:
            return PoseReport(error="empty image")

        nets = self._nets()
        if nets is None:
            return PoseReport(model_available=False, error=self._load_error)
        person_net, landmark_net = nets

        if image.ndim == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            bgr = image

        h, w = bgr.shape[:2]
        scale = 1.0
        if max(h, w) > ANALYSIS_MAX_SIDE:
            scale = ANALYSIS_MAX_SIDE / float(max(h, w))
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        try:
            people = person_net.detect(bgr)
        except Exception as exc:
            logger.warning("Person detection failed: %s", exc)
            return PoseReport(error=f"Person detection failed: {exc}")

        poses: list[Pose] = []
        for row in people[: self.max_people]:
            try:
                pose = landmark_net.estimate(bgr, row)
            except Exception as exc:
                logger.debug("Pose estimation failed for one body: %s", exc)
                continue
            if pose is not None:
                poses.append(pose)

        if scale != 1.0 and poses:
            inverse = 1.0 / scale
            for pose in poses:
                pose.landmarks[:, :2] *= inverse
                pose.box = tuple(float(v * inverse) for v in pose.box)

        return PoseReport(poses=poses)

    def analyze_path(self, path: str | Path) -> PoseReport:
        """Analyse an image file, including RAW and HEIC."""
        if not self.available:
            return PoseReport(model_available=False, error=self._load_error)

        from src.engine import fast_preview

        path = Path(path)
        preview = fast_preview.load_preview(path, max_side=ANALYSIS_MAX_SIDE)
        if preview is None:
            return PoseReport(error=f"Could not read {path.name}")
        # Landmarks come back in this preview's pixels.
        return self.analyze(cv2.cvtColor(preview.image, cv2.COLOR_RGB2BGR))

    def body_tilt(self, image: np.ndarray) -> tuple[float | None, float]:
        """Camera roll from the main subject's body: ``(degrees, confidence)``."""
        return self.analyze(image).body_tilt
