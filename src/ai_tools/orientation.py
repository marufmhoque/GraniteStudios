"""Auto-levelling: work out which way up a photograph should be.

Two independent signals, because neither is sufficient alone.

**Which way is up** is decided by the face detector. It will not recognise
an upside-down face at all, so each quarter turn is tried and the one the
detector responds to wins. This is reliable: every 90, 180 and 270 degree
turn was recovered correctly in testing.

**How far from level** is decided by the scene. Walls, door frames and
horizons are long straight lines that ought to sit square to the frame;
measuring how far they have drifted gives the tilt. Round-tripping a known
tilt through detection and correction leaves well under a degree.

**Which way is up, when there is no face.** A subject with their back to
the camera defeats the face detector entirely. The pose model fills that gap:
the direction from the hips to the shoulders is read straight off the
landmarks and rounded to a quarter turn. Every turn of a full-body frame was
recovered correctly this way in testing.

**Why head roll is still not used.** The obvious approach -- take the line
between the detected eyes and level it -- does not work with the face
detector, and the measurements are worth keeping so nobody tries again:
rotating a portrait through 24 degrees moved the measured eye line by about
5, because YuNet regresses landmarks towards an upright configuration.
Searching for the rotation of peak detection confidence fails for the same
underlying reason -- the detector is deliberately tolerant of in-plane
rotation, so confidence stayed flat across the entire plus or minus 20 degree
sweep. The pose model behaves the same way when swept (its confidence sat at
1.00 at all four quarter turns), which is exactly why the body signal above
reads the landmark geometry instead of sweeping for a peak.

**Levelling by the body.** Shoulders and hips give a second tilt estimate
for frames with no usable lines. It is only applied when both are visible and
agree: correcting by that angle and re-measuring leaves 0.53 degrees on
average. Shoulders alone leave 2.90 degrees and up to 9.54, so a
shoulders-only reading is reported but never applied -- see
``pose_analysis.MIN_APPLY_CONFIDENCE``.

**What is still missing.** There is no generative fill: rotating a frame
leaves wedge-shaped gaps at the corners, and the honest options are to crop
them away or leave them, both of which this module does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Below this a tilt is not worth correcting: re-encoding and cropping a
# frame to fix a third of a degree costs more than it gains.
MIN_CORRECTION_DEG = 0.75

# Beyond this, a "level the horizon" correction is almost certainly a
# misread rather than a genuinely tilted photograph.
MAX_FINE_CORRECTION_DEG = 20.0

ANALYSIS_MAX_SIDE = 1400


@dataclass
class OrientationResult:
    """What the detector concluded about a frame."""

    coarse: int = 0
    fine: float = 0.0
    method: str = "none"
    confidence: float = 0.0
    face_count: int = 0
    line_angle: float | None = None
    body_angle: float | None = None
    body_confidence: float = 0.0
    pose_count: int = 0
    framing: str | None = None
    coarse_method: str = "none"

    @property
    def total_angle(self) -> float:
        """Degrees to rotate the frame counter-clockwise to set it upright."""
        return float(self.coarse + self.fine)

    @property
    def needs_rotation(self) -> bool:
        return self.coarse != 0 or abs(self.fine) >= MIN_CORRECTION_DEG

    @property
    def summary(self) -> str:
        if not self.needs_rotation:
            return "Already upright"
        bits = []
        if self.coarse:
            bits.append(f"{self.coarse}° turn")
        if abs(self.fine) >= MIN_CORRECTION_DEG:
            bits.append(f"{self.fine:+.1f}° tilt")
        return f"{', '.join(bits)} ({self.method}, confidence {self.confidence:.2f})"


def _prepare(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) > ANALYSIS_MAX_SIDE:
        scale = ANALYSIS_MAX_SIDE / float(max(h, w))
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return image


def rotate_image(image: np.ndarray, angle: float, crop: bool = True) -> np.ndarray:
    """Rotate counter-clockwise by ``angle`` degrees.

    With ``crop`` the result is cut back to the largest upright rectangle
    that contains no blank corner, keeping the centre of the frame. Without
    it the canvas grows and the corners are left black -- there is no
    generative fill here, so those are the two honest choices.
    """
    if angle % 360 == 0:
        return image.copy()

    h, w = image.shape[:2]
    centre = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)

    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    matrix[0, 2] += new_w / 2.0 - centre[0]
    matrix[1, 2] += new_h / 2.0 - centre[1]

    rotated = cv2.warpAffine(
        image, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0)
    )
    if not crop:
        return rotated

    crop_w, crop_h = _largest_inscribed_rect(w, h, angle)
    if crop_w < 8 or crop_h < 8:
        return rotated
    cx, cy = new_w // 2, new_h // 2
    x0 = max(0, cx - crop_w // 2)
    y0 = max(0, cy - crop_h // 2)
    return rotated[y0 : y0 + crop_h, x0 : x0 + crop_w]


def _largest_inscribed_rect(w: int, h: int, angle: float) -> tuple[int, int]:
    """Largest upright rectangle fitting inside a rotated w x h rectangle.

    Standard result: for a rotation of a rectangle, the biggest axis-aligned
    rectangle inside it either touches all four sides (the wide case) or is
    limited by the shorter side (the narrow case).
    """
    radians = np.radians(abs(angle) % 180)
    if radians > np.pi / 2:
        radians = np.pi - radians
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    long_side, short_side = (w, h) if width_is_longer else (h, w)

    sin_a, cos_a = abs(np.sin(radians)), abs(np.cos(radians))
    if short_side <= 2.0 * sin_a * cos_a * long_side or abs(sin_a - cos_a) < 1e-10:
        # Half-constrained: the solution touches the shorter side twice.
        x = 0.5 * short_side
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr = (w * cos_a - h * sin_a) / cos_2a
        hr = (h * cos_a - w * sin_a) / cos_2a
    return max(0, int(wr)), max(0, int(hr))


def _blank_mask(gray: np.ndarray) -> np.ndarray | None:
    """Mask covering filled-in blank areas, or None if there are none.

    A frame that has been rotated before, letterboxed, or scanned carries
    wedges and bars of pure black. Their boundaries are the straightest lines
    in the picture and they sit at precisely the angle that produced them, so
    an edge detector reads them as overwhelming evidence that the frame is
    tilted by exactly the amount it was already corrected by.

    Only near-black is treated this way, and the mask is grown a little so it
    covers the transition rather than just the flat interior. A photograph
    with a genuinely black backdrop loses the edges along that backdrop, which
    is a fair trade: it discards evidence rather than inventing it.
    """
    blank = (gray <= 2).astype(np.uint8)
    if float(blank.mean()) < 0.002:
        return None
    return cv2.dilate(blank, np.ones((7, 7), np.uint8), iterations=1)


def line_tilt(image: np.ndarray) -> tuple[float | None, float]:
    """Scene tilt from long straight edges that ought to be square.

    Every detected segment is folded into the -45..45 range: a line is
    equally good evidence whether it is a drifting vertical or a drifting
    horizontal, since both should be square to the frame.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blank = _blank_mask(gray)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 180, apertureSize=3)
    if blank is not None:
        # The border of a blank region is a flawless straight line at exactly
        # the angle of whatever rotation produced it, so leaving it in means
        # measuring our own padding and calling it the scene. See
        # ``_blank_mask`` for why this is worth the edges it costs.
        edges[blank > 0] = 0

    min_length = max(40, int(min(gray.shape[:2]) * 0.22))
    segments = cv2.HoughLinesP(
        edges, 1, np.pi / 360, threshold=70, minLineLength=min_length, maxLineGap=12
    )
    if segments is None or len(segments) == 0:
        return None, 0.0

    # OpenCV 4 returns (N, 1, 4) here; OpenCV 5 returns (N, 4). Accept both.
    segments = np.asarray(segments)
    if segments.ndim == 3:
        segments = segments[:, 0, :]

    deviations, lengths = [], []
    for x1, y1, x2, y2 in segments:
        length = float(np.hypot(x2 - x1, y2 - y1))
        angle = np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1)))
        # Fold onto "how far from square", in -45..45.
        deviation = ((angle + 45.0) % 90.0) - 45.0
        # Segments already close to square carry the signal; a 40-degree
        # diagonal is a real diagonal, not a tilted wall.
        if abs(deviation) <= 20.0:
            deviations.append(deviation)
            lengths.append(length)

    if len(deviations) < 4:
        return None, 0.0

    deviations = np.array(deviations)
    lengths = np.array(lengths)
    # Length-weighted median: long architectural edges should outvote short
    # noisy ones.
    order = np.argsort(deviations)
    deviations, lengths = deviations[order], lengths[order]
    cumulative = np.cumsum(lengths)
    median_angle = float(deviations[int(np.searchsorted(cumulative, cumulative[-1] / 2.0))])

    agreement = float(np.sum(lengths[np.abs(deviations - median_angle) <= 2.0]) / cumulative[-1])
    # Returned as "rotate by this to square the frame up", matching the sign
    # convention rotate_image() takes, so the two compose directly.
    return float(median_angle), float(np.clip(agreement, 0.0, 1.0))


def coarse_orientation(analyzer, image: np.ndarray) -> tuple[int, float]:
    """Find the quarter turn that puts faces the right way up.

    A face detector simply does not fire on an upside-down face, so this
    cannot be measured -- each of the four turns has to be tried and the
    one the detector likes best wins.
    """
    if analyzer is None or not analyzer.available:
        return 0, 0.0

    best_rotation, best_score = 0, 0.0
    for rotation in (0, 90, 180, 270):
        candidate = image if rotation == 0 else rotate_image(image, rotation, crop=False)
        faces = analyzer.detect(candidate, measure_eyes=False)
        score = sum(f.confidence for f in faces)
        if score > best_score + 1e-6:
            best_rotation, best_score = rotation, score

    if best_score <= 0.0:
        return 0, 0.0
    return best_rotation, float(min(1.0, best_score))


def _pose_report(pose_analyzer, image):
    """Analyse ``image`` for bodies, tolerating an unavailable pose model."""
    if pose_analyzer is None or not pose_analyzer.available:
        return None
    try:
        report = pose_analyzer.analyze(image)
    except Exception as exc:  # a missing model must not break levelling
        logger.debug("Pose analysis failed: %s", exc)
        return None
    return report if report.count else None


def default_pose_analyzer():
    """A pose analyzer, but only if its models are already on disk.

    Levelling must not stall mid-batch on a 17 MB download that the user
    never asked for, so this never fetches. A caller that wants the body
    engine switched on passes its own analyzer, created with downloading
    enabled, after telling the user what that costs.
    """
    try:
        from src.ai_tools.pose_analysis import PoseAnalyzer

        if not PoseAnalyzer.is_cached():
            return None
        return PoseAnalyzer(download=False)
    except Exception:
        return None


def detect(image: np.ndarray, analyzer=None, pose_analyzer=None) -> OrientationResult:
    """Work out how far from upright a frame is.

    ``analyzer`` is a face analyzer and ``pose_analyzer`` a body one; either
    may be omitted, in which case a cached one is used if there is one. With
    neither, only the scene-line engine runs.
    """
    if image is None or image.size == 0:
        return OrientationResult()

    if analyzer is None:
        try:
            from src.ai_tools.face_analysis import FaceAnalyzer

            analyzer = FaceAnalyzer()
        except Exception:
            analyzer = None

    if pose_analyzer is None:
        pose_analyzer = default_pose_analyzer()

    small = _prepare(image)

    # --- which way up ---------------------------------------------------- #
    # Faces answer this best when there is one, because the detector simply
    # does not fire on an upside-down face. When there is not -- a subject
    # facing away, or too small a face -- the body answers it instead.
    coarse, coarse_conf = coarse_orientation(analyzer, small)
    coarse_method = "faces" if coarse_conf > 0.0 else "none"

    report = _pose_report(pose_analyzer, small)
    if coarse_conf <= 0.0 and report is not None:
        turn = report.quarter_turn
        if turn is not None:
            main = report.main
            coarse, coarse_method = turn, "body"
            coarse_conf = float(main.confidence) if main is not None else 0.0

    upright = small if coarse == 0 else rotate_image(small, coarse, crop=False)

    faces = 0
    if analyzer is not None and analyzer.available:
        faces = len(analyzer.detect(upright, measure_eyes=False))

    # The tilt has to be measured on the frame as it will be left, so a
    # quarter turn means re-reading the body rather than rotating the angle.
    if coarse and report is not None:
        report = _pose_report(pose_analyzer, upright)

    # --- how far from level ---------------------------------------------- #
    l_angle, l_conf = line_tilt(upright)
    body_angle, body_conf = (None, 0.0) if report is None else report.body_tilt
    applicable_body = None if report is None else report.applicable_tilt

    # Lines first where there are any: they measure the camera against the
    # world, whereas a body can genuinely be leaning. In practice the two
    # rarely compete -- the portraits in testing yielded no usable lines at
    # all, and the frames with strong lines had no usable body.
    if l_angle is not None and l_conf > 0.0:
        fine, method, confidence = float(l_angle), "lines", float(l_conf)
    elif applicable_body is not None:
        fine, method, confidence = float(applicable_body), "body", float(body_conf)
    else:
        fine, method, confidence = 0.0, "none", 0.0

    if abs(fine) > MAX_FINE_CORRECTION_DEG:
        # A plausibility cap, applied whichever engine produced the reading.
        # The body engine stayed accurate out to 24 degrees in testing, but a
        # correction that large is more often a misread than a rolled camera.
        logger.debug("Ignoring implausible tilt of %.1f degrees", fine)
        fine, method, confidence = 0.0, "none", 0.0

    if coarse and confidence == 0.0:
        confidence = coarse_conf

    main = None if report is None else report.main
    return OrientationResult(
        coarse=coarse,
        fine=float(fine),
        method=method,
        confidence=float(confidence),
        face_count=faces,
        line_angle=l_angle,
        body_angle=body_angle,
        body_confidence=float(body_conf),
        pose_count=0 if report is None else report.count,
        framing=None if main is None else main.framing,
        coarse_method=coarse_method,
    )


def auto_rotate(
    image: np.ndarray, analyzer=None, pose_analyzer=None, crop: bool = True
) -> tuple[np.ndarray, OrientationResult]:
    """Detect and apply the correction in one step."""
    result = detect(image, analyzer=analyzer, pose_analyzer=pose_analyzer)
    if not result.needs_rotation:
        return image.copy(), result
    return rotate_image(image, result.total_angle, crop=crop), result
