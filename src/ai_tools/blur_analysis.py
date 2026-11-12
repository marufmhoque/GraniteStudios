"""Blur analysis: how blurred a frame is, and what kind of blur it is.

A single sharpness number cannot tell a photographer what went wrong. Missing
focus and camera shake look alike on a Laplacian-variance score, but they
mean different things: a missed-focus frame is gone, whereas a slightly
shaken one is often still usable once sharpened. This module separates them.

How the two are told apart:

* **Out of focus** attenuates detail in every direction equally. Edges stay
  where they are, they just soften.
* **Motion blur** smears along one axis. Edges running parallel to the
  movement are destroyed, while edges crossing it survive, so the image's
  gradients bunch up in one orientation.

That bunching is measured by sweeping the gradient energy across every
direction. A smear leaves a pronounced dip in that profile at the angle of
travel; defocus leaves it comparatively flat. The depth of the dip, compared
against its own perpendicular, is the *directionality* score below, and the
angle of the dip is the direction of travel.

Directionality alone is not enough, because a perfectly sharp photograph of
a fence or a horizon is also strongly directional -- in testing, one sharp
reference frame scored more directional than its own defocused copy. So it
is consulted only once overall sharpness is already low enough for the frame
to be in question.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Work at this size so a 50MP frame and a phone snap score comparably;
# Laplacian variance is resolution-dependent otherwise.
ANALYSIS_MAX_SIDE = 1024

# Verdicts
SHARP = "sharp"
SOFT = "soft"
OUT_OF_FOCUS = "out_of_focus"
MOTION_BLUR = "motion_blur"

# How hard to be on a frame. "Aggressive" is for reclaiming storage and will
# flag mild softness; "conservative" only flags frames that are clearly
# unusable. The middle setting is the default.
#
#   blurry_below     - sharpness under this counts as blurred
#   soft_below       - under this is "soft", worth a look, not condemned
#   rescue_floor     - below this, sharpening cannot bring detail back, so
#                      offering a one-tap fix would be dishonest
STRICTNESS = {
    "aggressive": {"blurry_below": 180.0, "soft_below": 420.0, "rescue_floor": 45.0},
    "balanced": {"blurry_below": 95.0, "soft_below": 230.0, "rescue_floor": 30.0},
    "conservative": {"blurry_below": 42.0, "soft_below": 110.0, "rescue_floor": 18.0},
}
DEFAULT_STRICTNESS = "balanced"

# Above this directionality a low-sharpness frame is read as a directional
# smear rather than missed focus. Calibrated against synthetic motion and
# defocus blur on two reference images; the separation is real but not wide
# (0.45 vs 0.39 in the closest pair), so borderline frames are worth a look.
MOTION_DIRECTIONALITY = 0.42


@dataclass
class BlurReport:
    """What the analysis found for one frame."""

    sharpness: float
    directionality: float
    motion_angle: float | None
    verdict: str
    is_blurry: bool
    recoverable: bool
    strictness: str = DEFAULT_STRICTNESS
    error: str | None = None

    @property
    def is_sharp(self) -> bool:
        return self.verdict == SHARP

    @property
    def blur_kind(self) -> str:
        """Plain-language description, for the UI."""
        return {
            SHARP: "Sharp",
            SOFT: "Slightly soft",
            OUT_OF_FOCUS: "Out of focus",
            MOTION_BLUR: "Motion blur",
        }.get(self.verdict, self.verdict)

    @property
    def summary(self) -> str:
        bits = [self.blur_kind, f"sharpness {self.sharpness:,.0f}"]
        if self.verdict == MOTION_BLUR and self.motion_angle is not None:
            bits.append(f"{self.motion_angle:.0f}° smear")
        if self.recoverable:
            bits.append("sharpening may help")
        return " · ".join(bits)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if max(h, w) > ANALYSIS_MAX_SIDE:
        scale = ANALYSIS_MAX_SIDE / float(max(h, w))
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return gray


def sharpness_score(image: np.ndarray) -> float:
    """Laplacian variance: how much high-frequency detail survives."""
    gray = _to_gray(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def directional_blur(image: np.ndarray, step: int = 5) -> tuple[float, float]:
    """Return ``(directionality, motion_angle_degrees)``.

    Measures gradient energy as a function of direction. A motion smear
    destroys detail along the direction of travel while leaving detail
    across it intact, so the energy profile has a pronounced dip at the
    smear angle. Defocus attenuates every direction together, leaving the
    profile comparatively flat.

    Directionality is ``1 - min_energy / perpendicular_energy``: comparing
    the dip against its own perpendicular, rather than against the profile
    maximum, keeps the measure meaningful on images that are naturally
    lopsided.

    Note this cannot stand alone. A sharp photograph of a fence or a horizon
    is strongly directional too -- in testing, a sharp frame scored more
    directional than its own defocused copy. Only call it on a frame whose
    sharpness is already low enough to be in question.
    """
    gray = _to_gray(image).astype(np.float32) / 255.0
    # Light smoothing first, so sensor noise does not read as structure.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    angles = np.arange(0, 180, step)
    energy = np.empty(len(angles), dtype=np.float64)
    for index, degrees in enumerate(angles):
        radians = np.radians(float(degrees))
        directional = gx * np.cos(radians) + gy * np.sin(radians)
        energy[index] = float(np.mean(directional * directional))

    if not np.isfinite(energy).all() or energy.max() <= 1e-12:
        return 0.0, 0.0

    low = int(np.argmin(energy))
    perpendicular = energy[(low + len(angles) // 2) % len(angles)]
    directionality = float(np.clip(1.0 - energy[low] / max(perpendicular, 1e-12), 0.0, 1.0))

    # The measurement runs in image coordinates, where y increases downward,
    # so it comes out mirrored relative to the angle convention cv2's
    # rotation helpers use. Reflecting here keeps a reported "30 degrees"
    # meaning the same thing as a 30-degree rotation elsewhere in the app.
    motion_angle = float((180 - angles[low]) % 180)
    return directionality, motion_angle


def analyze(
    image: np.ndarray | str | Path,
    strictness: str = DEFAULT_STRICTNESS,
) -> BlurReport:
    """Score one frame and classify what kind of blur it has, if any."""
    thresholds = STRICTNESS.get(strictness, STRICTNESS[DEFAULT_STRICTNESS])

    if isinstance(image, (str, Path)):
        path = Path(image)
        array = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if array is None:
            # RAW and HEIC need the project loader, not cv2.imread.
            try:
                from src.engine.universal_loader import UniversalLoader

                rgb = UniversalLoader().load_array(path)
                array = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception as exc:
                return BlurReport(
                    sharpness=0.0,
                    directionality=0.0,
                    motion_angle=None,
                    verdict=OUT_OF_FOCUS,
                    is_blurry=False,
                    recoverable=False,
                    strictness=strictness,
                    error=f"Could not read {path.name}: {exc}",
                )
    else:
        array = image

    if array is None or array.size == 0:
        return BlurReport(0.0, 0.0, None, OUT_OF_FOCUS, False, False, strictness, "empty image")

    sharpness = sharpness_score(array)
    directionality, angle = directional_blur(array)

    if sharpness >= thresholds["soft_below"]:
        verdict = SHARP
    elif sharpness >= thresholds["blurry_below"]:
        verdict = SOFT
    elif directionality >= MOTION_DIRECTIONALITY:
        verdict = MOTION_BLUR
    else:
        verdict = OUT_OF_FOCUS

    is_blurry = verdict in (OUT_OF_FOCUS, MOTION_BLUR)

    # Worth offering a sharpen only when there is detail left to lift. Below
    # the floor the frame has no high-frequency content to recover, and a
    # one-tap "fix" would just add halos to mush.
    recoverable = (
        verdict in (SOFT, MOTION_BLUR, OUT_OF_FOCUS)
        and sharpness >= thresholds["rescue_floor"]
        and verdict != SHARP
    )

    return BlurReport(
        sharpness=sharpness,
        directionality=directionality,
        motion_angle=angle if verdict == MOTION_BLUR else None,
        verdict=verdict,
        is_blurry=is_blurry,
        recoverable=recoverable,
        strictness=strictness,
    )


def sharpen(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Unsharp-mask a frame — the "rescue" offered beside a delete suggestion.

    This is conventional sharpening, not deblurring: it raises local contrast
    at edges that are still present. It genuinely rescues mild softness. It
    cannot reconstruct detail that motion or a missed focus destroyed, and no
    amount of strength will — that is why ``recoverable`` gates the offer.
    """
    strength = float(np.clip(strength, 0.0, 3.0))
    if strength == 0.0:
        return image.copy()

    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def analyze_many(paths, strictness: str = DEFAULT_STRICTNESS, progress=None) -> dict:
    """Analyse a list of paths, returning ``{path_string: BlurReport}``."""
    results: dict[str, BlurReport] = {}
    total = len(paths)
    for index, path in enumerate(paths, start=1):
        results[str(path)] = analyze(path, strictness=strictness)
        if progress:
            progress(index, total)
    return results
