"""Masks: which pixels belong to the subject, the sky, the skin, the teeth.

A mask is a float image from 0 (not selected) to 1 (selected), the same
size as the photograph. Every edit that should apply to only part of a
frame goes through one.

Where each mask comes from, and how far it can be trusted
---------------------------------------------------------
There is no single model here that labels every pixel with a class. Each
kind of mask is built from whatever actually knows about that thing, and
the result says which method produced it.

==========  ===========================================  ==================
Mask        Method                                       Standing
==========  ===========================================  ==================
subject     U2Net salient-object model (if on disk),     Model-based
            else the person model
background  the complement of subject                    Model-based
person      PP-HumanSeg, a small portrait model          Model-based, coarse
                                                         (it works at 192 px)
skin        face-mesh outline, minus eyes, brows and     Geometry + colour
            lips, grown to neck and ears by a skin-
            colour model fitted to that face
lips        face-mesh outline                            Geometry
teeth       inside the inner lips, where it is bright    Geometry + colour
            and unsaturated
eyes, iris  face-mesh outline and iris ring              Geometry
click       EfficientSAM from the points clicked         Model-based
hair        EfficientSAM seeded above each forehead,     **Partial** --
            minus the face                               unmeasured
sky         bright, smooth, blue-or-grey region          **Partial** --
            touching the top of the frame                a heuristic
==========  ===========================================  ==================

"Partial" means it runs and is often useful, and that nobody has measured
how often. Hair depends on a click-to-select model guessing what was meant
from a few seed points; sky is colour and position, so a pale wall at the
top of a frame can be taken for it. Both are offered as a starting point to
refine with the brush, not as a finished selection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.ai_tools import face_mesh as fm
from src.ai_tools import model_store
from src.ai_tools.model_store import ModelSpec

logger = logging.getLogger(__name__)

HUMANSEG_SPEC = ModelSpec(
    filename="human_segmentation_pphumanseg_2023mar.onnx",
    sha256="552d8a984054e59b5d773d24b9b12022b22046ceb2bbc4c9aaeaceb36a9ddf24",
    size=6163938,
    zoo_dir="human_segmentation_pphumanseg",
    label="the person-segmentation model",
)
SAM_SPEC = ModelSpec(
    filename="image_segmentation_efficientsam_ti_2025april.onnx",
    sha256="4eb496e0a7259d435b49b66faf1754aa45a5c382a34558ddda9a8c6fe5915d77",
    size=48312857,
    zoo_dir="image_segmentation_efficientsam",
    label="the click-to-select model",
)
U2NET_PATH = Path.home() / ".u2net" / "u2net.onnx"

KINDS = ("subject", "background", "person", "skin", "lips", "teeth", "eyes", "iris", "hair", "sky")
PARTIAL_KINDS = ("hair", "sky")

RIGHT_BROW = (70, 63, 105, 66, 107, 55, 65, 52, 53, 46)
LEFT_BROW = (300, 293, 334, 296, 336, 285, 295, 282, 283, 276)
RIGHT_IRIS = (468, 469, 470, 471, 472)      # centre, then four rim points
LEFT_IRIS = (473, 474, 475, 476, 477)

SAM_SIDE = 1024
SAM_MAX_POINTS = 6
ANALYSIS_SIDE = 1280        # masks are built at this size and scaled up


@dataclass
class MaskResult:
    """A mask and where it came from."""

    mask: np.ndarray            # H x W float32, 0..1
    method: str
    note: str | None = None     # set when the mask is empty or approximate

    @property
    def coverage(self) -> float:
        return float(self.mask.mean()) if self.mask.size else 0.0

    @property
    def is_empty(self) -> bool:
        return self.coverage < 1e-4


def _empty(shape, method: str, note: str) -> MaskResult:
    return MaskResult(np.zeros(shape[:2], np.float32), method, note)


def feather(mask: np.ndarray, radius: float) -> np.ndarray:
    """Soften a mask's edge. ``radius`` is in pixels."""
    if radius <= 0.3:
        return mask.astype(np.float32)
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), float(radius))


def _fill(shape, polygons, value: float = 1.0) -> np.ndarray:
    canvas = np.zeros(shape[:2], np.float32)
    cv2.fillPoly(canvas, [np.asarray(p, np.int32) for p in polygons], float(value), lineType=cv2.LINE_AA)
    return canvas


# --------------------------------------------------------------------- #
# Face-derived masks
# --------------------------------------------------------------------- #

@dataclass
class FaceRegions:
    """One face's landmarks, with the outlines masks are cut from."""

    landmarks: np.ndarray       # 478 x 3 in image pixels
    box: tuple[int, int, int, int]

    def outline(self, indices) -> np.ndarray:
        return np.round(self.landmarks[list(indices), :2]).astype(np.int32)

    @property
    def eye_span(self) -> float:
        a, b = self.landmarks[fm.RIGHT_EYE_OUTER, :2], self.landmarks[fm.LEFT_EYE_OUTER, :2]
        return float(np.linalg.norm(a - b))

    def iris(self, indices) -> tuple[tuple[int, int], int]:
        points = self.landmarks[list(indices), :2]
        centre = points[0]
        radius = float(np.mean(np.linalg.norm(points[1:] - centre, axis=1)))
        return (int(round(centre[0])), int(round(centre[1]))), max(1, int(round(radius)))


def skin_mask(rgb: np.ndarray, face: FaceRegions) -> np.ndarray:
    """Skin for one face: the face itself, grown to matching skin nearby.

    The mesh gives the face outline exactly, but skin does not stop at the
    jaw. A colour model is fitted to this face's own skin -- so it adapts to
    the person and the light rather than assuming what skin looks like --
    and anything nearby that matches it is added.
    """
    h, w = rgb.shape[:2]
    span = max(face.eye_span, 8.0)

    core = _fill(rgb.shape, [face.outline(fm.FACE_OVAL)])
    holes = [face.outline(r) for r in (fm.RIGHT_EYE_RING, fm.LEFT_EYE_RING, fm.OUTER_LIPS, RIGHT_BROW, LEFT_BROW)]
    cut = _fill(rgb.shape, holes)
    grow = max(1, int(span * 0.06))
    cut = cv2.dilate(cut, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    face_skin = np.clip(core - cut, 0.0, 1.0)

    # Fit the colour model on the confidently-skin interior of the face.
    shrink = max(1, int(span * 0.10))
    sure = cv2.erode(face_skin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * shrink + 1, 2 * shrink + 1))) > 0.5
    if sure.sum() < 30:
        return face_skin

    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    chroma = ycrcb[:, :, 1:3]
    samples = chroma[sure]
    mean = samples.mean(axis=0)
    cov = np.cov(samples.T) + np.eye(2, dtype=np.float32) * 4.0
    inv = np.linalg.inv(cov)
    delta = chroma - mean
    distance2 = np.einsum("hwi,ij,hwj->hw", delta, inv, delta)
    similar = np.exp(-distance2 / (2.0 * 2.5 ** 2)).astype(np.float32)

    # Skin has colour. A white wall or grey backdrop sits at the neutral
    # point, which on a pale face is not far from the skin mean, so anything
    # much closer to neutral than this face's skin is ruled out.
    neutral = np.array([128.0, 128.0], dtype=np.float32)
    skin_chroma = float(np.linalg.norm(mean - neutral))
    if skin_chroma < 4.0:
        # A black-and-white photograph: every pixel has the same (absent)
        # colour, so colour cannot tell skin from anything. Geometry only.
        return face_skin
    coloured = np.linalg.norm(chroma - neutral, axis=2) >= 0.5 * skin_chroma
    similar = similar * coloured

    # Only near this face: forehead to collarbone, ear to ear and a bit.
    x, y, bw, bh = face.box
    reach = np.zeros((h, w), np.float32)
    x0, x1 = int(max(0, x - 0.6 * bw)), int(min(w, x + 1.6 * bw))
    y0, y1 = int(max(0, y - 0.35 * bh)), int(min(h, y + 2.0 * bh))
    reach[y0:y1, x0:x1] = 1.0
    nearby = similar * reach * (1.0 - cut)
    nearby = cv2.morphologyEx(nearby, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # Keep only what is joined to the face: a neck, an ear, a shoulder. A
    # skin-coloured patch elsewhere in the frame is not this person's skin.
    grown = ((nearby > 0.72) | (face_skin > 0.5)).astype(np.uint8)
    _, labels = cv2.connectedComponents(grown)
    attached = np.isin(labels, np.unique(labels[face_skin > 0.5]))
    nearby = nearby * attached
    # Within two standard deviations of this face's skin colour.
    return np.clip(np.maximum(face_skin, np.where(nearby > 0.72, nearby, 0.0)), 0.0, 1.0)


def teeth_mask(rgb: np.ndarray, face: FaceRegions) -> np.ndarray:
    """Teeth: bright, unsaturated pixels between the lips."""
    mouth = _fill(rgb.shape, [face.outline(fm.INNER_LIPS)])
    inside = mouth > 0.5
    if inside.sum() < 12:
        return np.zeros(rgb.shape[:2], np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation, value = hsv[:, :, 1] / 255.0, hsv[:, :, 2] / 255.0
    # Bright relative to this mouth, so a dim photograph still works.
    floor = max(0.35, float(np.percentile(value[inside], 55)))
    tooth = (value >= floor) & (saturation <= 0.45)
    return (mouth * tooth).astype(np.float32)


def _face_regions(rgb: np.ndarray, detector, mesh) -> list[FaceRegions]:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    regions = []
    for face in detector.detect_reduced(bgr, 800):
        measured = mesh.analyze_face(bgr, face)
        if measured is not None:
            regions.append(FaceRegions(measured.landmarks, face.box))
    return regions


# --------------------------------------------------------------------- #
# Sky
# --------------------------------------------------------------------- #

def sky_mask(rgb: np.ndarray) -> np.ndarray:
    """A heuristic sky selection: see the module docstring for its limits.

    Sky is taken to be a region that touches the top edge, is smooth, and
    is either blue or bright and nearly colourless (overcast). Candidate
    pixels are found by colour and texture, then only the part connected to
    the top of the frame is kept, so a blue car lower down is not sky.
    """
    h, w = rgb.shape[:2]
    scale = 480.0 / max(h, w)
    small = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue, sat, val = hsv[:, :, 0] * 2.0, hsv[:, :, 1] / 255.0, hsv[:, :, 2] / 255.0
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    texture = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (0, 0), 3)

    blue = (hue > 180) & (hue < 260) & (sat > 0.12) & (val > 0.35)
    overcast = (sat < 0.12) & (val > 0.72)
    candidate = ((blue | overcast) & (texture < 6.0)).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    count, labels = cv2.connectedComponents(candidate)
    keep = np.zeros_like(candidate)
    for label in set(labels[0, :].tolist()) - {0}:
        keep[labels == label] = 1
    if keep.sum() < 0.01 * keep.size:
        return np.zeros((h, w), np.float32)
    full = cv2.resize(keep.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(feather(full, max(1.0, max(h, w) / 400.0)), 0.0, 1.0)


# --------------------------------------------------------------------- #
# The segmenter
# --------------------------------------------------------------------- #

class Segmenter:
    """Produces every kind of mask, loading each model only when asked.

    ``download`` controls whether a missing model may be fetched. The
    interface asks the user first and passes True only after a yes; a
    render never downloads on its own.
    """

    def __init__(self, download: bool = False) -> None:
        self.download = download
        self._detector = None
        self._mesh = None
        self._humanseg = None
        self._sam = None
        self._remover = None

    # -- availability ---------------------------------------------------- #

    @staticmethod
    def models_for(kind: str) -> tuple[ModelSpec, ...]:
        """Models a mask kind needs beyond the face detector."""
        if kind in ("skin", "lips", "teeth", "eyes", "iris"):
            return (fm.MESH_SPEC,)
        if kind == "hair":
            return (fm.MESH_SPEC, SAM_SPEC)
        if kind in ("person",):
            return (HUMANSEG_SPEC,)
        if kind in ("subject", "background") and not U2NET_PATH.exists():
            return (HUMANSEG_SPEC,)
        return ()

    @classmethod
    def download_mb(cls, kind: str) -> float:
        return model_store.total_download_mb(cls.models_for(kind))

    # -- lazy models ------------------------------------------------------- #

    def _faces(self):
        if self._detector is None:
            from src.ai_tools.face_analysis import FaceAnalyzer

            self._detector = FaceAnalyzer(download=self.download)
            self._mesh = fm.FaceMesh(download=self.download)
        if not (self._detector.available and self._mesh.available):
            return None
        return self._detector, self._mesh

    def _person_net(self):
        if self._humanseg is None:
            path = model_store.ensure_model(HUMANSEG_SPEC, download=self.download)
            if path is not None:
                try:
                    self._humanseg = cv2.dnn.readNet(str(path))
                except Exception as exc:
                    logger.warning("Person model failed to load: %s", exc)
        return self._humanseg

    def _sam_net(self):
        if self._sam is None:
            path = model_store.ensure_model(SAM_SPEC, download=self.download)
            if path is not None:
                try:
                    self._sam = cv2.dnn.readNet(str(path))
                except Exception as exc:
                    logger.warning("Click-to-select model failed to load: %s", exc)
        return self._sam

    # -- model-based masks ------------------------------------------------- #

    def person(self, rgb: np.ndarray) -> MaskResult:
        net = self._person_net()
        if net is None:
            return _empty(rgb.shape, "person model", "the person-segmentation model is not available")
        h, w = rgb.shape[:2]
        tile = cv2.resize(rgb, (192, 192), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        net.setInput(cv2.dnn.blobFromImage((tile - 0.5) / 0.5))
        logits = np.asarray(net.forward(), dtype=np.float32)[0]
        exp = np.exp(logits - logits.max(axis=0, keepdims=True))
        probability = exp[1] / exp.sum(axis=0)
        # The model is unsure over large plain areas; firm the edge up so a
        # 40 percent guess does not become a 40 percent adjustment.
        probability = np.clip((probability - 0.35) / 0.30, 0.0, 1.0)
        probability = probability * probability * (3.0 - 2.0 * probability)
        full = cv2.resize(probability, (w, h), interpolation=cv2.INTER_LINEAR)
        return MaskResult(np.clip(full, 0.0, 1.0), "person model",
                          None if full.max() > 0.5 else "no person was found")

    def subject(self, rgb: np.ndarray) -> MaskResult:
        if U2NET_PATH.exists():
            try:
                if self._remover is None:
                    from src.ai_tools.background_removal import BackgroundRemover

                    self._remover = BackgroundRemover()
                rgba = self._remover.remove_background_array(rgb)
                return MaskResult(rgba[:, :, 3].astype(np.float32) / 255.0, "salient-object model")
            except Exception as exc:
                logger.warning("Subject model failed (%s); using the person model.", exc)
        result = self.person(rgb)
        if result.note is None:
            result.note = "from the person model; it will miss subjects that are not people"
        return result

    def click(self, rgb: np.ndarray, points, exclude=()) -> MaskResult:
        """Select the object under the clicked points.

        ``points`` are ``(x, y)`` positions on the object, at most six.
        ``exclude`` are positions that must *not* be selected; they are used
        to choose between the model's candidate masks.
        """
        net = self._sam_net()
        if net is None:
            return _empty(rgb.shape, "click-to-select", "the click-to-select model is not available")
        points = [tuple(p) for p in points][:SAM_MAX_POINTS]
        if not points:
            return _empty(rgb.shape, "click-to-select", "no points were given")
        h, w = rgb.shape[:2]
        tile = cv2.resize(rgb, (SAM_SIDE, SAM_SIDE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        coords = np.zeros((SAM_MAX_POINTS, 2), np.float32)
        labels = np.full((SAM_MAX_POINTS, 1), -1.0, np.float32)
        for index, (x, y) in enumerate(points):
            coords[index] = (x * SAM_SIDE / w, y * SAM_SIDE / h)
            labels[index] = 1.0
        try:
            net.setInput(cv2.dnn.blobFromImage(tile), "batched_images")
            net.setInput(coords[None, None], "batched_point_coords")
            net.setInput(labels[None, None], "batched_point_labels")
            masks, scores = net.forward(["output_masks", "iou_predictions"])
        except Exception as exc:
            logger.warning("Click-to-select inference failed: %s", exc)
            return _empty(rgb.shape, "click-to-select", "the model could not process this image")

        candidates = np.asarray(masks)[0, 0] >= 0
        order = np.argsort(np.asarray(scores)[0, 0])[::-1]
        chosen = candidates[order[0]]
        for index in order:
            candidate = candidates[index]
            inside = [
                candidate[int(np.clip(y * SAM_SIDE / h, 0, SAM_SIDE - 1)), int(np.clip(x * SAM_SIDE / w, 0, SAM_SIDE - 1))]
                for x, y in exclude
            ]
            if not any(inside):
                chosen = candidate
                break
        full = cv2.resize(chosen.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        return MaskResult(full, "click-to-select", None if full.max() > 0.5 else "nothing was selected there")

    # -- face-derived masks ------------------------------------------------ #

    def faces(self, rgb: np.ndarray) -> list[FaceRegions] | None:
        """Landmarks for every face, or None when the face models are absent."""
        engines = self._faces()
        return None if engines is None else _face_regions(rgb, *engines)

    def _per_face(self, rgb: np.ndarray, kind: str, build) -> MaskResult:
        faces = self.faces(rgb)
        if faces is None:
            return _empty(rgb.shape, "face mesh", "the face-mesh model is not available")
        if not faces:
            return _empty(rgb.shape, "face mesh", "no face was found")
        mask = np.zeros(rgb.shape[:2], np.float32)
        for face in faces:
            mask = np.maximum(mask, build(face))
        return MaskResult(mask, "face mesh", None if mask.max() > 0 else f"no {kind} is visible")

    def hair(self, rgb: np.ndarray) -> MaskResult:
        faces = self.faces(rgb)
        if not faces:
            return _empty(rgb.shape, "click-to-select", "no face was found to look for hair around")
        h, w = rgb.shape[:2]
        mask = np.zeros((h, w), np.float32)
        for face in faces:
            x, y, bw, bh = face.box
            crown = [
                (x + 0.5 * bw, y - 0.18 * bh), (x + 0.22 * bw, y - 0.08 * bh), (x + 0.78 * bw, y - 0.08 * bh),
            ]
            crown = [(float(np.clip(px, 0, w - 1)), float(np.clip(py, 0, h - 1))) for px, py in crown]
            nose = tuple(face.landmarks[1, :2])
            found = self.click(rgb, crown, exclude=[nose])
            if found.note and found.is_empty:
                return found
            face_area = _fill(rgb.shape, [face.outline(fm.FACE_OVAL)])
            mask = np.maximum(mask, np.clip(found.mask - face_area, 0.0, 1.0))
        return MaskResult(mask, "click-to-select, seeded above the forehead",
                          "approximate: refine with the brush")

    # -- dispatch ---------------------------------------------------------- #

    def mask(self, rgb: np.ndarray, kind: str) -> MaskResult:
        """Build a mask of the given kind for an RGB uint8 image.

        Never raises for a missing model or an absent subject: the result
        is an empty mask with ``note`` explaining why.
        """
        if kind not in KINDS:
            raise ValueError(f"Unknown mask kind {kind!r}. Choose from: {', '.join(KINDS)}")
        h, w = rgb.shape[:2]
        # Work at a bounded size; masks are smooth enough to scale up.
        scale = ANALYSIS_SIDE / float(max(h, w))
        work = rgb if scale >= 1.0 else cv2.resize(
            rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA
        )
        try:
            result = self._build(work, kind)
        except Exception as exc:  # a mask failing must not take the render down
            logger.warning("Mask %r failed: %s", kind, exc)
            result = _empty(work.shape, kind, f"could not build this mask ({type(exc).__name__})")
        if result.mask.shape[:2] != (h, w):
            result.mask = cv2.resize(result.mask, (w, h), interpolation=cv2.INTER_LINEAR)
        result.mask = np.clip(result.mask, 0.0, 1.0).astype(np.float32)
        return result

    def _build(self, rgb: np.ndarray, kind: str) -> MaskResult:
        if kind == "subject":
            return self.subject(rgb)
        if kind == "background":
            found = self.subject(rgb)
            return MaskResult(1.0 - found.mask, found.method, found.note)
        if kind == "person":
            return self.person(rgb)
        if kind == "sky":
            mask = sky_mask(rgb)
            return MaskResult(mask, "colour and position heuristic",
                              "approximate: refine with the brush" if mask.max() > 0 else "no sky was found")
        if kind == "hair":
            return self.hair(rgb)
        if kind == "skin":
            return self._per_face(rgb, kind, lambda f: skin_mask(rgb, f))
        if kind == "lips":
            return self._per_face(rgb, kind, lambda f: np.clip(
                _fill(rgb.shape, [f.outline(fm.OUTER_LIPS)]) - _fill(rgb.shape, [f.outline(fm.INNER_LIPS)]), 0, 1))
        if kind == "teeth":
            return self._per_face(rgb, kind, lambda f: teeth_mask(rgb, f))
        if kind == "eyes":
            return self._per_face(rgb, kind, lambda f: _fill(
                rgb.shape, [f.outline(fm.RIGHT_EYE_RING), f.outline(fm.LEFT_EYE_RING)]))

        def iris(face: FaceRegions) -> np.ndarray:
            canvas = np.zeros(rgb.shape[:2], np.float32)
            for indices in (RIGHT_IRIS, LEFT_IRIS):
                centre, radius = face.iris(indices)
                cv2.circle(canvas, centre, radius, 1.0, -1, lineType=cv2.LINE_AA)
            lids = _fill(rgb.shape, [face.outline(fm.RIGHT_EYE_RING), face.outline(fm.LEFT_EYE_RING)])
            return canvas * lids        # the part of the iris the lids leave showing

        return self._per_face(rgb, kind, iris)
