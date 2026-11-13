"""AI culling suite: sharpness, faces, burst grouping and a keeper score.

Designed for high-volume shoots. Every frame is decoded once; from that one
preview come its sharpness, a perceptual hash, its capture time and, when
face checking is on, a measurement of every face in it. Frames are grouped
into bursts by capture time and ranked within each burst.

The keeper score
----------------
    score = 0.40 * face sharpness + 0.35 * eyes + 0.15 * smile + 0.10 * composition

Each term is first brought to 0..1, because the raw quantities are not
comparable: Laplacian variance runs into the thousands and an eye aspect
ratio tops out near 0.3. Weighting them as they come would make the score a
restatement of sharpness.

* *Face sharpness* -- Laplacian variance inside the main faces, relative to
  the sharpest faces in the batch. Background detail does not count, so a
  frame focused on the wall behind someone no longer wins.
* *Eyes* -- eye openness of the least open main face, from the eye aspect
  ratio. One person blinking spoils a group shot.
* *Smile* -- mean smile of the main faces.
* *Composition* -- a stated heuristic, not a measurement of taste: is the
  main face clear of the frame edges and does it have room above it.

A frame with no faces has nothing to say about eyes or smiles. Its score is
its whole-frame sharpness, placed on the same 0..1 scale, so frames with
and without people can still be ranked against each other.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

from src.engine import fast_preview
from src.engine.universal_loader import UniversalLoader, RAW_EXTENSIONS

logger = logging.getLogger(__name__)

# One decode feeds every measurement. Sharpness is taken from the full
# preview (see fast_preview's note on why a reduced decode cannot be used
# for it). Faces are *found* on a small copy and *measured* on the full one.
FACE_MAX_SIDE = 1600      # perceptual hash, and the standalone eye check
DETECT_MAX_SIDE = 800     # face search

# Keeper-score weights (see the module docstring).
WEIGHT_SHARPNESS = 0.40
WEIGHT_EYES = 0.35
WEIGHT_SMILE = 0.15
WEIGHT_COMPOSITION = 0.10

# A face smaller than this fraction of the largest face in the frame is
# treated as a bystander and does not vote on eyes or smiles.
MAIN_FACE_RATIO = 0.5
FACE_SHARPNESS_SIDE = 256   # faces are measured at a common size

# EXIF tag ids we care about (avoid rebuilding the name map per file).
_EXIF_DATETIME_TAGS = {
    tag_id for tag_id, name in TAGS.items()
    if name in ("DateTimeOriginal", "DateTimeDigitized", "DateTime")
}


@dataclass
class CullResult:
    """Scoring result for one frame."""

    path: Path
    sharpness: float
    timestamp: datetime | None = None
    is_burst_pick: bool = False
    phash: int | None = None
    is_starred: bool = False
    is_duplicate: bool = False
    # Populated only when a cull runs with eye checking enabled.
    face_count: int = 0
    eyes_closed: bool = False
    min_eye_openness: float | None = None
    eyes_checked: bool = False
    # Added in 2.0.
    faces: list = field(default_factory=list)      # FaceObservation, one per face
    frame_size: tuple[int, int] | None = None      # pixels the face boxes refer to
    eye_method: str = "none"                       # "mesh", "heuristic" or "none"
    min_ear: float | None = None
    smile: float | None = None
    mouth_open: bool = False
    face_sharpness: float | None = None
    composition: float | None = None
    score: float = 0.0
    person_ids: list = field(default_factory=list)


@dataclass
class FaceObservation:
    """What was measured on one face in one frame."""

    box: tuple[int, int, int, int]
    confidence: float
    is_main: bool = True
    sharpness: float = 0.0
    ear: float | None = None
    eye_openness: float | None = None
    eyes_closed: bool = False
    smile: float | None = None
    mouth_open: bool = False
    roll_deg: float | None = None
    detector_row: np.ndarray | None = None     # for face recognition alignment
    embedding: np.ndarray | None = None
    person_id: int | None = None


def _region_sharpness(gray: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """Laplacian variance inside a box, measured at a common size.

    Shrinking large faces to one size keeps a close-up from scoring higher
    than a mid-shot merely for having more pixels. Small faces are left as
    they are: enlarging them would only add blur.
    """
    x, y, w, h = box
    height, width = gray.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0
    crop = fast_preview.fit(gray[y0:y1, x0:x1], FACE_SHARPNESS_SIDE)
    return float(cv2.Laplacian(crop, cv2.CV_64F).var())


def composition_score(box: tuple[int, int, int, int], frame_size: tuple[int, int]) -> float:
    """A 0..1 heuristic for how the main face sits in the frame.

    This is not a judgement of composition as a photographer means it. It
    checks two things that are almost always faults: a face cut by the edge
    of the frame, and a face jammed against the top with no room above it.
    Anything else scores full marks.
    """
    x, y, w, h = box
    width, height = frame_size
    if width <= 0 or height <= 0 or w <= 0 or h <= 0:
        return 1.0
    score = 1.0
    # Cut by an edge: how much of the box falls outside the frame.
    inside_w = max(0, min(width, x + w) - max(0, x))
    inside_h = max(0, min(height, y + h) - max(0, y))
    visible = (inside_w * inside_h) / float(w * h)
    score -= 0.6 * (1.0 - visible)
    # Touching an edge without being cut still crowds the subject.
    margin = min(x, y, width - (x + w), height - (y + h)) / float(max(w, h))
    if margin < 0.05:
        score -= 0.2
    # Headroom: the detector box starts near the brow, so the head needs
    # roughly a third of the box above it.
    if y < 0.3 * h:
        score -= 0.2 * (1.0 - max(0.0, y) / (0.3 * h))
    return float(np.clip(score, 0.0, 1.0))


class _FaceJudge:
    """Finds and measures faces from several threads.

    OpenCV networks hold per-call state, so each worker thread gets its own
    detector and mesh. A caller-supplied analyzer is a single object and is
    therefore used under a lock instead.

    With the face mesh available, eyes are judged by eye aspect ratio. When
    it is not, the older patch-darkness heuristic is used and the result
    says so in ``eye_method``.
    """

    def __init__(self, analyzer=None, use_mesh: bool = True, embedder=None):
        self._shared = analyzer
        self._lock = threading.Lock()
        self._local = threading.local()
        self._embedder = embedder
        if analyzer is None:
            from src.ai_tools.face_analysis import FaceAnalyzer

            # Created here, on the calling thread, so a first-run model
            # download happens once rather than racing across workers.
            self._factory = FaceAnalyzer
            self.available = FaceAnalyzer().available
        else:
            self._factory = None
            self.available = bool(getattr(analyzer, "available", True))

        self._mesh = None
        if use_mesh and self.available and analyzer is None:
            try:
                from src.ai_tools.face_mesh import ThreadLocalFaceMesh

                mesh = ThreadLocalFaceMesh()
                self._mesh = mesh if mesh.available else None
            except Exception as exc:  # the cull still works without it
                logger.info("Face mesh unavailable (%s); using the eye heuristic.", exc)

    @property
    def method(self) -> str:
        return "mesh" if self._mesh is not None else "heuristic"

    def _detector(self):
        analyzer = getattr(self._local, "analyzer", None)
        if analyzer is None:
            analyzer = self._local.analyzer = self._factory(download=False)
        return analyzer

    def annotate(
        self,
        frame: "CullResult",
        rgb: np.ndarray,
        sharp_gray: np.ndarray | None = None,
        sharp_scale: float = 1.0,
    ) -> None:
        """Measure every face in ``rgb`` and record it on ``frame``.

        Face boxes are reported in ``rgb``'s pixels. When ``sharp_gray`` is
        given -- a full-resolution greyscale of the same frame, larger by
        ``sharp_scale`` -- face sharpness is measured there instead, so it
        is not blunted by the reduced decode ``rgb`` came from.
        """
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        height, width = bgr.shape[:2]
        frame.eyes_checked = True
        frame.frame_size = (width, height)

        if self._shared is not None:
            with self._lock:
                report = self._shared.analyze(bgr)
            faces, detector = report.faces, None
        else:
            detector = self._detector()
            faces = detector.detect_reduced(bgr, DETECT_MAX_SIDE)

        frame.face_count = len(faces)
        if not faces:
            return

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if sharp_gray is None:
            sharp_gray, sharp_scale = gray, 1.0

        def region(box) -> float:
            return _region_sharpness(sharp_gray, tuple(int(round(v * sharp_scale)) for v in box))

        largest = max(max(f.w, f.h) for f in faces)
        observations: list[FaceObservation] = []
        for face in faces:
            seen = FaceObservation(
                box=face.box,
                confidence=face.confidence,
                is_main=max(face.w, face.h) >= MAIN_FACE_RATIO * largest,
                detector_row=face.detector_row() if hasattr(face, "detector_row") else None,
            )
            measured = self._mesh.analyze_face(bgr, face) if self._mesh is not None else None
            if measured is not None:
                seen.ear = measured.ear
                seen.eye_openness = measured.eye_openness
                seen.eyes_closed = measured.eyes_closed
                seen.smile = measured.smile
                seen.mouth_open = measured.mouth_is_open
                seen.roll_deg = measured.roll_deg
                seen.sharpness = region(measured.eye_box())
            else:
                if detector is not None:
                    detector.measure_eyes(gray, face)
                if face.eyes_measured:
                    seen.eye_openness = face.min_openness
                    seen.eyes_closed = face.eyes_closed
                seen.sharpness = region(face.box)
            if self._embedder is not None and seen.is_main:
                seen.embedding = self._embedder.embed(bgr, seen.detector_row)
            observations.append(seen)

        frame.faces = observations
        frame.eye_method = "mesh" if any(o.ear is not None for o in observations) else (
            "heuristic" if any(o.eye_openness is not None for o in observations) else "none"
        )
        main = [o for o in observations if o.is_main]
        frame.eyes_closed = any(o.eyes_closed for o in main)
        openness = [o.eye_openness for o in main if o.eye_openness is not None]
        frame.min_eye_openness = min(openness) if openness else None
        ears = [o.ear for o in main if o.ear is not None]
        frame.min_ear = min(ears) if ears else None
        smiles = [o.smile for o in main if o.smile is not None]
        frame.smile = float(np.mean(smiles)) if smiles else None
        frame.mouth_open = any(o.mouth_open for o in main)
        frame.face_sharpness = float(np.mean([o.sharpness for o in main])) if main else None
        biggest = max(main, key=lambda o: o.box[2] * o.box[3])
        frame.composition = composition_score(biggest.box, (width, height))


def assign_scores(frames: "list[CullResult]") -> None:
    """Fill in ``score`` for a batch. See the module docstring for the terms.

    Sharpness is scaled against the batch itself (its 95th percentile), so
    the score ranks frames from one shoot against each other. It is not an
    absolute quality figure and should not be compared across shoots.
    """
    if not frames:
        return

    def scale(values: list[float]) -> float:
        positive = [v for v in values if v and v > 0]
        return float(np.percentile(positive, 95)) if positive else 1.0

    frame_top = scale([f.sharpness for f in frames])
    face_top = scale([f.face_sharpness for f in frames if f.face_sharpness is not None])

    for frame in frames:
        whole = float(np.clip(frame.sharpness / frame_top, 0.0, 1.0))
        if frame.face_sharpness is None:
            frame.score = whole
            continue
        terms = [(WEIGHT_SHARPNESS, float(np.clip(frame.face_sharpness / face_top, 0.0, 1.0)))]
        if frame.min_eye_openness is not None:
            terms.append((WEIGHT_EYES, float(np.clip(frame.min_eye_openness, 0.0, 1.0))))
        if frame.smile is not None:
            terms.append((WEIGHT_SMILE, float(np.clip(frame.smile, 0.0, 1.0))))
        if frame.composition is not None:
            terms.append((WEIGHT_COMPOSITION, frame.composition))
        # A term that could not be measured is left out and the rest are
        # rescaled, rather than scoring the gap as zero.
        total = sum(weight for weight, _ in terms)
        frame.score = sum(weight * value for weight, value in terms) / total if total else whole


@dataclass
class BurstGroup:
    """A group of frames shot in rapid succession."""

    frames: list[CullResult] = field(default_factory=list)

    @property
    def best(self) -> CullResult | None:
        if not self.frames:
            return None
        return max(self.frames, key=lambda f: f.sharpness)


class SmartCull:
    """Sharpness analysis + burst grouping for rapid culling."""

    def __init__(self, loader: UniversalLoader | None = None, analysis_max_px: int = 1200):
        self.loader = loader or UniversalLoader()
        self.analysis_max_px = analysis_max_px

    # ------------------------------------------------------------------ #
    # Sharpness
    # ------------------------------------------------------------------ #

    def calculate_sharpness(self, image_path: str | Path) -> float:
        """Score focus quality using the variance of the Laplacian.

        Higher = sharper. Blurry/missed-focus frames typically score under
        ~100 on the downscaled preview; tack-sharp frames score in the
        hundreds to thousands. Returns 0.0 if the image cannot be read.
        """
        preview = fast_preview.load_preview(image_path, max_side=None, exact=True, mode="L")
        if preview is None:
            return 0.0
        return self._sharpness_of(preview.image)

    def _sharpness_of(self, image: np.ndarray) -> float:
        """Laplacian variance at a fixed analysis size.

        Takes a greyscale array (or RGB, which is converted). It should be
        a full decode: the single area-averaged shrink here is what keeps
        scores comparable between frames.
        """
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = fast_preview.fit(gray, self.analysis_max_px)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _dhash_of(rgb: np.ndarray) -> int:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        pixels = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
        value = 0
        for bit in (pixels[:, 1:] > pixels[:, :-1]).flatten():
            value = (value << 1) | int(bit)
        return value

    # ------------------------------------------------------------------ #
    # One pass over a batch
    # ------------------------------------------------------------------ #

    def score_frames(
        self,
        image_paths,
        check_eyes: bool = False,
        analyzer=None,
        progress: Callable[[int, int], None] | None = None,
        workers: int | None = None,
        use_mesh: bool = True,
        embedder=None,
    ) -> list[CullResult]:
        """Score a batch, decoding each file exactly once.

        Sharpness, the perceptual hash, the capture time and (optionally)
        the face measurements all come from the same decoded preview, on a
        pool of worker threads. Previously each of those opened the file
        again, and the eye check fully developed every RAW.

        ``embedder`` is an optional ``person_clustering.FaceEmbedder``; when
        given, each main face also gets an identity embedding.
        """
        paths = [Path(p) for p in image_paths]
        eyes = _FaceJudge(analyzer, use_mesh=use_mesh, embedder=embedder) if check_eyes else None
        if eyes is not None and not eyes.available:
            logger.info("Face model unavailable -- skipping eye checks.")
            eyes = None

        def measure(key: str, pair) -> CullResult:
            path = Path(key)
            if pair is None:
                return CullResult(path=path, sharpness=0.0, timestamp=self._file_time(path))
            result = CullResult(
                path=path,
                sharpness=self._sharpness_of(pair.gray),
                timestamp=pair.captured_at or self._file_time(path),
                phash=self._dhash_of(pair.colour),
            )
            if eyes is not None:
                eyes.annotate(result, pair.colour, pair.gray, pair.scale)
            return result

        # Greyscale in full for sharpness, reduced colour for faces: see
        # fast_preview for why one colour decode at full size is slower.
        scored = fast_preview.map_pairs(
            paths, measure, colour_max_side=FACE_MAX_SIDE, workers=workers, progress=progress
        )
        frames = [scored[str(p)] for p in paths]
        assign_scores(frames)
        return frames

    @staticmethod
    def _file_time(path: Path) -> datetime | None:
        try:
            return datetime.fromtimestamp(os.path.getmtime(path))
        except OSError:
            return None

    # ------------------------------------------------------------------ #
    # Perceptual hashing (near-duplicate detection)
    # ------------------------------------------------------------------ #

    def perceptual_hash(self, image_path: str | Path) -> int:
        """Compute a 64-bit dHash for visual-similarity comparison.

        dHash downsizes to 9x8 grayscale and encodes whether each pixel is
        brighter than its right-hand neighbour. Near-identical frames (a
        burst of the same scene) produce hashes a small Hamming distance
        apart, even across minor exposure/focus changes. Returns 0 on failure.
        """
        preview = fast_preview.load_preview(image_path, max_side=FACE_MAX_SIDE)
        if preview is None:
            return 0
        return self._dhash_of(preview.image)

    @staticmethod
    def hamming_distance(a: int, b: int) -> int:
        """Number of differing bits between two perceptual hashes."""
        return bin(a ^ b).count("1")

    # ------------------------------------------------------------------ #
    # Timestamps
    # ------------------------------------------------------------------ #

    def get_timestamp(self, image_path: str | Path) -> datetime | None:
        """Read the capture timestamp from EXIF, falling back to file mtime."""
        path = Path(image_path)
        try:
            if path.suffix.lower() in RAW_EXTENSIONS:
                # The embedded preview carries the camera's EXIF. Any size
                # will do; only its header is wanted.
                preview = fast_preview.load_preview(path, max_side=64, min_side=0)
                if preview is not None and preview.captured_at:
                    return preview.captured_at
            else:
                with Image.open(path) as img:
                    captured = fast_preview.read_capture_time(img)
                if captured:
                    return captured
        except Exception:
            logger.debug("EXIF read failed for %s", path, exc_info=True)

        # No EXIF time: file modification time is the closest stable proxy.
        return self._file_time(path)

    # ------------------------------------------------------------------ #
    # Burst grouping
    # ------------------------------------------------------------------ #

    def group_bursts(
        self,
        image_paths: list[str | Path],
        time_threshold: float = 2.0,
        check_eyes: bool = False,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[BurstGroup]:
        """Group photos into bursts by capture-time proximity.

        Frames whose timestamps are within ``time_threshold`` seconds of the
        previous frame join the same burst. Frames without timestamps each
        become their own group. Every group's sharpest frame is flagged as
        the burst pick.
        """
        results = self.score_frames(image_paths, check_eyes=check_eyes, progress=progress)

        # Sort by timestamp; undated frames go last as singletons.
        dated = sorted((r for r in results if r.timestamp), key=lambda r: r.timestamp)
        undated = [r for r in results if r.timestamp is None]

        groups: list[BurstGroup] = []
        for frame in dated:
            if (
                groups
                and groups[-1].frames
                and (frame.timestamp - groups[-1].frames[-1].timestamp).total_seconds()
                <= time_threshold
            ):
                groups[-1].frames.append(frame)
            else:
                groups.append(BurstGroup(frames=[frame]))

        groups.extend(BurstGroup(frames=[frame]) for frame in undated)

        for group in groups:
            best = group.best
            if best is not None:
                best.is_burst_pick = True

        return groups

    def cull_folder(
        self,
        folder: str | Path,
        time_threshold: float = 2.0,
        check_eyes: bool = False,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[BurstGroup]:
        """Convenience: scan a folder and return its burst groups."""
        paths = self.loader.scan_folder(folder)
        return self.group_bursts(
            list(paths), time_threshold=time_threshold, check_eyes=check_eyes, progress=progress
        )

    def check_eyes(
        self,
        frames: "list[CullResult]",
        analyzer=None,
        progress: Callable[[int, int], None] | None = None,
    ) -> bool:
        """Annotate frames with face count and closed-eye state.

        Returns False and leaves frames untouched when the face model is
        unavailable, so a cull still works on a machine that could not
        fetch it — the same graceful-degradation rule the rest of the app
        follows.
        """
        eyes = _FaceJudge(analyzer)
        if not eyes.available:
            logger.info("Face model unavailable — skipping eye checks.")
            return False

        by_path = {str(frame.path): frame for frame in frames}

        def measure(key: str, preview) -> None:
            if preview is not None:
                eyes.annotate(by_path[key], preview.image)

        fast_preview.map_previews(
            by_path, measure, max_side=FACE_MAX_SIDE, progress=progress
        )
        assign_scores(frames)
        return True

    def cull_to_target(
        self,
        folder: str | Path,
        target_count: int,
        time_threshold: float = 2.0,
        progress: Callable[[int, int], None] | None = None,
        check_eyes: bool = False,
    ) -> list[CullResult]:
        """Select roughly ``target_count`` keepers from a whole folder.

        A photographer usually knows the number they want to deliver ("give
        me the best 400 of these 3,000") rather than a sharpness threshold
        that means nothing in the abstract. This spreads the quota across
        bursts instead of taking a global top-N, because a global sort
        collapses onto whichever moments happened to be sharpest and can
        drop entire parts of the day.

        Each burst contributes its sharpest frame first; if the quota is
        larger than the number of bursts, further passes take each burst's
        next-best frame, and so on. Everything selected comes back starred,
        everything else marked as a duplicate, sharpest-first.

        Returns every scored frame — not just the keepers — so the caller
        can still show, and override, the ones that missed the cut.
        """
        if target_count < 0:
            raise ValueError("target_count cannot be negative")

        # Eyes are measured in the same pass as sharpness, before ranking, so
        # a blink can demote a frame that would otherwise win its burst.
        groups = self.cull_folder(
            folder, time_threshold=time_threshold, check_eyes=check_eyes, progress=progress
        )

        # Sharpest-first inside every burst, so round-robin passes take each
        # burst's best remaining frame. With eye checking on, an open-eyed
        # frame outranks a sharper one where everybody blinked — a crisp
        # photo of closed eyes is not a keeper.
        ranked: list[list[CullResult]] = []
        for group in groups:
            frames = sorted(
                group.frames,
                key=lambda f: (not f.eyes_closed, f.score if f.eyes_checked else f.sharpness),
                reverse=True,
            )
            for frame in frames:
                frame.is_starred = False
                frame.is_duplicate = False
            if frames:
                ranked.append(frames)

        all_frames = [f for frames in ranked for f in frames]
        total = len(all_frames)
        if total == 0:
            return []

        quota = min(target_count, total)
        selected = 0
        depth = 0
        # Round-robin across bursts: pass 0 takes every burst's best frame,
        # pass 1 their second-best, and so on until the quota is filled.
        while selected < quota:
            took_this_pass = False
            for frames in ranked:
                if selected >= quota:
                    break
                if depth < len(frames):
                    frames[depth].is_starred = True
                    selected += 1
                    took_this_pass = True
                    if progress:
                        progress(selected, quota)
            if not took_this_pass:
                break  # every burst exhausted before the quota was met
            depth += 1

        for frame in all_frames:
            if not frame.is_starred:
                frame.is_duplicate = True

        all_frames.sort(key=lambda f: (not f.is_starred, f.eyes_closed, -f.sharpness))
        return all_frames

    # ------------------------------------------------------------------ #
    # Smart grouping & top picks
    # ------------------------------------------------------------------ #

    @staticmethod
    def _as_results(image_group) -> list[CullResult]:
        """Normalize a BurstGroup / list of CullResult / list of paths."""
        if isinstance(image_group, BurstGroup):
            return image_group.frames
        frames: list[CullResult] = []
        for item in image_group:
            if isinstance(item, CullResult):
                frames.append(item)
            else:  # a path-like
                frames.append(CullResult(path=Path(item), sharpness=0.0))
        return frames

    def get_top_picks(self, image_group, count: int = 2) -> list[CullResult]:
        """Star the sharpest frames in a similar-image group, flag the rest.

        Given a group of visually similar images (produced by
        :meth:`group_similar`, which combines capture timestamps with a
        perceptual hash), this sorts the frames by sharpness, stars the top
        ``count``, and marks every remaining frame as a duplicate.

        Args:
            image_group: A ``BurstGroup``, a list of ``CullResult``, or a
                list of image paths.
            count: How many frames to star (default 2).

        Returns:
            The frames sorted sharpest-first, with ``is_starred`` and
            ``is_duplicate`` set on each.
        """
        frames = self._as_results(image_group)

        # Make sure every frame has a sharpness score and perceptual hash.
        for frame in frames:
            if frame.sharpness == 0.0:
                frame.sharpness = self.calculate_sharpness(frame.path)
            if frame.phash is None:
                frame.phash = self.perceptual_hash(frame.path)

        ordered = sorted(frames, key=lambda f: f.sharpness, reverse=True)

        for rank, frame in enumerate(ordered):
            frame.is_starred = rank < count
            frame.is_duplicate = rank >= count

        return ordered

    def group_similar(
        self,
        image_paths: list[str | Path],
        time_threshold: float = 3.0,
        hash_threshold: int = 12,
        top_count: int = 2,
    ) -> list[list[CullResult]]:
        """Group visually similar shots and auto-star the best of each.

        A frame joins an existing group when it is close in *both* capture
        time (within ``time_threshold`` seconds of the group's latest frame)
        and appearance (perceptual-hash Hamming distance within
        ``hash_threshold``). Each resulting group then has its top
        ``top_count`` frames starred and the rest marked as duplicates.

        Returns a list of groups, each a sharpness-sorted list of
        ``CullResult``.
        """
        results = self.score_frames(image_paths)

        # Order by time so bursts are contiguous; undated frames trail behind.
        dated = sorted((r for r in results if r.timestamp), key=lambda r: r.timestamp)
        undated = [r for r in results if r.timestamp is None]

        groups: list[list[CullResult]] = []
        for frame in dated:
            placed = False
            if groups:
                last = groups[-1][-1]
                close_in_time = (
                    last.timestamp is not None
                    and (frame.timestamp - last.timestamp).total_seconds() <= time_threshold
                )
                close_in_look = self.hamming_distance(frame.phash or 0, last.phash or 0) <= hash_threshold
                if close_in_time and close_in_look:
                    groups[-1].append(frame)
                    placed = True
            if not placed:
                groups.append([frame])

        groups.extend([frame] for frame in undated)

        return [self.get_top_picks(group, count=top_count) for group in groups]
