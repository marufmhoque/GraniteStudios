"""AI culling suite: sharpness scoring and burst grouping.

Designed for high-volume shoots: score every frame's sharpness with
Laplacian variance, group burst sequences by capture timestamp, and pick
the technical best of each burst automatically.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

from src.engine.universal_loader import UniversalLoader, RAW_EXTENSIONS

logger = logging.getLogger(__name__)

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
        preview = self.loader.load_preview(image_path)
        if preview is None:
            return 0.0

        arr = np.asarray(preview)
        # Bound analysis size so scoring is fast and scores are comparable.
        h, w = arr.shape[:2]
        scale = self.analysis_max_px / max(h, w)
        if scale < 1.0:
            arr = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

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
        preview = self.loader.load_preview(image_path)
        if preview is None:
            return 0

        small = preview.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = np.asarray(small, dtype=np.int16)
        diff = pixels[:, 1:] > pixels[:, :-1]  # 8x8 boolean grid

        value = 0
        for bit in diff.flatten():
            value = (value << 1) | int(bit)
        return value

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

        if path.suffix.lower() not in RAW_EXTENSIONS:
            try:
                with Image.open(path) as img:
                    exif = img.getexif()
                    for tag_id in _EXIF_DATETIME_TAGS:
                        raw_value = exif.get(tag_id)
                        if raw_value:
                            try:
                                return datetime.strptime(str(raw_value), "%Y:%m:%d %H:%M:%S")
                            except ValueError:
                                continue
            except Exception:
                logger.debug("EXIF read failed for %s", path, exc_info=True)

        # RAW files and EXIF-less images: file modification time is the
        # closest stable proxy for capture time.
        try:
            return datetime.fromtimestamp(os.path.getmtime(path))
        except OSError:
            return None

    # ------------------------------------------------------------------ #
    # Burst grouping
    # ------------------------------------------------------------------ #

    def group_bursts(
        self,
        image_paths: list[str | Path],
        time_threshold: float = 2.0,
    ) -> list[BurstGroup]:
        """Group photos into bursts by capture-time proximity.

        Frames whose timestamps are within ``time_threshold`` seconds of the
        previous frame join the same burst. Frames without timestamps each
        become their own group. Every group's sharpest frame is flagged as
        the burst pick.
        """
        results: list[CullResult] = []
        for p in image_paths:
            path = Path(p)
            results.append(
                CullResult(
                    path=path,
                    sharpness=self.calculate_sharpness(path),
                    timestamp=self.get_timestamp(path),
                )
            )

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
    ) -> list[BurstGroup]:
        """Convenience: scan a folder and return its burst groups."""
        paths = self.loader.scan_folder(folder)
        return self.group_bursts(list(paths), time_threshold=time_threshold)

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
        if analyzer is None:
            from src.ai_tools.face_analysis import FaceAnalyzer

            analyzer = FaceAnalyzer()
        if not analyzer.available:
            logger.info("Face model unavailable — skipping eye checks.")
            return False

        total = len(frames)
        for i, frame in enumerate(frames, start=1):
            report = analyzer.analyze_path(frame.path)
            frame.eyes_checked = True
            frame.face_count = report.face_count
            frame.eyes_closed = report.any_eyes_closed
            openness = [
                f.min_openness for f in report.faces if f.min_openness is not None
            ]
            frame.min_eye_openness = min(openness) if openness else None
            if progress:
                progress(i, total)
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

        groups = self.cull_folder(folder, time_threshold=time_threshold)

        if check_eyes:
            # Annotate before ranking, so a blink can demote a frame that
            # would otherwise win its burst on sharpness alone.
            self.check_eyes([f for g in groups for f in g.frames], progress=progress)

        # Sharpest-first inside every burst, so round-robin passes take each
        # burst's best remaining frame. With eye checking on, an open-eyed
        # frame outranks a sharper one where everybody blinked — a crisp
        # photo of closed eyes is not a keeper.
        ranked: list[list[CullResult]] = []
        for group in groups:
            frames = sorted(
                group.frames,
                key=lambda f: (not f.eyes_closed, f.sharpness),
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
        results: list[CullResult] = []
        for p in image_paths:
            path = Path(p)
            results.append(
                CullResult(
                    path=path,
                    sharpness=self.calculate_sharpness(path),
                    timestamp=self.get_timestamp(path),
                    phash=self.perceptual_hash(path),
                )
            )

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
