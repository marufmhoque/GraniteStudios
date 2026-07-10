"""AI culling suite: sharpness scoring and burst grouping.

Designed for high-volume shoots: score every frame's sharpness with
Laplacian variance, group burst sequences by capture timestamp, and pick
the technical best of each burst automatically.
"""

from __future__ import annotations

import logging
import os
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
