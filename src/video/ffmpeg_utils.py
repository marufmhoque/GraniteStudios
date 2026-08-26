"""FFmpeg/OpenCV plumbing shared by every module in the video engine.

Granite Studios ships its own FFmpeg binary via ``imageio-ffmpeg`` so video
editing works out of the box on a fresh install with no system FFmpeg
required — the same "just works on modest hardware" philosophy as the
photo side of the app. A system ``ffmpeg`` on PATH is preferred when
present, since it may carry hardware encoders the bundled static build
lacks.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


class FFmpegError(RuntimeError):
    """Raised when an ffmpeg invocation exits non-zero, or a probe fails."""


@lru_cache(maxsize=1)
def get_ffmpeg_exe() -> str:
    """Locate an ffmpeg binary: system PATH first, else the bundled one."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(args: list[str], overwrite: bool = True) -> None:
    """Run ffmpeg with the given args (excluding the 'ffmpeg' executable itself).

    Raises FFmpegError with the captured stderr tail on failure.
    """
    cmd = [get_ffmpeg_exe()]
    if overwrite:
        cmd.append("-y")
    cmd += ["-hide_banner", "-loglevel", "error", *args]
    logger.debug("ffmpeg: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-25:])
        raise FFmpegError(f"ffmpeg failed ({result.returncode}):\n{tail}")


@dataclass
class VideoProbe:
    """Lightweight probe result — enough to drive normalization + UI display."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float
    codec_fourcc: str
    has_audio: bool
    container: str = field(default="")

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def duration_label(self) -> str:
        m, s = divmod(int(self.duration_sec), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def probe(path: str | Path) -> VideoProbe:
    """Probe a video file for resolution/fps/duration/codec/audio presence.

    Uses OpenCV for the video stream (no ffprobe dependency needed) and a
    quick ffmpeg dry-run to detect whether an audio stream is present.
    """
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FFmpegError(f"Could not open video: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip()
        duration = frame_count / fps if fps else 0.0
    finally:
        cap.release()

    has_audio = _probe_has_audio(path)
    return VideoProbe(
        path=path,
        width=width,
        height=height,
        fps=round(fps, 3),
        frame_count=frame_count,
        duration_sec=round(duration, 3),
        codec_fourcc=fourcc or "?",
        has_audio=has_audio,
        container=path.suffix.lstrip(".").lower(),
    )


def _probe_has_audio(path: Path) -> bool:
    """Detect an audio stream by reading ffmpeg's own stream listing.

    ``ffmpeg -i <file>`` with no output prints the full stream table to
    stderr and exits non-zero (no output was requested) regardless of
    ffmpeg version, so we just look for an "Audio:" line rather than
    depending on ffprobe being installed separately.
    """
    cmd = [get_ffmpeg_exe(), "-hide_banner", "-i", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return "Audio:" in result.stderr
