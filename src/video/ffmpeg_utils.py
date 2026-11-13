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
import re
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
    def codec_label(self) -> str:
        """A readable codec name, e.g. "ProRes 422 HQ" for ``apch``."""
        return CODEC_NAMES.get(self.codec_fourcc.lower(), self.codec_fourcc)

    @property
    def is_intermediate_codec(self) -> bool:
        """True for editing codecs (ProRes, DNxHD/HR) rather than delivery ones."""
        return self.codec_label.startswith(("ProRes", "DNx"))

    @property
    def duration_label(self) -> str:
        m, s = divmod(int(self.duration_sec), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".mxf", ".avi"}

# FourCC (lower-cased) -> name. OpenCV reports the container's sample-entry
# code, which for the editing codecs identifies the exact flavour.
CODEC_NAMES = {
    "avc1": "H.264", "h264": "H.264", "x264": "H.264",
    "hvc1": "H.265 / HEVC", "hev1": "H.265 / HEVC", "hevc": "H.265 / HEVC",
    "av01": "AV1", "vp09": "VP9", "vp90": "VP9", "vp80": "VP8",
    "apco": "ProRes 422 Proxy", "apcs": "ProRes 422 LT", "apcn": "ProRes 422",
    "apch": "ProRes 422 HQ", "ap4h": "ProRes 4444", "ap4x": "ProRes 4444 XQ",
    "avdn": "DNxHD", "avdh": "DNxHR",
    "mjpg": "Motion JPEG", "mp4v": "MPEG-4 Part 2", "fmp4": "MPEG-4 Part 2",
}


def is_video(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


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

    listing = _stream_listing(path)
    has_audio = "Audio:" in listing
    # OpenCV reports one tag per codec family ("apch" for every ProRes
    # flavour). The container's own tag says which flavour it actually is.
    fourcc = _container_video_tag(listing) or fourcc
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


_VIDEO_TAG = re.compile(r"Video:.*?\((\w{4}) / 0x[0-9A-Fa-f]+\)")


def _stream_listing(path: Path) -> str:
    """ffmpeg's stream table for a file (see :func:`_probe_has_audio`)."""
    cmd = [get_ffmpeg_exe(), "-hide_banner", "-i", str(path)]
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace").stderr


def _container_video_tag(listing: str) -> str | None:
    match = _VIDEO_TAG.search(listing)
    return match.group(1) if match else None


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
