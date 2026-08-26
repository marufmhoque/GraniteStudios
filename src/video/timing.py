"""Time & pacing manipulation: trim, reverse, freeze frame, speed ramping,
time-lapse condensation, and jump cuts.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from .ffmpeg_utils import FFmpegError, probe, run_ffmpeg


def trim(
    input_path: str | Path,
    output_path: str | Path,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    reencode: bool = True,
) -> Path:
    """Cut a segment out of a clip. ``reencode=True`` (default) re-encodes for
    frame-accurate cuts; ``reencode=False`` stream-copies for speed but snaps
    to the nearest keyframe.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args = ["-ss", f"{start_sec:.3f}"]
    if end_sec is not None:
        args += ["-to", f"{end_sec:.3f}"]
    args += ["-i", str(input_path)]
    if reencode:
        args += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-c:a", "aac"]
    else:
        args += ["-c", "copy"]
    args += [str(output_path)]
    run_ffmpeg(args)
    return output_path


def reverse(input_path: str | Path, output_path: str | Path) -> Path:
    """Reverse a clip's timeline (video and, if present, audio)."""
    input_path, output_path = Path(input_path), Path(output_path)
    info = probe(input_path)
    args = ["-i", str(input_path), "-vf", "reverse"]
    if info.has_audio:
        args += ["-af", "areverse"]
    else:
        args += ["-an"]
    args += ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]
    if info.has_audio:
        args += ["-c:a", "aac"]
    args += [str(output_path)]
    run_ffmpeg(args)
    return output_path


def freeze_frame(
    input_path: str | Path, output_path: str | Path, at_sec: float, hold_sec: float = 2.0
) -> Path:
    """Freeze the frame at ``at_sec`` for ``hold_sec``, splicing it into the timeline."""
    input_path, output_path = Path(input_path), Path(output_path)
    info = probe(input_path)
    frame_dur = 1 / info.fps if info.fps else 0.033
    tpad = f"tpad=stop_mode=clone:stop_duration={hold_sec}"
    filter_complex = (
        f"[0:v]trim=0:{at_sec},setpts=PTS-STARTPTS[v1];"
        f"[0:v]trim={at_sec}:{at_sec + frame_dur},setpts=PTS-STARTPTS,{tpad}[vfreeze];"
        f"[0:v]trim={at_sec},setpts=PTS-STARTPTS[v2];"
        f"[v1][vfreeze][v2]concat=n=3:v=1:a=0[outv]"
    )
    args = ["-i", str(input_path), "-filter_complex", filter_complex, "-map", "[outv]"]
    # Audio dropped: a frozen frame has no clean audio equivalent to hold.
    args += ["-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", str(output_path)]
    run_ffmpeg(args)
    return output_path


def _atempo_chain(factor: float) -> str:
    """atempo only accepts factors in [0.5, 2.0]; chain filters for larger changes."""
    if factor <= 0:
        raise ValueError("Speed factor must be positive")
    filters = []
    remaining = factor
    while remaining < 0.5 or remaining > 2.0:
        step = 2.0 if remaining > 2.0 else 0.5
        filters.append(f"atempo={step}")
        remaining /= step
    filters.append(f"atempo={remaining:.6f}")
    return ",".join(filters)


def change_speed(input_path: str | Path, output_path: str | Path, factor: float) -> Path:
    """Constant speed change (>1 = faster, <1 = slower) applied to the whole clip."""
    input_path, output_path = Path(input_path), Path(output_path)
    info = probe(input_path)
    vf = f"setpts={1 / factor:.6f}*PTS"
    args = ["-i", str(input_path), "-vf", vf]
    if info.has_audio:
        args += ["-af", _atempo_chain(factor)]
    else:
        args += ["-an"]
    args += ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]
    if info.has_audio:
        args += ["-c:a", "aac"]
    args += [str(output_path)]
    run_ffmpeg(args)
    return output_path


def speed_ramp(
    input_path: str | Path, output_path: str | Path, segments: list[tuple[float, float, float]]
) -> Path:
    """Variable speed ramping: ``segments`` is a list of (start_sec, end_sec,
    factor) covering the timeline contiguously. Each segment is trimmed,
    re-timed at its own factor, then concatenated — giving true ramped
    pacing (e.g. slow-mo into normal speed into a fast whip) rather than one
    constant rate for the whole clip.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    with tempfile.TemporaryDirectory(prefix="gs_ramp_") as tmp:
        tmp_dir = Path(tmp)
        parts = []
        for i, (start, end, factor) in enumerate(segments):
            seg_path = tmp_dir / f"seg_{i:03d}.mp4"
            trim(input_path, seg_path, start, end, reencode=True)
            ramped_path = tmp_dir / f"ramp_{i:03d}.mp4"
            change_speed(seg_path, ramped_path, factor)
            parts.append(ramped_path)
        # Local import: structure.py imports trim() from this module, so
        # importing concat_clips at module load time would be circular.
        from .structure import concat_clips

        concat_clips(parts, output_path)
    return output_path


def extract_timelapse(
    input_path: str | Path, output_path: str | Path, target_duration_sec: float
) -> Path:
    """Condense long footage into a short highlight by uniformly speeding it
    up so the whole clip fits into ``target_duration_sec`` — e.g. a
    30-minute file down to a 30-second time-lapse.
    """
    input_path = Path(input_path)
    info = probe(input_path)
    if info.duration_sec <= 0:
        raise FFmpegError(f"Could not determine duration for {input_path}")
    factor = info.duration_sec / target_duration_sec
    return change_speed(input_path, output_path, factor)
