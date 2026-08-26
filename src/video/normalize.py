"""Pre-processing normalization: mismatched inputs become one standard
working format so downstream edit operations (concat, transitions, speed
ramps) can assume consistent fps, resolution, and audio sample rate
instead of fighting whatever the source camera happened to produce.
"""

from __future__ import annotations

from pathlib import Path

from .ffmpeg_utils import probe, run_ffmpeg

DEFAULT_FPS = 30
DEFAULT_AUDIO_RATE = 48000
DEFAULT_AUDIO_CHANNELS = 2


def normalize_video(
    input_path: str | Path,
    output_path: str | Path,
    target_fps: int = DEFAULT_FPS,
    target_sample_rate: int = DEFAULT_AUDIO_RATE,
    max_width: int | None = None,
) -> Path:
    """Re-encode to a standard fps / audio sample rate / (optional) resolution cap."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    info = probe(input_path)
    vf_parts = [f"fps={target_fps}"]
    if max_width and info.width > max_width:
        vf_parts.append(f"scale={max_width}:-2")
    vf = ",".join(vf_parts)

    args = [
        "-i", str(input_path),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
    ]
    if info.has_audio:
        args += [
            "-ar", str(target_sample_rate), "-ac", str(DEFAULT_AUDIO_CHANNELS),
            "-c:a", "aac", "-b:a", "192k",
        ]
    else:
        args += ["-an"]
    args += [str(output_path)]

    run_ffmpeg(args)
    return output_path
