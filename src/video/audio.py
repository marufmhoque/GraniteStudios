"""Audio enhancements: track mixing, volume balancing, auto-ducking, and
noise reduction.
"""

from __future__ import annotations

from pathlib import Path

from .ffmpeg_utils import run_ffmpeg


def extract_audio(input_path: str | Path, output_path: str | Path) -> Path:
    input_path, output_path = Path(input_path), Path(output_path)
    run_ffmpeg(["-i", str(input_path), "-vn", "-c:a", "pcm_s16le", str(output_path)])
    return output_path


def replace_audio(video_path: str | Path, audio_path: str | Path, output_path: str | Path) -> Path:
    """Swap a video's audio track for a different file entirely."""
    video_path, audio_path, output_path = Path(video_path), Path(audio_path), Path(output_path)
    args = [
        "-i", str(video_path), "-i", str(audio_path),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-c:a", "aac", "-shortest", str(output_path),
    ]
    run_ffmpeg(args)
    return output_path


def mix_audio_tracks(
    track_paths: list[str | Path], output_path: str | Path, volumes: list[float] | None = None
) -> Path:
    """Mix N audio tracks into one, with optional per-track volume weights (0..1+)."""
    track_paths = [Path(p) for p in track_paths]
    output_path = Path(output_path)
    if volumes is None:
        volumes = [1.0] * len(track_paths)
    if len(volumes) != len(track_paths):
        raise ValueError("volumes must match track_paths in length")

    args = []
    for p in track_paths:
        args += ["-i", str(p)]

    per_track = "".join(f"[{i}:a]volume={v}[a{i}]" for i, v in enumerate(volumes))
    mix_in = "".join(f"[a{i}]" for i in range(len(track_paths)))
    filter_complex = f"{per_track};{mix_in}amix=inputs={len(track_paths)}:duration=longest[out]"

    args += ["-filter_complex", filter_complex, "-map", "[out]", "-c:a", "aac", str(output_path)]
    run_ffmpeg(args)
    return output_path


def normalize_loudness(
    input_path: str | Path, output_path: str | Path, target_lufs: float = -16.0
) -> Path:
    """EBU R128 loudness normalization (single-pass, streaming/broadcast standard)."""
    input_path, output_path = Path(input_path), Path(output_path)
    af = f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11"
    args = ["-i", str(input_path), "-af", af, "-c:v", "copy", "-c:a", "aac", str(output_path)]
    run_ffmpeg(args)
    return output_path


def auto_duck(
    voice_path: str | Path,
    music_path: str | Path,
    output_path: str | Path,
    threshold: float = 0.05,
    ratio: float = 8.0,
) -> Path:
    """Auto-duck music underneath a voice track using sidechain compression —
    music volume automatically drops whenever the voice track is speaking.
    """
    voice_path, music_path, output_path = Path(voice_path), Path(music_path), Path(output_path)
    filter_complex = (
        f"[1:a][0:a]sidechaincompress=threshold={threshold}:ratio={ratio}:"
        f"attack=20:release=250[ducked];[0:a][ducked]amix=inputs=2:duration=first:weights='1 1'[out]"
    )
    args = [
        "-i", str(voice_path), "-i", str(music_path),
        "-filter_complex", filter_complex, "-map", "[out]", "-c:a", "aac", str(output_path),
    ]
    run_ffmpeg(args)
    return output_path


def reduce_noise(input_path: str | Path, output_path: str | Path, strength: float = 25.0) -> Path:
    """Reduce background noise via ffmpeg's FFT denoiser (afftdn).

    ``strength`` is the assumed noise floor in dB below full scale, higher
    = more aggressive denoising. afftdn requires this in [20, 80].
    """
    strength = max(20.0, min(80.0, strength))
    input_path, output_path = Path(input_path), Path(output_path)
    args = [
        "-i", str(input_path), "-af", f"afftdn=nf=-{strength}",
        "-c:v", "copy", "-c:a", "aac", str(output_path),
    ]
    run_ffmpeg(args)
    return output_path


def adjust_volume(input_path: str | Path, output_path: str | Path, factor: float) -> Path:
    """Scale audio volume by a linear factor (1.0 = unchanged, 2.0 = +6dB, 0.5 = -6dB)."""
    input_path, output_path = Path(input_path), Path(output_path)
    args = ["-i", str(input_path), "-af", f"volume={factor}", "-c:v", "copy", "-c:a", "aac", str(output_path)]
    run_ffmpeg(args)
    return output_path
