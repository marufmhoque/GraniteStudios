"""Structural & scene operations: split/trim, concat/stitch with ripple
timeline shifting, transitions, and split audio/video (J-cuts and L-cuts).
"""

from __future__ import annotations

from pathlib import Path

from .ffmpeg_utils import probe, run_ffmpeg
from .timing import trim


def split_at(
    input_path: str | Path, output_dir: str | Path, cut_points: list[float]
) -> list[Path]:
    """Split one clip into N+1 pieces at the given cut points (seconds).

    Each piece is written as an independent file starting at its own t=0, so
    this is a "ripple" split in effect: dropping or re-trimming any one piece
    automatically shifts everything downstream when the pieces are later
    concatenated, since there's no shared in-place timeline to keep in sync.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    info = probe(input_path)
    bounds = [0.0, *sorted(cut_points), info.duration_sec]
    outputs = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        out_path = output_dir / f"{input_path.stem}_part{i + 1:02d}{input_path.suffix}"
        trim(input_path, out_path, start, end, reencode=True)
        outputs.append(out_path)
    return outputs


def concat_clips(input_paths: list[str | Path], output_path: str | Path) -> Path:
    """Concatenate clips in order via filter_complex concat (re-encodes, so
    mismatched fps/resolution/codec across sources is handled safely —
    unlike the ffmpeg concat demuxer, which requires identical parameters).
    """
    input_paths = [Path(p) for p in input_paths]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not input_paths:
        raise ValueError("concat_clips needs at least one input")

    if len(input_paths) == 1:
        run_ffmpeg(
            [
                "-i", str(input_paths[0]),
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "aac", str(output_path),
            ]
        )
        return output_path

    all_audio = all(probe(p).has_audio for p in input_paths)

    cmd = []
    for p in input_paths:
        cmd += ["-i", str(p)]

    if all_audio:
        av_pairs = "".join(f"[{i}:v][{i}:a]" for i in range(len(input_paths)))
        filter_complex = f"{av_pairs}concat=n={len(input_paths)}:v=1:a=1[outv][outa]"
        maps = ["-map", "[outv]", "-map", "[outa]"]
    else:
        v_labels = "".join(f"[{i}:v]" for i in range(len(input_paths)))
        filter_complex = f"{v_labels}concat=n={len(input_paths)}:v=1:a=0[outv]"
        maps = ["-map", "[outv]"]

    args = cmd + ["-filter_complex", filter_complex] + maps
    args += ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]
    if all_audio:
        args += ["-c:a", "aac"]
    args += [str(output_path)]
    run_ffmpeg(args)
    return output_path


def add_transition(
    clip_a: str | Path,
    clip_b: str | Path,
    output_path: str | Path,
    transition: str = "fade",
    duration: float = 1.0,
) -> Path:
    """Join exactly two clips with a transition (crossfade by default), using
    ffmpeg's xfade (video) / acrossfade (audio) filters. The offset is
    computed from clip_a's own duration so the transition starts
    ``duration`` seconds before clip_a ends.
    """
    clip_a, clip_b, output_path = Path(clip_a), Path(clip_b), Path(output_path)
    info_a = probe(clip_a)
    info_b = probe(clip_b)
    offset = max(0.0, info_a.duration_sec - duration)

    known_transitions = {
        "fade", "wipeleft", "wiperight", "wipeup", "wipedown",
        "slideup", "slidedown", "slideleft", "slideright",
        "circleopen", "circleclose", "dissolve",
    }
    xfade_name = transition if transition in known_transitions else "fade"

    filter_complex = (
        f"[0:v][1:v]xfade=transition={xfade_name}:duration={duration}:offset={offset:.3f}[outv]"
    )
    maps = ["-map", "[outv]"]
    both_audio = info_a.has_audio and info_b.has_audio
    if both_audio:
        filter_complex += f";[0:a][1:a]acrossfade=d={duration}[outa]"
        maps += ["-map", "[outa]"]

    args = ["-i", str(clip_a), "-i", str(clip_b), "-filter_complex", filter_complex, *maps]
    args += ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]
    if both_audio:
        args += ["-c:a", "aac"]
    args += [str(output_path)]
    run_ffmpeg(args)
    return output_path


def hard_cut(clip_a: str | Path, clip_b: str | Path, output_path: str | Path) -> Path:
    """Join two clips with a hard cut (no transition)."""
    return concat_clips([clip_a, clip_b], output_path)


def apply_j_or_l_cut(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    audio_offset_sec: float,
) -> Path:
    """Overlay ``audio_path``'s audio onto ``video_path``'s video, shifted by
    ``audio_offset_sec`` — negative values make the audio lead the picture
    (a J-cut: audio from the next scene starts early); positive values make
    it lag behind (an L-cut: audio from the previous scene lingers).
    """
    video_path, audio_path, output_path = Path(video_path), Path(audio_path), Path(output_path)
    if audio_offset_sec >= 0:
        adelay_ms = int(audio_offset_sec * 1000)
        args = [
            "-i", str(video_path), "-i", str(audio_path),
            "-filter_complex", f"[1:a]adelay={adelay_ms}|{adelay_ms}[a]",
            "-map", "0:v", "-map", "[a]",
        ]
    else:
        trim_sec = -audio_offset_sec
        args = [
            "-i", str(video_path),
            "-ss", f"{trim_sec:.3f}", "-i", str(audio_path),
            "-map", "0:v", "-map", "1:a",
        ]
    args += ["-c:v", "copy", "-c:a", "aac", "-shortest", str(output_path)]
    run_ffmpeg(args)
    return output_path
