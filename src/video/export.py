"""Export presets — container/codec/bitrate/resolution combinations for the
common distribution targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .ffmpeg_utils import probe, run_ffmpeg


@dataclass(frozen=True)
class ExportPreset:
    name: str
    container: str  # file extension, no dot
    video_codec: str
    audio_codec: str
    crf: int | None = None
    video_bitrate: str | None = None
    audio_bitrate: str = "192k"
    max_width: int | None = None
    # Exact output frame. Unlike max_width (which only ever shrinks a source
    # that is already too wide), this pins the frame for targets that mandate
    # one -- the source is fitted inside it and padded, never stretched.
    target_resolution: tuple[int, int] | None = None
    extra_args: tuple[str, ...] = ()


PRESETS: dict[str, ExportPreset] = {
    "web_mp4": ExportPreset(
        name="Web MP4 (H.264)", container="mp4", video_codec="libx264", audio_codec="aac",
        crf=20, max_width=1920, extra_args=("-pix_fmt", "yuv420p", "-movflags", "+faststart"),
    ),
    "mov_high_quality": ExportPreset(
        name="MOV (High Quality H.264)", container="mov", video_codec="libx264", audio_codec="aac",
        crf=15, extra_args=("-pix_fmt", "yuv420p", "-profile:v", "high"),
    ),
    "webm_vp9": ExportPreset(
        name="WebM (VP9/Opus)", container="webm", video_codec="libvpx-vp9", audio_codec="libopus",
        crf=30, extra_args=("-b:v", "0", "-row-mt", "1"),
    ),
    "instagram_reel": ExportPreset(
        name="Instagram Reel (1080x1920 MP4)", container="mp4", video_codec="libx264", audio_codec="aac",
        crf=20, target_resolution=(1080, 1920),
        extra_args=("-pix_fmt", "yuv420p", "-movflags", "+faststart"),
    ),
}


def export(
    input_path: str | Path,
    output_path: str | Path,
    preset: str | ExportPreset = "web_mp4",
    resolution: tuple[int, int] | None = None,
    bitrate: str | None = None,
) -> Path:
    """Export a clip through a named preset (or a custom ExportPreset), with
    optional resolution/bitrate overrides.
    """
    input_path = Path(input_path)
    preset_obj = PRESETS[preset] if isinstance(preset, str) else preset
    output_path = Path(output_path).with_suffix(f".{preset_obj.container}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    info = probe(input_path)
    vf_parts = []
    if resolution:
        vf_parts.append(f"scale={resolution[0]}:{resolution[1]}")
    elif preset_obj.target_resolution:
        tw, th = preset_obj.target_resolution
        vf_parts.append(
            f"scale={tw}:{th}:force_original_aspect_ratio=decrease,"
            f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2,setsar=1"
        )
    elif preset_obj.max_width and info.width > preset_obj.max_width:
        vf_parts.append(f"scale={preset_obj.max_width}:-2")

    args = ["-i", str(input_path)]
    if vf_parts:
        args += ["-vf", ",".join(vf_parts)]
    args += ["-c:v", preset_obj.video_codec]
    if bitrate or preset_obj.video_bitrate:
        args += ["-b:v", bitrate or preset_obj.video_bitrate]
    elif preset_obj.crf is not None:
        args += ["-crf", str(preset_obj.crf)]
    args += list(preset_obj.extra_args)
    if info.has_audio:
        args += ["-c:a", preset_obj.audio_codec, "-b:a", preset_obj.audio_bitrate]
    else:
        args += ["-an"]
    args += [str(output_path)]
    run_ffmpeg(args)
    return output_path
