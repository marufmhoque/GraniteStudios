"""Color correction and LUT application for video."""

from __future__ import annotations

from pathlib import Path

from .ffmpeg_utils import probe, run_ffmpeg


def color_correct(
    input_path: str | Path,
    output_path: str | Path,
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    gamma: float = 1.0,
    white_balance_temp: float = 0.0,
) -> Path:
    """Exposure/contrast/saturation/gamma + a simple temperature-based white
    balance shift (positive = warmer, negative = cooler), all in one filter
    chain so it's a single encode pass.

    ``brightness`` is additive in [-1, 1]; ``contrast``/``saturation``/``gamma``
    are multiplicative around 1.0; ``white_balance_temp`` is in [-1, 1].
    """
    input_path, output_path = Path(input_path), Path(output_path)
    info = probe(input_path)
    filters = [f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation}:gamma={gamma}"]
    if abs(white_balance_temp) > 1e-6:
        shift = max(-1.0, min(1.0, white_balance_temp)) * 0.3
        filters.append(f"colorbalance=rs={shift:.3f}:bs={-shift:.3f}")
    vf = ",".join(filters)
    args = ["-i", str(input_path), "-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "18"]
    args += ["-c:a", "copy"] if info.has_audio else ["-an"]
    args += [str(output_path)]
    run_ffmpeg(args)
    return output_path


def apply_lut(input_path: str | Path, output_path: str | Path, cube_path: str | Path) -> Path:
    """Apply a 3D LUT (.cube file) via ffmpeg's lut3d filter."""
    input_path, output_path, cube_path = Path(input_path), Path(output_path), Path(cube_path)
    if not cube_path.exists():
        raise FileNotFoundError(f"LUT file not found: {cube_path}")
    info = probe(input_path)
    # ffmpeg filter paths need forward slashes and escaped colons on Windows.
    safe_lut_path = str(cube_path).replace("\\", "/").replace(":", "\\:")
    vf = f"lut3d=file='{safe_lut_path}'"
    args = ["-i", str(input_path), "-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "18"]
    args += ["-c:a", "copy"] if info.has_audio else ["-an"]
    args += [str(output_path)]
    run_ffmpeg(args)
    return output_path
