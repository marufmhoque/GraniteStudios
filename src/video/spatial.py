"""Spatial & framing adjustments: aspect ratio conversion, cropping,
subject-aware auto-crop, basic masking, and frame stabilization.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .ffmpeg_utils import FFmpegError, probe, run_ffmpeg

ASPECT_PRESETS = {
    "16:9": (16, 9),
    "9:16": (9, 16),
    "1:1": (1, 1),
    "4:5": (4, 5),
    "4:3": (4, 3),
}


def _parse_ratio(text: str) -> tuple[int, int]:
    try:
        w, h = text.split(":")
        return int(w), int(h)
    except Exception as exc:
        raise ValueError(f"Bad aspect ratio: {text!r} (expected e.g. '9:16')") from exc


def convert_aspect_ratio(
    input_path: str | Path, output_path: str | Path, target: str, mode: str = "crop"
) -> Path:
    """Convert to a target aspect ratio (e.g. "9:16") either by center-cropping
    (``mode="crop"``, fills the frame, loses edges) or letterbox padding
    (``mode="pad"``, keeps the whole frame, adds black bars).
    """
    input_path, output_path = Path(input_path), Path(output_path)
    info = probe(input_path)
    tw, th = ASPECT_PRESETS.get(target, _parse_ratio(target))
    target_ratio = tw / th
    src_ratio = info.width / info.height

    if mode == "crop":
        if src_ratio > target_ratio:
            new_w = int(info.height * target_ratio) // 2 * 2
            vf = f"crop={new_w}:{info.height}:(iw-{new_w})/2:0"
        else:
            new_h = int(info.width / target_ratio) // 2 * 2
            vf = f"crop={info.width}:{new_h}:0:(ih-{new_h})/2"
    else:  # pad / letterbox
        if src_ratio > target_ratio:
            out_w = info.width
            out_h = int(info.width / target_ratio) // 2 * 2
        else:
            out_h = info.height
            out_w = int(info.height * target_ratio) // 2 * 2
        vf = f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:color=black"

    args = ["-i", str(input_path), "-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "18"]
    args += ["-c:a", "aac"] if info.has_audio else ["-an"]
    args += [str(output_path)]
    run_ffmpeg(args)
    return output_path


def crop_box(
    input_path: str | Path,
    output_path: str | Path,
    x: float,
    y: float,
    w: float,
    h: float,
    normalized: bool = True,
) -> Path:
    """Crop to an explicit bounding box. Coordinates are fractions of the
    frame (0..1) by default (``normalized=True``); pass pixel values with
    ``normalized=False``.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    info = probe(input_path)
    if normalized:
        px, py = int(x * info.width), int(y * info.height)
        pw, ph = int(w * info.width), int(h * info.height)
    else:
        px, py, pw, ph = int(x), int(y), int(w), int(h)
    pw, ph = pw // 2 * 2, ph // 2 * 2
    vf = f"crop={pw}:{ph}:{px}:{py}"
    args = ["-i", str(input_path), "-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "18"]
    args += ["-c:a", "aac"] if info.has_audio else ["-an"]
    args += [str(output_path)]
    run_ffmpeg(args)
    return output_path


def _get_face_cascade():
    """Best-effort Haar-cascade face detector.

    Returns None if this OpenCV build doesn't ship CascadeClassifier or its
    cascade data files (true of some recent opencv-python-headless releases,
    which dropped the bundled objdetect data) — callers fall back to a
    center crop in that case rather than crashing.
    """
    if not hasattr(cv2, "CascadeClassifier"):
        return None
    cascade_dir = getattr(getattr(cv2, "data", None), "haarcascades", None)
    if not cascade_dir:
        return None
    cascade_path = Path(cascade_dir) / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        return None
    cascade = cv2.CascadeClassifier(str(cascade_path))
    return None if cascade.empty() else cascade


def auto_crop_subject(
    input_path: str | Path,
    output_path: str | Path,
    target_aspect: str = "9:16",
    sample_frames: int = 20,
) -> Path:
    """Subject-aware crop: samples frames, runs Haar-cascade face detection,
    and centers the crop box on the median detected face position — falling
    back to a plain center crop if no faces are found (or face detection
    isn't available in this OpenCV build) anywhere in the sample.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    info = probe(input_path)
    tw, th = ASPECT_PRESETS.get(target_aspect, _parse_ratio(target_aspect))
    target_ratio = tw / th

    centers = []
    cascade = _get_face_cascade()
    if cascade is not None:
        cap = cv2.VideoCapture(str(input_path))
        try:
            step = max(1, info.frame_count // sample_frames) if info.frame_count else 1
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % step == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))
                    for (fx, fy, fw, fh) in faces:
                        centers.append((fx + fw / 2, fy + fh / 2))
                idx += 1
        finally:
            cap.release()

    if centers:
        cx = float(np.median([c[0] for c in centers]))
        cy = float(np.median([c[1] for c in centers]))
    else:
        cx, cy = info.width / 2, info.height / 2

    if info.width / info.height > target_ratio:
        crop_h = info.height
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = info.width
        crop_h = int(crop_w / target_ratio)
    crop_w, crop_h = crop_w // 2 * 2, crop_h // 2 * 2

    px = int(min(max(cx - crop_w / 2, 0), max(0, info.width - crop_w)))
    py = int(min(max(cy - crop_h / 2, 0), max(0, info.height - crop_h)))

    return crop_box(input_path, output_path, px, py, crop_w, crop_h, normalized=False)


def apply_static_mask(
    input_path: str | Path,
    output_path: str | Path,
    x: float,
    y: float,
    w: float,
    h: float,
    feather: int = 0,
) -> Path:
    """Blank out (or blur) a rectangular region for the whole clip — privacy
    redaction. Coordinates are normalized fractions of the frame.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    info = probe(input_path)
    px, py = int(x * info.width), int(y * info.height)
    pw, ph = int(w * info.width), int(h * info.height)
    if feather > 0:
        vf = (
            f"split[base][blurred];"
            f"[blurred]crop={pw}:{ph}:{px}:{py},boxblur=20:5[blurredcrop];"
            f"[base][blurredcrop]overlay={px}:{py}"
        )
    else:
        vf = f"drawbox=x={px}:y={py}:w={pw}:h={ph}:color=black:t=fill"
    args = ["-i", str(input_path), "-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "18"]
    args += ["-c:a", "copy"] if info.has_audio else ["-an"]
    args += [str(output_path)]
    run_ffmpeg(args)
    return output_path


def _moving_average(trajectory: np.ndarray, window: int) -> np.ndarray:
    smoothed = np.copy(trajectory)
    radius = window // 2
    for i in range(len(trajectory)):
        lo = max(0, i - radius)
        hi = min(len(trajectory), i + radius + 1)
        smoothed[i] = trajectory[lo:hi].mean(axis=0)
    return smoothed


def stabilize(input_path: str | Path, output_path: str | Path, smoothing_window: int = 30) -> Path:
    """Frame stabilization via OpenCV point-feature tracking + trajectory
    smoothing (a moving average over ``smoothing_window`` frames). Writes a
    stabilized silent video with OpenCV, then remuxes the original audio
    back on with ffmpeg.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    info = probe(input_path)
    cap = cv2.VideoCapture(str(input_path))

    transforms = []
    ok, prev = cap.read()
    if not ok:
        cap.release()
        raise FFmpegError(f"Could not read frames from {input_path}")
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
        dx, dy, da = 0.0, 0.0, 0.0
        if prev_pts is not None:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
            if curr_pts is not None:
                valid_prev = prev_pts[status == 1]
                valid_curr = curr_pts[status == 1]
                if len(valid_prev) >= 6:
                    m, _ = cv2.estimateAffinePartial2D(valid_prev, valid_curr)
                    if m is not None:
                        dx, dy = m[0, 2], m[1, 2]
                        da = np.arctan2(m[1, 0], m[0, 0])
        transforms.append((dx, dy, da))
        prev_gray = gray
    cap.release()

    trajectory = np.cumsum(np.array(transforms), axis=0) if transforms else np.zeros((1, 3))
    smoothed = _moving_average(trajectory, smoothing_window)
    corrections = smoothed - trajectory

    cap = cv2.VideoCapture(str(input_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_silent = output_path.with_suffix(".silent.mp4")
    writer = cv2.VideoWriter(str(tmp_silent), fourcc, info.fps, (info.width, info.height))

    ok, frame = cap.read()
    if ok:
        writer.write(frame)
    for dx, dy, da in corrections[1:] if len(corrections) > 1 else []:
        ok, frame = cap.read()
        if not ok:
            break
        m = np.array([[np.cos(da), -np.sin(da), dx], [np.sin(da), np.cos(da), dy]], dtype=np.float32)
        stabilized = cv2.warpAffine(frame, m, (info.width, info.height))
        writer.write(stabilized)
    cap.release()
    writer.release()

    args = ["-i", str(tmp_silent)]
    if info.has_audio:
        args += ["-i", str(input_path), "-map", "0:v", "-map", "1:a", "-c:a", "aac"]
    args += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", str(output_path)]
    run_ffmpeg(args)
    tmp_silent.unlink(missing_ok=True)
    return output_path
