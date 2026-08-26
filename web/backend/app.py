"""Granite Studios web service — HTTP wrapper around the desktop app's
photo and video engines.

The processing code is imported from ``src/`` unchanged: this module only
handles uploads, job bookkeeping, and serving results, so the desktop app
and the website always run the exact same image/video pipeline.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

# The engines live in the repo root's src/ package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ai_tools.background_removal import BackgroundRemover  # noqa: E402
from src.ai_tools.color_engine import ColorEngine  # noqa: E402
from src.engine.universal_loader import UniversalLoader  # noqa: E402
from src.social.ig_sync import IgFormatter  # noqa: E402
from src.social.reel_maker import ScrlReelMaker  # noqa: E402
from src.video import audio as video_audio  # noqa: E402
from src.video import color as video_color  # noqa: E402
from src.video import export as video_export  # noqa: E402
from src.video import spatial as video_spatial  # noqa: E402
from src.video import structure as video_structure  # noqa: E402
from src.video import timing as video_timing  # noqa: E402
from src.video.ffmpeg_utils import probe as video_probe  # noqa: E402

from storage import SessionStorage  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STORAGE_ROOT = Path(os.environ.get("GRANITE_STORAGE_ROOT", "/tmp/granite-web"))
MAX_UPLOAD_MB = int(os.environ.get("GRANITE_MAX_UPLOAD_MB", "200"))
# Comma-separated list; defaults to the portfolio site plus local dev.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "GRANITE_ALLOWED_ORIGINS",
        "https://marufhoque.com,https://www.marufhoque.com,http://localhost:5173",
    ).split(",")
    if o.strip()
]

storage = SessionStorage(STORAGE_ROOT)
loader = UniversalLoader()
color_engine = ColorEngine()
bg_remover = BackgroundRemover()
ig_formatter = IgFormatter(loader=loader)
reel_maker = ScrlReelMaker(loader=loader)

app = FastAPI(title="Granite Studios Web API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    # Any localhost port, so a dev server on whatever port works without
    # reconfiguring the backend. Safe because a localhost origin is only
    # reachable from the developer's own machine.
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------- #
# Job tracking — video work is far too slow for a request/response cycle,
# so those endpoints return a job id the client polls.
# --------------------------------------------------------------------- #


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"  # queued | running | done | failed
    message: str = ""
    error: str | None = None
    result_file_id: str | None = None
    created_at: float = field(default_factory=time.time)


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()


def _start_job(kind: str, work: Callable[[Job], str]) -> Job:
    """Run ``work`` on a daemon thread; it returns the result file id."""
    job = Job(id=uuid.uuid4().hex, kind=kind)
    with _jobs_lock:
        _jobs[job.id] = job

    def runner() -> None:
        job.status = "running"
        try:
            job.result_file_id = work(job)
            job.status = "done"
            job.message = "Complete"
        except Exception as exc:
            logger.exception("Job %s (%s) failed", job.id, kind)
            job.status = "failed"
            job.error = str(exc)

    threading.Thread(target=runner, daemon=True).start()
    return job


# --------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------- #


class PhotoEditRequest(BaseModel):
    file_id: str
    operation: str  # remove_background | cinematic_filter | selective_color | auto_fix | adjust
    filter_name: str | None = None
    color_name: str | None = None
    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0


class IgFormatRequest(BaseModel):
    file_id: str
    target: str = "feed"  # feed | story


class ReelRequest(BaseModel):
    file_ids: list[str]
    duration: float = 10.0


class VideoEditRequest(BaseModel):
    file_id: str
    operations: list[dict[str, Any]] = []
    preset: str = "web_mp4"


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _require_file(file_id: str, kind: str | None = None):
    stored = storage.get(file_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="File not found or session expired")
    if kind and stored.kind != kind:
        raise HTTPException(status_code=400, detail=f"Expected a {kind} file")
    return stored


def _load_image_array(path: Path) -> np.ndarray:
    """Decode any supported still (RAW/HEIC/JPEG/PNG) to an RGB array."""
    array = loader.load_array(path)
    if array is None:
        raise HTTPException(status_code=400, detail="Could not decode that image")
    return array


def _save_image(array_or_image, dest: Path) -> Path:
    image = (
        array_or_image
        if isinstance(array_or_image, Image.Image)
        else Image.fromarray(array_or_image)
    )
    if dest.suffix.lower() == ".png":
        image.save(dest, "PNG", optimize=True)
    else:
        image.save(dest, "JPEG", quality=94)
    return dest


# --------------------------------------------------------------------- #
# Meta endpoints
# --------------------------------------------------------------------- #


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/capabilities")
def capabilities():
    """Everything the UI needs to render its controls — kept server-side so
    the frontend never drifts from what the engines actually support."""
    return {
        "photo": {
            "filters": list(ColorEngine.FILTERS),
            "colors": list(ColorEngine.COLORS),
            "operations": [
                "remove_background",
                "cinematic_filter",
                "selective_color",
                "auto_fix",
                "adjust",
            ],
        },
        "video": {
            "operations": [
                "trim", "reverse", "change_speed", "timelapse", "freeze_frame",
                "convert_aspect_ratio", "auto_crop_subject", "stabilize",
                "color_correct", "reduce_noise", "normalize_loudness",
            ],
            "aspect_ratios": list(video_spatial.ASPECT_PRESETS.keys()),
            "presets": {k: v.name for k, v in video_export.PRESETS.items()},
            "transitions": ["fade", "dissolve", "wipeleft", "wiperight", "slideup", "circleopen"],
        },
        "limits": {"max_upload_mb": MAX_UPLOAD_MB},
    }


@app.post("/api/session")
def create_session():
    storage.sweep()
    return {"session_id": storage.new_session_id()}


# --------------------------------------------------------------------- #
# Upload / download
# --------------------------------------------------------------------- #


@app.post("/api/upload")
async def upload(session_id: str = Form(...), file: UploadFile = File(...)):
    data = await file.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=413, detail=f"File is {size_mb:.0f}MB; the limit is {MAX_UPLOAD_MB}MB"
        )
    try:
        stored = storage.save_upload(session_id, file.filename or "upload", data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    info: dict[str, Any] = {
        "file_id": stored.file_id,
        "name": stored.original_name,
        "kind": stored.kind,
        "size_mb": round(size_mb, 2),
    }
    if stored.kind == "video":
        try:
            probed = video_probe(stored.path)
            info["video"] = {
                "width": probed.width,
                "height": probed.height,
                "fps": probed.fps,
                "duration_sec": probed.duration_sec,
                "duration_label": probed.duration_label,
                "has_audio": probed.has_audio,
                "codec": probed.codec_fourcc,
            }
        except Exception:
            logger.warning("Could not probe uploaded video %s", stored.path, exc_info=True)
    else:
        meta = loader.read_metadata(stored.path)
        info["metadata"] = {"gear": meta.gear_label, "settings": meta.settings_label}
    return info


@app.get("/api/file/{file_id}")
def get_file(file_id: str, download: bool = False):
    stored = _require_file(file_id)
    return FileResponse(
        stored.path,
        filename=stored.original_name if download else None,
        content_disposition_type="attachment" if download else "inline",
    )


@app.get("/api/preview/{file_id}")
def preview(file_id: str, max_px: int = 1400):
    """Downscaled JPEG preview — keeps RAW/HEIC and huge stills viewable in a
    browser that can't decode them natively."""
    stored = _require_file(file_id, kind="image")
    preview_path = stored.path.with_name(f"{stored.path.stem}_preview.jpg")
    if not preview_path.exists():
        image = loader.load_preview(stored.path)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode that image")
        image = image.copy()
        image.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
        image.convert("RGB").save(preview_path, "JPEG", quality=88)
    return FileResponse(preview_path)


# --------------------------------------------------------------------- #
# Photo endpoints — fast enough to answer inline
# --------------------------------------------------------------------- #


@app.post("/api/photo/edit")
def photo_edit(req: PhotoEditRequest):
    stored = _require_file(req.file_id, kind="image")
    array = _load_image_array(stored.path)
    op = req.operation
    stem = Path(stored.original_name).stem

    if op == "remove_background":
        rgba = bg_remover.remove_background_array(array)
        dest = storage.output_path(stored.session_id, stored.path.stem, "nobg", ".png")
        _save_image(Image.fromarray(rgba, "RGBA"), dest)
        out_name = f"{stem}_nobg.png"
    elif op == "cinematic_filter":
        if not req.filter_name:
            raise HTTPException(status_code=400, detail="filter_name is required")
        try:
            result = color_engine.apply_lut_filter(array, req.filter_name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        dest = storage.output_path(stored.session_id, stored.path.stem, req.filter_name, ".jpg")
        _save_image(result, dest)
        out_name = f"{stem}_{req.filter_name}.jpg"
    elif op == "selective_color":
        if not req.color_name:
            raise HTTPException(status_code=400, detail="color_name is required")
        try:
            result = color_engine.selective_color(array, req.color_name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        dest = storage.output_path(stored.session_id, stored.path.stem, f"only-{req.color_name}", ".jpg")
        _save_image(result, dest)
        out_name = f"{stem}_only-{req.color_name}.jpg"
    elif op == "auto_fix":
        result = color_engine.smart_auto_fix(array)
        dest = storage.output_path(stored.session_id, stored.path.stem, "autofix", ".jpg")
        _save_image(result, dest)
        out_name = f"{stem}_autofix.jpg"
    elif op == "adjust":
        result = color_engine.apply_adjustments(
            array, brightness=req.brightness, contrast=req.contrast, saturation=req.saturation
        )
        dest = storage.output_path(stored.session_id, stored.path.stem, "adjusted", ".jpg")
        _save_image(result, dest)
        out_name = f"{stem}_adjusted.jpg"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown operation: {op}")

    result_file = storage.register(stored.session_id, dest, out_name, "image")
    return {"file_id": result_file.file_id, "name": out_name, "kind": "image"}


@app.post("/api/photo/instagram")
def photo_instagram(req: IgFormatRequest):
    stored = _require_file(req.file_id, kind="image")
    if req.target not in ("feed", "story"):
        raise HTTPException(status_code=400, detail="target must be 'feed' or 'story'")
    dest = storage.output_path(stored.session_id, stored.path.stem, f"ig-{req.target}", ".jpg")
    ig_formatter.export(stored.path, dest, target=req.target)
    out_name = f"{Path(stored.original_name).stem}_ig-{req.target}.jpg"
    result_file = storage.register(stored.session_id, dest, out_name, "image")
    return {"file_id": result_file.file_id, "name": out_name, "kind": "image"}


# --------------------------------------------------------------------- #
# Video + reel endpoints — job-based
# --------------------------------------------------------------------- #

_VIDEO_OPS: dict[str, Callable[..., Path]] = {
    "trim": video_timing.trim,
    "reverse": video_timing.reverse,
    "change_speed": video_timing.change_speed,
    "timelapse": video_timing.extract_timelapse,
    "freeze_frame": video_timing.freeze_frame,
    "convert_aspect_ratio": video_spatial.convert_aspect_ratio,
    "crop_box": video_spatial.crop_box,
    "auto_crop_subject": video_spatial.auto_crop_subject,
    "stabilize": video_spatial.stabilize,
    "static_mask": video_spatial.apply_static_mask,
    "color_correct": video_color.color_correct,
    "reduce_noise": video_audio.reduce_noise,
    "normalize_loudness": video_audio.normalize_loudness,
    "adjust_volume": video_audio.adjust_volume,
}


@app.post("/api/video/edit")
def video_edit(req: VideoEditRequest):
    stored = _require_file(req.file_id, kind="video")
    if req.preset not in video_export.PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {req.preset}")
    for spec in req.operations:
        if spec.get("name") not in _VIDEO_OPS:
            raise HTTPException(status_code=400, detail=f"Unknown operation: {spec.get('name')}")

    def work(job: Job) -> str:
        current = stored.path
        session_dir = storage.session_dir(stored.session_id)
        for i, spec in enumerate(req.operations):
            name = spec["name"]
            params = spec.get("params", {}) or {}
            job.message = f"{name} ({i + 1}/{len(req.operations)})"
            step_out = session_dir / f"{stored.path.stem}_s{i:02d}_{name}.mp4"
            current = _VIDEO_OPS[name](current, step_out, **params)

        job.message = "Encoding final export"
        final_out = session_dir / f"{stored.path.stem}_final"
        result_path = video_export.export(current, final_out, preset=req.preset)
        out_name = f"{Path(stored.original_name).stem}_edited{result_path.suffix}"
        return storage.register(stored.session_id, result_path, out_name, "video").file_id

    job = _start_job("video_edit", work)
    return {"job_id": job.id, "status": job.status}


@app.post("/api/video/reel")
def video_reel(req: ReelRequest):
    if not 3 <= len(req.file_ids) <= 7:
        raise HTTPException(status_code=400, detail="Select between 3 and 7 images")
    stored_files = [_require_file(fid, kind="image") for fid in req.file_ids]
    session_id = stored_files[0].session_id

    def work(job: Job) -> str:
        job.message = "Stitching panorama and rendering reel"
        dest = storage.session_dir(session_id) / f"panoramic_reel_{uuid.uuid4().hex[:8]}.mp4"
        reel_maker.create_panoramic_reel(
            [f.path for f in stored_files], dest, duration=req.duration
        )
        return storage.register(session_id, dest, "panoramic_reel.mp4", "video").file_id

    job = _start_job("reel", work)
    return {"job_id": job.id, "status": job.status}


@app.get("/api/job/{job_id}")
def get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.id,
        "kind": job.kind,
        "status": job.status,
        "message": job.message,
        "error": job.error,
        "result_file_id": job.result_file_id,
    }
