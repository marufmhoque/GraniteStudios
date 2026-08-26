"""Optional local REST API for programmatic video editing — exposes the
video engine over HTTP so external scripts/tools can drive it without
importing Python directly. Only runs if the user starts it (see the
"Start Local API Server" toggle on the Video Editor tab); the desktop app
works fully without it.

Requires: pip install fastapi uvicorn
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from .ffmpeg_utils import probe
from .pipeline import JobManager, VideoJob, VideoOperation
from .structure import add_transition, concat_clips

if TYPE_CHECKING:
    from pydantic import BaseModel
else:
    try:
        from pydantic import BaseModel
    except ImportError:
        BaseModel = None

_job_manager: JobManager | None = None


def get_job_manager() -> JobManager:
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager


def _job_summary(job: VideoJob) -> dict:
    return {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress,
        "error": job.error,
        "output_path": str(job.result_path) if job.result_path else None,
    }


# Request models live at module level (not nested inside create_app()) so
# their type hints resolve against real module globals — this file uses
# postponed annotation evaluation (PEP 563), and pydantic/FastAPI can't
# resolve a forward reference like list["OperationSpec"] against a class
# that only exists as a local variable inside a function.
if BaseModel is not None:

    class OperationSpec(BaseModel):
        name: str
        params: dict[str, Any] = {}

    class JobRequest(BaseModel):
        input_path: str
        output_path: str
        operations: list[OperationSpec] = []
        preset: str = "web_mp4"

    class ConcatRequest(BaseModel):
        input_paths: list[str]
        output_path: str

    class TransitionRequest(BaseModel):
        clip_a: str
        clip_b: str
        output_path: str
        transition: str = "fade"
        duration: float = 1.0


def create_app():
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise ImportError(
            "The local API server needs fastapi + uvicorn: pip install fastapi uvicorn"
        ) from exc

    app = FastAPI(title="Granite Studios Video Engine", version="1.0")
    jobs = get_job_manager()

    @app.get("/probe")
    def probe_endpoint(path: str):
        try:
            info = probe(path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "duration_sec": info.duration_sec,
            "frame_count": info.frame_count,
            "codec_fourcc": info.codec_fourcc,
            "has_audio": info.has_audio,
            "container": info.container,
        }

    @app.post("/jobs")
    def submit_job(req: JobRequest):
        ops = [VideoOperation(name=o.name, params=o.params) for o in req.operations]
        job = jobs.submit(req.input_path, ops, req.output_path, preset=req.preset)
        return {"job_id": job.id, "status": job.status}

    @app.get("/jobs")
    def list_jobs():
        return [_job_summary(j) for j in jobs.list_jobs()]

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return _job_summary(job)

    @app.post("/concat")
    def concat_endpoint(req: ConcatRequest):
        try:
            out = concat_clips(req.input_paths, req.output_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"output_path": str(out)}

    @app.post("/transition")
    def transition_endpoint(req: TransitionRequest):
        try:
            out = add_transition(req.clip_a, req.clip_b, req.output_path, req.transition, req.duration)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"output_path": str(out)}

    return app


def run_api_server(host: str = "127.0.0.1", port: int = 8756, blocking: bool = False):
    """Start the REST API. By default runs uvicorn in a daemon background
    thread so it doesn't block the calling process (e.g. the PyQt UI).
    Returns the uvicorn.Server instance — set ``.should_exit = True`` on it
    to stop the server.
    """
    import uvicorn

    app = create_app()
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    if blocking:
        server.run()
        return server

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server
