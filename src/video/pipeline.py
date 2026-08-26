"""Job orchestration: chain video operations into one pipeline, run them
asynchronously through a hardware-sized thread pool, and track status for
both the desktop UI and the local REST API.
"""

from __future__ import annotations

import logging
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import psutil

from . import audio as audio_ops
from . import color as color_ops
from . import export as export_ops
from . import spatial as spatial_ops
from . import timing as timing_ops
from .normalize import normalize_video

logger = logging.getLogger(__name__)

# Operation name -> implementing function. Every entry here takes
# (input_path, output_path, **params) and returns the output Path, so the
# pipeline can dispatch generically. Multi-clip operations (concat,
# transitions, J/L cuts — see structure.py) don't fit that single-input
# shape and are called directly instead of through this linear chain.
_OPERATIONS: dict[str, Callable[..., Path]] = {
    "normalize": normalize_video,
    "trim": timing_ops.trim,
    "reverse": timing_ops.reverse,
    "freeze_frame": timing_ops.freeze_frame,
    "change_speed": timing_ops.change_speed,
    "speed_ramp": timing_ops.speed_ramp,
    "timelapse": timing_ops.extract_timelapse,
    "convert_aspect_ratio": spatial_ops.convert_aspect_ratio,
    "crop_box": spatial_ops.crop_box,
    "auto_crop_subject": spatial_ops.auto_crop_subject,
    "stabilize": spatial_ops.stabilize,
    "static_mask": spatial_ops.apply_static_mask,
    "color_correct": color_ops.color_correct,
    "apply_lut": color_ops.apply_lut,
    "normalize_loudness": audio_ops.normalize_loudness,
    "reduce_noise": audio_ops.reduce_noise,
    "adjust_volume": audio_ops.adjust_volume,
}


@dataclass
class VideoOperation:
    """One step in a pipeline: a registered op name + its kwargs (everything
    except input_path/output_path, which the pipeline fills in as it chains
    each step's output into the next step's input).
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoJob:
    id: str
    input_path: Path
    operations: list[VideoOperation]
    output_path: Path
    preset: str = "web_mp4"
    status: str = "queued"  # queued -> running -> done | failed
    progress: float = 0.0
    error: str | None = None
    result_path: Path | None = None
    created_at: float = field(default_factory=time.time)


class VideoPipeline:
    """Executes one VideoJob's operations in sequence through temp intermediates."""

    def run_job(self, job: VideoJob, on_progress: Callable[[float], None] | None = None) -> Path:
        job.status = "running"
        current: Path = Path(job.input_path)
        with tempfile.TemporaryDirectory(prefix="gs_video_") as tmp:
            tmp_dir = Path(tmp)
            total_steps = len(job.operations) + 1  # +1 for the final export step
            for i, op in enumerate(job.operations):
                fn = _OPERATIONS.get(op.name)
                if fn is None:
                    raise ValueError(f"Unknown video operation: {op.name!r}")
                step_out = tmp_dir / f"step_{i:03d}_{op.name}.mp4"
                current = fn(current, step_out, **op.params)
                job.progress = (i + 1) / total_steps
                if on_progress:
                    on_progress(job.progress)

            result = export_ops.export(current, job.output_path, preset=job.preset)
            job.progress = 1.0
            if on_progress:
                on_progress(1.0)

        job.result_path = result
        job.status = "done"
        return result


def _default_worker_count() -> int:
    """Video encoding is CPU-bound and memory-hungry; leave headroom for the
    rest of the app instead of using every core — the same conservative
    philosophy as get_safe_export_ram_gb() in engine/hardware.py.
    """
    cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 2
    return max(1, min(cores - 1, 4))


class JobManager:
    """In-memory job queue + thread pool, shared by the desktop UI and the
    optional local REST API (api.py) so both see the same job state.
    """

    def __init__(self, max_workers: int | None = None):
        self._jobs: dict[str, VideoJob] = {}
        self._pipeline = VideoPipeline()
        self._executor = ThreadPoolExecutor(max_workers=max_workers or _default_worker_count())

    def submit(
        self,
        input_path: str | Path,
        operations: list[VideoOperation],
        output_path: str | Path,
        preset: str = "web_mp4",
    ) -> VideoJob:
        job = VideoJob(
            id=str(uuid.uuid4()),
            input_path=Path(input_path),
            operations=operations,
            output_path=Path(output_path),
            preset=preset,
        )
        self._jobs[job.id] = job
        self._executor.submit(self._run, job)
        return job

    def _run(self, job: VideoJob) -> None:
        try:
            self._pipeline.run_job(job)
        except Exception as exc:  # surface every failure through job state
            logger.exception("Video job %s failed", job.id)
            job.status = "failed"
            job.error = str(exc)

    def get(self, job_id: str) -> VideoJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[VideoJob]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
