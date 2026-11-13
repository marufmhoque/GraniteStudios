"""Turning edited frames into files: presets, batches and delivery copies.

This is the one place recipes become pixels on disk. Everything else in the
app edits settings; Export renders them. The source files are only ever
read, and a rejected frame is never exported (the caller passes the frames
that are left once rejects are dropped).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from src.engine import develop_pipeline as dp
from src.engine import fast_preview
from src.engine import layer_stack
from src.engine.parametric_recipe import Recipe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Preset:
    name: str
    fmt: str                    # see develop_pipeline.EXPORT_FORMATS
    quality: int = 92
    max_side: int | None = None
    dpi: int | None = None
    suffix: str = ""            # added to the filename, e.g. "_web"

    @property
    def extension(self) -> str:
        return {"jpeg": ".jpg", "png": ".png", "png16": ".png", "tiff": ".tif", "tiff16": ".tif"}[self.fmt]


PRESETS = {
    "web": Preset("Web JPEG (2048 px)", "jpeg", 85, 2048, None, "_web"),
    "full": Preset("Full-size JPEG", "jpeg", 95),
    "tiff16": Preset("16-bit TIFF (for further editing)", "tiff16", suffix="_16bit"),
    "print": Preset("Print, 300 dpi JPEG", "jpeg", 97, None, 300, "_print"),
    "png16": Preset("16-bit PNG", "png16", suffix="_16bit"),
}


def watermark_items(kind: str, name: str = ""):
    """Vector items for a batch watermark, or an empty list for none."""
    from src.graphics import vector_canvas as vc

    if kind == "signature" and name.strip():
        return vc.watermark_signature(name.strip())
    if kind == "seal" and name.strip():
        return vc.watermark_seal(name.strip().upper())
    if kind == "exif_strip":
        return vc.watermark_exif_strip()
    return []


@dataclass
class FrameJob:
    """Everything needed to render one frame."""

    path: Path
    recipe: Recipe
    context: layer_stack.RenderContext | None = None
    inpainter: object | None = None
    metadata: object | None = None


def render_frame(job: FrameJob, bits: int = 8) -> np.ndarray:
    """A frame developed through its recipe, layers and overlays included."""
    source = dp.open_source(job.path)
    if source is None:
        raise IOError(f"Could not read {job.path.name}")
    context = job.context
    if context is not None and context.metadata is None:
        context.metadata = job.metadata
    return dp.render(source, job.recipe, bits=bits, layers=context, inpainter=job.inpainter)


def _names(paths: list[Path]) -> dict[Path, str]:
    """Output stems, made unique when two sources share one (IMG_1.CR2 + IMG_1.JPG)."""
    counts: dict[str, int] = {}
    for path in paths:
        counts[path.stem.lower()] = counts.get(path.stem.lower(), 0) + 1
    return {p: (f"{p.stem}_{p.suffix[1:].lower()}" if counts[p.stem.lower()] > 1 else p.stem) for p in paths}


def export_batch(
    jobs: list[FrameJob],
    destination: str | Path,
    preset: Preset,
    watermark=None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Render each job with ``preset`` into ``destination``.

    ``watermark`` is a list of vector items drawn over every frame, filled
    in from that frame's own metadata. Returns ``(written, failures)``. One
    unreadable frame does not stop the rest.
    """
    from src.graphics import vector_canvas as vc

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    names = _names([job.path for job in jobs])
    written, failures = [], []
    bits = 16 if preset.fmt.endswith("16") else 8
    for index, job in enumerate(jobs, start=1):
        target = destination / f"{names[job.path]}{preset.suffix}{preset.extension}"
        try:
            if target.resolve() == job.path.resolve():
                raise ValueError("refusing to export over the original file")
            image = render_frame(job, bits=bits)
            if preset.max_side:
                image = fast_preview.fit(image, preset.max_side)
            if watermark:
                image = vc.composite_onto(image, watermark, job.metadata)
            dp.write_image(image, target, preset.fmt, preset.quality, preset.dpi)
            written.append(target)
        except Exception as exc:
            logger.warning("Export of %s failed: %s", job.path.name, exc)
            failures.append((job.path, str(exc)))
        if progress:
            progress(index, len(jobs))
    return written, failures


def delivery_copy(job: FrameJob, edit_folder: str | Path, quality: int = 95) -> Path:
    """A full-size JPEG of an edited frame in the edit folder.

    For the publishing tools (gallery, reel, Instagram), which read finished
    images. Re-rendered every time, so it always matches the current recipe.
    """
    target = Path(edit_folder) / f"{job.path.stem}_edit.jpg"
    dp.write_image(render_frame(job), target, "jpeg", quality)
    return target
