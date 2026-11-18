"""Turning edited frames into files: presets, batches and delivery copies.

This is the one place recipes become pixels on disk. Everything else in the
app edits settings; Export renders them. The source files are only ever
read, and a rejected frame is never exported (the caller passes the frames
that are left once rejects are dropped).

The export queue
----------------
An :class:`ExportBundle` is one delivery: the photos a set of people picks
out (John and Jane together, Sam alone, everyone), a preset, a watermark and
a subfolder. :class:`MultiExportRunner` runs a list of them in one go. A
photo that two bundles need with the same preset and watermark is rendered
once and copied to the second folder, and the renders run on several cores
within the RAM budget from :mod:`src.engine.hardware`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
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
    canvas: tuple[int, int] | None = None   # padded to exactly this size (Instagram), nothing cropped

    @property
    def extension(self) -> str:
        return {"jpeg": ".jpg", "png": ".png", "png16": ".png", "tiff": ".tif", "tiff16": ".tif"}[self.fmt]


PRESETS = {
    "web": Preset("Web JPEG (2048 px)", "jpeg", 85, 2048, None, "_web"),
    "full": Preset("Full-size JPEG", "jpeg", 95),
    "tiff16": Preset("16-bit TIFF (for further editing)", "tiff16", suffix="_16bit"),
    "print": Preset("Print, 300 dpi JPEG", "jpeg", 97, None, 300, "_print"),
    "png16": Preset("16-bit PNG", "png16", suffix="_16bit"),
    "ig45": Preset("Instagram 4:5 (1080 \u00d7 1350)", "jpeg", 92, suffix="_ig", canvas=(1080, 1350)),
}

# Other names a preset is known by (the export queue's, and older ones).
PRESET_ALIASES = {
    "web jpeg": "web", "full-size jpeg": "full", "full size jpeg": "full", "16-bit tiff": "tiff16",
    "print 300 dpi": "print", "instagram 4:5": "ig45", "instagram": "ig45", "16-bit png": "png16",
}


def preset_key(name: str) -> str:
    """The ``PRESETS`` key for a key, a preset's display name or an alias."""
    if name in PRESETS:
        return name
    folded = str(name).strip().casefold()
    for key, preset in PRESETS.items():
        if preset.name.casefold() == folded:
            return key
    if folded in PRESET_ALIASES:
        return PRESET_ALIASES[folded]
    raise KeyError(f"Unknown export preset {name!r}")


def preset_for(name: str) -> Preset:
    return PRESETS[preset_key(name)]


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
    variant: str = ""           # added to the file name for a copy, e.g. "v2-moody"


def version_tag(version: int, name: str) -> str:
    """The file-name tag for a copy, numbered like its grid chip (the master
    is v1, so copy 1 is ``v2-moody-edit``). Empty for the master."""
    if version == 0:
        return ""
    slug = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")[:32]
    number = int(version) + 1
    return f"v{number}-{slug}" if slug else f"v{number}"


def render_frame(job: FrameJob, bits: int = 8) -> np.ndarray:
    """A frame developed through its recipe, layers and overlays included."""
    source = dp.open_source(job.path)
    if source is None:
        raise IOError(f"Could not read {job.path.name}")
    context = job.context
    if context is not None and context.metadata is None:
        context.metadata = job.metadata
    return dp.render(source, job.recipe, bits=bits, layers=context, inpainter=job.inpainter)


def finish_and_write(job: FrameJob, target: Path, preset: Preset, watermark=None) -> Path:
    """Render one frame with ``preset`` (size, watermark, canvas) to ``target``."""
    from src.graphics import vector_canvas as vc

    target = Path(target)
    if target.resolve() == job.path.resolve():
        raise ValueError("refusing to export over the original file")
    bits = 16 if preset.fmt.endswith("16") else 8
    image = render_frame(job, bits=bits)
    if preset.max_side:
        image = fast_preview.fit(image, preset.max_side)
    if watermark:
        image = vc.composite_onto(image, watermark, job.metadata)
    if preset.canvas:
        from PIL import Image

        from src.social.ig_sync import pad_to_canvas

        if image.dtype != np.uint8:
            image = (image.astype(np.float32) / 257.0).round().astype(np.uint8)
        image = np.asarray(pad_to_canvas(Image.fromarray(image), preset.canvas))
    dp.write_image(image, target, preset.fmt, preset.quality, preset.dpi)
    return target


def _names(paths: list[Path]) -> dict[Path, str]:
    """Output stems, made unique when two sources share one (IMG_1.CR2 + IMG_1.JPG)."""
    counts: dict[str, int] = {}
    for path in dict.fromkeys(paths):       # several versions of one file are one source
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
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    names = _names([job.path for job in jobs])
    written, failures = [], []
    for index, job in enumerate(jobs, start=1):
        variant = f"_{job.variant}" if job.variant else ""
        target = destination / f"{names[job.path]}{variant}{preset.suffix}{preset.extension}"
        try:
            written.append(finish_and_write(job, target, preset, watermark))
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


# --------------------------------------------------------------------- #
# The export queue: several deliveries, by person, in one run
# --------------------------------------------------------------------- #

QUEUE_FILENAME = "export-queue.granite.json"
EVERYONE = "Everyone"
MATCH_LABELS = {          # how the queue's segmented control names each mode
    "ALL_PROJECT": "Everyone",
    "ANY": "Any of these",
    "ALL": "Together (All)",
    "SOLO": "Solo portraits",
}


def folder_slug(name: str) -> str:
    """A bundle name as a folder name: ``John & Jane`` -> ``John_and_Jane``."""
    text = re.sub(r"\s*[&+]\s*", " and ", str(name))
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", text)
    text = re.sub(r"\s+", "_", text.strip()).strip("._")
    return text[:60] or "Export"


def bundle_title(names: list[str]) -> str:
    """The default name of a bundle for these people: ``John & Jane``,
    ``Ann, Bo & Cy``, or ``Everyone`` for nobody."""
    names = [n for n in names if n]
    if not names:
        return EVERYONE
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " & " + names[-1]


@dataclass
class ExportBundle:
    """One job in the export queue: whose photos, how, and where.

    ``person_ids`` empty means every keeper, whatever ``match_mode`` says.
    ``watermark_template`` is a :func:`watermark_items` kind (``signature``,
    ``seal``, ``exif_strip``) or None; ``watermark_text`` is the name a
    signature or seal shows.
    """

    bundle_name: str = EVERYONE
    person_ids: list[str] = field(default_factory=list)
    match_mode: str = "ALL_PROJECT"
    preset_name: str = "web"
    watermark_template: str | None = None
    destination_subfolder: str = ""
    watermark_text: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @property
    def mode(self) -> str:
        """The match mode that applies: everyone when nobody is chosen."""
        return self.match_mode if self.person_ids else "ALL_PROJECT"

    @property
    def preset(self) -> Preset:
        return preset_for(self.preset_name)

    @property
    def subfolder(self) -> str:
        return folder_slug(self.destination_subfolder or self.bundle_name)

    def watermark(self):
        return watermark_items(self.watermark_template or "", self.watermark_text)

    def render_key(self) -> tuple:
        """Two bundles with the same key make the same file from a photo."""
        named = self.watermark_template in ("signature", "seal")
        return (preset_key(self.preset_name), self.watermark_template or "",
                self.watermark_text.strip() if named else "")

    def to_dict(self) -> dict:
        return {
            "id": self.id, "bundle_name": self.bundle_name, "person_ids": [str(p) for p in self.person_ids],
            "match_mode": self.match_mode, "preset_name": self.preset_name,
            "watermark_template": self.watermark_template, "watermark_text": self.watermark_text,
            "destination_subfolder": self.destination_subfolder,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "ExportBundle":
        mode = str(raw.get("match_mode", "ALL_PROJECT"))
        preset = str(raw.get("preset_name", "web"))
        try:
            preset_key(preset)
        except KeyError:
            preset = "web"
        return cls(
            bundle_name=str(raw.get("bundle_name") or EVERYONE),
            person_ids=[str(p) for p in raw.get("person_ids", []) if str(p).strip().isdigit()],
            match_mode=mode if mode in MATCH_LABELS else "ANY",
            preset_name=preset,
            watermark_template=raw.get("watermark_template") or None,
            destination_subfolder=str(raw.get("destination_subfolder", "")),
            watermark_text=str(raw.get("watermark_text", "")),
            id=str(raw.get("id") or uuid.uuid4().hex),
        )


def save_queue(edit_folder: str | Path, bundles: list[ExportBundle], base_dir: str | Path | None = None) -> Path:
    """Keep the queue with the folder's edits, so it is there next time."""
    target = Path(edit_folder) / QUEUE_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "base_dir": str(base_dir) if base_dir else "",
               "bundles": [b.to_dict() for b in bundles]}
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(temporary, target)
    return target


def load_queue(edit_folder: str | Path) -> tuple[list[ExportBundle], str]:
    """``(bundles, base_dir)``; a missing or damaged file is an empty queue."""
    try:
        raw = json.loads((Path(edit_folder) / QUEUE_FILENAME).read_text(encoding="utf-8"))
        bundles = [ExportBundle.from_dict(b) for b in raw.get("bundles", []) if isinstance(b, dict)]
        return bundles, str(raw.get("base_dir") or "")
    except Exception:
        return [], ""


def export_workers(bits: int = 8, budget_gb: float | None = None) -> int:
    """How many frames to render at once. One core is left for the window,
    and each render gets its share of the safe RAM budget (a 16-bit render
    holds about twice what an 8-bit one does)."""
    if budget_gb is None:
        try:
            from src.engine.hardware import get_safe_export_ram_gb

            budget_gb = get_safe_export_ram_gb()
        except Exception:
            budget_gb = 2.0
    per_render = 2.5 if bits > 8 else 1.25
    cores = max(1, (os.cpu_count() or 2) - 1)
    return max(1, min(cores, 6, int(budget_gb // per_render)))


@dataclass
class QueueProgress:
    """Where a queue run is, for a status line such as *Exporting Bundle 2
    of 3 (Sam & Joe): photo 42 of 92…*."""

    bundle_name: str = ""
    bundle_number: int = 0
    bundle_count: int = 0
    photo: int = 0
    photos_in_bundle: int = 0
    done: int = 0
    total: int = 0

    def describe(self) -> str:
        return (f"Exporting Bundle {self.bundle_number} of {self.bundle_count} ({self.bundle_name}): "
                f"photo {self.photo} of {self.photos_in_bundle}…")


@dataclass
class BundleResult:
    bundle: ExportBundle
    folder: Path
    written: list[Path] = field(default_factory=list)
    failures: list[tuple[Path, str]] = field(default_factory=list)
    copied: int = 0             # files copied from another bundle's render, not rendered again


@dataclass
class QueueResult:
    bundles: list[BundleResult] = field(default_factory=list)
    rendered: int = 0
    copied: int = 0

    @property
    def written(self) -> int:
        return sum(len(b.written) for b in self.bundles)

    @property
    def failures(self) -> list[tuple[Path, str]]:
        return [f for b in self.bundles for f in b.failures]


class MultiExportRunner:
    """Run every bundle of an export queue.

    ``session`` answers which photos each bundle needs
    (:meth:`LibrarySession.images_for_people`), and ``job_factory(path)``
    says how to render one (the Export workspace passes its own, which
    knows about layers and fills). ``protected`` folders, by default the
    open photo folder, are never written into, so an export can never land
    beside or over an original (INV-1).
    """

    def __init__(self, session, job_factory: Callable[[Path], FrameJob] | None = None,
                 workers: int | None = None, protected=None) -> None:
        self.session = session
        self.job_factory = job_factory or self._plain_job
        self.workers = workers
        if protected is None:
            folder = getattr(session, "current_folder", None)
            protected = [folder] if folder is not None else []
        self.protected = {Path(p).resolve() for p in protected}
        self.progress = QueueProgress()

    def _plain_job(self, path: Path) -> FrameJob:
        return FrameJob(Path(path), self.session.recipe(path), metadata=self.session.metadata.get(str(path)))

    def images_for(self, bundle: ExportBundle) -> list[Path]:
        return self.session.images_for_people(bundle.person_ids, bundle.mode)

    def plan(self, bundles: list[ExportBundle]) -> list[tuple[ExportBundle, list[Path]]]:
        return [(bundle, self.images_for(bundle)) for bundle in bundles]

    def total_files(self, bundles: list[ExportBundle]) -> int:
        return sum(len(paths) for _, paths in self.plan(bundles))

    def run_bundles(self, bundles: list[ExportBundle], base_export_dir: str | Path,
                    progress_callback: Callable[[str, int, int], None] | None = None) -> QueueResult:
        """Export every bundle into ``base_export_dir / <its subfolder>``.

        ``progress_callback(bundle_name, files_done, files_in_queue)`` is
        called after every file, and :attr:`progress` then holds the detail
        (which bundle of how many, which photo of that bundle's). One
        unreadable photo, or one bundle that cannot be written, does not
        stop the rest.
        """
        base = Path(base_export_dir)
        plan = self.plan(bundles)
        total = sum(len(paths) for _, paths in plan)
        result = QueueResult()
        rendered: dict[tuple, Path] = {}      # (photo, render key) -> the file first written for it
        failed: dict[tuple, str] = {}
        done = 0

        def tick(bundle: ExportBundle, number: int, photo: int, count: int) -> None:
            nonlocal done
            done += 1
            self.progress = QueueProgress(bundle.bundle_name, number, len(plan), photo, count, done, total)
            if progress_callback:
                progress_callback(bundle.bundle_name, done, total)

        for number, (bundle, paths) in enumerate(plan, start=1):
            folder = base / bundle.subfolder
            outcome = BundleResult(bundle, folder)
            result.bundles.append(outcome)
            try:
                preset = bundle.preset
                watermark = bundle.watermark()
                if folder.resolve() in self.protected:
                    raise ValueError(f"{folder} is a photo folder; exports never go beside the originals")
                folder.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                for photo, path in enumerate(paths, start=1):
                    outcome.failures.append((Path(path), str(exc)))
                    tick(bundle, number, photo, len(paths))
                continue

            names = _names(paths)
            targets = {path: folder / f"{names[path]}{preset.suffix}{preset.extension}" for path in paths}
            render_key = bundle.render_key()
            photo = 0

            # What an earlier bundle already rendered the same way is copied.
            to_render = []
            for path in paths:
                key = (str(path), render_key)
                if key not in rendered and key not in failed:
                    to_render.append(path)
                    continue
                photo += 1
                try:
                    if key in failed:
                        raise IOError(failed[key])
                    target = targets[path]
                    if target.resolve() == Path(path).resolve():
                        raise ValueError("refusing to export over the original file")
                    if rendered[key].resolve() != target.resolve():
                        shutil.copy2(rendered[key], target)
                    outcome.written.append(target)
                    outcome.copied += 1
                    result.copied += 1
                except Exception as exc:
                    outcome.failures.append((Path(path), str(exc)))
                tick(bundle, number, photo, len(paths))

            jobs = {path: self.job_factory(path) for path in to_render}     # made here, not on the pool
            workers = self.workers or export_workers(16 if preset.fmt.endswith("16") else 8)
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = {pool.submit(finish_and_write, jobs[path], targets[path], preset, watermark): path
                           for path in to_render}
                for future in as_completed(futures):
                    path = futures[future]
                    key = (str(path), render_key)
                    photo += 1
                    try:
                        rendered[key] = future.result()
                        outcome.written.append(rendered[key])
                        result.rendered += 1
                    except Exception as exc:
                        logger.warning("Export of %s failed: %s", Path(path).name, exc)
                        failed[key] = str(exc)
                        outcome.failures.append((Path(path), str(exc)))
                    tick(bundle, number, photo, len(paths))
            order = {targets[p]: i for i, p in enumerate(paths)}
            outcome.written.sort(key=lambda target: order.get(target, len(order)))
        return result
