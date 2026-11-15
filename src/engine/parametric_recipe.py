"""The edit recipe: every adjustment to a photograph, stored as data.

An edit is a list of settings, not a new image. Moving a slider changes a
number in a :class:`Recipe`; nothing is rendered to a file until an export
is asked for, and the original is only ever read. That is what makes editing
non-destructive: there is no chain of re-saved JPEGs to lose quality along,
and any setting can be changed or undone later, in any order.

Where recipes live
------------------
``<Folder>/<Folder>-edit/<filename>.granite.json`` -- in the edit subfolder,
never beside the originals, and named after the whole filename so that
``IMG_1.CR2`` and ``IMG_1.JPG`` shot together do not share a recipe.

Versions ("virtual copies")
---------------------------
A photograph can carry several independent edits of one original. The
first is the master and keeps the name above; each copy is another small
recipe, ``<filename>.granite-v<N>.json``, beside it. A copy duplicates only
the settings, never the pixels: every version renders from the same source
file. The two name forms cannot collide, because a master's sidecar always
ends ``.granite.json`` and a copy's never does.

Robustness
----------
A recipe file can be damaged, hand-edited, or written by a newer version of
the app. Reading one therefore never raises and never guesses: an unknown
field is ignored, a value of the wrong type falls back to its default, a
number outside its range is clamped, and a file that is not JSON at all
yields a neutral recipe with ``load_error`` set so the interface can say so.
A photograph must not become unopenable because its sidecar is bad.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RECIPE_VERSION = 1
SIDECAR_SUFFIX = ".granite.json"
VERSION_MARK = ".granite-v"          # <filename>.granite-v<N>.json
MASTER = 0                           # the version id of the master edit

HSL_BANDS = ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta")
# Centre of each band on the hue circle, in degrees.
HSL_CENTRES = (0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 280.0, 320.0)

# (minimum, maximum) for every numeric setting. One table, so the interface
# and the loader cannot disagree about what a slider may hold.
RANGES: dict[str, tuple[float, float]] = {
    "exposure": (-5.0, 5.0),
    "contrast": (-100.0, 100.0),
    "highlights": (-100.0, 100.0),
    "shadows": (-100.0, 100.0),
    "whites": (-100.0, 100.0),
    "blacks": (-100.0, 100.0),
    "temperature": (-100.0, 100.0),
    "tint": (-100.0, 100.0),
    "texture": (-100.0, 100.0),
    "clarity": (-100.0, 100.0),
    "dehaze": (-100.0, 100.0),
    "vibrance": (-100.0, 100.0),
    "saturation": (-100.0, 100.0),
    "hsl": (-100.0, 100.0),
    "grade_hue": (0.0, 360.0),
    "grade_saturation": (0.0, 100.0),
    "grade_luminance": (-100.0, 100.0),
    "grade_balance": (-100.0, 100.0),
    "grade_blending": (0.0, 100.0),
    "rotation": (-45.0, 45.0),
    "keystone": (-100.0, 100.0),
    "sharpen_amount": (0.0, 150.0),
    "sharpen_radius": (0.5, 3.0),
    "noise_reduction": (0.0, 100.0),
    "lens_distortion": (-100.0, 100.0),
    "lens_vignette": (-100.0, 100.0),
    "lut_strength": (0.0, 100.0),
    "look_strength": (0.0, 100.0),
    "deblur_angle": (0.0, 180.0),
    "deblur_length": (1.0, 30.0),
    "deblur_radius": (1.0, 20.0),
    "deblur_damping": (0.0, 1.0),
}

DEBLUR_ENGINES = ("parametric", "wiener_fast", "neural")
DEBLUR_TYPES = ("motion", "defocus")
DEBLUR_MASKS = ("full", "face_only", "subject_only")


def clamp(name: str, value: Any, default: float = 0.0) -> float:
    """Coerce ``value`` to a float inside the range registered for ``name``."""
    low, high = RANGES[name]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return min(high, max(low, number))


# --------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------- #

@dataclass
class Basic:
    """Tone and presence. Every slider is neutral at zero."""

    exposure: float = 0.0        # stops
    contrast: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0
    temperature: float = 0.0     # negative cooler, positive warmer
    tint: float = 0.0            # negative green, positive magenta
    texture: float = 0.0
    clarity: float = 0.0
    dehaze: float = 0.0
    vibrance: float = 0.0
    saturation: float = 0.0


@dataclass
class ToneCurve:
    """A point curve on the 0..1 square, per channel.

    Each curve is a list of ``[input, output]`` points. The straight line
    ``[[0, 0], [1, 1]]`` is the identity.
    """

    master: list = field(default_factory=lambda: [[0.0, 0.0], [1.0, 1.0]])
    red: list = field(default_factory=lambda: [[0.0, 0.0], [1.0, 1.0]])
    green: list = field(default_factory=lambda: [[0.0, 0.0], [1.0, 1.0]])
    blue: list = field(default_factory=lambda: [[0.0, 0.0], [1.0, 1.0]])


@dataclass
class HSL:
    """Hue, saturation and luminance offsets for eight colour bands,
    in the order of :data:`HSL_BANDS`."""

    hue: list = field(default_factory=lambda: [0.0] * 8)
    saturation: list = field(default_factory=lambda: [0.0] * 8)
    luminance: list = field(default_factory=lambda: [0.0] * 8)


@dataclass
class GradeWheel:
    hue: float = 0.0             # degrees
    saturation: float = 0.0      # 0 leaves the tones uncoloured
    luminance: float = 0.0


@dataclass
class ColorGrade:
    """Three-way colour grading: a tint each for shadows, midtones, highlights."""

    shadows: GradeWheel = field(default_factory=GradeWheel)
    midtones: GradeWheel = field(default_factory=GradeWheel)
    highlights: GradeWheel = field(default_factory=GradeWheel)
    balance: float = 0.0         # shifts the shadow/highlight split
    blending: float = 50.0       # overlap between the three ranges


@dataclass
class Geometry:
    """Framing. ``crop`` is ``[x, y, w, h]`` as fractions of the frame after
    rotation, or None for the whole frame."""

    quarter_turns: int = 0       # clockwise
    flip_horizontal: bool = False
    flip_vertical: bool = False
    rotation: float = 0.0        # degrees, counter-clockwise; straightening
    keystone_vertical: float = 0.0
    keystone_horizontal: float = 0.0
    crop: list | None = None
    constrain_to_image: bool = True   # trim the blank wedges rotation leaves
    fill_blank: bool = False          # or, when not trimming, paint them in


@dataclass
class Lens:
    profile: bool = False        # look the lens up and correct it
    distortion: float = 0.0      # manual, on top of or instead of a profile
    vignette: float = 0.0        # positive brightens the corners


@dataclass
class Detail:
    sharpen_amount: float = 0.0
    sharpen_radius: float = 1.0
    noise_reduction: float = 0.0


@dataclass
class Deblur:
    """Recovering motion smear or missed focus (see ``ai_tools/deblur.py``).

    Lengths and radii are in pixels of the full-size original, so a setting
    means the same thing in the preview and the export. ``angle`` is the
    direction of the smear, counter-clockwise from horizontal on screen.
    """

    enabled: bool = False
    engine: str = "parametric"   # "parametric" | "wiener_fast" | "neural"
    blur_type: str = "motion"    # "motion" | "defocus"
    angle: float = 0.0
    length: float = 5.0
    radius: float = 3.0
    damping: float = 0.05        # artifact suppression, 0..1
    mask_mode: str = "full"      # "full" | "face_only" | "subject_only"


@dataclass
class Look:
    """A colour look: a channel-mixing matrix with a lift and gamma.

    This is how the six built-in grades are expressed, so they stay
    adjustable settings rather than baked-in pixels.
    """

    name: str = ""
    matrix: list | None = None   # 3 x 3, rows are output channels
    lift: float = 0.0            # on a 0..255 scale, as the grades were authored
    gamma: float = 1.0
    strength: float = 100.0


@dataclass
class Lut:
    path: str = ""               # a .cube file
    strength: float = 100.0


@dataclass
class Recipe:
    version: int = RECIPE_VERSION
    basic: Basic = field(default_factory=Basic)
    curve: ToneCurve = field(default_factory=ToneCurve)
    hsl: HSL = field(default_factory=HSL)
    grade: ColorGrade = field(default_factory=ColorGrade)
    geometry: Geometry = field(default_factory=Geometry)
    lens: Lens = field(default_factory=Lens)
    detail: Detail = field(default_factory=Detail)
    deblur: Deblur = field(default_factory=Deblur)
    look: Look = field(default_factory=Look)
    lut: Lut = field(default_factory=Lut)
    layers: list = field(default_factory=list)       # see layer_stack.py
    vectors: list = field(default_factory=list)      # see vector_canvas.py
    # Not saved: set when a sidecar could not be read cleanly.
    load_error: str | None = field(default=None, compare=False)

    # -- convenience ------------------------------------------------------ #

    def copy(self) -> "Recipe":
        return from_dict(to_dict(self))

    @property
    def is_neutral(self) -> bool:
        """True when rendering this recipe would change nothing."""
        return to_dict(self) == to_dict(Recipe())

    def reset(self, section: str) -> None:
        """Return one section (``"basic"``, ``"hsl"``, ...) to neutral."""
        fresh = Recipe()
        if not hasattr(fresh, section) or section in ("version", "load_error"):
            raise KeyError(section)
        setattr(self, section, getattr(fresh, section))


# --------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------- #

def to_dict(recipe: Recipe) -> dict:
    data = dataclasses.asdict(recipe)
    data.pop("load_error", None)
    return data


def _number(name: str, raw: Any, default: float) -> float:
    return clamp(name, raw, default) if raw is not None else default


def _points(raw: Any) -> list:
    """A valid, sorted curve; the identity if ``raw`` is unusable."""
    identity = [[0.0, 0.0], [1.0, 1.0]]
    if not isinstance(raw, (list, tuple)):
        return identity
    points: dict[float, float] = {}
    for item in raw:
        try:
            x, y = float(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            points[min(1.0, max(0.0, x))] = min(1.0, max(0.0, y))
    if len(points) < 2:
        return identity
    return [[x, points[x]] for x in sorted(points)]


def _eight(raw: Any) -> list:
    values = [0.0] * 8
    if isinstance(raw, (list, tuple)):
        for index, item in enumerate(raw[:8]):
            values[index] = clamp("hsl", item)
    return values


def _matrix(raw: Any) -> list | None:
    try:
        rows = [[float(v) for v in row] for row in raw]
    except (TypeError, ValueError):
        return None
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        return None
    if not all(math.isfinite(v) for row in rows for v in row):
        return None
    return rows


def _crop(raw: Any) -> list | None:
    try:
        x, y, w, h = (float(v) for v in raw)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        return None
    x, y = min(1.0, max(0.0, x)), min(1.0, max(0.0, y))
    w, h = min(1.0 - x, max(0.0, w)), min(1.0 - y, max(0.0, h))
    return [x, y, w, h] if w > 0.01 and h > 0.01 else None


def _section(raw: Any) -> dict:
    return raw if isinstance(raw, dict) else {}


def from_dict(data: Any) -> Recipe:
    """Build a recipe from parsed JSON, tolerating anything."""
    recipe = Recipe()
    if not isinstance(data, dict):
        recipe.load_error = "recipe is not an object"
        return recipe

    basic = _section(data.get("basic"))
    for name in dataclasses.fields(Basic):
        setattr(recipe.basic, name.name, _number(name.name, basic.get(name.name), 0.0))

    curve = _section(data.get("curve"))
    for channel in ("master", "red", "green", "blue"):
        setattr(recipe.curve, channel, _points(curve.get(channel)))

    hsl = _section(data.get("hsl"))
    recipe.hsl = HSL(_eight(hsl.get("hue")), _eight(hsl.get("saturation")), _eight(hsl.get("luminance")))

    grade = _section(data.get("grade"))
    for zone in ("shadows", "midtones", "highlights"):
        wheel = _section(grade.get(zone))
        setattr(recipe.grade, zone, GradeWheel(
            _number("grade_hue", wheel.get("hue"), 0.0) % 360.0,
            _number("grade_saturation", wheel.get("saturation"), 0.0),
            _number("grade_luminance", wheel.get("luminance"), 0.0),
        ))
    recipe.grade.balance = _number("grade_balance", grade.get("balance"), 0.0)
    recipe.grade.blending = _number("grade_blending", grade.get("blending"), 50.0)

    geometry = _section(data.get("geometry"))
    try:
        recipe.geometry.quarter_turns = int(geometry.get("quarter_turns", 0)) % 4
    except (TypeError, ValueError):
        recipe.geometry.quarter_turns = 0
    recipe.geometry.flip_horizontal = geometry.get("flip_horizontal") is True
    recipe.geometry.flip_vertical = geometry.get("flip_vertical") is True
    recipe.geometry.rotation = _number("rotation", geometry.get("rotation"), 0.0)
    recipe.geometry.keystone_vertical = _number("keystone", geometry.get("keystone_vertical"), 0.0)
    recipe.geometry.keystone_horizontal = _number("keystone", geometry.get("keystone_horizontal"), 0.0)
    recipe.geometry.crop = _crop(geometry.get("crop"))
    recipe.geometry.constrain_to_image = geometry.get("constrain_to_image", True) is not False
    recipe.geometry.fill_blank = geometry.get("fill_blank") is True

    lens = _section(data.get("lens"))
    recipe.lens = Lens(
        lens.get("profile") is True,
        _number("lens_distortion", lens.get("distortion"), 0.0),
        _number("lens_vignette", lens.get("vignette"), 0.0),
    )

    detail = _section(data.get("detail"))
    recipe.detail = Detail(
        _number("sharpen_amount", detail.get("sharpen_amount"), 0.0),
        _number("sharpen_radius", detail.get("sharpen_radius"), 1.0),
        _number("noise_reduction", detail.get("noise_reduction"), 0.0),
    )

    deblur = _section(data.get("deblur"))

    def choice(value, allowed: tuple, default: str) -> str:
        return value if value in allowed else default

    recipe.deblur = Deblur(
        deblur.get("enabled") is True,
        choice(deblur.get("engine"), DEBLUR_ENGINES, "parametric"),
        choice(deblur.get("blur_type"), DEBLUR_TYPES, "motion"),
        _number("deblur_angle", deblur.get("angle"), 0.0),
        _number("deblur_length", deblur.get("length"), 5.0),
        _number("deblur_radius", deblur.get("radius"), 3.0),
        _number("deblur_damping", deblur.get("damping"), 0.05),
        choice(deblur.get("mask_mode"), DEBLUR_MASKS, "full"),
    )

    look = _section(data.get("look"))
    matrix = _matrix(look.get("matrix"))
    if matrix is not None:
        try:
            gamma = float(look.get("gamma", 1.0))
            lift = float(look.get("lift", 0.0))
        except (TypeError, ValueError):
            gamma, lift = 1.0, 0.0
        recipe.look = Look(
            str(look.get("name", ""))[:80], matrix,
            lift if math.isfinite(lift) else 0.0,
            gamma if math.isfinite(gamma) and 0.1 <= gamma <= 10.0 else 1.0,
            _number("look_strength", look.get("strength"), 100.0),
        )

    lut = _section(data.get("lut"))
    if isinstance(lut.get("path"), str):
        recipe.lut = Lut(lut["path"], _number("lut_strength", lut.get("strength"), 100.0))

    # Layers and vector overlays are validated by their own modules; here
    # they only need to be lists of objects to survive a round trip.
    for name in ("layers", "vectors"):
        raw = data.get(name)
        setattr(recipe, name, [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else [])

    try:
        recipe.version = int(data.get("version", RECIPE_VERSION))
    except (TypeError, ValueError):
        recipe.version = RECIPE_VERSION
    if recipe.version > RECIPE_VERSION:
        recipe.load_error = (
            f"written by a newer version (format {recipe.version}); "
            "settings this version does not know were ignored"
        )
    return recipe


def dumps(recipe: Recipe) -> str:
    return json.dumps(to_dict(recipe), indent=1, sort_keys=True)


def loads(text: str) -> Recipe:
    try:
        return from_dict(json.loads(text))
    except (TypeError, ValueError) as exc:
        recipe = Recipe()
        recipe.load_error = f"not valid JSON: {exc}"
        return recipe


# --------------------------------------------------------------------- #
# Sidecar files
# --------------------------------------------------------------------- #

def sidecar_path(source: str | Path, edit_folder: str | Path, version: int = MASTER) -> Path:
    """Where a version's recipe for ``source`` is kept. Always inside the edit folder."""
    name = Path(source).name
    if version == MASTER:
        return Path(edit_folder) / f"{name}{SIDECAR_SUFFIX}"
    if int(version) < 0:
        raise ValueError("version ids are positive")
    return Path(edit_folder) / f"{name}{VERSION_MARK}{int(version)}.json"


def version_files(edit_folder: str | Path) -> dict[str, list[int]]:
    """Every copy's recipe in an edit folder, as ``{source filename: [ids]}``.

    Read from the file names alone, so copies survive a lost session file.
    """
    found: dict[str, list[int]] = {}
    try:
        entries = list(Path(edit_folder).iterdir())
    except OSError:
        return found
    for entry in entries:
        name = entry.name
        if not name.endswith(".json") or VERSION_MARK not in name:
            continue
        stem, _, number = name[:-5].rpartition(VERSION_MARK)
        if stem and number.isdigit() and int(number) > 0:
            found.setdefault(stem, []).append(int(number))
    return {k: sorted(v) for k, v in found.items()}


def load(source: str | Path, edit_folder: str | Path, version: int = MASTER) -> Recipe:
    """The saved recipe for a photograph, or a neutral one. Never raises."""
    path = sidecar_path(source, edit_folder, version)
    try:
        # Decode leniently: a sidecar holding binary rubbish should fail as
        # "not valid JSON" below, not as a decoding error here.
        text = path.read_bytes().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return Recipe()
    except OSError as exc:
        recipe = Recipe()
        recipe.load_error = f"could not read {path.name}: {exc}"
        return recipe
    recipe = loads(text)
    if recipe.load_error:
        logger.warning("Recipe %s: %s", path.name, recipe.load_error)
    return recipe


def save(recipe: Recipe, source: str | Path, edit_folder: str | Path,
         version: int = MASTER) -> Path | None:
    """Write a version's recipe for ``source``. A neutral recipe removes the file.

    This module only ever writes inside ``edit_folder``.
    """
    path = sidecar_path(source, edit_folder, version)
    if recipe.is_neutral:
        if path.exists():
            path.unlink()
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(dumps(recipe), encoding="utf-8")
    temporary.replace(path)     # never leave a half-written recipe behind
    return path


def discard(source: str | Path, edit_folder: str | Path, version: int) -> bool:
    """Remove a copy's recipe. The master's cannot be removed this way, and
    nothing outside the edit folder is ever touched."""
    if version == MASTER:
        raise ValueError("the master edit is not a copy")
    path = sidecar_path(source, edit_folder, version)
    if path.exists():
        path.unlink()
        return True
    return False


# --------------------------------------------------------------------- #
# Built-in looks
# --------------------------------------------------------------------- #

def builtin_looks() -> dict[str, Look]:
    """The six grades from the original colour engine, as recipe looks."""
    from src.ai_tools.color_engine import _FILTER_MATRICES, _FILTER_TONE

    looks = {}
    for name, matrix in _FILTER_MATRICES.items():
        lift, gamma = _FILTER_TONE[name]
        looks[name] = Look(name, [[float(v) for v in row] for row in matrix], float(lift), float(gamma), 100.0)
    return looks


def with_look(name: str, base: Recipe | None = None) -> Recipe:
    """A copy of ``base`` (or a neutral recipe) with a built-in look applied."""
    looks = builtin_looks()
    if name not in looks:
        raise KeyError(f"Unknown look {name!r}. Choose from: {', '.join(looks)}")
    recipe = base.copy() if base is not None else Recipe()
    recipe.look = looks[name]
    return recipe


# --------------------------------------------------------------------- #
# Style versions
# --------------------------------------------------------------------- #

def _fade(lift: float, roll: float) -> list:
    """A master curve with lifted blacks and softened whites."""
    return [[0.0, lift], [0.25, 0.25 + lift * 0.4], [0.75, 0.76], [1.0, 1.0 - roll]]


def _nudge(recipe: Recipe, **changes: float) -> None:
    """Add to Basic sliders, staying inside their ranges."""
    for name, delta in changes.items():
        setattr(recipe.basic, name, clamp(name, getattr(recipe.basic, name) + delta))


def _cinematic(recipe: Recipe) -> None:
    recipe.look = builtin_looks()["cinematic_teal_orange"]
    recipe.lut = Lut()
    _nudge(recipe, contrast=12, highlights=-20, shadows=10, vibrance=-8)
    recipe.curve.master = _fade(0.04, 0.03)
    recipe.grade.shadows = GradeWheel(200.0, 18.0, 0.0)
    recipe.grade.highlights = GradeWheel(35.0, 14.0, 0.0)


def _warm_analog(recipe: Recipe) -> None:
    recipe.look = builtin_looks()["warm_film"]
    recipe.lut = Lut()
    _nudge(recipe, temperature=14, tint=4, highlights=-15, vibrance=6, saturation=-6)
    recipe.curve.master = _fade(0.07, 0.04)
    recipe.grade.highlights = GradeWheel(45.0, 12.0, 0.0)


def _clean_mono(recipe: Recipe) -> None:
    # Colour settings would only tint a monochrome result, so they go.
    recipe.look, recipe.lut, recipe.grade, recipe.hsl = Look(), Lut(), ColorGrade(), HSL()
    _nudge(recipe, contrast=18, clarity=10, whites=10, blacks=-8)
    recipe.basic.saturation = RANGES["saturation"][0]
    recipe.basic.vibrance = 0.0


# Name -> how that style changes a copy of the master. Tone, framing, lens,
# detail, layers and overlays are kept, so a crop or a retouch carries over.
VERSION_STYLES = {
    "Cinematic": _cinematic,
    "Warm Analog": _warm_analog,
    "Clean B&W": _clean_mono,
}


def styled(name: str, base: Recipe | None = None) -> Recipe:
    """A copy of ``base`` (or a neutral recipe) in one of :data:`VERSION_STYLES`."""
    if name not in VERSION_STYLES:
        raise KeyError(f"Unknown style {name!r}. Choose from: {', '.join(VERSION_STYLES)}")
    recipe = base.copy() if base is not None else Recipe()
    VERSION_STYLES[name](recipe)
    return recipe
