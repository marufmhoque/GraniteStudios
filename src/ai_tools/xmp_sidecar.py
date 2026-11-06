"""XMP sidecar read/write for star ratings, colour labels and rejects.

Culling is only useful if the decisions leave the app. Lightroom Classic,
Capture One, Photo Mechanic and Bridge all read an ``.xmp`` sidecar sitting
next to the image, so writing one lets a photographer cull here and then open
their existing catalogue with the picks already rated.

The sidecar is the only thing written — the RAW file is never touched, which
is the same non-destructive rule the rest of the app follows.

An existing sidecar is *merged*, not replaced: a file may already carry
develop settings, keywords or crops from another tool, and clobbering those
would silently destroy work. Only the rating/label attributes are rewritten.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

NS_X = "adobe:ns:meta/"
NS_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
NS_XMP = "http://ns.adobe.com/xap/1.0/"

TOOLKIT = "Granite Studios"

# Lightroom's colour labels are stored as these exact strings; anything else
# shows up as a custom label rather than a swatch.
VALID_LABELS = ("Red", "Yellow", "Green", "Blue", "Purple")

# xmp:Rating uses -1 for "rejected", which Lightroom shows as a flag rather
# than a star count.
RATING_REJECTED = -1
RATING_MIN = 0
RATING_MAX = 5

_XPACKET_HEAD = '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
_XPACKET_TAIL = '\n<?xpacket end="w"?>\n'


@dataclass
class XmpRating:
    """Rating state for one image."""

    rating: int = 0
    label: str | None = None

    @property
    def rejected(self) -> bool:
        return self.rating == RATING_REJECTED


def sidecar_path_for(image_path: str | Path) -> Path:
    """Return the sidecar path for an image.

    Uses the ``IMG_1234.xmp`` form (extension replaced) rather than
    ``IMG_1234.CR2.xmp`` — that is what Lightroom Classic writes for RAW
    files and therefore what it looks for on import.
    """
    return Path(image_path).with_suffix(".xmp")


def _clamp_rating(rating: int) -> int:
    if rating == RATING_REJECTED:
        return RATING_REJECTED
    return max(RATING_MIN, min(RATING_MAX, int(rating)))


def _register_namespaces() -> None:
    ET.register_namespace("x", NS_X)
    ET.register_namespace("rdf", NS_RDF)
    ET.register_namespace("xmp", NS_XMP)


def _new_tree() -> tuple[ET.Element, ET.Element]:
    """Build an empty xmpmeta/RDF/Description skeleton."""
    root = ET.Element(f"{{{NS_X}}}xmpmeta", {f"{{{NS_X}}}xmptk": TOOLKIT})
    rdf = ET.SubElement(root, f"{{{NS_RDF}}}RDF")
    desc = ET.SubElement(rdf, f"{{{NS_RDF}}}Description", {f"{{{NS_RDF}}}about": ""})
    return root, desc


def _find_description(root: ET.Element) -> ET.Element | None:
    rdf = root.find(f"{{{NS_RDF}}}RDF")
    if rdf is None:
        return None
    return rdf.find(f"{{{NS_RDF}}}Description")


def read_rating(image_path: str | Path) -> XmpRating:
    """Read rating/label from an image's sidecar.

    Returns a default (unrated, unlabelled) result when there is no sidecar
    or it cannot be parsed — a damaged sidecar should not break a cull.
    """
    sidecar = sidecar_path_for(image_path)
    if not sidecar.exists():
        return XmpRating()

    try:
        text = sidecar.read_text(encoding="utf-8", errors="replace")
        # Strip the xpacket processing instructions; ElementTree rejects the
        # BOM-bearing PI that Adobe tools wrap sidecars in.
        body = re.sub(r"<\?xpacket[^>]*\?>", "", text).strip()
        root = ET.fromstring(body)
    except (OSError, ET.ParseError):
        logger.warning("Could not parse sidecar %s", sidecar)
        return XmpRating()

    desc = _find_description(root)
    if desc is None:
        return XmpRating()

    rating = 0
    label = None

    raw_rating = desc.get(f"{{{NS_XMP}}}Rating")
    if raw_rating is None:
        node = desc.find(f"{{{NS_XMP}}}Rating")
        raw_rating = node.text if node is not None else None
    if raw_rating is not None:
        try:
            rating = _clamp_rating(int(float(raw_rating)))
        except (TypeError, ValueError):
            rating = 0

    raw_label = desc.get(f"{{{NS_XMP}}}Label")
    if raw_label is None:
        node = desc.find(f"{{{NS_XMP}}}Label")
        raw_label = node.text if node is not None else None
    if raw_label:
        label = raw_label.strip() or None

    return XmpRating(rating=rating, label=label)


def write_rating(
    image_path: str | Path,
    rating: int = 0,
    label: str | None = None,
) -> Path:
    """Write rating/label to an image's XMP sidecar and return its path.

    ``rating`` is 0-5, or ``RATING_REJECTED`` (-1) to mark a reject.
    ``label`` should be one of VALID_LABELS to show as a colour swatch;
    other strings are still written, just as a custom label. Passing None
    leaves any existing label untouched; pass "" to clear it.

    An existing sidecar keeps everything it already had apart from the
    rating and label attributes.
    """
    image_path = Path(image_path)
    sidecar = sidecar_path_for(image_path)
    rating = _clamp_rating(rating)
    _register_namespaces()

    root: ET.Element | None = None
    desc: ET.Element | None = None

    if sidecar.exists():
        try:
            text = sidecar.read_text(encoding="utf-8", errors="replace")
            body = re.sub(r"<\?xpacket[^>]*\?>", "", text).strip()
            root = ET.fromstring(body)
            desc = _find_description(root)
        except (OSError, ET.ParseError):
            logger.warning("Replacing unparseable sidecar %s", sidecar)
            root, desc = None, None

    if root is None or desc is None:
        root, desc = _new_tree()

    desc.set(f"{{{NS_XMP}}}Rating", str(rating))
    # Some writers store these as child elements rather than attributes;
    # drop those so the attribute we just set is the single source of truth.
    for child in list(desc.findall(f"{{{NS_XMP}}}Rating")):
        desc.remove(child)

    if label is not None:
        if label == "":
            desc.attrib.pop(f"{{{NS_XMP}}}Label", None)
        else:
            desc.set(f"{{{NS_XMP}}}Label", label)
        for child in list(desc.findall(f"{{{NS_XMP}}}Label")):
            desc.remove(child)

    payload = ET.tostring(root, encoding="unicode")
    sidecar.write_text(_XPACKET_HEAD + payload + _XPACKET_TAIL, encoding="utf-8")
    return sidecar


def write_cull_results(results, stars_for_picks: int = 2, label_rejects: bool = False) -> int:
    """Write sidecars for a list of CullResult objects.

    Picks (``is_starred`` or ``is_burst_pick``) get ``stars_for_picks``
    stars. Duplicates are labelled Yellow so they are easy to filter in
    Lightroom without being destructive. With ``label_rejects`` set,
    non-picks are marked rejected (-1) instead of simply left at zero.

    Returns the number of sidecars written.
    """
    written = 0
    for result in results:
        is_pick = bool(getattr(result, "is_starred", False) or getattr(result, "is_burst_pick", False))
        is_dupe = bool(getattr(result, "is_duplicate", False))

        if is_pick:
            rating = stars_for_picks
        elif label_rejects:
            rating = RATING_REJECTED
        else:
            rating = 0

        label = "Yellow" if is_dupe and not is_pick else None

        try:
            write_rating(result.path, rating=rating, label=label)
            written += 1
        except OSError:
            logger.exception("Could not write sidecar for %s", result.path)
    return written
