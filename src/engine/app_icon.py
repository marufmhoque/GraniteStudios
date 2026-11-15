"""The app icon in the formats each platform's shell needs.

``icon.png`` is the single source of truth. Windows shortcuts cannot show a
PNG: a .lnk whose IconLocation names one renders as a blank page. So
Windows gets ``icon.ico``, written here from the PNG, square-padded so the
logo is not stretched, at every size Explorer asks for.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.engine.resources import resource_path

logger = logging.getLogger(__name__)

ICO_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

# Where an icon is written when the install folder has none (a packaged
# build's folder may be read-only, so never beside the app).
USER_ICON = Path.home() / ".granitestudios" / "icon.ico"


def square(image):
    """The image centred on a transparent square, so no size distorts it."""
    from PIL import Image

    image = image.convert("RGBA")
    side = max(image.size)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2), image)
    return canvas


def write_ico(png: Path, ico: Path) -> Path:
    """Write ``ico`` from ``png`` at every size in :data:`ICO_SIZES`."""
    from PIL import Image

    with Image.open(png) as image:
        logo = square(image)
    ico.parent.mkdir(parents=True, exist_ok=True)
    logo.save(ico, format="ICO", sizes=ICO_SIZES)
    return ico


def windows_icon() -> Path | None:
    """An .ico of the logo for shortcuts: the bundled one, else one made
    from ``icon.png`` in the user's folder. None if neither can be had."""
    bundled = resource_path("icon.ico")
    if bundled.exists():
        return bundled
    png = resource_path("icon.png")
    if USER_ICON.exists() and png.exists() and USER_ICON.stat().st_mtime >= png.stat().st_mtime:
        return USER_ICON
    if not png.exists():
        return None
    try:
        return write_ico(png, USER_ICON)
    except Exception:
        logger.exception("Could not make an .ico from %s", png)
        return None
