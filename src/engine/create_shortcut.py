"""Desktop / Start Menu shortcut creation for Granite Studios.

Uses pyshortcuts to drop a "Granite Studios" launcher on the user's Desktop
and Start Menu the first time the app runs, so non-technical users get a
normal double-click icon instead of having to dig through a dist/ folder
for the .exe.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.engine.resources import get_base_path, is_frozen, resource_path

logger = logging.getLogger(__name__)

APP_NAME = "Granite Studios"

# Marker file in the user's home directory — cheaper and more portable than
# querying the OS for whether a shortcut already exists, and works the same
# way on every platform pyshortcuts supports.
_MARKER_FILENAME = ".granitestudios_shortcut_created"


def _marker_path() -> Path:
    return Path.home() / _MARKER_FILENAME


def shortcut_already_created() -> bool:
    """True once create_desktop_shortcut() has successfully run before."""
    return _marker_path().exists()


def create_desktop_shortcut() -> bool:
    """Create Desktop + Start Menu shortcuts for Granite Studios.

    Idempotent: no-ops (returns False) once a marker file confirms a
    shortcut was already created on a prior launch. Also no-ops quietly if
    pyshortcuts isn't installed (e.g. a bare source checkout) or shortcut
    creation fails for any reason — this is a convenience, never something
    that should block the app from starting.

    Meant to be called once, automatically, on every launch (see main.py);
    the marker file makes repeat calls cheap no-ops.
    """
    if shortcut_already_created():
        return False

    try:
        from pyshortcuts import make_shortcut
    except ImportError:
        logger.info("pyshortcuts is not installed — skipping shortcut creation.")
        return False

    icon_path = resource_path("icon.ico")
    if not icon_path.exists():
        icon_path = resource_path("icon.png")

    if is_frozen():
        # Packaged build: point the shortcut straight at the .exe itself.
        # noexe=True tells pyshortcuts to treat `target` as the whole
        # command rather than a script to hand to a Python interpreter.
        target = sys.executable
        noexe = True
    else:
        # Running from source: launch main.py with the current interpreter.
        target = str(get_base_path() / "main.py")
        noexe = False

    try:
        make_shortcut(
            target,
            name=APP_NAME,
            description="Local AI photo & video editing suite",
            icon=str(icon_path) if icon_path.exists() else None,
            terminal=False,
            desktop=True,
            startmenu=True,
            noexe=noexe,
        )
    except Exception:
        logger.exception("Shortcut creation failed")
        return False

    try:
        _marker_path().write_text("created", encoding="utf-8")
    except OSError:
        pass

    logger.info("Created a %r shortcut on the Desktop and Start Menu.", APP_NAME)
    return True
