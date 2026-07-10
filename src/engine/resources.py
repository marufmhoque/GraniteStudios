"""Resource path resolution for both source runs and frozen bundles.

When Granite Studios is packaged with PyInstaller, bundled data files (added
via ``--add-data``) live under ``sys._MEIPASS`` at runtime rather than next to
the source tree. This module hides that difference: call ``resource_path`` with
a path relative to the project root and it resolves correctly whether the app
is running from source, a ``--onedir`` build, or a ``.app`` bundle.
"""

from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    """True when running inside a PyInstaller (or similar) frozen bundle."""
    return bool(getattr(sys, "frozen", False)) and hasattr(sys, "_MEIPASS")


def get_base_path() -> Path:
    """Return the root directory that bundled resources are relative to.

    In a frozen app this is ``sys._MEIPASS`` (where ``--add-data`` unpacks
    files). Running from source it is the project root (three levels up from
    this file: resources.py -> engine -> src -> project root).
    """
    if is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent.parent


def resource_path(relative: str | Path) -> Path:
    """Resolve a bundled resource path.

    Example:
        resource_path("assets/icon.png")

    Resolves to ``<project root>/assets/icon.png`` from source and to
    ``<_MEIPASS>/assets/icon.png`` inside a packaged build.
    """
    return get_base_path() / relative
