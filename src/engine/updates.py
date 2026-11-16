"""Knowing when a newer Granite Studios is on GitHub, and installing it.

The check downloads one small file, ``src/__init__.py`` from the ``main``
branch, and reads its ``__version__``. Nothing is sent but the request
itself: no photos, no settings, no identifiers. It runs at most once a day
and can be turned off (:func:`save_prefs`).

Updating hands over to the installer copied beside the app as
``Update Granite Studios.bat``. It is told, through ``GRANITE_RELAUNCH``, to
reopen the app when it finishes.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
import time
import urllib.request
import webbrowser
from decimal import Decimal, InvalidOperation
from pathlib import Path

from src import __version__
from src.engine.resources import get_base_path, is_frozen

logger = logging.getLogger(__name__)

REPO = "marufmhoque/GraniteStudios"
VERSION_URL = f"https://raw.githubusercontent.com/{REPO}/main/src/__init__.py"
INSTALLER_URL = f"https://github.com/{REPO}/releases/latest/download/Install-GraniteStudios.bat"
USER_AGENT = "GraniteStudios-updater"
CHECK_INTERVAL_S = 24 * 60 * 60
UPDATER_NAME = "Update Granite Studios.bat"
PREFS_PATH = Path.home() / ".granitestudios" / "updates.json"

_VERSION = re.compile(r'__version__\s*=\s*"(\d+\.\d{2})"')


def parse_version(text: str) -> str | None:
    """The two-decimal ``__version__`` in a ``src/__init__.py``, or None."""
    match = _VERSION.search(text or "")
    return match.group(1) if match else None


def _decimal(version: str | None) -> Decimal | None:
    if not version or not re.fullmatch(r"\d+\.\d{2}", version):
        return None
    try:
        return Decimal(version)
    except InvalidOperation:
        return None


def is_newer(remote: str | None, local: str | None = __version__) -> bool:
    """True when ``remote`` is a later version than ``local``. A local
    version from before 2.00 (not two decimals) counts as older."""
    theirs = _decimal(remote)
    if theirs is None:
        return False
    ours = _decimal(local)
    return ours is None or theirs > ours


def latest_version(timeout: float = 5.0) -> str | None:
    """The version on GitHub's ``main`` branch, or None when it cannot be
    read (offline, blocked, or an unexpected reply). Never raises."""
    try:
        request = urllib.request.Request(VERSION_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read(16_384).decode("utf-8", errors="replace")
        return parse_version(text)
    except Exception as exc:
        logger.info("Could not check for updates: %s", exc)
        return None


# -- preferences -------------------------------------------------------------- #

def load_prefs(path: Path | None = None) -> dict:
    """``{"auto_check": bool, "last_check": float}``; defaults when the file
    is missing or damaged."""
    prefs = {"auto_check": True, "last_check": 0.0}
    try:
        stored = json.loads((path or PREFS_PATH).read_text(encoding="utf-8"))
        if isinstance(stored.get("auto_check"), bool):
            prefs["auto_check"] = stored["auto_check"]
        if isinstance(stored.get("last_check"), (int, float)):
            prefs["last_check"] = float(stored["last_check"])
    except (OSError, ValueError, AttributeError):
        pass
    return prefs


def save_prefs(prefs: dict, path: Path | None = None) -> None:
    target = path or PREFS_PATH
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(prefs, indent=2), encoding="utf-8")
    except OSError:
        logger.warning("Could not save update preferences to %s", target)


def check_due(prefs: dict, now: float | None = None) -> bool:
    """True when automatic checks are on and the last was a day ago or more."""
    now = time.time() if now is None else now
    return bool(prefs.get("auto_check", True)) and now - float(prefs.get("last_check", 0.0)) >= CHECK_INTERVAL_S


# -- updating ----------------------------------------------------------------- #

def install_dir() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Granite Studios"


def updater_path() -> Path:
    return get_base_path() / UPDATER_NAME


def can_self_update() -> bool:
    """True for a copy the installer put in place: Windows, run from source
    in the install folder, with the updater beside it. A development
    checkout or a packaged build never replaces itself."""
    if platform.system() != "Windows" or is_frozen():
        return False
    try:
        same = get_base_path().resolve() == install_dir().resolve()
    except OSError:
        return False
    return same and updater_path().exists()


def launch_update() -> bool:
    """Start the update. Returns True when the installer was started (the
    caller should then close the app), False when the download page was
    opened instead."""
    if not can_self_update():
        webbrowser.open(INSTALLER_URL)
        return False
    env = dict(os.environ, GRANITE_RELAUNCH="1")
    updater = updater_path()
    subprocess.Popen(
        ["cmd.exe", "/c", str(updater)],
        cwd=str(updater.parent),
        env=env,
        creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
    )
    return True
