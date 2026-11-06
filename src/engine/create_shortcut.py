"""Desktop / Start Menu shortcut creation for Granite Studios.

On Windows the shortcut is written directly through the Shell's COM
interface rather than via pyshortcuts. pyshortcuts drops the script path
into the .lnk's Arguments field unquoted, so any install path containing a
space (``C:\\Users\\Jo Smith\\...``, or a Google Drive ``G:\\My Drive\\...``)
is split at the space by the shell: Python receives a truncated path,
exits 2 before a window ever appears, and the app looks like it "starts and
then doesn't open". Quoting the argument ourselves is the whole fix.

Running from source also needs pythonw.exe, not python.exe — the latter
attaches a console window that sits behind the GUI for the app's lifetime.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

from src.engine.resources import get_base_path, is_frozen, resource_path

logger = logging.getLogger(__name__)

APP_NAME = "Granite Studios"

# Marker file in the user's home directory — cheaper and more portable than
# querying the OS for whether a shortcut already exists, and works the same
# way on every platform.
_MARKER_FILENAME = ".granitestudios_shortcut_created"


def _marker_path() -> Path:
    return Path.home() / _MARKER_FILENAME


def shortcut_already_created() -> bool:
    """True once create_desktop_shortcut() has successfully run before."""
    return _marker_path().exists()


def _gui_python() -> str:
    """pythonw.exe beside the running interpreter, so no console tags along.

    Falls back to sys.executable when pythonw is missing (a stripped
    embeddable distribution, or any non-Windows platform).
    """
    exe = Path(sys.executable)
    if platform.system() == "Windows" and exe.name.lower() == "python.exe":
        pythonw = exe.with_name("pythonw.exe")
        if pythonw.exists():
            return str(pythonw)
    return str(exe)


def launch_command() -> tuple[str, str]:
    """Return ``(target, arguments)`` for launching this installation.

    A frozen build is its own executable and takes no arguments; a source
    checkout is ``pythonw.exe "<project>/main.py"``. The argument is
    returned already quoted, ready to drop into a .lnk.
    """
    if is_frozen():
        return str(Path(sys.executable).resolve()), ""
    script = (get_base_path() / "main.py").resolve()
    return _gui_python(), f'"{script}"'


def _ps_quote(value: str) -> str:
    """Quote a string for a PowerShell single-quoted literal."""
    return "'" + value.replace("'", "''") + "'"


def _make_windows_shortcut(
    lnk_path: Path, target: str, arguments: str, working_dir: str, icon: str | None
) -> None:
    """Write a .lnk through WScript.Shell, with Arguments properly quoted."""
    lines = [
        "$ErrorActionPreference = 'Stop'",
        "$sh = New-Object -ComObject WScript.Shell",
        f"$lnk = $sh.CreateShortcut({_ps_quote(str(lnk_path))})",
        f"$lnk.TargetPath = {_ps_quote(target)}",
        f"$lnk.Arguments = {_ps_quote(arguments)}",
        f"$lnk.WorkingDirectory = {_ps_quote(working_dir)}",
        f"$lnk.Description = {_ps_quote('Local AI photo & video editing suite')}",
    ]
    if icon:
        lines.append(f"$lnk.IconLocation = {_ps_quote(icon + ',0')}")
    lines.append("$lnk.Save()")

    lnk_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            "; ".join(lines),
        ],
        check=True,
        capture_output=True,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def _windows_shortcut_dirs() -> list[Path]:
    home = Path.home()
    desktop = Path(os.environ.get("USERPROFILE", home)) / "Desktop"
    start = (
        Path(os.environ.get("APPDATA", home / "AppData/Roaming"))
        / "Microsoft/Windows/Start Menu/Programs"
    )
    return [desktop, start]


def create_desktop_shortcut(force: bool = False) -> bool:
    """Create Desktop + Start Menu shortcuts for Granite Studios.

    Idempotent: no-ops (returns False) once a marker file confirms a
    shortcut was created on a prior launch, unless ``force`` is set — the
    installer passes force=True to refresh a stale shortcut. Never raises;
    a shortcut is a convenience and must not stop the app from starting.
    """
    if shortcut_already_created() and not force:
        return False

    icon_path = resource_path("icon.ico")
    if not icon_path.exists():
        icon_path = resource_path("icon.png")
    icon = str(icon_path) if icon_path.exists() else None

    target, arguments = launch_command()
    working_dir = str(get_base_path().resolve())

    try:
        if platform.system() == "Windows":
            for directory in _windows_shortcut_dirs():
                _make_windows_shortcut(
                    directory / f"{APP_NAME}.lnk", target, arguments, working_dir, icon
                )
        else:
            from pyshortcuts import make_shortcut

            make_shortcut(
                f"{target} {arguments}".strip(),
                name=APP_NAME,
                description="Local AI photo & video editing suite",
                icon=icon,
                terminal=False,
                desktop=True,
                startmenu=True,
                noexe=True,
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
