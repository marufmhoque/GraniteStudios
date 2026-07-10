"""Build a standalone Windows distribution of Granite Studios with PyInstaller.

Usage
-----
    pip install pyinstaller
    python package_windows.py

Output
------
    dist/Granite Studios/            <- the app folder (--onedir)
        Granite Studios.exe          <- double-click to launch

--onedir is used (instead of --onefile) for much faster startup, and
--noconsole hides the terminal window so no black command box appears.
Zip the whole ``dist/Granite Studios`` folder to distribute it.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
APP_NAME = "Granite Studios"
ENTRY_POINT = "main.py"

# Modules PyInstaller's static analysis can miss (native / lazily imported).
HIDDEN_IMPORTS = [
    "rawpy",
    "PyQt6.QtCore",
    "PyQt6.QtWidgets",
    "PyQt6.QtGui",
    "moviepy",
    "onnxruntime",
    # Defensive extras: these are imported by the app but rely on native
    # backends that PyInstaller occasionally fails to follow.
    "pillow_heif",
    "cv2",
]


def build() -> None:
    try:
        import PyInstaller.__main__ as pyi
    except ImportError:
        sys.exit(
            "PyInstaller is not installed.\n"
            "Run:  pip install pyinstaller\n"
            "Then: python package_windows.py"
        )

    # On Windows the --add-data separator is ';' (source;destination_in_bundle).
    sep = os.pathsep  # ';' on Windows

    args: list[str] = [
        ENTRY_POINT,
        "--name", APP_NAME,
        "--onedir",        # folder build -> fast startup
        "--noconsole",     # no black console window on launch
        "--noconfirm",     # overwrite a previous build without prompting
        "--clean",         # clear PyInstaller's cache for a fresh build
        # Bundle the entire /src package as data so resource_path() can find it.
        "--add-data", f"{PROJECT_ROOT / 'src'}{sep}src",
    ]

    # Include the assets folder (icons, etc.) only if it exists.
    assets_dir = PROJECT_ROOT / "assets"
    if assets_dir.is_dir():
        args += ["--add-data", f"{assets_dir}{sep}assets"]
        icon_ico = assets_dir / "icon.ico"
        if icon_ico.exists():
            args += ["--icon", str(icon_ico)]

    for mod in HIDDEN_IMPORTS:
        args += ["--hidden-import", mod]

    print("Running PyInstaller with:")
    print("  " + " ".join(args))
    print()

    pyi.run(args)

    dist_dir = PROJECT_ROOT / "dist" / APP_NAME
    print()
    print("=" * 60)
    if dist_dir.exists():
        print(f"Build complete: {dist_dir}")
        print(f"Launch: {dist_dir / (APP_NAME + '.exe')}")
        print(f"Distribute: zip the folder '{dist_dir}'.")
    else:
        print("Build finished, but the expected output folder was not found.")
        print("Check the PyInstaller output above for errors.")
    print("=" * 60)


if __name__ == "__main__":
    # PyInstaller must run from the project root so relative paths resolve.
    os.chdir(PROJECT_ROOT)
    build()
