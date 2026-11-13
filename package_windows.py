"""Build a standalone Windows distribution of Granite Studios with PyInstaller.

Usage
-----
    pip install pyinstaller
    python package_windows.py

Output
------
    dist/GraniteStudios/            <- the app folder (--onedir)
        GraniteStudios.exe          <- double-click to launch

--onedir is used (instead of --onefile) for much faster startup, and
--noconsole hides the terminal window so no black command box appears.
Zip the whole ``dist/GraniteStudios`` folder to distribute it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
APP_NAME = "GraniteStudios"          # PyInstaller --name -> exe filename
DISPLAY_NAME = "Granite Studios"     # human-facing name for shortcuts/prints
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
    "pyshortcuts",
    "psutil",
    "rembg",
    "onnxruntime",
    "imageio_ffmpeg",
    "fastapi",
    "uvicorn",
    "pydantic",
    # Added in 2.0.
    "lensfunpy",
    "tifffile",
    "scipy.interpolate",
    "PyQt6.QtSvg",
]

ICO_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def convert_icons(project_root: Path, source_png: Path) -> tuple[Path | None, Path | None]:
    """Convert icon.png into icon.ico (Windows) and icon.icns (macOS) via Pillow.

    Each OS reads its own native icon format for the exe/app-bundle/taskbar/
    Dock to render correctly — Explorer wants .ico, Finder wants .icns.
    icon.png stays the single source of truth; both files are regenerated
    on every build so they can never drift out of sync with it.

    Returns (ico_path, icns_path), with either entry None if source_png is
    missing.
    """
    if not source_png.exists():
        print(f"Warning: {source_png} not found — skipping icon conversion.")
        return None, None

    from PIL import Image

    image = Image.open(source_png).convert("RGBA")

    ico_path = project_root / "icon.ico"
    image.save(ico_path, format="ICO", sizes=ICO_SIZES)

    icns_path = project_root / "icon.icns"
    image.save(icns_path, format="ICNS")

    print(f"Generated {ico_path.name} (Windows) and {icns_path.name} (macOS) from {source_png.name}.")
    return ico_path, icns_path


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

    # App icon: convert icon.png -> icon.ico/.icns, use the .ico for the
    # .exe itself, and bundle both the source PNG and the ICO as runtime
    # data so resource_path() can find them (splash screen, window icon,
    # and the shortcut icon created by src/engine/create_shortcut.py).
    icon_png = PROJECT_ROOT / "icon.png"
    ico_path, _icns_path = convert_icons(PROJECT_ROOT, icon_png)
    if ico_path is not None:
        args += ["--icon", str(ico_path)]
        args += ["--add-data", f"{ico_path}{sep}."]
    if icon_png.exists():
        args += ["--add-data", f"{icon_png}{sep}."]

    # Include the assets folder (extra art, fonts, etc.) only if it exists.
    assets_dir = PROJECT_ROOT / "assets"
    if assets_dir.is_dir():
        args += ["--add-data", f"{assets_dir}{sep}assets"]

    for mod in HIDDEN_IMPORTS:
        args += ["--hidden-import", mod]

    # imageio_ffmpeg ships its bundled ffmpeg.exe as package data, not code —
    # PyInstaller's import analysis can't see it, so it has to be collected
    # explicitly or the packaged app has no working video engine.
    args += ["--collect-data", "imageio_ffmpeg"]
    # The lens-correction database is XML shipped as package data, likewise.
    args += ["--collect-data", "lensfunpy"]

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
        print(f"On first launch, {DISPLAY_NAME} adds its own Desktop and Start Menu shortcut.")
    else:
        print("Build finished, but the expected output folder was not found.")
        print("Check the PyInstaller output above for errors.")
    print("=" * 60)


if __name__ == "__main__":
    # PyInstaller must run from the project root so relative paths resolve.
    os.chdir(PROJECT_ROOT)
    build()
