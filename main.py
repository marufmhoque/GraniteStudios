"""Granite Studios — entry point.

Run with:  python main.py

Boot sequence:
  1. Fix the Windows taskbar icon (must happen before any window exists).
  2. Show a frameless splash screen immediately.
  3. Import the UI package (this pulls in the heavier engine modules —
     rawpy, cv2, onnxruntime — while the splash is already on screen).
  4. Build the main window, then reveal it once the splash's minimum
     display time has elapsed.
"""

import ctypes
import platform
import sys
from pathlib import Path

# Ensure the project root is importable regardless of invocation directory.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.resources import resource_path

SPLASH_DURATION_MS = 3000
APP_USER_MODEL_ID = "GraniteStudios.PhotoSuite.1.0"


def _fix_windows_taskbar_icon() -> None:
    """Give the taskbar our own icon instead of the generic Python one.

    Windows groups taskbar buttons by an "AppUserModelID". A python.exe
    -launched GUI app otherwise inherits Python's own ID, so Explorer shows
    the stock Python logo instead of icon.png. This must run before any
    window (including the splash) is created.
    """
    if platform.system() != "Windows":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except (AttributeError, OSError):
        pass


def main() -> int:
    _fix_windows_taskbar_icon()

    from PyQt6.QtCore import QEventLoop, Qt, QTimer
    from PyQt6.QtGui import QFont, QPixmap
    from PyQt6.QtWidgets import QApplication, QSplashScreen

    app = QApplication(sys.argv)
    app.setApplicationName("Granite Studios")

    app_font = QFont()
    app_font.setFamilies(["Inter", "Segoe UI", "Roboto", "sans-serif"])
    app_font.setPointSize(10)
    app.setFont(app_font)

    # --- Splash screen: frameless, centered, up immediately ------------ #
    splash_pixmap = QPixmap(str(resource_path("icon.png")))
    if not splash_pixmap.isNull():
        splash_pixmap = splash_pixmap.scaled(
            400,
            400,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    splash = QSplashScreen(
        splash_pixmap,
        Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint,
    )
    screen_geometry = app.primaryScreen().geometry()
    splash.move(
        screen_geometry.center().x() - splash.width() // 2,
        screen_geometry.center().y() - splash.height() // 2,
    )
    splash.show()
    app.processEvents()

    # Heavy imports (engine + AI modules) happen here, while the splash is
    # already visible, so the delay is hidden rather than dead air.
    from src.engine.create_shortcut import create_desktop_shortcut
    from src.ui.main_window import MainWindow

    window = MainWindow()

    # First-run convenience: drop a Desktop/Start Menu shortcut so future
    # launches don't require digging through a dist/ folder for the .exe.
    # No-ops instantly on every launch after the first (marker-file check).
    create_desktop_shortcut()

    # Hold the splash for its full duration without blocking the event
    # loop, then hand off to the main window.
    wait_loop = QEventLoop()
    QTimer.singleShot(SPLASH_DURATION_MS, wait_loop.quit)
    wait_loop.exec()

    window.show()
    splash.finish(window)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
