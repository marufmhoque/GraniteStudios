"""Shared fixtures for the Granite Studios test suite.

Everything here builds its media from scratch in a temporary directory, so
the suite needs no sample files and never touches a real photo library.
Tests that do need a real camera file or a cached model are marked
``media`` / ``model`` and skip themselves when it is absent.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

# Qt must be told to run headless before anything imports it.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"


# --------------------------------------------------------------------- #
# Synthetic media
# --------------------------------------------------------------------- #

def make_scene(seed: int = 0, size: tuple[int, int] = (640, 480)) -> np.ndarray:
    """A textured BGR frame with hard edges, so sharpness is well defined."""
    rng = np.random.default_rng(seed)
    w, h = size
    base = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    image = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    for _ in range(14):
        x0, y0 = int(rng.integers(0, w - 60)), int(rng.integers(0, h - 60))
        x1, y1 = x0 + int(rng.integers(20, 160)), y0 + int(rng.integers(20, 120))
        colour = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.rectangle(image, (x0, y0), (x1, y1), colour, -1)
    fine = rng.integers(-18, 18, (h, w, 3), dtype=np.int16)
    return np.clip(image.astype(np.int16) + fine, 0, 255).astype(np.uint8)


def make_architecture(size: tuple[int, int] = (800, 600)) -> np.ndarray:
    """A frame of long straight horizontals and verticals, like a facade."""
    w, h = size
    image = np.full((h, w, 3), 205, np.uint8)
    for y in range(40, h, 55):
        cv2.line(image, (0, y), (w, y), (40, 40, 40), 3)
    for x in range(50, w, 90):
        cv2.line(image, (x, 0), (x, h), (70, 70, 70), 2)
    return image


def rotate_keep_size(image: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate about the centre, filling with edge pixels (no blank wedges)."""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def write_jpeg(path: Path, image: np.ndarray, quality: int = 92) -> Path:
    ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok, f"could not write {path}"
    return path


# --------------------------------------------------------------------- #
# The invariant helpers
# --------------------------------------------------------------------- #

def snapshot(folder: Path) -> dict[str, str]:
    """SHA-256 of every file directly inside ``folder`` (not its subfolders).

    The edit subfolder lives inside the source folder by design, so only the
    top level is the "originals" set.
    """
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.iterdir())
        if p.is_file()
    }


def assert_originals_untouched(before: dict[str, str], folder: Path) -> None:
    """INV-1: every original still exists, byte for byte.

    New files may appear (sidecars, the edit subfolder); nothing that was
    there may change, move or vanish.
    """
    after = snapshot(folder)
    missing = sorted(set(before) - set(after))
    assert not missing, f"originals removed or moved: {missing}"
    changed = sorted(name for name in before if after[name] != before[name])
    assert not changed, f"originals modified: {changed}"


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #

@pytest.fixture
def library(tmp_path: Path) -> Path:
    """A small shoot folder: sharp frames, a near-duplicate, a blurred one,
    a tilted facade and a PNG."""
    folder = tmp_path / "Shoot"
    folder.mkdir()
    scene = make_scene(1)
    write_jpeg(folder / "IMG_0001.jpg", scene)
    write_jpeg(folder / "IMG_0002.jpg", cv2.convertScaleAbs(scene, alpha=1.03, beta=2))
    write_jpeg(folder / "IMG_0003.jpg", make_scene(2))
    write_jpeg(folder / "IMG_0004.jpg", cv2.GaussianBlur(make_scene(3), (0, 0), 9))
    write_jpeg(folder / "IMG_0005.jpg", rotate_keep_size(make_architecture(), 6.0))
    write_jpeg(folder / "IMG_0006.jpg", make_scene(4))
    cv2.imwrite(str(folder / "IMG_0007.png"), make_scene(5))
    return folder


@pytest.fixture(scope="session")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def pump(app, condition, timeout: float = 120.0) -> None:
    """Spin the event loop until ``condition()`` is true."""
    deadline = time.monotonic() + timeout
    while not condition():
        app.processEvents()
        if time.monotonic() > deadline:
            raise TimeoutError("timed out waiting for the UI")
        time.sleep(0.005)
    # Queued signals from a finished worker still need one more turn.
    for _ in range(5):
        app.processEvents()


class Harness:
    """Drives a real MainWindow headlessly, the way a user would."""

    def __init__(self, app, window, folder: Path, out_dir: Path):
        self.app = app
        self.window = window
        self.folder = folder
        self.out_dir = out_dir
        self.messages: list[tuple[str, str]] = []

    def wait(self) -> None:
        w = self.window
        pump(self.app, lambda: w._task_worker is None or not w._task_worker.isRunning())

    def run(self, handler) -> None:
        """Call a slot and wait for the background task it starts."""
        self.wait()
        handler()
        self.wait()

    def select(self, *names: str) -> None:
        w = self.window
        wanted = set(names) if names else {p.name for p in w.image_paths}
        w.selected = {str(p) for p in w.image_paths if p.name in wanted}

    def reject(self, *names: str) -> None:
        w = self.window
        w._set_rejected([p for p in w.image_paths if p.name in names], True)

    def edit_files(self) -> set[str]:
        return {p.name for p in self.window.edit_folder.iterdir() if p.is_file()}


@pytest.fixture
def harness(qapp, library, tmp_path, monkeypatch):
    """A MainWindow with the synthetic library loaded through the real
    load-folder path, and every modal dialog answered automatically."""
    from PyQt6.QtWidgets import QFileDialog, QMessageBox

    from src.ui.main_window import MainWindow

    out_dir = tmp_path / "exports"
    out_dir.mkdir()
    messages: list[tuple[str, str]] = []

    def _note(kind):
        def _inner(parent=None, title="", text="", *args, **kwargs):
            messages.append((kind, f"{title}: {text}"))
            return QMessageBox.StandardButton.Yes
        return staticmethod(_inner)

    for kind in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, kind, _note(kind))

    chosen = {"dir": str(library)}
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory",
        staticmethod(lambda *a, **k: chosen["dir"]),
    )
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (str(out_dir / "reel.mp4"), "")),
    )

    window = MainWindow()
    window._on_load_folder()
    pump(qapp, lambda: window._thumb_worker is None or not window._thumb_worker.isRunning())
    # From here on, folder pickers mean "where to export".
    chosen["dir"] = str(out_dir)

    h = Harness(qapp, window, library, out_dir)
    h.messages = messages
    yield h

    h.wait()
    window.close()
    qapp.processEvents()


def model_cached(*specs) -> bool:
    from src.ai_tools import model_store

    return all(model_store.is_cached(spec) for spec in specs)
