"""Background rendering that keeps up with a slider.

Dragging a slider asks for a new preview dozens of times a second, far
faster than a frame can be rendered. Queuing every request would leave the
picture lagging seconds behind the hand. Instead only the newest request
is kept: while one render runs, later requests replace each other, and when
it finishes the latest one starts. A result that has been overtaken is
dropped rather than shown.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


class LatestOnlyRenderer(QObject):
    """Runs one job at a time on a worker thread, always the newest."""

    ready = pyqtSignal(object)           # the job's result
    failed = pyqtSignal(str)
    _finished = pyqtSignal(int, object, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._generation = 0
        self._pending: tuple[int, Callable] | None = None
        self._busy = False
        self._finished.connect(self._on_finished)

    @property
    def busy(self) -> bool:
        return self._busy or self._pending is not None

    def request(self, job: Callable[[], object]) -> None:
        """Render ``job()`` on a worker thread when the current one is done."""
        self._generation += 1
        self._pending = (self._generation, job)
        if not self._busy:
            self._start()

    def cancel(self) -> None:
        """Forget anything queued and ignore the result of what is running."""
        self._generation += 1
        self._pending = None

    def _start(self) -> None:
        generation, job = self._pending
        self._pending = None
        self._busy = True
        threading.Thread(target=self._run, args=(generation, job), daemon=True).start()

    def _run(self, generation: int, job: Callable) -> None:
        try:
            self._finished.emit(generation, job(), None)
        except Exception as exc:     # reported, never raised into Qt
            logger.exception("Preview render failed")
            self._finished.emit(generation, None, exc)

    def _on_finished(self, generation: int, result, error) -> None:
        self._busy = False
        if generation == self._generation:
            if error is not None:
                self.failed.emit(f"{type(error).__name__}: {error}")
            else:
                self.ready.emit(result)
        if self._pending is not None:
            self._start()

    def wait_idle(self, app, timeout: float = 60.0) -> None:
        """Spin the event loop until nothing is rendering (for tests)."""
        import time

        deadline = time.monotonic() + timeout
        while self.busy:
            app.processEvents()
            if time.monotonic() > deadline:
                raise TimeoutError("a preview render did not finish")
            time.sleep(0.005)
        for _ in range(3):
            app.processEvents()
