"""Background rendering that keeps up with a slider.

Dragging a slider asks for a new preview dozens of times a second, far
faster than a frame can be rendered. Queuing every request would leave the
picture lagging seconds behind the hand. Instead only the newest request
is kept: while one render runs, later requests replace each other, and when
it finishes the latest one starts. A result that has been overtaken is
dropped rather than shown, unless ``show_progress`` is set: then it is shown
when it is newer than anything shown so far, so a preview keeps moving with a
slider that never pauses (each frame a little behind the hand, the last one
exact). :meth:`cancel` (a different photo) drops everything older either way.
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
        # Something that will request a render shortly (Develop's settle
        # timer) and should count as busy, so tests wait for it too.
        self.hold = None
        self.show_progress = False
        self._shown = 0                  # generation of the last result emitted
        self._floor = 0                  # nothing older than this is ever emitted
        self._finished.connect(self._on_finished)

    @property
    def busy(self) -> bool:
        return self._busy or self._pending is not None or bool(self.hold is not None and self.hold())

    def request(self, job: Callable[[], object]) -> None:
        """Render ``job()`` on a worker thread when the current one is done."""
        self._generation += 1
        self._pending = (self._generation, job)
        if not self._busy:
            self._start()

    def cancel(self) -> None:
        """Forget anything queued and ignore the result of what is running."""
        self._generation += 1
        self._floor = self._generation
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
        latest = generation == self._generation
        progress = (self.show_progress and error is None
                    and generation > max(self._shown, self._floor))
        if latest or progress:
            self._shown = generation
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
