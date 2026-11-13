"""What every workspace has in common.

A workspace is one task-shaped view of the open folder: culling it,
developing a frame, retouching, designing overlays, exporting. They all look
at the same :class:`~src.engine.session.LibrarySession`, so the attributes
that used to live on the single main window (``self.selected``,
``self.rejected`` and so on) are forwarded to it here. That is what lets the
selection made while culling still be the selection in Export.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QLabel, QMessageBox, QVBoxLayout, QWidget


def _shared(name: str):
    """A property that reads and writes one field of the shared session."""
    return property(
        lambda self: getattr(self.session, name),
        lambda self, value: setattr(self.session, name, value),
    )


class Workspace(QWidget):
    """Base class for the workspaces in the bar across the top."""

    title = ""
    shortcut_hint = ""

    def __init__(self, shell) -> None:
        super().__init__()
        self.shell = shell
        self.session = shell.session
        # Engines are created once by the shell and shared.
        self.loader = shell.loader
        self.culler = shell.culler
        self.color_engine = shell.color_engine
        self.bg_remover = shell.bg_remover
        self.reel_maker = shell.reel_maker
        self.ig_formatter = shell.ig_formatter
        self.status_label = shell.status_label

    # -- state that belongs to the folder, not to any one workspace ------- #
    current_folder = _shared("current_folder")
    edit_folder = _shared("edit_folder")
    image_paths = _shared("image_paths")
    selected = _shared("selected")
    rejected = _shared("rejected")
    last_cull_frames = _shared("last_cull_frames")
    blur_reports = _shared("blur_reports")
    web_selected = _shared("web_selected")
    metadata = _shared("metadata")
    labels = _shared("labels")

    # -- lifecycle --------------------------------------------------------- #

    def on_enter(self) -> None:
        """Called each time the workspace is shown."""

    def on_leave(self) -> None:
        """Called when another workspace is about to be shown."""

    def on_session(self, topic: str) -> None:
        """Called when the shared session changes (see ``session.notify``)."""

    def shutdown(self) -> None:
        """Called once as the application closes."""

    # -- helpers ----------------------------------------------------------- #

    def _run_task(self, fn, on_done, busy_button) -> None:
        self.shell._run_task(fn, on_done, busy_button)

    @property
    def _task_worker(self):
        return self.shell._task_worker

    @staticmethod
    def _page_header(title: str, subtitle: str) -> QVBoxLayout:
        header = QVBoxLayout()
        title_label = QLabel(title)
        title_label.setObjectName("PageTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("PageSubtitle")
        header.addWidget(title_label)
        header.addWidget(subtitle_label)
        return header

    def _drop_rejected(self, paths, action: str) -> list:
        """Filter rejected frames out of an export.

        This is the whole meaning of "rejected": the photo is skipped when
        anything is exported. The original file on disk is never deleted,
        moved or altered by this app.
        """
        kept = [p for p in paths if str(p) not in self.rejected]
        dropped = len(paths) - len(kept)
        if dropped:
            self.status_label.setText(
                f"{action}: skipping {dropped} rejected photo(s). "
                f"Originals remain on disk untouched."
            )
        return kept

    def _selected_paths(self) -> list[Path]:
        return [p for p in self.image_paths if str(p) in self.selected]

    def _require_library(self) -> bool:
        if not self.image_paths:
            QMessageBox.information(self, "No images", "Load a folder in the Cull workspace first.")
            return False
        return True

    def _targets(self) -> list[Path]:
        """What a batch action applies to: the selection, else the current frame."""
        chosen = self._selected_paths()
        if chosen:
            return chosen
        current = self.session.current
        return [Path(current)] if current else []
