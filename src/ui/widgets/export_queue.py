"""The Export Queue: several deliveries, picked by who is in the photos, in one run.

Each :class:`BundleCard` edits one :class:`ExportBundle`: a title, the people
(chosen from the faces Cull recognised), how they must appear (any of them,
together, alone, or everyone), a preset, a watermark and a subfolder, with a
live count of the photos it matches. :class:`ExportQueuePanel` holds the
cards, the base folder and the **Run All Exports** button, and runs the
queue with :class:`MultiExportRunner` off the interface thread.

The queue is kept in the folder's ``-edit`` folder, so it is there the next
time the folder or project is opened. Originals are only read.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from pathlib import Path

from PyQt6.QtCore import QObject, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from src.engine import export_presets as ep
from src.engine.export_presets import ExportBundle, MATCH_LABELS, MultiExportRunner

logger = logging.getLogger(__name__)

FACE_SIZE = 40
MODES = ("ALL_PROJECT", "ANY", "ALL", "SOLO")
WATERMARKS = (("No watermark", ""), ("Signature", "signature"), ("Seal", "seal"),
              ("Camera details strip", "exif_strip"))
QUEUE_STYLE = """
QFrame#BundleCard { background: #1c1c1f; border: 1px solid #2c2c30; border-radius: 10px; }
QFrame#BundleCard QLabel { background: transparent; }
QPushButton#ModeButton {
    background: #232326; color: #c9c9ce; border: 1px solid #34343a; padding: 5px 12px; border-radius: 0;
}
QPushButton#ModeButton:checked { background: #E31B23; color: #ffffff; border-color: #E31B23; }
QPushButton#ModeButton:disabled { color: #55555b; }
QPushButton#PersonChip {
    background: #2a2a2e; color: #f2f3f5; border: 1px solid #3a3a40; border-radius: 12px; padding: 3px 10px;
}
QPushButton#PersonChip:hover { border-color: #E31B23; }
QPushButton#RemoveBundle { background: transparent; color: #8f8f94; border: none; font-size: 16px; padding: 2px 8px; }
QPushButton#RemoveBundle:hover { color: #ffffff; }
QLabel#MatchBadge { padding: 3px 10px; border-radius: 10px; background: #202a22; color: #8fe39b; }
QLabel#MatchBadge[empty="true"] { background: #2a2020; color: #e38f8f; }
"""


# --------------------------------------------------------------------- #
# Face thumbnails, cut from the preview the faces were found in
# --------------------------------------------------------------------- #

class FaceThumbs(QObject):
    """Square face crops for the people of a folder, made off the
    interface thread and cached. ``ready(person_id, QImage)`` as each is
    done."""

    ready = pyqtSignal(int, QImage)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.images: dict[tuple[str, int], QImage] = {}
        self._pending: set[tuple[str, int]] = set()

    def get(self, folder: str, person: int) -> QImage | None:
        return self.images.get((folder, person))

    def request(self, folder: str, people) -> None:
        """Start making the crops this folder's people do not have yet."""
        if people is None:
            return
        wanted = []
        for pid in people.by_size():
            key = (folder, pid)
            if key in self.images or key in self._pending:
                continue
            refs = people.people.get(pid, [])[:12]
            if refs:
                wanted.append((pid, max(refs, key=lambda r: r.box[2] * r.box[3])))
                self._pending.add(key)
        if not wanted:
            return

        def work() -> None:
            for pid, ref in wanted:
                try:
                    image = face_image(ref)
                except Exception as exc:             # a face we cannot show is only a missing icon
                    logger.debug("No face thumbnail for %s: %s", ref.path, exc)
                    image = None
                self._pending.discard((folder, pid))
                if image is not None:
                    self.images[(folder, pid)] = image
                    try:
                        self.ready.emit(pid, image)
                    except RuntimeError:        # the window closed meanwhile
                        return

        threading.Thread(target=work, name="face-thumbs", daemon=True).start()


def face_image(ref, size: int = FACE_SIZE * 2) -> QImage | None:
    """A square crop around one face, or None if the photo cannot be read."""
    from src.ai_tools.culling import FACE_MAX_SIDE
    from src.engine import fast_preview

    preview = fast_preview.load_preview(ref.path, max_side=FACE_MAX_SIDE)
    if preview is None:
        return None
    array = preview.image
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    height, width = array.shape[:2]
    x, y, w, h = ref.box
    side = int(max(w, h) * 1.5)
    cx, cy = x + w // 2, y + h // 2
    left, top = max(0, cx - side // 2), max(0, cy - side // 2)
    crop = array[top:min(height, top + side), left:min(width, left + side)]
    if crop.size == 0:
        return None
    import cv2

    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA).copy()
    return QImage(crop.data, size, size, size * 3, QImage.Format.Format_RGB888).copy()


def _discard(widget) -> None:
    """Take a widget off the screen now; Qt deletes it later."""
    if widget is not None:
        widget.hide()
        widget.setParent(None)
        widget.deleteLater()


def circle_icon(image: QImage | None, size: int = FACE_SIZE) -> QIcon:
    if image is None:
        return QIcon()
    return QIcon(QPixmap.fromImage(image).scaled(size, size, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                                 Qt.TransformationMode.SmoothTransformation))


# --------------------------------------------------------------------- #
# The people picker: search, faces, tick boxes
# --------------------------------------------------------------------- #

class PeoplePicker(QFrame):
    """A pop-up list of the recognised people, with a search field. Ticking
    someone adds them to the bundle at once."""

    def __init__(self, card: "BundleCard") -> None:
        super().__init__(card, Qt.WindowType.Popup)
        self.card = card
        self.setObjectName("PeoplePicker")
        self.setStyleSheet("QFrame#PeoplePicker { background: #1a1a1d; border: 1px solid #3a3a40; }")
        self.resize(300, 360)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search people…")
        self.search.textChanged.connect(self._filter)
        layout.addWidget(self.search)
        self.list = QListWidget()
        self.list.setIconSize(QSize(FACE_SIZE, FACE_SIZE))
        layout.addWidget(self.list, stretch=1)
        people = card.people()
        chosen = {ep_id for ep_id in card.bundle.person_ids}
        if people is not None:
            for pid in people.by_size():
                count = len(people.paths_of(pid))
                item = QListWidgetItem(f"{people.label(pid)}   · {count:,} photo{'s' if count != 1 else ''}")
                item.setData(Qt.ItemDataRole.UserRole, pid)
                item.setData(Qt.ItemDataRole.UserRole + 1, people.label(pid))
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if str(pid) in chosen else Qt.CheckState.Unchecked)
                item.setIcon(circle_icon(card.face(pid)))
                self.list.addItem(item)
        if self.list.count() == 0:
            empty = QListWidgetItem("Nobody recognised yet: in Cull, press “Group by Person”.")
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            self.list.addItem(empty)
        self.list.itemChanged.connect(self._on_changed)
        self.list.itemClicked.connect(self._on_clicked)

    def set_face(self, pid: int, image: QImage) -> None:
        for row in range(self.list.count()):
            item = self.list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == pid:
                item.setIcon(circle_icon(image))

    def _filter(self, text: str) -> None:
        needle = text.strip().casefold()
        for row in range(self.list.count()):
            item = self.list.item(row)
            name = str(item.data(Qt.ItemDataRole.UserRole + 1) or "")
            item.setHidden(bool(needle) and needle not in name.casefold())

    def _on_clicked(self, item: QListWidgetItem) -> None:
        if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:      # a click anywhere on the row ticks it
            item.setCheckState(Qt.CheckState.Unchecked if item.checkState() == Qt.CheckState.Checked
                               else Qt.CheckState.Checked)

    def checked(self) -> list[str]:
        return [str(self.list.item(r).data(Qt.ItemDataRole.UserRole)) for r in range(self.list.count())
                if self.list.item(r).flags() & Qt.ItemFlag.ItemIsUserCheckable
                and self.list.item(r).checkState() == Qt.CheckState.Checked]

    def _on_changed(self, _item) -> None:
        self.card.set_people(self.checked())


# --------------------------------------------------------------------- #
# One bundle
# --------------------------------------------------------------------- #

class BundleCard(QFrame):
    """Edits one :class:`ExportBundle`."""

    changed = pyqtSignal()
    remove_requested = pyqtSignal(object)

    def __init__(self, bundle: ExportBundle, panel: "ExportQueuePanel") -> None:
        super().__init__()
        self.bundle = bundle
        self.panel = panel
        self.picker: PeoplePicker | None = None
        # The title and folder follow the people until the user types their own.
        self._auto_title = bundle.bundle_name == self._default_title()
        self._auto_folder = not bundle.destination_subfolder or \
            bundle.destination_subfolder == ep.folder_slug(bundle.bundle_name)
        self.setObjectName("BundleCard")
        grid = QGridLayout(self)
        grid.setContentsMargins(14, 12, 10, 12)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        self.title_edit = QLineEdit(bundle.bundle_name)
        self.title_edit.setPlaceholderText("Bundle name, e.g. John & Jane")
        self.title_edit.setStyleSheet("font-size: 15px; font-weight: 700;")
        self.title_edit.textEdited.connect(self._on_title_edited)
        self.badge = QLabel()
        self.badge.setObjectName("MatchBadge")
        self.remove_button = QPushButton("✕")
        self.remove_button.setObjectName("RemoveBundle")
        self.remove_button.setToolTip("Remove this bundle from the queue")
        self.remove_button.clicked.connect(lambda: self.remove_requested.emit(self))
        top = QHBoxLayout()
        top.addWidget(self.title_edit, stretch=1)
        top.addWidget(self.badge)
        top.addWidget(self.remove_button)
        grid.addLayout(top, 0, 0, 1, 2)

        grid.addWidget(self._label("People"), 1, 0)
        people_row = QHBoxLayout()
        people_row.setSpacing(6)
        self.chips = QHBoxLayout()
        self.chips.setSpacing(6)
        people_row.addLayout(self.chips)
        self.add_people_button = QPushButton("+ People…")
        self.add_people_button.setObjectName("SecondaryButton")
        self.add_people_button.setToolTip("Choose who this bundle is for, from the faces found in Cull")
        self.add_people_button.clicked.connect(self.open_picker)
        people_row.addWidget(self.add_people_button)
        people_row.addStretch()
        grid.addLayout(people_row, 1, 1)

        grid.addWidget(self._label("Match"), 2, 0)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(0)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_buttons: dict[str, QPushButton] = {}
        tips = {
            "ALL_PROJECT": "Every keeper in the folder, whoever is in it",
            "ANY": "Photos with at least one of these people",
            "ALL": "Photos with all of these people in them together",
            "SOLO": "Photos of one of these people alone",
        }
        for mode in MODES:
            button = QPushButton(MATCH_LABELS[mode])
            button.setObjectName("ModeButton")
            button.setCheckable(True)
            button.setToolTip(tips[mode])
            button.clicked.connect(lambda _c=False, m=mode: self.set_mode(m))
            self.mode_group.addButton(button)
            self.mode_buttons[mode] = button
            mode_row.addWidget(button)
        mode_row.addStretch()
        grid.addLayout(mode_row, 2, 1)

        grid.addWidget(self._label("Preset"), 3, 0)
        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        for key, preset in ep.PRESETS.items():
            self.preset_combo.addItem(preset.name, key)
        try:
            self.preset_combo.setCurrentIndex(max(0, self.preset_combo.findData(ep.preset_key(bundle.preset_name))))
        except KeyError:
            pass
        self.preset_combo.currentIndexChanged.connect(self._on_preset)
        preset_row.addWidget(self.preset_combo)
        self.watermark_combo = QComboBox()
        for label, kind in WATERMARKS:
            self.watermark_combo.addItem(label, kind)
        self.watermark_combo.setCurrentIndex(max(0, self.watermark_combo.findData(bundle.watermark_template or "")))
        self.watermark_combo.currentIndexChanged.connect(self._on_watermark)
        preset_row.addWidget(self.watermark_combo)
        self.watermark_name = QLineEdit(bundle.watermark_text)
        self.watermark_name.setPlaceholderText("Name for the signature or seal")
        self.watermark_name.textEdited.connect(self._on_watermark)
        preset_row.addWidget(self.watermark_name, stretch=1)
        grid.addLayout(preset_row, 3, 1)

        grid.addWidget(self._label("Subfolder"), 4, 0)
        self.folder_edit = QLineEdit(bundle.destination_subfolder or bundle.subfolder)
        self.folder_edit.setPlaceholderText("Folder inside the export folder")
        self.folder_edit.textEdited.connect(self._on_folder_edited)
        grid.addWidget(self.folder_edit, 4, 1)
        grid.setColumnStretch(1, 1)
        self._sync()

    @staticmethod
    def _label(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("color: #8f8f94;")
        return label

    # -- what the card knows ------------------------------------------------- #

    def people(self):
        return self.panel.session.people

    def face(self, pid: int) -> QImage | None:
        return self.panel.faces.get(self.panel.folder_key(), pid)

    def names(self) -> list[str]:
        people = self.people()
        if people is None:
            return [f"Person {int(p) + 1}" for p in self.bundle.person_ids]
        return [people.label(int(p)) for p in self.bundle.person_ids]

    def _default_title(self) -> str:
        if self.bundle.mode == "ALL_PROJECT":
            return ep.EVERYONE
        return ep.bundle_title(self.names())

    # -- editing ---------------------------------------------------------------- #

    def set_people(self, person_ids) -> None:
        ids = [str(p) for p in dict.fromkeys(str(p) for p in person_ids)]
        self.bundle.person_ids = ids
        if ids and self.bundle.match_mode == "ALL_PROJECT":
            self.bundle.match_mode = "ANY"
        elif not ids:
            self.bundle.match_mode = "ALL_PROJECT"
        self._follow_names()
        self._sync()
        self.changed.emit()

    def set_mode(self, mode: str) -> None:
        if mode != "ALL_PROJECT" and not self.bundle.person_ids:
            mode = "ALL_PROJECT"
        self.bundle.match_mode = mode
        self._follow_names()
        self._sync()
        self.changed.emit()

    def _follow_names(self) -> None:
        if self._auto_title:
            self.bundle.bundle_name = self._default_title()
        if self._auto_folder:
            self.bundle.destination_subfolder = ep.folder_slug(self.bundle.bundle_name)

    def _on_title_edited(self, text: str) -> None:
        self._auto_title = not text.strip()
        self.bundle.bundle_name = text.strip() or self._default_title()
        if self._auto_folder:
            self.bundle.destination_subfolder = ep.folder_slug(self.bundle.bundle_name)
            self.folder_edit.setText(self.bundle.destination_subfolder)
        self.changed.emit()

    def _on_folder_edited(self, text: str) -> None:
        self._auto_folder = not text.strip()
        self.bundle.destination_subfolder = text.strip() or ep.folder_slug(self.bundle.bundle_name)
        self.changed.emit()

    def _on_preset(self, _index: int) -> None:
        self.bundle.preset_name = self.preset_combo.currentData()
        self.changed.emit()

    def _on_watermark(self, *_args) -> None:
        self.bundle.watermark_template = self.watermark_combo.currentData() or None
        self.bundle.watermark_text = self.watermark_name.text()
        self.watermark_name.setVisible(self.bundle.watermark_template in ("signature", "seal"))
        self.changed.emit()

    def remove_person(self, pid: str) -> None:
        self.set_people([p for p in self.bundle.person_ids if p != str(pid)])

    def open_picker(self) -> PeoplePicker:
        self.picker = PeoplePicker(self)
        self.picker.move(self.add_people_button.mapToGlobal(self.add_people_button.rect().bottomLeft()))
        self.picker.show()
        self.picker.search.setFocus()
        return self.picker

    def set_face(self, pid: int, image: QImage) -> None:
        if self.picker is not None and self.picker.isVisible():
            self.picker.set_face(pid, image)
        self._sync_chips()

    # -- showing it ------------------------------------------------------------- #

    def _sync_chips(self) -> None:
        while self.chips.count():
            _discard(self.chips.takeAt(0).widget())
        for pid, name in zip(self.bundle.person_ids, self.names()):
            chip = QPushButton(f"{name}  ✕")
            chip.setObjectName("PersonChip")
            chip.setToolTip(f"Remove {name} from this bundle")
            image = self.face(int(pid))
            chip.setIcon(circle_icon(image, 20) if image is not None else QIcon())
            if image is None:
                chip.setText(f"\U0001F464 {name}  ✕")
            chip.setIconSize(QSize(20, 20))
            chip.clicked.connect(lambda _c=False, p=pid: self.remove_person(p))
            self.chips.addWidget(chip)

    def _sync(self) -> None:
        self._sync_chips()
        mode = self.bundle.mode
        for key, button in self.mode_buttons.items():
            button.setChecked(key == mode)
            button.setEnabled(key == "ALL_PROJECT" or bool(self.bundle.person_ids))
        if self.title_edit.text() != self.bundle.bundle_name:
            self.title_edit.setText(self.bundle.bundle_name)
        if self.folder_edit.text() != self.bundle.destination_subfolder:
            self.folder_edit.setText(self.bundle.destination_subfolder)
        self.watermark_name.setVisible(self.bundle.watermark_template in ("signature", "seal"))
        self.add_people_button.setText("+ People…" if not self.bundle.person_ids else "+ More…")

    def set_count(self, count: int) -> None:
        self.count = count
        dot = "\U0001F7E2" if count else "⚪"
        self.badge.setText(f"{dot} Matches: {count:,} photo{'s' if count != 1 else ''}")
        self.badge.setProperty("empty", "true" if not count else "false")
        self.badge.style().unpolish(self.badge)
        self.badge.style().polish(self.badge)


# --------------------------------------------------------------------- #
# The queue
# --------------------------------------------------------------------- #

class _QueueSignals(QObject):
    """Carries a run's progress from its thread to the window."""

    progress = pyqtSignal(str, int, int)


class ExportQueuePanel(QWidget):
    """The Export Queue Builder in the Export workspace."""

    def __init__(self, workspace) -> None:
        super().__init__()
        self.workspace = workspace
        self.session = workspace.session
        self.cards: list[BundleCard] = []
        self.base_dir = ""
        self.last_result = None
        self._loaded_for: str | None = None
        self.faces = FaceThumbs(self)
        self.faces.ready.connect(self._on_face)
        self._signals = _QueueSignals(self)
        self._signals.progress.connect(self._on_progress)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(self.save)
        self.setStyleSheet(QUEUE_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 16, 28, 8)
        layout.setSpacing(10)
        layout.addLayout(workspace._page_header(
            "Export Queue",
            "Several deliveries in one run: each bundle picks the keepers by who is in them, with its "
            "own preset, watermark and folder. A photo two bundles share is rendered once."))
        self.cards_layout = QVBoxLayout()
        self.cards_layout.setSpacing(10)
        layout.addLayout(self.cards_layout)
        self.add_button = QPushButton("+ Add Export Bundle")
        self.add_button.setObjectName("SecondaryButton")
        self.add_button.clicked.connect(lambda: self.add_bundle())
        add_row = QHBoxLayout()
        add_row.addWidget(self.add_button)
        add_row.addStretch()
        layout.addLayout(add_row)

        bar = QFrame()
        bar.setStyleSheet("QFrame { border-top: 1px solid #2a2a2a; } QLabel { border: none; }")
        bottom = QVBoxLayout(bar)
        bottom.setContentsMargins(0, 10, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("Export to:"))
        self.base_label = QLabel()
        self.base_label.setObjectName("PageSubtitle")
        self.base_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.base_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        row.addWidget(self.base_label, stretch=1)
        self.base_button = QPushButton("Choose Folder…")
        self.base_button.setObjectName("SecondaryButton")
        self.base_button.clicked.connect(self.choose_base_dir)
        row.addWidget(self.base_button)
        self.run_button = QPushButton()
        self.run_button.setObjectName("ActionButton")
        self.run_button.clicked.connect(self.run_all)
        row.addWidget(self.run_button)
        bottom.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        bottom.addWidget(self.progress)
        self.queue_status = QLabel()
        self.queue_status.setObjectName("PageSubtitle")
        self.queue_status.setWordWrap(True)
        bottom.addWidget(self.queue_status)
        layout.addWidget(bar)
        self._show_base()
        self.refresh_counts()

    # -- the bundles ---------------------------------------------------------------- #

    def folder_key(self) -> str:
        return str(self.session.current_folder or "")

    def bundles(self) -> list[ExportBundle]:
        return [card.bundle for card in self.cards]

    def add_bundle(self, bundle: ExportBundle | None = None, save: bool = True) -> BundleCard:
        card = BundleCard(bundle or ExportBundle(destination_subfolder=ep.folder_slug(ep.EVERYONE)), self)
        card.changed.connect(self._on_card_changed)
        card.remove_requested.connect(self.remove_bundle)
        self.cards.append(card)
        self.cards_layout.addWidget(card)
        self.refresh_counts()
        if save:
            self._save_soon()
        return card

    def remove_bundle(self, card: BundleCard) -> None:
        if card in self.cards:
            self.cards.remove(card)
            self.cards_layout.removeWidget(card)
            _discard(card)
            self.refresh_counts()
            self._save_soon()

    def _on_card_changed(self) -> None:
        self.refresh_counts()
        self._save_soon()

    def refresh_counts(self) -> tuple[int, int]:
        """Each card's match badge, and the Run button's totals."""
        total = 0
        for card in self.cards:
            count = len(self.session.images_for_people(card.bundle.person_ids, card.bundle.mode)) \
                if self.session.is_open else 0
            card.set_count(count)
            total += count
        bundles = len(self.cards)
        self.run_button.setText(f"\U0001F680 Run All Exports ({total:,} total file{'s' if total != 1 else ''} "
                                f"across {bundles} bundle{'s' if bundles != 1 else ''})")
        self.run_button.setEnabled(total > 0)
        return total, bundles

    def refresh_people(self) -> None:
        """People were found, named or forgotten in Cull."""
        for card in self.cards:
            people = self.session.people
            if people is not None:
                card.bundle.person_ids = [p for p in card.bundle.person_ids if int(p) in people.people]
            card._follow_names()
            card._sync()
        self.faces.request(self.folder_key(), self.session.people)
        self.refresh_counts()

    def _on_face(self, pid: int, image: QImage) -> None:
        for card in self.cards:
            if str(pid) in card.bundle.person_ids or (card.picker is not None and card.picker.isVisible()):
                card.set_face(pid, image)

    # -- the folder they go to ---------------------------------------------------------- #

    def _show_base(self) -> None:
        self.base_label.setText(self.base_dir or "No folder chosen yet")
        self.base_label.setToolTip(self.base_dir)

    def set_base_dir(self, folder: str | Path) -> None:
        self.base_dir = str(folder) if folder else ""
        self._show_base()
        self._save_soon()

    def choose_base_dir(self) -> str:
        chosen = QFileDialog.getExistingDirectory(self, "Export the bundles into which folder?", self.base_dir)
        if chosen:
            self.set_base_dir(chosen)
        return chosen

    # -- keeping it with the folder -------------------------------------------------------- #

    def _save_soon(self) -> None:
        if self.session.edit_folder is not None:
            self._save_timer.start()

    def save(self) -> None:
        self._save_timer.stop()
        if self.session.edit_folder is None or self._loaded_for != self.folder_key():
            return
        try:
            ep.save_queue(self.session.edit_folder, self.bundles(), self.base_dir)
        except OSError as exc:
            logger.warning("Could not save the export queue: %s", exc)

    def load_for_folder(self) -> None:
        """Show the queue saved for the folder that is open (one "Everyone"
        bundle for a folder that has none)."""
        if self._loaded_for == self.folder_key():
            self.refresh_people()
            return
        self.save()
        for card in list(self.cards):
            self.cards.remove(card)
            self.cards_layout.removeWidget(card)
            _discard(card)
        self._loaded_for = self.folder_key()
        bundles, base = ep.load_queue(self.session.edit_folder) if self.session.edit_folder else ([], "")
        self.base_dir = base
        self._show_base()
        for bundle in bundles or [ExportBundle(destination_subfolder=ep.folder_slug(ep.EVERYONE))]:
            self.add_bundle(bundle, save=False)
        self.refresh_people()

    # -- running ------------------------------------------------------------------------- #

    def run_all(self) -> bool:
        """Run every bundle. False if nothing was started."""
        ws = self.workspace
        bundles = self.bundles()
        if not ws._require_library():
            return False
        total, _ = self.refresh_counts()
        if not bundles or total == 0:
            QMessageBox.information(self, "Export Queue", "No bundle matches any photos yet.")
            return False
        for bundle in bundles:
            if bundle.watermark_template in ("signature", "seal") and not bundle.watermark_text.strip():
                QMessageBox.information(self, "Watermark",
                                        f"Type the name to put on the watermark of “{bundle.bundle_name}”.")
                return False
        if not self.base_dir and not self.choose_base_dir():
            return False
        base = Path(self.base_dir)
        self.session.save_recipes()
        self.save()
        runner = MultiExportRunner(self.session)
        # The jobs are made here, on the interface thread, once per photo.
        jobs = {path: ws._job(path) for _, paths in runner.plan(bundles) for path in paths}
        runner.job_factory = jobs.__getitem__
        signals = self._signals
        self.progress.setRange(0, total)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.queue_status.setText(f"Starting {total:,} file(s) across {len(bundles)} bundle(s)…")

        def work():
            def tick(_name, done, count):
                try:
                    signals.progress.emit(runner.progress.describe(), done, count)
                except RuntimeError:            # the window closed meanwhile
                    pass
            return runner.run_bundles(bundles, base, tick)

        def done(result):
            self.last_result = result
            self.progress.setVisible(False)
            failures = result.failures
            message = (f"Exported {result.written:,} file(s) across {len(result.bundles)} bundle(s) to {base}.")
            if result.copied:
                message += f" {result.copied:,} were shared between bundles and rendered only once."
            if failures:
                message += f" {len(failures)} could not be exported: " + \
                    ", ".join(sorted({p.name for p, _ in failures})[:5])
            self.queue_status.setText(message + " Originals untouched.")
            ws.status_label.setText(message)
            log = getattr(type(ws.shell), "log_action", None)
            if log is not None and result.written:
                names = ", ".join(b.bundle.bundle_name for b in result.bundles if b.written)
                log(ws.shell, "export", f"Export queue: {result.written:,} files in "
                    f"{sum(1 for b in result.bundles if b.written)} bundles ({names})")

        ws._run_task(work, done, self.run_button)
        return True

    def _on_progress(self, text: str, done: int, total: int) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        self.queue_status.setText(text)
