"""Colours, the stylesheet and small drawing helpers shared by the interface."""

from __future__ import annotations

from PIL import Image
from PyQt6.QtGui import QImage, QPixmap

THUMB_SIZE = 200

# --- "Crimson Granite" theme — Nikon-red accents on deep charcoal --- #
NIKON_RED = "#E31B23"
NIKON_RED_DARK = "#8B0000"

CRIMSON_GRANITE_STYLESHEET = """
* { font-family: "Inter", "Segoe UI", "Roboto", sans-serif; }

QMainWindow, QWidget { background-color: #121212; color: #e8e8ea; font-size: 13px; }

QFrame#Sidebar { background-color: #1a1a1a; border-right: 1px solid #2a2a2a; }

QPushButton#NavButton {
    background: transparent; border: none; border-left: 3px solid transparent;
    border-radius: 8px; padding: 12px 16px; text-align: left;
    color: #a8a8ac; font-size: 14px;
}
QPushButton#NavButton:hover { background-color: #232323; color: #ffffff; }
QPushButton#NavButton:checked {
    background-color: rgba(227, 27, 35, 32);
    border-left: 3px solid #E31B23;
    color: #ffffff; font-weight: 600;
}

QPushButton#ActionButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #E31B23, stop:1 #8B0000);
    border: none; border-radius: 12px;
    padding: 10px 22px; color: #ffffff; font-weight: 700;
}
QPushButton#ActionButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #FF3A42, stop:1 #B00000);
}
QPushButton#ActionButton:pressed {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #C21620, stop:1 #6E0000);
}
QPushButton#ActionButton:disabled { background: #2a2a2a; color: #6a6a6e; }

QPushButton#SecondaryButton {
    background-color: #1e1e1e; border: 1px solid #333333; border-radius: 12px;
    padding: 9px 18px; color: #e8e8ea;
}
QPushButton#SecondaryButton:hover { background-color: #262626; border-color: #E31B23; }

QScrollArea { border: none; }
QLabel#PageTitle { font-size: 22px; font-weight: 700; color: #ffffff; }
QLabel#PageSubtitle { color: #949498; font-size: 13px; }

QLabel#Thumb {
    background-color: #1a1a1a; border: 1px solid #333333; border-radius: 12px;
}
QLabel#Thumb:hover { border: 1px solid #E31B23; }
QLabel#ThumbSelected {
    border: 2px solid #E31B23; border-radius: 12px; background-color: #1a1a1a;
}

QComboBox {
    background-color: #1e1e1e; border: 1px solid #333333; border-radius: 8px;
    padding: 6px 12px; color: #e8e8ea;
}
QComboBox:hover { border-color: #E31B23; }
QComboBox QAbstractItemView {
    background-color: #1a1a1a; color: #e8e8ea; selection-background-color: #E31B23;
    border-radius: 8px;
}

QProgressBar {
    background-color: #1a1a1a; border: none; border-radius: 6px; height: 8px; text-align: center;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #E31B23, stop:1 #8B0000);
    border-radius: 6px;
}

QCheckBox { color: #a8a8ac; spacing: 8px; }
QCheckBox::indicator {
    width: 16px; height: 16px; border-radius: 4px; border: 1px solid #333333;
    background-color: #1a1a1a;
}
QCheckBox::indicator:checked { background-color: #E31B23; border-color: #E31B23; }

QScrollBar:vertical { background: #121212; width: 10px; }
QScrollBar::handle:vertical { background: #333333; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #E31B23; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
"""

# Backward-compatible alias — kept in case other modules import the old name.
DARK_STYLESHEET = CRIMSON_GRANITE_STYLESHEET


def pil_to_pixmap(image: Image.Image) -> QPixmap:
    """Convert a PIL RGB image into a QPixmap."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    data = image.tobytes("raw", "RGB")
    qimage = QImage(data, image.width, image.height, image.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


# ---------------------------------------------------------------------- #
# Worker threads
# ---------------------------------------------------------------------- #
