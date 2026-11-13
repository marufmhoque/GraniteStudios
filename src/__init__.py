"""Granite Studios — local-first, hardware-adaptive AI photo & video suite."""

__version__ = "0.1.0"

# OpenCV 5 prints a warning every time a network is created ("Targets are
# not supported by the new graph engine"). It is harmless and, with one
# network per worker thread, very noisy. Errors are still reported.
try:
    import cv2 as _cv2

    _cv2.utils.logging.setLogLevel(_cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:  # logging control is a nicety, never a requirement
    pass
