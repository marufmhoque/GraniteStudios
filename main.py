"""Granite Studios — entry point.

Run with:  python main.py
"""

import sys
from pathlib import Path

# Ensure the project root is importable regardless of invocation directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ui.main_window import run

if __name__ == "__main__":
    sys.exit(run())
