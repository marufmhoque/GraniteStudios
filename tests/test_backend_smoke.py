"""The hosted backend imports the desktop engines; keep that surface intact.

The desktop environment does not install the backend's own dependencies, so
the first test reads the backend's imports and resolves each one against
``src`` directly. It runs everywhere. The second builds the real app and
runs only where the backend's requirements are installed.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1] / "web" / "backend"


def _engine_imports() -> list[tuple[str, str]]:
    tree = ast.parse((BACKEND / "app.py").read_text(encoding="utf-8"))
    wanted = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src"):
            wanted += [(node.module, alias.name) for alias in node.names]
    return wanted


def test_backend_actually_imports_from_the_engines():
    assert len(_engine_imports()) >= 10, "the import scan found nothing to check"


@pytest.mark.parametrize("module,name", _engine_imports())
def test_every_engine_name_the_backend_imports_still_exists(module, name):
    loaded = importlib.import_module(module)
    if not hasattr(loaded, name):
        importlib.import_module(f"{module}.{name}")


def test_backend_builds_its_app(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("multipart", reason="backend requirements are not installed here")
    monkeypatch.setenv("GRANITE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.syspath_prepend(str(BACKEND))
    sys.modules.pop("app", None)
    import app as backend

    routes = {route.path for route in backend.app.routes}
    assert any(path.startswith("/api") for path in routes)
