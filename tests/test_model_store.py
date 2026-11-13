"""INV-3 and INV-5: pinned digests, and a missing model disables its feature."""

from __future__ import annotations

import hashlib

import pytest

from src.ai_tools import model_store
from src.ai_tools.model_store import ModelSpec


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(model_store, "MODEL_DIR", tmp_path)
    return tmp_path


def _spec(payload: bytes, name: str = "tiny.onnx") -> ModelSpec:
    return ModelSpec(
        filename=name,
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        zoo_dir="nowhere",
        label="a test model",
    )


def test_cached_model_that_matches_is_returned(store):
    payload = b"model-bytes"
    spec = _spec(payload)
    spec.path.write_bytes(payload)
    assert model_store.ensure_model(spec, download=False) == spec.path
    assert model_store.is_cached(spec)


def test_missing_model_returns_none_without_downloading(store):
    assert model_store.ensure_model(_spec(b"x"), download=False) is None


def test_tampered_cache_is_rejected_and_removed(store):
    spec = _spec(b"the real model")
    spec.path.write_bytes(b"something else!")
    assert not model_store.verify(spec)
    assert model_store.ensure_model(spec, download=False) is None
    assert not spec.path.exists()


def test_substituted_download_is_rejected(store, monkeypatch):
    spec = _spec(b"the real model")
    monkeypatch.setattr(model_store, "_lfs_download_url", lambda s: None)
    monkeypatch.setattr(model_store, "_fetch", lambda *a, **k: b"an impostor model")
    assert model_store.ensure_model(spec, download=True) is None
    assert not spec.path.exists()


def test_good_download_is_saved(store, monkeypatch):
    payload = b"the real model"
    spec = _spec(payload)
    monkeypatch.setattr(model_store, "_lfs_download_url", lambda s: None)
    monkeypatch.setattr(model_store, "_fetch", lambda *a, **k: payload)
    assert model_store.ensure_model(spec, download=True) == spec.path
    assert spec.path.read_bytes() == payload


def test_several_models_are_all_or_nothing(store):
    present = _spec(b"here", "a.onnx")
    present.path.write_bytes(b"here")
    absent = _spec(b"gone", "b.onnx")
    assert model_store.ensure_models([present, absent], download=False) is None


def test_engines_degrade_when_their_models_are_absent(store):
    from src.ai_tools import orientation
    from src.ai_tools.face_analysis import FaceAnalyzer
    from src.ai_tools.pose_analysis import PoseAnalyzer
    from tests.conftest import make_architecture, rotate_keep_size

    assert not FaceAnalyzer(download=False).available
    assert not PoseAnalyzer(download=False).available
    assert orientation.default_pose_analyzer() is None
    # Levelling still works from scene lines alone.
    _, result = orientation.auto_rotate(rotate_keep_size(make_architecture(), 5.0))
    assert result.needs_rotation
