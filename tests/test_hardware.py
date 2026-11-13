"""Execution-provider selection. The choice is a pure function, so every
platform's behaviour is tested here regardless of what this machine has."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.engine import hardware as hw

ALL = [hw.DIRECTML, hw.CUDA, hw.COREML, hw.OPENVINO, hw.CPU]


@pytest.mark.parametrize("system,available,nvidia,expected", [
    ("Windows", [hw.DIRECTML, hw.CPU], False, [hw.DIRECTML, hw.CPU]),
    ("Windows", [hw.CPU], False, [hw.CPU]),
    ("Windows", [hw.DIRECTML, hw.CUDA, hw.CPU], True, [hw.DIRECTML, hw.CUDA, hw.CPU]),
    ("Windows", [hw.CUDA, hw.CPU], False, [hw.CPU]),
    ("Darwin", [hw.COREML, hw.CPU], False, [hw.COREML, hw.CPU]),
    ("Darwin", ALL, True, [hw.COREML, hw.CPU]),
    ("Linux", [hw.CUDA, hw.OPENVINO, hw.CPU], True, [hw.CUDA, hw.OPENVINO, hw.CPU]),
    ("Linux", [hw.CUDA, hw.OPENVINO, hw.CPU], False, [hw.OPENVINO, hw.CPU]),
    ("Linux", [hw.DIRECTML, hw.CPU], False, [hw.CPU]),
])
def test_provider_choice(system, available, nvidia, expected):
    assert hw.select_providers(available, system=system, nvidia=nvidia) == expected


def test_the_cpu_is_always_the_last_resort():
    for system in ("Windows", "Darwin", "Linux"):
        for nvidia in (True, False):
            chosen = hw.select_providers(ALL, system=system, nvidia=nvidia)
            assert chosen[-1] == hw.CPU and chosen.count(hw.CPU) == 1


def test_the_directml_provider_is_named_as_onnx_runtime_names_it():
    assert hw.DIRECTML == "DmlExecutionProvider"


def test_an_override_wins_when_it_is_available():
    assert hw.select_providers(ALL, system="Windows", override="cpu") == [hw.CPU]
    assert hw.select_providers(ALL, system="Windows", override=hw.OPENVINO) == [hw.OPENVINO, hw.CPU]
    assert hw.select_providers([hw.CPU], system="Windows", override=hw.DIRECTML) == [hw.CPU]


def test_this_machine_reports_a_usable_provider_list():
    import onnxruntime as ort

    providers = hw.get_execution_providers()
    assert providers[-1] == hw.CPU
    assert set(providers) <= set(ort.get_available_providers())
    summary = hw.get_hardware_summary()
    assert summary["provider"] == providers[0] and summary["tier"]


@pytest.mark.model
def test_a_session_runs_and_falls_back_when_a_provider_cannot_start():
    """Uses the background-removal model, since no smaller ONNX file ships
    with the app. Skipped where that model has not been downloaded."""
    model = Path.home() / ".u2net" / "u2net.onnx"
    if not model.exists():
        pytest.skip("background-removal model not cached")
    x = np.random.default_rng(0).random((1, 3, 320, 320), dtype=np.float32)

    cpu = hw.create_session(model, providers=[hw.CPU])
    name = cpu.get_inputs()[0].name
    reference = cpu.run(None, {name: x})[0]

    # A provider that does not exist must not take the feature down.
    fallback = hw.create_session(model, providers=["NoSuchExecutionProvider", hw.CPU])
    assert fallback.get_providers()[0] == hw.CPU
    assert np.allclose(fallback.run(None, {name: x})[0], reference, atol=1e-5)

    # Whatever this machine prefers must agree with the CPU.
    preferred = hw.create_session(model)
    assert np.abs(preferred.run(None, {name: x})[0] - reference).max() < 1e-3
