"""Hardware detection and ONNX Runtime execution-provider selection.

Granite Studios is hardware-adaptive: it prefers CUDA on Nvidia machines,
falls back to OpenVINO on Intel machines (e.g. i5-8350U + integrated GPU),
and uses plain CPU execution everywhere else.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from functools import lru_cache

import onnxruntime as ort


def _nvidia_gpu_present() -> bool:
    """Detect an Nvidia GPU by probing nvidia-smi (works on Windows/Linux)."""
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return False
    try:
        result = subprocess.run(
            [smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


@lru_cache(maxsize=1)
def get_execution_provider() -> str:
    """Return the best available ONNX Runtime execution provider.

    Priority:
      1. CUDAExecutionProvider   — Nvidia GPU present and ORT built with CUDA.
      2. CoreMLExecutionProvider — Apple Silicon / macOS (MPS-class hardware).
      3. OpenVINOExecutionProvider — Intel CPU/iGPU (ideal for i5-8350U).
      4. CPUExecutionProvider    — universal fallback.
    """
    available = set(ort.get_available_providers())

    if "CUDAExecutionProvider" in available and _nvidia_gpu_present():
        return "CUDAExecutionProvider"

    if platform.system() == "Darwin" and "CoreMLExecutionProvider" in available:
        return "CoreMLExecutionProvider"

    if "OpenVINOExecutionProvider" in available:
        return "OpenVINOExecutionProvider"

    return "CPUExecutionProvider"


@lru_cache(maxsize=1)
def get_hardware_summary() -> dict:
    """Return a human-readable summary of the detected hardware tier."""
    provider = get_execution_provider()
    tier = {
        "CUDAExecutionProvider": "High (Nvidia CUDA)",
        "CoreMLExecutionProvider": "High (Apple CoreML/MPS)",
        "OpenVINOExecutionProvider": "Medium (Intel OpenVINO)",
        "CPUExecutionProvider": "Baseline (CPU)",
    }[provider]
    return {
        "os": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "provider": provider,
        "tier": tier,
        "onnxruntime_version": ort.__version__,
    }


if __name__ == "__main__":
    for key, value in get_hardware_summary().items():
        print(f"{key:>22}: {value}")
