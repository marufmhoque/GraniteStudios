"""Hardware detection and ONNX Runtime execution-provider selection.

Granite Studios is hardware-adaptive: it prefers CUDA on Nvidia machines,
falls back to OpenVINO on Intel machines (e.g. i5-8350U + integrated GPU),
and uses plain CPU execution everywhere else. It also profiles system RAM
so memory-heavy operations (batch export, reel rendering) can size
themselves to what's actually available instead of assuming a fixed budget.
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from functools import lru_cache

import onnxruntime as ort
import psutil

logger = logging.getLogger(__name__)


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


def get_ram_info() -> dict:
    """Live snapshot of system RAM, in GB, plus a safe export budget.

    Not cached (unlike get_hardware_summary) since available RAM changes
    throughout a session as the user opens folders and runs exports.
    """
    vm = psutil.virtual_memory()
    total_gb = vm.total / (1024**3)
    available_gb = vm.available / (1024**3)
    return {
        "total_gb": round(total_gb, 1),
        "available_gb": round(available_gb, 1),
        "safe_export_budget_gb": round(get_safe_export_ram_gb(available_gb), 1),
    }


def get_safe_export_ram_gb(available_gb: float | None = None) -> float:
    """How much RAM the export pipeline may safely use right now.

    Conservative rule of thumb: at most half of currently available RAM,
    capped at 8GB, so a heavy batch export never starves the OS or whatever
    else the user has open — especially on modest hardware (e.g. an 8GB
    laptop). Call this fresh at export time rather than trusting a cached
    snapshot, since available RAM changes as the user works.
    """
    if available_gb is None:
        available_gb = psutil.virtual_memory().available / (1024**3)
    return max(0.5, min(available_gb * 0.5, 8.0))


@lru_cache(maxsize=1)
def get_hardware_summary() -> dict:
    """Return a human-readable summary of the detected hardware tier.

    Cached — this is a startup snapshot for display/logging, not a live
    resource monitor (use get_ram_info() / get_safe_export_ram_gb() for
    up-to-date numbers during a long-running session).
    """
    provider = get_execution_provider()
    tier = {
        "CUDAExecutionProvider": "High (Nvidia CUDA)",
        "CoreMLExecutionProvider": "High (Apple CoreML/MPS)",
        "OpenVINOExecutionProvider": "Medium (Intel OpenVINO)",
        "CPUExecutionProvider": "Baseline (CPU)",
    }[provider]
    ram = get_ram_info()
    return {
        "os": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_cores": psutil.cpu_count(logical=True),
        "provider": provider,
        "tier": tier,
        "onnxruntime_version": ort.__version__,
        "ram_total_gb": ram["total_gb"],
        "ram_available_gb": ram["available_gb"],
        "safe_export_ram_gb": ram["safe_export_budget_gb"],
    }


def log_hardware_diagnostics() -> dict:
    """Log the detected hardware profile once, at app startup.

    Meant to be called during the boot splash screen (see main.py) so a
    RAM-aware export budget is known before the user does anything.
    Returns the summary dict for reuse by the caller.
    """
    summary = get_hardware_summary()
    logger.info(
        "Hardware: %s/%s | %s | RAM %.1f/%.1f GB available (safe export "
        "budget: %.1f GB) | %d CPU cores | onnxruntime %s",
        summary["os"],
        summary["machine"],
        summary["tier"],
        summary["ram_available_gb"],
        summary["ram_total_gb"],
        summary["safe_export_ram_gb"],
        summary["cpu_cores"],
        summary["onnxruntime_version"],
    )
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for key, value in log_hardware_diagnostics().items():
        print(f"{key:>22}: {value}")
