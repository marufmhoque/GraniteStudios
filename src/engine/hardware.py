"""Hardware detection and ONNX Runtime execution-provider selection.

Every ONNX Runtime session in the app is created through this module, so
there is one place that decides what hardware a model runs on:

* **Windows** -- DirectML first. It drives any DirectX 12 graphics adapter,
  whether Intel, AMD or NVIDIA, without a vendor-specific install.
* **macOS** -- Core ML, which reaches the Apple Neural Engine.
* **Anywhere** -- CUDA when an NVIDIA card and a CUDA build are present,
  OpenVINO when that build is, and the CPU as the last resort.

The CPU is always last in the list, so a provider that fails to start on a
given machine falls through rather than failing the feature.

What was measured (13 November 2026, Intel UHD Graphics 620, the
background-removal model):

* DirectML output matched the CPU's to within 3e-7, and the final cut-out
  to within one level in 255.
* Inference took 461 ms against 625 ms on the CPU: 1.36 times faster. A
  whole background removal went from 742 ms to 470 ms.
* The *first* DirectML session on a machine took 76 seconds while the
  graphics driver compiled its shaders. Every start after that took 0.6 s.

That is a modest gain on integrated graphics. A discrete card should do
much better, but none was available to measure, so no figure is claimed.

Scope: this covers models run through ONNX Runtime, which today is
background removal. The face, pose, mesh, segmentation and inpainting
models run through OpenCV's own engine on the CPU and are not affected.

It also profiles system RAM so memory-heavy operations (batch export, reel
rendering) can size themselves to what is actually available.
"""

from __future__ import annotations

import logging
import os
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


DIRECTML = "DmlExecutionProvider"      # not "DirectMLExecutionProvider"
CUDA = "CUDAExecutionProvider"
COREML = "CoreMLExecutionProvider"
OPENVINO = "OpenVINOExecutionProvider"
CPU = "CPUExecutionProvider"

PROVIDER_TIERS = {
    DIRECTML: "High (DirectML GPU)",
    CUDA: "High (Nvidia CUDA)",
    COREML: "High (Apple CoreML/MPS)",
    OPENVINO: "Medium (Intel OpenVINO)",
    CPU: "Baseline (CPU)",
}

# Set to a provider name to force it, or to "cpu" to switch acceleration off.
OVERRIDE_ENV = "GRANITE_ORT_PROVIDER"


def provider_priority(system: str | None = None) -> list[str]:
    """The order providers are preferred in on an operating system."""
    system = system or platform.system()
    if system == "Windows":
        return [DIRECTML, CUDA, OPENVINO, CPU]
    if system == "Darwin":
        return [COREML, CPU]
    return [CUDA, OPENVINO, CPU]


def select_providers(available, system: str | None = None, nvidia: bool = False,
                     override: str | None = None) -> list[str]:
    """Choose providers from those a build offers. Pure, so it can be tested.

    Returns the usable providers in preference order, always ending with
    the CPU. CUDA is only chosen when an NVIDIA card is actually present:
    a CUDA build on a machine without one fails at session creation.
    """
    available = list(available)
    if override:
        wanted = CPU if override.strip().lower() == "cpu" else override.strip()
        if wanted in available:
            return [wanted] if wanted == CPU else [wanted, CPU]
        logger.warning("%s=%s is not available in this build; ignoring it.", OVERRIDE_ENV, override)
    chosen = [
        provider for provider in provider_priority(system)
        if provider in available and provider != CPU and (provider != CUDA or nvidia)
    ]
    return chosen + [CPU]


@lru_cache(maxsize=1)
def get_execution_providers() -> tuple[str, ...]:
    """Providers for this machine, best first, CPU last."""
    return tuple(select_providers(
        ort.get_available_providers(), nvidia=_nvidia_gpu_present(),
        override=os.environ.get(OVERRIDE_ENV),
    ))


def get_execution_provider() -> str:
    """The single best provider for this machine."""
    return get_execution_providers()[0]


def session_options() -> "ort.SessionOptions":
    """Options every session shares: full graph optimisation, and one
    thread per physical core for whatever runs on the CPU."""
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
    options.intra_op_num_threads = max(1, cores)
    return options


def create_session(model_path, providers=None) -> "ort.InferenceSession":
    """Create an inference session the standard way.

    Tries the preferred providers together first. If the accelerated one
    cannot start on this machine -- an old driver, a remote desktop session
    with no graphics adapter -- it falls back to the CPU alone rather than
    letting the feature fail.
    """
    providers = list(providers or get_execution_providers())
    try:
        return ort.InferenceSession(str(model_path), sess_options=session_options(), providers=providers)
    except Exception as exc:
        if providers == [CPU]:
            raise
        logger.warning("%s could not start (%s); using the CPU.", providers[0], exc)
        return ort.InferenceSession(str(model_path), sess_options=session_options(), providers=[CPU])


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
    tier = PROVIDER_TIERS.get(provider, provider)
    ram = get_ram_info()
    return {
        "os": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_cores": psutil.cpu_count(logical=True),
        "provider": provider,
        "providers": list(get_execution_providers()),
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
