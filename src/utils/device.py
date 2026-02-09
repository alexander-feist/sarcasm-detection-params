import platform
from typing import Any

import psutil
import torch
from cpuinfo import cpuinfo


def _get_accelerator_info() -> dict[str, Any] | None:
    """Return a dictionary containing hardware properties of the accelerator."""
    info = {}

    # For CUDA (NVIDIA)
    if torch.cuda.is_available():
        num = torch.cuda.device_count()
        info["type"] = "cuda"
        info["count"] = num
        info["devices"] = []
        for i in range(num):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append(
                {
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
        return info

    # For ROCm (AMD)
    if torch.version.hip is not None:
        num = torch.cuda.device_count()  # Works for ROCm too
        info["type"] = "rocm"
        info["count"] = num
        info["devices"] = []
        for i in range(num):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append(
                {
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                }
            )
        return info

    # For Apple Silicon
    if torch.backends.mps.is_available():
        info["type"] = "mps"
        info["count"] = 1
        info["devices"] = [{"name": "Apple Silicon GPU"}]
        return info

    return None


def get_device_info() -> dict[str, Any]:
    """Return a dictionary containing system and hardware properties."""
    return {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "processor": cpuinfo.get_cpu_info().get("brand_raw", "Unknown Processor"),
        "accelerator": _get_accelerator_info(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "cpu_frequency_mhz": psutil.cpu_freq().max,
        "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "torch_num_threads": torch.get_num_threads(),
    }
