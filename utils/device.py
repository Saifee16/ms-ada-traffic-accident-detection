"""utils/device.py — Auto device detection with CPU/CUDA path selection."""
from __future__ import annotations

import torch


def resolve_device(preference: str = "auto") -> torch.device:
    """
    Returns torch.device based on preference.
    'auto' → CUDA if available, else CPU.
    """
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference.startswith("cuda"):
        if not torch.cuda.is_available():
            import warnings
            warnings.warn("CUDA requested but not available; falling back to CPU.")
            return torch.device("cpu")
        return torch.device(preference)
    return torch.device("cpu")


def device_info() -> dict:
    d: dict = {"device": str(resolve_device())}
    if torch.cuda.is_available():
        d["cuda_device_name"] = torch.cuda.get_device_name(0)
        d["cuda_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return d
