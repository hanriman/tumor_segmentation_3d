import contextlib
from collections.abc import Generator

import torch


def get_device(preference: str = "auto") -> torch.device:
    r"""
    Resolves execution device across NVIDIA CUDA, Apple Silicon MPS, and CPU.
    """
    pref = preference.lower()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif pref == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif pref == "cpu":
        return torch.device("cpu")

    # Auto detection
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_autocast_context(
    device: torch.device, enabled: bool = True
) -> Generator[None, None, None] | contextlib.AbstractContextManager:
    r"""
    Returns an appropriate mixed-precision autocast context for the active hardware.

    Theoretical Justification:
    --------------------------
    3D volumetric inputs ([B, 4, 128, 128, 128]) and dense 3D convolutions require substantial VRAM.
    Using Automatic Mixed Precision (AMP) reduces activation memory by ~50% and leverages hardware
    Tensor Cores (NVIDIA Ampere/Hopper FP16/BF16, Apple Silicon Neural Engine), accelerating 3D SSL
    pre-training while maintaining full float32 numerical precision for sensitive statistical tests.
    """
    if not enabled:
        return contextlib.nullcontext()

    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    elif device.type == "mps":
        # PyTorch MPS autocast support
        if hasattr(torch.amp, "autocast"):
            try:
                return torch.amp.autocast(device_type="mps", dtype=torch.float16)
            except (RuntimeError, ValueError, AttributeError):
                return contextlib.nullcontext()
        return contextlib.nullcontext()
    elif device.type == "cpu":
        return torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
    return contextlib.nullcontext()
