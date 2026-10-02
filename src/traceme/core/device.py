"""Compute-device selection, kept free of model imports so it's testable."""
from __future__ import annotations

from typing import TYPE_CHECKING

from traceme.core.logging import get_logger

if TYPE_CHECKING:
    import torch

log = get_logger("traceme.core.device")

DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")


def _mps_available() -> bool:
    import torch

    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def pick_device(preference: str) -> "torch.device":
    """Auto prefers CUDA, then Apple MPS, then CPU. An explicit choice that
    isn't available falls back to CPU with a warning rather than failing."""
    import torch  # deferred: keeps CLI parsing (--help) fast

    pref = preference.strip().lower()
    if pref not in DEVICE_CHOICES:
        raise ValueError(f"Unknown device '{preference}'. Choose from: {list(DEVICE_CHOICES)}")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps") if _mps_available() else torch.device("cpu")
    if pref == "cuda" and not torch.cuda.is_available():
        log.warning("Device 'cuda' requested but CUDA is not available; using CPU.")
        return torch.device("cpu")
    if pref == "mps" and not _mps_available():
        log.warning("Device 'mps' requested but Apple MPS is not available; using CPU.")
        return torch.device("cpu")
    return torch.device(pref)
