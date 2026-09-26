"""Device selection: MPS, then CUDA, then CPU."""

import os

import torch


def select_device() -> torch.device:
    """Pick the best available device.

    Order is MPS (Apple Silicon) first, then CUDA, then CPU. The original
    project ordered CUDA first; this machine is a MacBook, so MPS wins here.
    """
    if os.environ.get("AUTOLAB_FORCE_CPU") == "1":  # AUTOLAB: keep tests off the shared GPU
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
