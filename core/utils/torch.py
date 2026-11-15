from __future__ import annotations

import torch


def default_device() -> str:
    """Best available device: CUDA (Modal GPUs), then Apple MPS (local Mac), then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
