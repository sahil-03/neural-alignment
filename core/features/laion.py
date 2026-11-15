from __future__ import annotations

import hashlib
import io
import urllib.request

import numpy as np
import torch
from torch import nn

from core.features.gold import GoldParams

WEIGHTS_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/"
    "6934dd81792f086e613a121dbce43082cb8be85e/sac%2Blogos%2Bava1-l14-linearMSE.pth"
)
WEIGHTS_SHA256 = "21dd590f3ccdc646f0d53120778b296013b096a035a2718c9cb0d511bff0f1e0"
LINEAR_LAYERS = (0, 2, 4, 6, 7)  # indices in the released nn.Sequential; 1, 3, 5 are Dropout
DIM = 768
MAX_COLLAPSE_ERR = 1e-6


def fetch_weights() -> bytes:
    with urllib.request.urlopen(WEIGHTS_URL) as r:
        raw = r.read()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != WEIGHTS_SHA256:
        raise ValueError(f"unexpected aesthetic weights sha256 {sha}")
    return raw


def load_state(raw: bytes) -> dict[str, np.ndarray]:
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    return {k: v.numpy().astype(np.float64) for k, v in state.items()}


def collapse(state: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    """The released MLP has no activations (its ReLUs are commented out), so it is one affine map."""
    W, b = np.eye(DIM), np.zeros(DIM)
    for i in LINEAR_LAYERS:
        Wi, bi = state[f"layers.{i}.weight"], state[f"layers.{i}.bias"]
        W, b = Wi @ W, Wi @ b + bi
    return W[0], float(b[0])


def reference_forward(state: dict[str, np.ndarray], z: np.ndarray) -> np.ndarray:
    """The released network, rebuilt exactly as in LAION's simple_inference.py, in eval mode."""
    mlp = nn.Sequential(
        nn.Linear(DIM, 1024), nn.Dropout(0.2), nn.Linear(1024, 128), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.Dropout(0.1), nn.Linear(64, 16), nn.Linear(16, 1),
    ).double().eval()
    mlp.load_state_dict({k.removeprefix("layers."): torch.from_numpy(v) for k, v in state.items()})
    with torch.no_grad():
        return mlp(torch.from_numpy(z.astype(np.float64))).numpy()[:, 0]


def build_params(raw: bytes | None = None) -> GoldParams:
    """Collapse the weights to (w, b) and check the collapse against the original network."""
    raw = raw if raw is not None else fetch_weights()
    state = load_state(raw)
    w, b = collapse(state)
    z = np.random.default_rng(0).standard_normal((4096, DIM))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    err = float(np.abs(z @ w + b - reference_forward(state, z)).max())
    if err > MAX_COLLAPSE_ERR:
        raise AssertionError(f"collapsed gold differs from the network by {err}")
    return GoldParams(weights_sha256=hashlib.sha256(raw).hexdigest(), w=w.tolist(), b=b, max_abs_err=err)
