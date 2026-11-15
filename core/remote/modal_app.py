from __future__ import annotations

from typing import Any

import modal

from core.config import Settings
from core.features.registry import EXTRACTORS
from core.storage.store import Store

APP_NAME = "neural-alignment"
COCO_MOUNT = "/coco"
GB = 1024  # Modal memory is in MiB

app = modal.App(APP_NAME)


cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extra_options="--no-dev")  # project dependencies pinned by uv.lock
    .env({"NA_DATA_ROOT": Settings().data_root, "NA_COCO_ROOT": COCO_MOUNT})
    .add_local_python_source("core")
)


def _bake_weights(models: list[tuple[str, str]]) -> None:
    """Runs at image build. Needs no project code, so editing core/ never re-downloads weights."""
    from huggingface_hub import snapshot_download

    for repo, revision in models:
        snapshot_download(repo, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.txt"])


# Torch + transformers, with every registered extractor's pinned weights cached in the image.
# Used for GPU extraction and for anything that needs torch on CPU (e.g. reading the gold weights).
torch_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(groups=["gpu"], extra_options="--no-dev")
    .run_function(_bake_weights, kwargs={"models": [(e.hf_repo, e.hf_revision) for e in EXTRACTORS.values()]})
    .env({"NA_DATA_ROOT": Settings().data_root, "NA_COCO_ROOT": COCO_MOUNT, "HF_HUB_OFFLINE": "1"})
    .add_local_python_source("core")
)


r2_secret = modal.Secret.from_name("r2")
coco_volume = modal.Volume.from_name("neural-alignment-coco", create_if_missing=True)


cpu_defaults: dict[str, Any] = {
    "image": cpu_image,
    "secrets": [r2_secret],
    "retries": modal.Retries(max_retries=2),
}

torch_defaults: dict[str, Any] = {**cpu_defaults, "image": torch_image}


def data_root() -> str:
    root = Settings().data_root
    if not root.startswith("r2://"):
        # Container disks are ephemeral; anything written to a local root would be lost.
        raise RuntimeError(f"remote jobs need NA_DATA_ROOT=r2://..., got {root!r}")
    return root


def remote_store() -> Store:
    data_root()
    return Store.from_settings(Settings())
