from __future__ import annotations

from typing import Any

import modal

from core.config import Settings
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


r2_secret = modal.Secret.from_name("r2")
coco_volume = modal.Volume.from_name("neural-alignment-coco", create_if_missing=True)


cpu_defaults: dict[str, Any] = {
    "image": cpu_image,
    "secrets": [r2_secret],
    "retries": modal.Retries(max_retries=2),
}


def data_root() -> str:
    root = Settings().data_root
    if not root.startswith("r2://"):
        # Container disks are ephemeral; anything written to a local root would be lost.
        raise RuntimeError(f"remote jobs need NA_DATA_ROOT=r2://..., got {root!r}")
    return root


def remote_store() -> Store:
    data_root()
    return Store.from_settings(Settings())
