"""
Remote data preparation: Modal wrappers around the idempotent steps also run by core.pipelines.data.

    uv run modal run core/remote/data.py::prepare --subject 1
    uv run modal run core/remote/data.py::prepare --subject 1 --steps betas,aggregate
    uv run modal run core/remote/data.py::coco          # COCO -> volume -> candidate table
"""

from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from core.remote.modal_app import (
    COCO_MOUNT,
    GB,
    app,
    coco_volume,
    cpu_defaults,
    data_root,
    remote_store,
)

COCO_ZIP_URL = "http://images.cocodataset.org/zips/{split}.zip"
STEPS = ("metadata", "verify", "betas", "aggregate", "splits")


@app.function(**cpu_defaults, timeout=15 * 60)
def metadata() -> dict:
    from core.nsd.meta import build_metadata

    return build_metadata(remote_store()).model_dump()


@app.function(**cpu_defaults, memory=8 * GB, timeout=30 * 60)
def verify(subject: int) -> dict:
    from core.nsd.betas import verify_orientation

    return verify_orientation(subject).model_dump()


@app.function(**cpu_defaults, cpu=2, memory=6 * GB, timeout=30 * 60, max_containers=20)
def session(subject: int, session: int) -> bool:
    """One NSD session: download ~1 GB, mask, z-score, write (750, V) float16. False if already done."""
    from core.nsd.betas import process_session

    return process_session(remote_store(), subject, session)


@app.function(**cpu_defaults, cpu=4, memory=24 * GB, timeout=60 * 60)
def aggregate(subject: int) -> dict:
    from core.nsd.betas import aggregate_subject

    return aggregate_subject(remote_store(), subject).model_dump()


@app.function(**cpu_defaults, memory=8 * GB, timeout=15 * 60)
def splits(subject: int) -> dict:
    from core.datasets.splits import build_splits

    return build_splits(remote_store(), subject).model_dump()


# --- COCO --------------------------------------------------------------------------------------

@app.function(**cpu_defaults, volumes={COCO_MOUNT: coco_volume}, cpu=4, memory=4 * GB, timeout=3 * 60 * 60)
def download_coco() -> dict:
    """Fetch COCO 2017 train+val (~19 GB) into the volume once; skipped if already complete."""
    from core.datasets.images import COCO_SPLITS, CocoDir

    coco = CocoDir(root=Path(COCO_MOUNT))
    if not coco.is_complete():
        for split in COCO_SPLITS:
            with tempfile.TemporaryDirectory() as d:
                zpath = Path(d) / f"{split}.zip"
                print(f"downloading {split}")
                with urllib.request.urlopen(COCO_ZIP_URL.format(split=split)) as r, zpath.open("wb") as f:
                    shutil.copyfileobj(r, f, length=16 * 1024**2)
                print(f"extracting {split}")
                shutil.rmtree(Path(COCO_MOUNT) / split, ignore_errors=True)  # drop any partial extract
                with zipfile.ZipFile(zpath) as z:
                    z.extractall(COCO_MOUNT)  # zips contain a top-level <split>/ directory
            coco_volume.commit()
        if not coco.is_complete():
            raise RuntimeError("COCO still incomplete after download")
    return {s: sum(1 for _ in (Path(COCO_MOUNT) / s).glob("*.jpg")) for s in COCO_SPLITS}


@app.function(**cpu_defaults, volumes={COCO_MOUNT: coco_volume}, memory=4 * GB, timeout=15 * 60)
def candidates() -> dict:
    from core.datasets.images import CocoDir, build_candidates

    return {"n_candidates": build_candidates(remote_store(), CocoDir(root=Path(COCO_MOUNT)))}


# --- Entrypoints -------------------------------------------------------------------------------

def _show(name: str, result: Any) -> None:
    print(f"[{name}] {result}")


@app.local_entrypoint()
def prepare(subject: int = 1, steps: str = ",".join(STEPS)) -> None:
    """Run the NSD steps for one subject, in order. Sessions run in parallel."""
    from core.nsd.constants import NSD_SESSIONS

    print(f"data root: {data_root()}")
    chosen = [s.strip() for s in steps.split(",") if s.strip()]
    unknown = set(chosen) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown steps {sorted(unknown)}; choose from {STEPS}")

    for step in STEPS:  # fixed order regardless of how --steps was written
        if step not in chosen:
            continue
        if step == "metadata":
            _show(step, metadata.remote())
        elif step == "verify":
            _show(step, verify.remote(subject))
        elif step == "betas":
            ks = list(range(1, NSD_SESSIONS[subject] + 1))
            wrote = list(session.starmap([(subject, k) for k in ks]))
            _show(step, {"processed": [k for k, w in zip(ks, wrote, strict=True) if w],
                         "skipped": [k for k, w in zip(ks, wrote, strict=True) if not w]})
        elif step == "aggregate":
            _show(step, aggregate.remote(subject))
        elif step == "splits":
            _show(step, splits.remote(subject))


@app.local_entrypoint()
def coco() -> None:
    """Download COCO into the volume (once), then write the candidate table."""
    print(f"data root: {data_root()}")
    _show("download_coco", download_coco.remote())
    _show("candidates", candidates.remote())
