from __future__ import annotations

from pathlib import Path
from typing import Any

from core.datasets.coco import download_coco as download
from core.datasets.images import CocoDir, build_candidates
from core.datasets.splits import build_splits
from core.nsd.betas import aggregate_subject, process_session, verify_orientation
from core.nsd.constants import NSD_SESSIONS
from core.nsd.meta import build_metadata
from core.remote.modal_app import (
    COCO_MOUNT,
    GB,
    app,
    coco_volume,
    cpu_defaults,
    data_root,
    remote_store,
)

STEPS = ("metadata", "verify", "betas", "aggregate", "splits")


@app.function(**cpu_defaults, timeout=15 * 60)
def metadata() -> dict:
    return build_metadata(remote_store()).model_dump()


@app.function(**cpu_defaults, memory=8 * GB, timeout=30 * 60)
def verify(subject: int) -> dict:
    return verify_orientation(subject).model_dump()


@app.function(**cpu_defaults, cpu=2, memory=6 * GB, timeout=30 * 60, max_containers=20)
def session(subject: int, session: int) -> bool:
    """One NSD session: download ~1 GB, mask, z-score, write (750, V) float16. False if already done."""
    return process_session(remote_store(), subject, session)


@app.function(**cpu_defaults, cpu=4, memory=24 * GB, timeout=60 * 60)
def aggregate(subject: int) -> dict:
    return aggregate_subject(remote_store(), subject).model_dump()


@app.function(**cpu_defaults, memory=8 * GB, timeout=15 * 60)
def splits(subject: int) -> dict:
    return build_splits(remote_store(), subject).model_dump()


# --- COCO --------------------------------------------------------------------------------------

@app.function(**cpu_defaults, volumes={COCO_MOUNT: coco_volume}, cpu=4, memory=4 * GB, timeout=3 * 60 * 60)
def download_coco() -> dict:
    """Fetch COCO 2017 train+val (~19 GB) into the volume once; complete splits are skipped."""
    return download(Path(COCO_MOUNT), on_split_done=coco_volume.commit).model_dump()


@app.function(**cpu_defaults, volumes={COCO_MOUNT: coco_volume}, memory=4 * GB, timeout=15 * 60)
def candidates() -> dict:
    return {"n_candidates": build_candidates(remote_store(), CocoDir(root=Path(COCO_MOUNT)))}


# --- Entrypoints -------------------------------------------------------------------------------

def _show(name: str, result: Any) -> None:
    print(f"[{name}] {result}")


@app.local_entrypoint()
def prepare(subject: int = 1, steps: str = ",".join(STEPS)) -> None:
    """Run the NSD steps for one subject, in order. Sessions run in parallel."""
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
