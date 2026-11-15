from __future__ import annotations

import argparse

from pydantic import BaseModel
from tqdm import tqdm

from core.config import Settings
from core.datasets.images import CocoDir, build_candidates
from core.datasets.splits import build_splits
from core.nsd.betas import aggregate_subject, process_session, verify_orientation
from core.nsd.constants import NSD_SESSIONS
from core.nsd.meta import build_metadata
from core.nsd.voxels import VoxelMask
from core.storage.store import Store


class SessionsSummary(BaseModel):
    subject: int
    processed: list[int]
    skipped: list[int]


def run_sessions(store: Store, subject: int, sessions: list[int] | None = None, force: bool = False) -> SessionsSummary:
    sessions = sessions or list(range(1, NSD_SESSIONS[subject] + 1))
    mask = VoxelMask.nsdgeneral(subject)
    done: list[int] = []
    skipped: list[int] = []
    for k in tqdm(sessions, desc=f"subj{subject:02d} sessions"):
        (done if process_session(store, subject, k, force=force, mask=mask) else skipped).append(k)
    return SessionsSummary(subject=subject, processed=done, skipped=skipped)


class CandidatesSummary(BaseModel):
    n_candidates: int


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="core.pipelines.data")
    parser.add_argument("step", choices=["metadata", "verify", "betas", "aggregate", "candidates", "splits"])
    parser.add_argument("--subject", type=int, default=1, choices=sorted(NSD_SESSIONS))
    parser.add_argument("--sessions", type=int, nargs="*")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    settings = Settings()
    store = Store.from_settings(settings)
    print(f"store: {store}")

    result: BaseModel
    match args.step:
        case "metadata":
            result = build_metadata(store, force=args.force)
        case "verify":
            result = verify_orientation(args.subject)
        case "betas":
            result = run_sessions(store, args.subject, args.sessions, force=args.force)
        case "aggregate":
            result = aggregate_subject(store, args.subject, force=args.force)
        case "candidates":
            n = build_candidates(store, CocoDir(root=settings.coco_root), force=args.force)
            result = CandidatesSummary(n_candidates=n)
        case "splits":
            result = build_splits(store, args.subject, force=args.force)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
