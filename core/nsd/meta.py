from __future__ import annotations

import ast

import numpy as np
import pandas as pd
from pydantic import BaseModel

from core.datasets.records import NSDImageRecord, TrialRecord
from core.nsd.constants import N_NSD_IMAGES, NSD_SESSIONS, SUBJECTS, TRIALS_PER_SESSION
from core.nsd.remote import NSDBucket, NSDKeys, nsd_bucket
from core.storage import backends
from core.storage.store import Store

_CROP_COLS = ["crop_top", "crop_bottom", "crop_left", "crop_right"]


class MetadataSummary(BaseModel):
    n_images: int
    n_shared: int
    trials_per_subject: dict[int, int]


def read_stim_info(bucket: NSDBucket | None = None) -> pd.DataFrame:
    return (bucket or nsd_bucket()).read_csv(NSDKeys.STIM_INFO, index_col=0)


def image_table(stim: pd.DataFrame) -> pd.DataFrame:
    """One row per NSD image. NSD's cropBox is (top, bottom, left, right) as fractions of the COCO image."""
    crops = pd.DataFrame(stim["cropBox"].map(ast.literal_eval).tolist(), columns=_CROP_COLS, index=stim.index)
    df = pd.DataFrame({
        "nsd_id": stim["nsdId"].astype(np.int64),
        "coco_id": stim["cocoId"].astype(np.int64),
        "coco_split": stim["cocoSplit"].astype(str),
        **{c: crops[c].astype(np.float64) for c in _CROP_COLS},
        "shared1000": stim["shared1000"].astype(bool),
    })
    df = NSDImageRecord.validate_frame(df)
    if len(df) != N_NSD_IMAGES or df["coco_id"].duplicated().any():
        raise ValueError("expected 73,000 NSD images with unique COCO ids")
    return df


def trial_table(stim: pd.DataFrame, subject: int) -> pd.DataFrame:
    """One row per trial the subject actually completed, sorted by trial number.

    The CSV lists the planned 40 sessions for everyone; subjects 3/4/6/8 stopped early, so trials
    past their last completed session are dropped.
    """
    parts = []
    for rep in range(3):
        col = stim[f"subject{subject}_rep{rep}"]
        shown = col > 0
        parts.append(pd.DataFrame({
            "trial": col[shown].astype(np.int64),
            "nsd_id": stim.loc[shown, "nsdId"].astype(np.int64),
            "coco_id": stim.loc[shown, "cocoId"].astype(np.int64),
        }))
    df = pd.concat(parts, ignore_index=True)
    df["session"] = (df["trial"] - 1) // TRIALS_PER_SESSION + 1
    df["session_index"] = (df["trial"] - 1) % TRIALS_PER_SESSION
    df = df[df["session"] <= NSD_SESSIONS[subject]].sort_values("trial")
    df = TrialRecord.validate_frame(df)

    expected = NSD_SESSIONS[subject] * TRIALS_PER_SESSION
    if not np.array_equal(df["trial"].to_numpy(), np.arange(1, expected + 1)):
        raise ValueError(f"subj{subject:02d}: trials are not a contiguous 1..{expected}")
    return df


def build_metadata(store: Store, *, force: bool = False, bucket: NSDBucket | None = None) -> MetadataSummary:
    keys = [backends.NSD_IMAGES, *(backends.trials(s) for s in SUBJECTS)]
    if force or not all(store.exists(k) for k in keys):
        stim = read_stim_info(bucket)
        store.save_frame(backends.NSD_IMAGES, image_table(stim))
        for s in SUBJECTS:
            store.save_frame(backends.trials(s), trial_table(stim, s))
    images = load_nsd_images(store)
    return MetadataSummary(
        n_images=len(images),
        n_shared=int(images["shared1000"].sum()),
        trials_per_subject={s: len(load_trials(store, s)) for s in SUBJECTS},
    )


def load_nsd_images(store: Store) -> pd.DataFrame:
    return NSDImageRecord.validate_frame(store.load_frame(backends.NSD_IMAGES))


def load_trials(store: Store, subject: int) -> pd.DataFrame:
    return TrialRecord.validate_frame(store.load_frame(backends.trials(subject)))
