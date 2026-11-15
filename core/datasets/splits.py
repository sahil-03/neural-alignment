"""Per-subject splits, computed once and shared by every model so comparisons use identical images."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel

from core.datasets.arrays import ImageBetas
from core.datasets.records import NSDImageRecord, SplitRecord
from core.storage import backends
from core.storage.store import Store


class SplitConfig(BaseModel):
    n_folds: int = 5
    seed: int = 0


class SplitSummary(BaseModel):
    subject: int
    n_train: int
    n_test: int
    fold_sizes: list[int]


def make_splits(nsd_images: pd.DataFrame, betas: ImageBetas, config: SplitConfig | None = None) -> pd.DataFrame:
    """test = shared1000 images this subject saw (>= 1 rep); train = the rest, with fixed CV folds.

    The folds serve both hyperparameter selection and cross-fitting the encoder.
    """
    config = config or SplitConfig()
    shared = nsd_images.loc[nsd_images["shared1000"], "coco_id"]
    df = pd.DataFrame({"coco_id": betas.ids, "n_reps": betas.n_reps.astype(np.int64)})
    df["split"] = np.where(df["coco_id"].isin(shared), "test", "train")
    df["fold"] = -1
    train = np.flatnonzero(df["split"].to_numpy() == "train")
    perm = np.random.default_rng(config.seed).permutation(train)
    df.loc[perm, "fold"] = np.arange(len(perm)) % config.n_folds
    return SplitRecord.validate_frame(df)


def build_splits(
    store: Store, subject: int, config: SplitConfig | None = None, *, force: bool = False
) -> SplitSummary:
    key = backends.splits(subject)
    if force or not store.exists(key):
        nsd = NSDImageRecord.validate_frame(store.load_frame(backends.NSD_IMAGES))
        betas = ImageBetas.load(store, backends.image_betas(subject))
        store.save_frame(key, make_splits(nsd, betas, config))
    df = load_splits(store, subject)
    train = df[df["split"] == "train"]
    return SplitSummary(
        subject=subject,
        n_train=len(train),
        n_test=int((df["split"] == "test").sum()),
        fold_sizes=train["fold"].value_counts().sort_index().tolist(),
    )


def load_splits(store: Store, subject: int) -> pd.DataFrame:
    return SplitRecord.validate_frame(store.load_frame(backends.splits(subject)))
