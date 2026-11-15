from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from core.datasets.arrays import FeatureTable
from core.datasets.images import IMAGE_SOURCES
from core.features.shards import load_features
from core.scoring import TableScorer
from core.storage import backends
from core.storage.store import Store

NAME = "laion_aesthetic_v2"
CLIP_FEATURE = "clip_l14"


class GoldParams(BaseModel):
    weights_sha256: str
    w: list[float]  # (768,)
    b: float
    max_abs_err: float  # collapsed vs the original network, on random unit vectors


class GoldSummary(BaseModel):
    n: dict[str, int]
    mean: dict[str, float]
    std: dict[str, float]
    max_abs_err: float
    skipped: bool


def score(params: GoldParams, z: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(z, axis=1)
    if np.abs(norms - 1).max() > 1e-3:
        raise ValueError("gold expects unit-norm CLIP embeddings")
    return (z.astype(np.float64) @ np.asarray(params.w) + params.b).astype(np.float32)


def build_gold(store: Store, params: GoldParams, *, force: bool = False) -> GoldSummary:
    """Score every image source from its CLIP features; writes gold/<NAME>/<source> and the params."""
    sources = list(IMAGE_SOURCES)
    done = store.exists(backends.GOLD_PARAMS) and all(FeatureTable.exists(store, backends.gold(s)) for s in sources)
    if force or not done:
        for s in sources:
            z = load_features(store, CLIP_FEATURE, [s])
            FeatureTable(ids=z.ids, X=score(params, z.X)).save(store, backends.gold(s))
        store.save_model(backends.GOLD_PARAMS, params)

    params = store.load_model(backends.GOLD_PARAMS, GoldParams)
    tables = {s: FeatureTable.load(store, backends.gold(s)) for s in sources}
    return GoldSummary(
        n={s: len(t) for s, t in tables.items()},
        mean={s: float(t.X.mean()) for s, t in tables.items()},
        std={s: float(t.X.std()) for s, t in tables.items()},
        max_abs_err=params.max_abs_err,
        skipped=not force and done,
    )


def gold_scorer(store: Store) -> TableScorer:
    return TableScorer(NAME, FeatureTable.concat([FeatureTable.load(store, backends.gold(s)) for s in IMAGE_SOURCES]))
