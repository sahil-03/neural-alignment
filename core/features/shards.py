from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from core.datasets.arrays import FeatureTable
from core.datasets.images import IMAGE_SOURCES, source_ids
from core.features.registry import EXTRACTORS, all_features, spec
from core.storage import backends
from core.storage.store import Store

SHARD_SIZE = 10_000


class Shard(BaseModel):
    source: str
    index: int
    ids: list[int]
    extractors: list[str]  # the extractors still to run on this shard


class MergeSummary(BaseModel):
    feature: str
    source: str
    n: int
    dim: int
    skipped: bool


def plan_shards(
    store: Store, extractors: list[str] | None = None, sources: list[str] | None = None, size: int = SHARD_SIZE
) -> list[Shard]:
    """Every shard of every source, each listing all requested extractors."""
    extractors = extractors or list(EXTRACTORS)
    shards = []
    for source in sources or list(IMAGE_SOURCES):
        ids = source_ids(store, source)
        for i, start in enumerate(range(0, len(ids), size)):
            shards.append(Shard(source=source, index=i, ids=ids[start:start + size].tolist(), extractors=extractors))
    return shards


def _shard_key(feature: str, shard: Shard) -> str:
    return backends.feature_shard(feature, shard.source, SHARD_SIZE, shard.index)


def pending_shards(store: Store, shards: list[Shard]) -> list[Shard]:
    """Shards with at least one extractor whose outputs are not all saved, narrowed to those extractors."""
    out = []
    for shard in shards:
        missing = [
            e for e in shard.extractors
            if not all(FeatureTable.exists(store, _shard_key(f, shard)) for f in spec(e).outputs)
        ]
        if missing:
            out.append(shard.model_copy(update={"extractors": missing}))
    return out


def save_shard(store: Store, shard: Shard, features: dict[str, np.ndarray]) -> None:
    for feature, X in features.items():
        FeatureTable(ids=np.asarray(shard.ids), X=X).save(store, _shard_key(feature, shard))


def merge_features(store: Store, shards: list[Shard], *, force: bool = False) -> list[MergeSummary]:
    """Concatenate each (feature, source)'s shards into features/<feature>/<source>, validating ids."""
    by_source: dict[str, list[Shard]] = {}
    for s in shards:
        by_source.setdefault(s.source, []).append(s)

    summaries = []
    for source, group in by_source.items():
        group = sorted(group, key=lambda s: s.index)
        expected = np.concatenate([np.asarray(s.ids, dtype=np.int64) for s in group])
        for feature in all_features(group[0].extractors):
            key = backends.features(feature, source)
            skipped = not force and FeatureTable.exists(store, key)
            if not skipped:
                table = FeatureTable.concat([FeatureTable.load(store, _shard_key(feature, s)) for s in group])
                if not np.array_equal(table.ids, expected):
                    raise ValueError(f"{feature}/{source}: merged ids do not match the planned shards")
                if not np.isfinite(table.X).all():
                    raise ValueError(f"{feature}/{source}: non-finite values")
                table.save(store, key)
            meta = FeatureTable.load_manifest(store, key).arrays["X"]
            summaries.append(MergeSummary(
                feature=feature, source=source, n=meta.shape[0], dim=meta.shape[1] if len(meta.shape) > 1 else 1,
                skipped=skipped,
            ))
    return summaries


def load_features(store: Store, feature: str, sources: list[str] | None = None) -> FeatureTable:
    """One table over the requested sources (default: all), sorted by COCO id."""
    tables = [FeatureTable.load(store, backends.features(feature, s)) for s in (sources or list(IMAGE_SOURCES))]
    table = FeatureTable.concat(tables)
    return table.take(np.sort(table.ids))
