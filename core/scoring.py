from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from core.datasets.arrays import FeatureTable


@runtime_checkable
class Scorer(Protocol):
    name: str

    def score(self, ids: np.ndarray) -> np.ndarray:
        """(N,) COCO ids -> (N,) float scores, in the same order."""
        ...


class TableScorer:
    """Scores precomputed for a fixed set of images (e.g. gold)."""

    def __init__(self, name: str, table: FeatureTable):
        if table.X.ndim != 1:
            raise ValueError(f"{name}: expected one score per image, got shape {table.X.shape}")
        self.name = name
        self.table = table

    def __repr__(self) -> str:
        return f"TableScorer({self.name!r}, n={len(self.table)})"

    def score(self, ids: np.ndarray) -> np.ndarray:
        return self.table.take(ids).X
