from __future__ import annotations

import numpy as np
from pydantic import field_validator, model_validator

from core.datasets.bundle import ArrayBundle, positions


def _as_ids(v: object) -> np.ndarray:
    arr = np.asarray(v)
    if arr.ndim != 1:
        raise ValueError(f"ids must be 1-D, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"ids must be integers, got {arr.dtype}")
    return arr.astype(np.int64, copy=False)


def _require_unique(ids: np.ndarray, what: str) -> None:
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{what}: ids must be unique")


class FeatureTable(ArrayBundle):
    """Per-image vectors (features) or scalars (scores), keyed by COCO id."""

    ids: np.ndarray  # (N,) int64, unique
    X: np.ndarray    # (N, D) or (N,)

    @field_validator("ids", mode="before")
    @classmethod
    def _coerce_ids(cls, v: object) -> np.ndarray:
        return _as_ids(v)

    @model_validator(mode="after")
    def _check(self) -> FeatureTable:
        if len(self.ids) != len(self.X):
            raise ValueError(f"ids {self.ids.shape} and X {self.X.shape} are misaligned")
        _require_unique(self.ids, "FeatureTable")
        return self

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def dim(self) -> int:
        return 1 if self.X.ndim == 1 else self.X.shape[1]

    def take(self, ids: np.ndarray) -> FeatureTable:
        ids = _as_ids(ids)
        return FeatureTable(ids=ids, X=self.X[positions(self.ids, ids)])

    @staticmethod
    def concat(tables: list[FeatureTable]) -> FeatureTable:
        return FeatureTable(
            ids=np.concatenate([t.ids for t in tables]),
            X=np.concatenate([t.X for t in tables]),
        )


class TrialBetas(ArrayBundle):
    """Single-trial responses, z-scored per voxel within each session.

    Kept alongside ImageBetas because noise ceilings and noise-injection experiments need repeats.
    """

    trial: np.ndarray  # (T,) 1-based NSD trial number
    ids: np.ndarray    # (T,) COCO id shown on that trial (repeats allowed)
    Y: np.ndarray      # (T, V) float16

    @field_validator("ids", "trial", mode="before")
    @classmethod
    def _coerce_ids(cls, v: object) -> np.ndarray:
        return _as_ids(v)

    @model_validator(mode="after")
    def _check(self) -> TrialBetas:
        if not (len(self.trial) == len(self.ids) == len(self.Y)) or self.Y.ndim != 2:
            raise ValueError("trial, ids and Y are misaligned")
        _require_unique(self.trial, "TrialBetas.trial")
        return self

    @property
    def n_voxels(self) -> int:
        return self.Y.shape[1]


class ImageBetas(ArrayBundle):
    """Responses averaged over repeats: the (image -> brain response) table the models train on."""

    ids: np.ndarray        # (N,) unique COCO ids
    Y: np.ndarray          # (N, V) float32
    n_reps: np.ndarray     # (N,) number of trials averaged
    ncsnr: np.ndarray      # (V,) NSD noise-ceiling SNR per voxel
    voxel_ijk: np.ndarray  # (V, 3) voxel coordinates in the NIfTI func1pt8mm grid

    @field_validator("ids", mode="before")
    @classmethod
    def _coerce_ids(cls, v: object) -> np.ndarray:
        return _as_ids(v)

    @model_validator(mode="after")
    def _check(self) -> ImageBetas:
        if self.Y.ndim != 2:
            raise ValueError(f"Y must be 2-D, got {self.Y.shape}")
        if not (len(self.ids) == len(self.Y) == len(self.n_reps)):
            raise ValueError("ids, Y and n_reps are misaligned")
        if not (self.Y.shape[1] == len(self.ncsnr) == len(self.voxel_ijk)):
            raise ValueError("Y columns, ncsnr and voxel_ijk are misaligned")
        _require_unique(self.ids, "ImageBetas")
        return self

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def n_voxels(self) -> int:
        return self.Y.shape[1]

    def take(self, ids: np.ndarray) -> ImageBetas:
        idx = positions(self.ids, _as_ids(ids))
        return ImageBetas(
            ids=self.ids[idx], Y=self.Y[idx], n_reps=self.n_reps[idx],
            ncsnr=self.ncsnr, voxel_ijk=self.voxel_ijk,
        )
