from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from core.storage.store import Store

MANIFEST = "_manifest.json"


class ArrayInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    shape: tuple[int, ...]
    dtype: str


class BundleManifest(BaseModel):
    """Written last; its presence marks a bundle as complete."""

    model_config = ConfigDict(frozen=True)

    kind: str
    arrays: dict[str, ArrayInfo]


class ArrayBundle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    def manifest(self) -> BundleManifest:
        arrays = {}
        for name in type(self).model_fields:
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{type(self).__name__}.{name} is not an ndarray")
            arrays[name] = ArrayInfo(shape=arr.shape, dtype=str(arr.dtype))
        return BundleManifest(kind=type(self).__name__, arrays=arrays)

    def save(self, store: Store, key: str) -> None:
        manifest = self.manifest()
        for name in manifest.arrays:
            store.save_array(f"{key}/{name}.npy", getattr(self, name))
        store.save_model(f"{key}/{MANIFEST}", manifest)

    @classmethod
    def exists(cls, store: Store, key: str) -> bool:
        return store.exists(f"{key}/{MANIFEST}")

    @classmethod
    def load_manifest(cls, store: Store, key: str) -> BundleManifest:
        """Shapes and dtypes without downloading the arrays."""
        manifest = store.load_model(f"{key}/{MANIFEST}", BundleManifest)
        if manifest.kind != cls.__name__:
            raise TypeError(f"{key} holds a {manifest.kind}, not a {cls.__name__}")
        return manifest

    @classmethod
    def load(cls, store: Store, key: str) -> Self:
        cls.load_manifest(store, key)
        return cls(**{name: store.load_array(f"{key}/{name}.npy") for name in cls.model_fields})


def positions(have: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Row positions of `want` ids within `have`. Raises KeyError if any are missing."""
    idx = pd.Index(have).get_indexer(want)
    missing = idx < 0
    if missing.any():
        raise KeyError(f"{int(missing.sum())} ids not found, e.g. {np.asarray(want)[missing][:5].tolist()}")
    return idx
