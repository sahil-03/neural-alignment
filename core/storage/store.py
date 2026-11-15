from __future__ import annotations

import io
import json
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from pydantic import BaseModel

from core.config import Settings
from core.storage.backends import R2, Disk, StorageBackend
from core.storage.r2 import R2Client

M = TypeVar("M", bound=BaseModel)


class Store:

    def __init__(self, backend: StorageBackend):
        self.backend = backend

    def __repr__(self) -> str:
        return f"Store({self.backend!r})"

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> Store:
        settings = settings or Settings()
        root = settings.root
        if root.kind == "local":
            assert root.path is not None
            return cls(Disk(root.path))
        assert root.bucket is not None
        return cls(R2(R2Client(root.bucket), root.prefix))

    def exists(self, key: str) -> bool:
        return self.backend.exists(key)

    def list_keys(self, prefix: str = "") -> list[str]:
        return self.backend.list_keys(prefix)

    def put_bytes(self, key: str, data: bytes) -> None:
        self.backend.put_bytes(key, data)

    def get_bytes(self, key: str) -> bytes:
        return self.backend.get_bytes(key)

    def save_array(self, key: str, arr: np.ndarray) -> None:
        buf = io.BytesIO()
        np.save(buf, np.asarray(arr), allow_pickle=False)
        self.put_bytes(key, buf.getvalue())

    def load_array(self, key: str) -> np.ndarray:
        return np.load(io.BytesIO(self.get_bytes(key)), allow_pickle=False)

    def save_frame(self, key: str, df: pd.DataFrame) -> None:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        self.put_bytes(key, buf.getvalue())

    def load_frame(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(io.BytesIO(self.get_bytes(key)))

    def save_json(self, key: str, obj: Any) -> None:
        self.put_bytes(key, json.dumps(obj, indent=2).encode())

    def load_json(self, key: str) -> Any:
        return json.loads(self.get_bytes(key))

    def save_model(self, key: str, model: BaseModel) -> None:
        self.put_bytes(key, model.model_dump_json(indent=2).encode())

    def load_model(self, key: str, cls: type[M]) -> M:
        return cls.model_validate_json(self.get_bytes(key))
