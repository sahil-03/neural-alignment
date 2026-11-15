from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Protocol, runtime_checkable

from core.kit.r2.client import R2Client

NSD_IMAGES = "meta/nsd_images.parquet"
CANDIDATE_IMAGES = "meta/candidate_images.parquet"
GOLD = "gold/laion_aesthetic_v2"



@runtime_checkable
class StorageBackend(Protocol):

    def put_bytes(self, key: str, data: bytes) -> None: ...
    def get_bytes(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def list_keys(self, prefix: str = "") -> list[str]: ...


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _normalize(key: str) -> str:
    key = key.strip("/")
    if not key or any(part in ("", ".", "..") for part in key.split("/")):
        raise ValueError(f"invalid storage key: {key!r}")
    return key


def _subj(subject: int) -> str:
    return f"subj{subject:02d}"


def trials(subject: int) -> str:
    return f"meta/trials/{_subj(subject)}.parquet"


def session_betas(subject: int, session: int) -> str:
    return f"betas/{_subj(subject)}/sessions/session{session:02d}.npy"


def trial_betas(subject: int) -> str:
    return f"betas/{_subj(subject)}/trials"


def image_betas(subject: int) -> str:
    return f"betas/{_subj(subject)}/images"


def features(feature: str, source: str) -> str:
    return f"features/{feature}/{source}"


def feature_shard(feature: str, source: str, shard_size: int, index: int) -> str:
    # shard_size is part of the key so changing it can never mix incompatible shards
    return f"features/{feature}/{source}/shards/size{shard_size}/{index:04d}"


def splits(subject: int) -> str:
    return f"splits/{_subj(subject)}.parquet"


class Disk:
    """A directory on disk. Writes are atomic (temp file + rename)."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def __repr__(self) -> str:
        return f"Disk({str(self.root)!r})"

    def _path(self, key: str) -> Path:
        return self.root / _normalize(key)

    def put_bytes(self, key: str, data: bytes) -> None:
        _atomic_write(self._path(key), data)

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list_keys(self, prefix: str = "") -> list[str]:
        base = self.root / prefix.strip("/") if prefix.strip("/") else self.root
        if not base.exists():
            return []
        return sorted(
            p.relative_to(self.root).as_posix()
            for p in base.rglob("*")
            if p.is_file() and not p.name.endswith(".tmp")
        )


class R2:
    """An R2 bucket, optionally under a key prefix."""

    def __init__(self, client: R2Client, prefix: str = ""):
        self.client = client
        self.prefix = prefix.strip("/")

    def __repr__(self) -> str:
        return f"R2(r2://{self.client.bucket}/{self.prefix})"

    def _key(self, key: str) -> str:
        key = _normalize(key)
        return f"{self.prefix}/{key}" if self.prefix else key

    def put_bytes(self, key: str, data: bytes) -> None:
        self.client.put_bytes(self._key(key), data)

    def get_bytes(self, key: str) -> bytes:
        return self.client.get_bytes(self._key(key))

    def exists(self, key: str) -> bool:
        return self.client.exists(self._key(key))

    def list_keys(self, prefix: str = "") -> list[str]:
        full = f"{self.prefix}/{prefix.strip('/')}" if self.prefix else prefix.strip("/")
        strip = len(self.prefix) + 1 if self.prefix else 0
        return sorted(obj.key[strip:] for obj in self.client.list(full))