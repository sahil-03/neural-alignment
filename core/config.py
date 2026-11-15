from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parent / ".env"  # core/.env; gitignored, never uploaded to Modal


class DataRoot(BaseModel):
    """Parsed NA_DATA_ROOT."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["local", "r2"]
    path: Path | None = None    # kind == "local"
    bucket: str | None = None   # kind == "r2"
    prefix: str = ""            # kind == "r2"

    @model_validator(mode="after")
    def _check(self) -> DataRoot:
        if self.kind == "local" and self.path is None:
            raise ValueError("local data root needs a path")
        if self.kind == "r2" and not self.bucket:
            raise ValueError("r2 data root needs a bucket")
        return self

    @classmethod
    def parse(cls, value: str) -> DataRoot:
        if value.startswith("r2://"):
            bucket, _, prefix = value.removeprefix("r2://").partition("/")
            return cls(kind="r2", bucket=bucket, prefix=prefix.strip("/"))
        return cls(kind="local", path=Path(value).expanduser().resolve())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NA_", env_file=ENV_FILE, extra="ignore")

    data_root: str = "./data/store"
    coco_root: Path = Path("./data/coco")

    @field_validator("coco_root", mode="after")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return v.expanduser()

    @property
    def root(self) -> DataRoot:
        return DataRoot.parse(self.data_root)

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()