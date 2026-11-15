from __future__ import annotations

from typing import Any, Literal, Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from core.nsd.constants import N_NSD_IMAGES, TRIALS_PER_SESSION

CocoSplit = Literal["train2017", "val2017"]

_ADAPTERS: dict[type, TypeAdapter[Any]] = {}


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    @classmethod
    def columns(cls) -> list[str]:
        return list(cls.model_fields)

    @classmethod
    def validate_frame(cls, df: pd.DataFrame) -> pd.DataFrame:
        cols = cls.columns()
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"{cls.__name__}: missing columns {sorted(missing)}")
        out = df[cols].reset_index(drop=True)
        adapter = _ADAPTERS.get(cls) or _ADAPTERS.setdefault(cls, TypeAdapter(cls))
        for row in out.to_dict("records"):
            adapter.validate_python(row)
        return out

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> list[Self]:
        return [cls.model_validate(r) for r in cls.validate_frame(df).to_dict("records")]


class CropBox(BaseModel):
    """NSD crop: fraction of the COCO image trimmed from each side."""

    model_config = ConfigDict(frozen=True)

    top: float = Field(ge=0.0, lt=1.0)
    bottom: float = Field(ge=0.0, lt=1.0)
    left: float = Field(ge=0.0, lt=1.0)
    right: float = Field(ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _check(self) -> CropBox:
        if self.top + self.bottom >= 1 or self.left + self.right >= 1:
            raise ValueError(f"crop removes the whole image: {self}")
        return self


class NSDImageRecord(Record):
    nsd_id: int = Field(ge=0, lt=N_NSD_IMAGES)
    coco_id: int
    coco_split: CocoSplit
    crop_top: float = Field(ge=0.0, lt=1.0)
    crop_bottom: float = Field(ge=0.0, lt=1.0)
    crop_left: float = Field(ge=0.0, lt=1.0)
    crop_right: float = Field(ge=0.0, lt=1.0)
    shared1000: bool

    @property
    def crop(self) -> CropBox:
        return CropBox(top=self.crop_top, bottom=self.crop_bottom, left=self.crop_left, right=self.crop_right)


class CandidateImageRecord(Record):
    coco_id: int
    coco_split: CocoSplit


class TrialRecord(Record):
    trial: int = Field(ge=1)
    nsd_id: int = Field(ge=0, lt=N_NSD_IMAGES)
    coco_id: int
    session: int = Field(ge=1)
    session_index: int = Field(ge=0, lt=TRIALS_PER_SESSION)


class SplitRecord(Record):
    coco_id: int
    n_reps: int = Field(ge=1)
    split: Literal["train", "test"]
    fold: int = Field(ge=-1)  # -1 for test images

    @model_validator(mode="after")
    def _check(self) -> SplitRecord:
        if (self.split == "test") != (self.fold == -1):
            raise ValueError("test rows must have fold -1 and train rows a fold >= 0")
        return self
