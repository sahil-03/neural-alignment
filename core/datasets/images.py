from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pydantic import BaseModel, ConfigDict, model_validator

from core.datasets.records import (
    CandidateImageRecord,
    CocoSplit,
    CropBox,
    NSDImageRecord,
)
from core.nsd.constants import STIM_SIZE
from core.storage import backends
from core.storage.store import Store

COCO_SPLITS: tuple[CocoSplit, ...] = ("train2017", "val2017")
COCO_COUNTS: dict[str, int] = {"train2017": 118_287, "val2017": 5_000}


class CocoDir(BaseModel):
    """Extracted COCO 2017: <root>/{train2017,val2017}/<12-digit id>.jpg"""

    model_config = ConfigDict(frozen=True)

    root: Path

    def path(self, coco_id: int, split: CocoSplit) -> Path:
        return self.root / split / f"{coco_id:012d}.jpg"

    def open(self, coco_id: int, split: CocoSplit) -> Image.Image:
        return Image.open(self.path(coco_id, split)).convert("RGB")

    def index(self) -> pd.DataFrame:
        rows = [(int(p.stem), s) for s in COCO_SPLITS for p in (self.root / s).glob("*.jpg")]
        return pd.DataFrame(rows, columns=["coco_id", "coco_split"])

    def is_complete(self) -> bool:
        return all(
            (self.root / s).is_dir() and sum(1 for _ in (self.root / s).glob("*.jpg")) == n
            for s, n in COCO_COUNTS.items()
        )


def center_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def apply_crop(img: Image.Image, crop: CropBox) -> Image.Image:
    """NSD's crop. Verified to reproduce NSD stimuli to ~0.5/255 mean abs error.

    Rounding can leave the crop 1px off square (e.g. 428x427); NSD resized that directly, so we
    deliberately do not trim it (trimming raised the error to ~3.7/255).
    """
    w, h = img.size
    box = (round(crop.left * w), round(crop.top * h), round(w * (1 - crop.right)), round(h * (1 - crop.bottom)))
    return img.crop(box)


class ImageSpec(BaseModel):
    """How to render one image: which COCO file, and which crop (None = center square)."""

    model_config = ConfigDict(frozen=True)

    coco_id: int
    coco_split: CocoSplit
    crop: CropBox | None = None


class ImageSet(BaseModel):
    """A named set of COCO images, each rendered as a STIM_SIZE x STIM_SIZE NSD-style square."""

    model_config = ConfigDict(frozen=True)

    name: str
    coco: CocoDir
    specs: dict[int, ImageSpec]

    @model_validator(mode="after")
    def _check(self) -> ImageSet:
        bad = [k for k, s in self.specs.items() if k != s.coco_id]
        if bad:
            raise ValueError(f"spec keys must equal coco ids, e.g. {bad[:5]}")
        return self

    @classmethod
    def nsd(cls, table: pd.DataFrame, coco: CocoDir) -> ImageSet:
        specs = {
            r.coco_id: ImageSpec(coco_id=r.coco_id, coco_split=r.coco_split, crop=r.crop)
            for r in NSDImageRecord.from_frame(table)
        }
        return cls(name="nsd", coco=coco, specs=specs)

    @classmethod
    def candidates(cls, table: pd.DataFrame, coco: CocoDir) -> ImageSet:
        specs = {
            r.coco_id: ImageSpec(coco_id=r.coco_id, coco_split=r.coco_split)
            for r in CandidateImageRecord.from_frame(table)
        }
        return cls(name="candidates", coco=coco, specs=specs)

    def __len__(self) -> int:
        return len(self.specs)

    @property
    def ids(self) -> np.ndarray:
        return np.array(sorted(self.specs), dtype=np.int64)

    def load(self, coco_id: int) -> Image.Image:
        spec = self.specs[int(coco_id)]
        img = self.coco.open(spec.coco_id, spec.coco_split)
        img = center_square(img) if spec.crop is None else apply_crop(img, spec.crop)
        return img.resize((STIM_SIZE, STIM_SIZE), Image.Resampling.BICUBIC)


def candidate_table(coco: CocoDir, nsd_images: pd.DataFrame) -> pd.DataFrame:
    """COCO 2017 train+val minus NSD's 73k.

    Candidates get a center crop; NSD's crops were object-aware. That is a known distribution
    difference between NSD and candidate images, worth checking when inspecting high-n winners.
    """
    if not coco.is_complete():
        raise FileNotFoundError(f"COCO 2017 is incomplete under {coco.root}")
    idx = coco.index()
    cand = idx[~idx["coco_id"].isin(nsd_images["coco_id"])]
    if len(cand) != len(idx) - len(nsd_images):
        raise ValueError("some NSD images are missing from the COCO directory")
    return CandidateImageRecord.validate_frame(cand.sort_values("coco_id"))


def build_candidates(store: Store, coco: CocoDir, *, force: bool = False) -> int:
    if force or not store.exists(backends.CANDIDATE_IMAGES):
        nsd = NSDImageRecord.validate_frame(store.load_frame(backends.NSD_IMAGES))
        store.save_frame(backends.CANDIDATE_IMAGES, candidate_table(coco, nsd))
    return len(store.load_frame(backends.CANDIDATE_IMAGES))


def load_image_set(store: Store, name: str, coco: CocoDir) -> ImageSet:
    if name == "nsd":
        return ImageSet.nsd(store.load_frame(backends.NSD_IMAGES), coco)
    if name == "candidates":
        return ImageSet.candidates(store.load_frame(backends.CANDIDATE_IMAGES), coco)
    raise ValueError(f"unknown image set {name!r}")
