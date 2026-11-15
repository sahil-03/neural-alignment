from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from core.datasets.images import COCO_SPLITS, CocoDir

COCO_ZIP_URL = "http://images.cocodataset.org/zips/{split}.zip"
_CHUNK = 16 * 1024**2


class CocoSummary(BaseModel):
    counts: dict[str, int]
    downloaded: list[str]


def _count(root: Path, split: str) -> int:
    return sum(1 for _ in (root / split).glob("*.jpg"))


def download_coco(root: Path, *, on_split_done: Callable[[], None] = lambda: None) -> CocoSummary:
    """Download and extract any incomplete split (~18 GB train, ~1 GB val); complete splits are skipped.

    `on_split_done` runs after each split is extracted (e.g. to commit a Modal Volume).
    """
    coco = CocoDir(root=root)
    downloaded: list[str] = []
    for split in COCO_SPLITS:
        if coco.split_complete(split):
            continue
        with tempfile.TemporaryDirectory() as d:
            zpath = Path(d) / f"{split}.zip"
            print(f"downloading {split}")
            with urllib.request.urlopen(COCO_ZIP_URL.format(split=split)) as r, zpath.open("wb") as f:
                shutil.copyfileobj(r, f, length=_CHUNK)
            print(f"extracting {split}")
            shutil.rmtree(root / split, ignore_errors=True)  # drop any partial extract
            with zipfile.ZipFile(zpath) as z:
                z.extractall(root)  # the zips contain a top-level <split>/ directory
        on_split_done()
        downloaded.append(split)
    if not coco.is_complete():
        raise RuntimeError(f"COCO still incomplete under {root}")
    return CocoSummary(counts={s: _count(root, s) for s in COCO_SPLITS}, downloaded=downloaded)
