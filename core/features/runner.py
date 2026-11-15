from __future__ import annotations

import time

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel
from torch.utils.data import DataLoader, Dataset

from core.datasets.images import CocoDir, ImageSet, load_image_set
from core.features.extractors import INPUT_SIZE, build
from core.features.registry import EXTRACTORS
from core.features.shards import Shard, save_shard
from core.storage.store import Store


class RenderedImages(Dataset[torch.Tensor]):
    """COCO ids -> (3, 224, 224) float in [0, 1], from the 425px NSD-style render."""

    def __init__(self, images: ImageSet, ids: list[int]):
        self.images = images
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> torch.Tensor:
        # PIL bicubic, as CLIP's own preprocessing does
        img = self.images.load(self.ids[i]).resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BICUBIC)
        return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div_(255.0)
        

class FeatureRunner:
    """Loads extractors once (e.g. per container) and reuses them across shards."""

    def __init__(
        self,
        extractors: list[str] | None = None,
        device: str = "cuda",
        batch_size: int = 128,
        num_workers: int = 7,
    ):
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.extractors = {name: build(name, device) for name in (extractors or list(EXTRACTORS))}

    def embed(self, images: ImageSet, ids: list[int], extractors: list[str] | None = None) -> dict[str, np.ndarray]:
        """Features for `ids`, in that order: feature name -> (N, D) float32."""
        names = extractors or list(self.extractors)
        missing = set(names) - set(self.extractors)
        if missing:
            raise ValueError(f"extractors not loaded: {sorted(missing)}")

        loader = DataLoader(
            RenderedImages(images, ids), 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            pin_memory=self.device == "cuda", 
            persistent_workers=False,
        )
        chunks: dict[str, list[np.ndarray]] = {}
        for pixels in loader:
            pixels = pixels.to(self.device, non_blocking=True)
            for name in names:
                for feature, t in self.extractors[name](pixels).items():
                    chunks.setdefault(feature, []).append(t.float().cpu().numpy())
        out = {f: np.concatenate(c) for f, c in chunks.items()}
        for f, X in out.items():
            if len(X) != len(ids):
                raise RuntimeError(f"{f}: got {len(X)} rows for {len(ids)} ids")
        return out

    def run_shard(self, store: Store, coco: CocoDir, shard: Shard) -> ShardSummary:
        start = time.perf_counter()
        images = load_image_set(store, shard.source, coco)
        save_shard(store, shard, self.embed(images, shard.ids, shard.extractors))
        seconds = time.perf_counter() - start
        return ShardSummary(
            source=shard.source, index=shard.index, n=len(shard.ids), extractors=shard.extractors,
            seconds=round(seconds, 1), images_per_second=round(len(shard.ids) / seconds, 1),
        )


class ShardSummary(BaseModel):
    source: str
    index: int
    n: int
    extractors: list[str]
    seconds: float
    images_per_second: float
