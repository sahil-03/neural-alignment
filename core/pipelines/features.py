from __future__ import annotations

import argparse
import time

import numpy as np
from pydantic import BaseModel

from core.config import Settings
from core.datasets.images import IMAGE_SOURCES, CocoDir, load_image_set
from core.features.gold import build_gold
from core.features.shards import merge_features, pending_shards, plan_shards
from core.storage.store import Store


class FeatureStats(BaseModel):
    n: int
    dim: int
    mean_norm: float
    finite: bool


class ProbeSummary(BaseModel):
    source: str
    device: str
    seconds: float
    features: dict[str, FeatureStats]


class PendingSummary(BaseModel):
    n_shards: int
    by_source: dict[str, int]
    extractors: dict[str, int]


def probe(store: Store, coco: CocoDir, source: str, n: int, device: str | None) -> ProbeSummary:
    """Embed the first `n` images of `source` that exist under the local COCO root."""
    from core.features.runner import FeatureRunner  # torch: needs the gpu group
    from core.utils.torch import default_device  # torch: needs the gpu group

    images = load_image_set(store, source, coco)
    ids = [int(i) for i in images.ids if coco.path(i, images.specs[int(i)].coco_split).exists()][:n]
    if not ids:
        raise FileNotFoundError(f"no {source} images found under {coco.root}")
    device = device or default_device()
    runner = FeatureRunner(device=device, batch_size=32, num_workers=0)
    t = time.perf_counter()
    out = runner.embed(images, ids)
    return ProbeSummary(
        source=source, device=device, seconds=round(time.perf_counter() - t, 1),
        features={
            f: FeatureStats(n=len(X), dim=X.shape[1], mean_norm=float(np.linalg.norm(X, axis=1).mean()),
                            finite=bool(np.isfinite(X).all()))
            for f, X in out.items()
        },
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="core.pipelines.features")
    parser.add_argument("step", choices=["probe", "pending", "merge", "gold"])
    parser.add_argument("--source", default="nsd", choices=sorted(IMAGE_SOURCES))
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--device")
    parser.add_argument("--extractors", nargs="*")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    settings = Settings()
    store = Store.from_settings(settings)
    print(f"store: {store}")

    result: BaseModel
    match args.step:
        case "probe":
            result = probe(store, CocoDir(root=settings.coco_root), args.source, args.n, args.device)
        case "pending":
            todo = pending_shards(store, plan_shards(store, args.extractors))
            result = PendingSummary(
                n_shards=len(todo),
                by_source={s: sum(t.source == s for t in todo) for s in IMAGE_SOURCES},
                extractors={e: sum(e in t.extractors for t in todo) for e in {e for t in todo for e in t.extractors}},
            )
        case "merge":
            summaries = merge_features(store, plan_shards(store, args.extractors), force=args.force)
            print("\n".join(m.model_dump_json() for m in summaries))
            return
        case "gold":
            from core.features.laion import build_params  # torch: needs the gpu group

            result = build_gold(store, build_params(), force=args.force)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
