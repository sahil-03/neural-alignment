from pathlib import Path

import modal

from core.datasets.images import CocoDir
from core.features.gold import build_gold
from core.features.shards import Shard, merge_features, pending_shards, plan_shards
from core.remote.modal_app import (
    COCO_MOUNT,
    GB,
    app,
    coco_volume,
    cpu_defaults,
    remote_store,
    torch_defaults,
)


@app.cls(**torch_defaults, gpu="L4", cpu=8, memory=16 * GB, volumes={COCO_MOUNT: coco_volume},
         timeout=2 * 60 * 60, max_containers=4)
class Embedder:
    extractors: str = modal.parameter(default="")  # comma-separated; empty = all registered

    @modal.enter()
    def load(self) -> None:
        from core.features.runner import FeatureRunner  # torch: only in the torch image

        self.runner = FeatureRunner([e for e in self.extractors.split(",") if e] or None, device="cuda")

    @modal.method()
    def run(self, shard: dict) -> dict:
        return self.runner.run_shard(remote_store(), CocoDir(root=Path(COCO_MOUNT)), Shard(**shard)).model_dump()


@app.function(**cpu_defaults, memory=8 * GB, timeout=30 * 60)
def merge(extractors: list[str] | None = None) -> list[dict]:
    store = remote_store()
    return [m.model_dump() for m in merge_features(store, plan_shards(store, extractors))]


@app.function(**torch_defaults, memory=8 * GB, timeout=30 * 60)
def gold() -> dict:
    from core.features.laion import build_params  # torch: only in the torch image

    return build_gold(remote_store(), build_params()).model_dump()


@app.local_entrypoint()
def extract(extractors: str = "") -> None:
    """Embed every shard still missing an extractor's outputs, then merge shards per feature."""
    names = [e for e in extractors.split(",") if e] or None
    store = remote_store()
    todo = pending_shards(store, plan_shards(store, names))
    print(f"{len(todo)} shards to run")
    embedder = Embedder(extractors=extractors)  # type: ignore[call-arg]  # modal.parameter constructor
    for summary in embedder.run.map([s.model_dump() for s in todo]):
        print(summary)
    for m in merge.remote(names):
        print(m)


@app.local_entrypoint()
def score_gold() -> None:
    print(gold.remote())
