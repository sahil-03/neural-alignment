from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ExtractorSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    outputs: dict[str, int]  # feature name -> dim
    hf_repo: str
    hf_revision: str  # pinned commit, so features are reproducible


EXTRACTORS: dict[str, ExtractorSpec] = {
    s.name: s
    for s in (
        ExtractorSpec(
            name="dinov2_b", outputs={"dinov2_b_cls": 768, "dinov2_b_mean": 768},
            hf_repo="facebook/dinov2-base", hf_revision="f9e44c814b77203eaa57a6bdbbd535f21ede1415",
        ),
        ExtractorSpec(  # gold only; never a reward-model input
            name="clip_l14", outputs={"clip_l14": 768},
            hf_repo="openai/clip-vit-large-patch14", hf_revision="32bd64288804d66eefd0ccbe215aa642df71cc41",
        ),
    )
}


def spec(name: str) -> ExtractorSpec:
    if name not in EXTRACTORS:
        raise ValueError(f"unknown extractor {name!r}; choose from {sorted(EXTRACTORS)}")
    return EXTRACTORS[name]


def all_features(extractors: list[str] | None = None) -> list[str]:
    return [f for e in (extractors or list(EXTRACTORS)) for f in spec(e).outputs]
