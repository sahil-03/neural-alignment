from __future__ import annotations

from typing import ClassVar, Protocol

import torch
from transformers import CLIPVisionModelWithProjection, Dinov2Model

from core.features.registry import ExtractorSpec, spec

INPUT_SIZE = 224


class Extractor(Protocol):
    spec: ClassVar[ExtractorSpec]

    def __call__(self, pixels: torch.Tensor) -> dict[str, torch.Tensor]: ...


def _normalize(x: torch.Tensor, mean: tuple[float, ...], std: tuple[float, ...]) -> torch.Tensor:
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - m) / s


class DINOv2Base:
    spec: ClassVar[ExtractorSpec] = spec("dinov2_b")
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    def __init__(self, device: str):
        s = self.spec
        self.model = Dinov2Model.from_pretrained(s.hf_repo, revision=s.hf_revision).to(device).eval()  # type: ignore[arg-type]

    @torch.inference_mode()
    def __call__(self, pixels: torch.Tensor) -> dict[str, torch.Tensor]:
        # last_hidden_state is already layer-normed: (B, 1 + 16*16, 768) at 224px
        h = self.model(pixel_values=_normalize(pixels, self.MEAN, self.STD)).last_hidden_state
        return {"dinov2_b_cls": h[:, 0], "dinov2_b_mean": h[:, 1:].mean(1)}


class CLIPL14:
    """OpenAI CLIP ViT-L/14 image embedding, unit-normalized as the LAION predictor expects."""

    spec: ClassVar[ExtractorSpec] = spec("clip_l14")
    MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, device: str):
        s = self.spec
        model = CLIPVisionModelWithProjection.from_pretrained(s.hf_repo, revision=s.hf_revision)
        self.model = model.to(device).eval()  # type: ignore[arg-type]

    @torch.inference_mode()
    def __call__(self, pixels: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.model(pixel_values=_normalize(pixels, self.MEAN, self.STD)).image_embeds
        return {"clip_l14": torch.nn.functional.normalize(z, dim=-1)}


IMPLEMENTATIONS: dict[str, type[DINOv2Base | CLIPL14]] = {
    "dinov2_b": DINOv2Base,
    "clip_l14": CLIPL14,
}


def build(name: str, device: str) -> DINOv2Base | CLIPL14:
    spec(name)  # validates the name against the registry
    return IMPLEMENTATIONS[name](device)

