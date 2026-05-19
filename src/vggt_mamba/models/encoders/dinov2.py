"""DINOv2 ViT-L/14 wrapper for single-image patch features.

Built against the cloned `third_party/dinov2` package, loading the official
fbaipublicfiles ViT-L/14 checkpoint via plain `torch.load`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

from .base import EncoderOutput

_REPO = Path(__file__).resolve().parents[4]
_DINOV2 = _REPO / "third_party" / "dinov2"
if str(_DINOV2) not in sys.path:
    sys.path.insert(0, str(_DINOV2))


class DINOv2Encoder(nn.Module):
    """DINOv2 ViT-L/14 frozen feature extractor.

    Input:  (B, 3, H, W) RGB in [0, 1]. H, W should be divisible by 14 and
            equal `img_size`.
    Output: EncoderOutput with patches (B, N, D), grid = (img_size/14)^2.
    """

    # DINOv2 uses ImageNet normalization.
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, weights: str | Path, img_size: int = 518, freeze: bool = True):
        super().__init__()
        from dinov2.models.vision_transformer import vit_large  # type: ignore

        assert img_size % 14 == 0, f"DINOv2 img_size must be divisible by 14, got {img_size}"

        self.img_size = img_size
        self.patch_size = 14
        self.grid = img_size // 14

        # block_chunks=0 keeps blocks as a flat list, matching the checkpoint key layout
        # (the chunked form `blocks.X.Y.<param>` is a perf-only restructuring).
        # init_values=1e-5 enables LayerScale (gamma params in the checkpoint).
        # block_chunks=0 keeps blocks flat to match the checkpoint key layout.
        self.backbone = vit_large(
            patch_size=14,
            img_size=img_size,
            num_register_tokens=0,
            block_chunks=0,
            init_values=1.0e-5,
        )
        state = torch.load(Path(weights), map_location="cpu", weights_only=True)
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        if len(missing) > 0:
            print(f"[dinov2] missing keys: {len(missing)} (first 5: {missing[:5]})")
        if len(unexpected) > 0:
            print(f"[dinov2] unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")

        # infer feature dim
        with torch.no_grad():
            x = torch.zeros(1, 3, img_size, img_size)
            out = self.backbone.forward_features(x)
            self.dim = int(out["x_norm_patchtokens"].shape[-1])

        if freeze:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.STD).view(1, 3, 1, 1))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    @torch.no_grad()
    def _frozen_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        out = self.backbone.forward_features(x)
        return out["x_norm_patchtokens"]  # (B, N, D)

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        feats = self._frozen_forward(x)
        return EncoderOutput(patches=feats, grid_h=self.grid, grid_w=self.grid, dim=self.dim)


if __name__ == "__main__":
    import os
    root = Path(os.environ.get("VGGT_MAMBA_DATA_ROOT", "/workspace/datasets"))
    w = root / "weights/dinov2-large/dinov2_vitl14_pretrain.pth"
    enc = DINOv2Encoder(w, img_size=518).cuda()
    x = torch.rand(2, 3, 518, 518, device="cuda")
    out = enc(x)
    print(f"[dinov2] patches={tuple(out.patches.shape)} grid={out.grid_h}x{out.grid_w} dim={out.dim}")
