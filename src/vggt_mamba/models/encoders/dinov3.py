"""DINOv3 ViT-L/16 wrapper.

Loaded via HuggingFace Transformers — DINOv3 weights are gated; HF auth
token at ~/.cache/huggingface/token grants access. Container mounts that
cache, so transformers.AutoModel just works.

DINOv3 ViT-L/16 outputs (B, N+5, D) — 1 CLS + 4 register tokens prepended
to N patch tokens. We slice off the first 5.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .base import EncoderOutput


class DINOv3Encoder(nn.Module):
    """DINOv3 ViT-L/16 frozen feature extractor.

    Input:  (B, 3, H, W) RGB in [0, 1]. H,W must be divisible by 16.
    Output: EncoderOutput with patches (B, N, D), grid = (H/16, W/16).
    """

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    # 5 prepended tokens: 1 CLS + 4 register tokens
    N_SPECIAL_TOKENS = 5

    def __init__(
        self,
        repo_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        img_size: int = 512,
        freeze: bool = True,
    ):
        super().__init__()
        from transformers import AutoModel  # type: ignore

        assert img_size % 16 == 0, f"DINOv3 img_size must be divisible by 16, got {img_size}"

        self.img_size = img_size
        self.patch_size = 16
        self.grid = img_size // 16

        self.backbone = AutoModel.from_pretrained(repo_id)

        # infer feature dim
        with torch.no_grad():
            x = torch.zeros(1, 3, img_size, img_size)
            out = self.backbone(x)
            hs = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
            self.dim = int(hs.shape[-1])
            assert hs.shape[1] == self.grid * self.grid + self.N_SPECIAL_TOKENS, (
                f"unexpected DINOv3 token count: got {hs.shape[1]}, "
                f"expected {self.grid * self.grid + self.N_SPECIAL_TOKENS}"
            )

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
        out = self.backbone(x)
        hs = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        return hs[:, self.N_SPECIAL_TOKENS:]  # drop CLS + register tokens

    def forward(self, x: torch.Tensor) -> EncoderOutput:
        feats = self._frozen_forward(x)
        return EncoderOutput(patches=feats, grid_h=self.grid, grid_w=self.grid, dim=self.dim)


if __name__ == "__main__":
    enc = DINOv3Encoder(img_size=512).cuda()
    x = torch.rand(2, 3, 512, 512, device="cuda")
    out = enc(x)
    print(f"[dinov3] patches={tuple(out.patches.shape)} "
          f"grid={out.grid_h}x{out.grid_w} dim={out.dim}")
