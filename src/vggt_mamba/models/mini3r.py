"""Mini-3R: tiny baseline for Phase 1 (tokenizer probe) and Phase 2
(Mamba vs attention aggregator).

Encoder (frozen) → cross-frame aggregator (transformer or Mamba) →
per-frame DPT pointmap head.

Same architecture across all variants; only the encoder and/or aggregator
swap. This isolates one variable at a time:
- Phase 1: hold aggregator fixed (transformer), swap encoder.
- Phase 2: hold encoder fixed (DINOv3), swap aggregator.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .aggregators import CrossFrameMamba, CrossFrameTransformer
from .encoders import DINOv2Encoder, DINOv3Encoder, EncoderOutput, VJEPAEncoder
from .heads.dpt import PointmapHead


class Mini3R(nn.Module):
    """Frozen encoder → cross-frame aggregator → DPT pointmap head."""

    def __init__(
        self,
        encoder: VJEPAEncoder | DINOv2Encoder | DINOv3Encoder,
        aggregator: nn.Module,
        head_hidden: int = 256,
    ):
        super().__init__()
        self.encoder = encoder
        self.img_size = encoder.img_size
        self.grid_h = encoder.grid
        self.grid_w = encoder.grid
        self.dim = encoder.dim

        # Learnable frame-index embedding so the aggregator can distinguish
        # which frame each patch came from (Mamba's recurrence handles order
        # implicitly but the embedding is cheap and stays consistent across
        # aggregator choices).
        self.frame_embed = nn.Parameter(torch.zeros(1, 64, 1, self.dim))
        nn.init.trunc_normal_(self.frame_embed, std=0.02)

        self.xfm = aggregator
        self.head = PointmapHead(in_dim=self.dim, hidden=head_hidden, out_size=self.img_size)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb: (B, T, 3, H, W) -> pointmap (B, T, 3, H, W) in each frame's camera frame."""
        b, t, _, h, w = rgb.shape
        assert h == self.img_size and w == self.img_size, \
            f"expected {self.img_size}x{self.img_size}, got {h}x{w}"

        flat_rgb = rgb.reshape(b * t, 3, h, w)
        with torch.no_grad():
            enc_out: EncoderOutput = self.encoder(flat_rgb)
        patches = enc_out.patches  # (B*T, N, D)
        n = patches.shape[1]

        patches = patches.reshape(b, t, n, self.dim)
        patches = patches + self.frame_embed[:, :t]
        tokens = patches.reshape(b, t * n, self.dim)
        tokens = self.xfm(tokens)
        tokens = tokens.reshape(b * t, n, self.dim)

        grid = tokens.transpose(1, 2).reshape(b * t, self.dim, self.grid_h, self.grid_w)

        # Chunk the per-frame DPT head along the frame axis so we don't hit
        # the upsample_bilinear2d INT_MAX limit (B*T × 256 × H × W must fit
        # under 2.1B elements per call). With H=W=512 and 256 hidden channels
        # the per-frame footprint is 67M elements, so a chunk of 8 = 537M
        # fits comfortably.
        chunk = 8
        outs = [self.head(grid[i:i + chunk]) for i in range(0, b * t, chunk)]
        pmap = torch.cat(outs, dim=0)
        return pmap.reshape(b, t, 3, h, w)


def build_mini3r(
    encoder_name: Literal["vjepa", "dinov2", "dinov3"],
    weights_root: str,
    aggregator_name: Literal["attention", "mamba"] = "attention",
    n_xfm_layers: int = 4,
    d_state: int = 128,
) -> Mini3R:
    """Construct Mini3R with the selected encoder + aggregator.

    Encoders:
      V-JEPA 2.1 ViT-L native: 384x384, patch 16, grid 24  (576 patches).
      DINOv2 ViT-L/14 native:  518x518, patch 14, grid 37  (1369 patches).
      DINOv3 ViT-L/16 we run:  512x512, patch 16, grid 32  (1024 patches).

    Aggregators:
      "attention" — plain transformer encoder over T*N tokens (Phase 1 baseline).
      "mamba"     — Mamba-2 SSD over T*N tokens, fixed state (Phase 2).
    """
    from pathlib import Path

    weights_root = Path(weights_root)
    if encoder_name == "vjepa":
        w = weights_root / "vjepa2-vitl/vjepa2_1_vitl_dist_vitG_384.pt"
        enc = VJEPAEncoder(w, img_size=384, freeze=True)
    elif encoder_name == "dinov2":
        w = weights_root / "dinov2-large/dinov2_vitl14_pretrain.pth"
        enc = DINOv2Encoder(w, img_size=518, freeze=True)
    elif encoder_name == "dinov3":
        enc = DINOv3Encoder(
            repo_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
            img_size=512, freeze=True,
        )
    else:
        raise ValueError(f"unknown encoder {encoder_name!r}")

    if aggregator_name == "attention":
        agg = CrossFrameTransformer(dim=enc.dim, n_layers=n_xfm_layers)
    elif aggregator_name == "mamba":
        agg = CrossFrameMamba(dim=enc.dim, n_layers=n_xfm_layers, d_state=d_state)
    else:
        raise ValueError(f"unknown aggregator {aggregator_name!r}")

    return Mini3R(enc, aggregator=agg)


if __name__ == "__main__":
    import os
    root = os.environ.get("VGGT_MAMBA_DATA_ROOT", "/workspace/datasets") + "/weights"
    for agg in ("attention", "mamba"):
        m = build_mini3r("dinov3", root, aggregator_name=agg).cuda()
        s = m.img_size
        x = torch.rand(1, 4, 3, s, s, device="cuda")
        y = m(x)
        n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"[mini3r:dinov3:{agg}] in {tuple(x.shape)} -> pmap {tuple(y.shape)}, "
              f"trainable {n_train/1e6:.2f}M")
