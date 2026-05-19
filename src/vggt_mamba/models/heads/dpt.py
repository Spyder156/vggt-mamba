"""Tiny DPT-style pointmap head.

Takes per-frame patch features (B*T, D, gh, gw) and produces a pointmap
(B*T, 3, H, W) at the original image resolution. Intentionally simple —
real DPT does multi-scale fusion across encoder blocks; Phase 1 just needs
something that works and is the same across V-JEPA / DINOv2 variants.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointmapHead(nn.Module):
    """patches (B, D, gh, gw) -> pointmap (B, 3, H, W)."""

    def __init__(self, in_dim: int, hidden: int = 256, out_size: int = 384):
        super().__init__()
        self.out_size = out_size
        self.project = nn.Conv2d(in_dim, hidden, kernel_size=1)
        self.block1 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, hidden // 2, 3, padding=1),
            nn.GELU(),
        )
        self.out = nn.Conv2d(hidden // 2, 3, kernel_size=1)

    def forward(self, patches_grid: torch.Tensor) -> torch.Tensor:
        # patches_grid: (B, D, gh, gw)
        x = self.project(patches_grid)
        x = self.block1(x)
        x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        x = self.block2(x)
        return self.out(x)  # (B, 3, H, W)


if __name__ == "__main__":
    h = PointmapHead(in_dim=1024, hidden=256, out_size=384).cuda()
    grid = torch.randn(2, 1024, 24, 24, device="cuda")
    y = h(grid)
    print(f"[dpt] in={tuple(grid.shape)} out={tuple(y.shape)} "
          f"params={sum(p.numel() for p in h.parameters())/1e6:.2f}M")
