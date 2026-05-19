"""Camera head — state-only.

Reads from the per-frame state tokens (after Mamba scan), pools across K,
and predicts a 9-dim camera vector per frame:
    [tx, ty, tz, qx, qy, qz, qw, fovx, fovy]

The quaternion is L2-normalized at the end of the forward.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CameraHead(nn.Module):
    """state_per_frame (B, T, K, D) -> camera (B, T, 9)."""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 9),
        )

    def forward(self, state_per_frame: torch.Tensor) -> torch.Tensor:
        # mean-pool over the K summary tokens per frame
        pooled = state_per_frame.mean(dim=2)              # (B, T, D)
        raw = self.mlp(pooled)                            # (B, T, 9)

        tx_ty_tz = raw[..., :3]
        quat = F.normalize(raw[..., 3:7], dim=-1)         # unit quaternion
        fov = raw[..., 7:]                                # (B, T, 2)
        return torch.cat([tx_ty_tz, quat, fov], dim=-1)


if __name__ == "__main__":
    h = CameraHead(dim=1024).cuda()
    s = torch.randn(1, 4, 4, 1024, device="cuda")
    y = h(s)
    print(f"[camera_head] state {tuple(s.shape)} -> camera {tuple(y.shape)}")
    print(f"[camera_head] quat norms (should be 1): {y[..., 3:7].norm(dim=-1).flatten().tolist()}")
    print(f"[camera_head] params: {sum(p.numel() for p in h.parameters())/1e6:.2f}M")
