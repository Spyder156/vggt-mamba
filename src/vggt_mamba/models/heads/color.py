"""TerraWM-D: voxel-feature → RGB color head.

The drift-blindness diagnostic (terrawm_d_drift_blind_check.py) confirmed the
geometric render-vs-current channel INVERTS with drift: β_disp = -2.18
(95% CI [-2.99, -1.38]) after controlling for coverage. The channel doesn't
just fail to see drift — it actively reports lower mismatch when drift is
higher, because in dense self-similar scenes a drifted view matches some
nearby surface even better than the correct view matches the true one.

This head is the primary fix candidate: predict per-patch RGB from the
rendered voxel features at the corrected pose, supervise against the current
frame's RGB. If photometric mismatch is positively correlated with drift,
the pose head's loss has a signal that points the RIGHT way for the first
time.

Bypass guard (structural): this head reads ONLY rendered features, never
encoder patches. It cannot skip the voxel grid by routing through the
encoder. The scene-state ablation (post-reset Δt ≥ 0.10m) is the pre-
registered behavioral co-guard against subtler bypasses.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ColorHead(nn.Module):
    """Render-feature → predicted RGB.

    Args:
      voxel_dim: dim of rendered features (== voxel feature dim).
      hidden:    MLP hidden width.

    forward: (B, P, voxel_dim) → (B, P, 3) in [0, 1].
    """
    def __init__(self, voxel_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(voxel_dim),
            nn.Linear(voxel_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
            nn.Sigmoid(),                                                # [0, 1] color
        )

    def forward(self, rendered_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(rendered_features)


if __name__ == "__main__":
    torch.manual_seed(0)
    head = ColorHead(voxel_dim=32).cuda()
    B, P = 2, 1024
    rendered = torch.randn(B, P, 32, device="cuda") * 0.3
    rgb = head(rendered)
    print(f"[color-head] in: {tuple(rendered.shape)} -> out: {tuple(rgb.shape)}")
    print(f"[color-head] range: [{rgb.min():.3f}, {rgb.max():.3f}]")
    assert rgb.shape == (B, P, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
    loss = rgb.sum()
    loss.backward()
    print(f"[color-head] backward OK")
    print(f"[color-head] params: {sum(p.numel() for p in head.parameters())/1e3:.1f}K")
    print(f"[color-head] PASS")
