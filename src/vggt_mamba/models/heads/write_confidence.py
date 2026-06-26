"""TerraWM-D: write-confidence head.

Per-patch scalar weight applied at write time so each patch's contribution
to voxel mass is *learnable* — making mass differentiable. Without this
head, mass is purely a function of write frequency (a non-differentiable
proxy for occupancy); the render path's gradient flows only through voxel
features, not through depth/density. That's the gradient-starvation root
cause behind:
  - render_l1 stagnation (rendered depth has no gradient)
  - dead mass channel (no gradient to learn occupancy from)
  - pose-head bypass under tracking loss (grid render too weak to compete
    with frame features as a pose signal)

With this head:
  weights = sigmoid(WriteConfidenceHead(patches))     # (B, P) in [0, 1]
  write_voxels_trilinear(state, world_pts, voxel_feat, weights=weights)

Now state.write_mass = trilinear_weights × confidence, accumulated.
The render path's `density = relu(sampled_mass)` becomes differentiable
through `confidence`, which is differentiable through patches → encoder.

Design choices:
  - Sigmoid bound [0, 1] keeps per-write influence bounded for stability.
  - Zero-init final bias → initial confidence ≈ 0.5 (neutral).
  - Single scalar per patch (not per-corner): keeps the trilinear scatter
    semantically unchanged, only scales the strength.
  - Geometry is still detached at write time (world_pts ← bootstrap.detach()
    + pose.detach()). Confidence is the ONLY new gradient channel — we're
    making write-STRENGTH learnable, not write-POSITION.

Firewall property: bootstrap depth is still write-only; pose is still
detached for writes. The grid is still a one-way deposit on the geometry
side. We're only adding "how much to deposit" as a learnable scalar.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class WriteConfidenceHead(nn.Module):
    """Per-patch write confidence in [0, 1].

    in:  (B, T, P, D) or (B, P, D)  refined patch features
    out: (B, T, P) or (B, P)        scalar weight per patch
    """

    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Zero-init final layer → initial pre-sigmoid = 0 → confidence = 0.5
        # (neutral starting point; model learns to push up or down from there).
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        raw = self.mlp(patches).squeeze(-1)           # (..., P)
        return torch.sigmoid(raw)


if __name__ == "__main__":
    torch.manual_seed(0)
    head = WriteConfidenceHead(dim=1024).cuda()
    patches = torch.randn(1, 1024, 1024, device="cuda")
    conf = head(patches)
    print(f"[write-conf] in {tuple(patches.shape)} → out {tuple(conf.shape)}")
    print(f"[write-conf] init confidence: min={conf.min():.4f}, max={conf.max():.4f}, "
          f"mean={conf.mean():.4f} (should be ~0.5)")
    assert abs(conf.mean().item() - 0.5) < 1e-3, "zero-init should give mean confidence 0.5"

    # Backward sanity.
    loss = ((conf - 0.7) ** 2).sum()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in head.parameters() if p.grad is not None)
    print(f"[write-conf] backward OK, total grad norm: {grad_norm:.4f}")
    print(f"[write-conf] params: {sum(p.numel() for p in head.parameters()) / 1e3:.1f}K")
    print(f"[write-conf] PASS")
