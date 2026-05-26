"""TerraWM-D: bootstrap depth head.

Per-patch depth predictor used WRITE-ONLY. Never an output path. The job is
to give each patch a depth hypothesis so we can project it to 3D for writing
into the voxel grid. The actual output depth at frame t comes from rendering
the voxel grid (which by then contains contributions from frames 0..t).

Supervised at every frame independently with GT per-patch depth (avg-pooled
GT depth over the patch's pixel footprint). The supervision is local and
per-frame — simple, no temporal coupling. That's by design: the bootstrap
head's only job is "where to write the current patch in 3D," not "what
depth should be output." Output depth is rendered from the grid.

Firewall property: the bootstrap depth is NEVER returned as the dense head's
output. The dense output is rendered_depth, not bootstrap_depth. This is the
non-negotiable that keeps the voxel grid load-bearing — if bootstrap_depth
ever became a fallback path for output, it'd be a bypass and the grid would
go inert (per A's lesson).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BootstrapDepthHead(nn.Module):
    """Per-patch depth from intra-attended patch features.

    in:  (B, T, P, D) or (B, P, D) refined patch features
    out: (B, T, P) or (B, P)        per-patch depth in metres, strictly positive
    """

    def __init__(self, dim: int, hidden: int = 128, max_depth: float = 10.0):
        super().__init__()
        self.max_depth = max_depth
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        raw = self.mlp(patches).squeeze(-1)            # (..., P)
        # Map to (0, max_depth) via sigmoid * max_depth. Smooth, bounded, positive.
        return torch.sigmoid(raw) * self.max_depth


def gt_per_patch_depth(
    full_depth: torch.Tensor,      # (B, T, H, W) ground-truth metric depth (0 = invalid)
    grid_h: int,
    grid_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Avg-pool GT depth to the patch grid for per-patch supervision.

    Returns:
      patch_depth: (B, T, P)  mean depth over each patch's footprint
      patch_valid: (B, T, P)  bool — True iff at least one valid (>0) GT pixel
                              in the footprint (so we can mask the loss).
    """
    B, T, H, W = full_depth.shape
    P = grid_h * grid_w
    ph = H // grid_h
    pw = W // grid_w
    valid_mask = (full_depth > 1e-3).float()           # (B, T, H, W)
    d_flat = (full_depth * valid_mask).view(B * T, 1, H, W)
    v_flat = valid_mask.view(B * T, 1, H, W)
    sum_d = F.avg_pool2d(d_flat, kernel_size=(ph, pw), divisor_override=1).view(B, T, P)
    sum_v = F.avg_pool2d(v_flat, kernel_size=(ph, pw), divisor_override=1).view(B, T, P)
    mean_d = sum_d / sum_v.clamp_min(1.0)
    return mean_d, (sum_v > 0)


def bootstrap_depth_loss(
    pred_depth: torch.Tensor,     # (B, T, P) predicted per-patch depth
    gt_depth: torch.Tensor,       # (B, T, P) GT per-patch depth
    gt_valid: torch.Tensor,       # (B, T, P) bool
) -> torch.Tensor:
    """Smooth-L1 in log space, masked on valid GT only.

    Log space because depth spans ~2 orders of magnitude (0.1m – 10m) and
    L1 in linear space over-penalizes far-depth errors.
    """
    valid = gt_valid.float()
    eps = 1e-3
    log_pred = (pred_depth.clamp_min(eps)).log()
    log_gt = (gt_depth.clamp_min(eps)).log()
    diff = log_pred - log_gt
    loss_per = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none")
    return (loss_per * valid).sum() / valid.sum().clamp_min(1.0)


if __name__ == "__main__":
    torch.manual_seed(0)
    head = BootstrapDepthHead(dim=1024).cuda()
    patches = torch.randn(1, 8, 1024, 1024, device="cuda")
    pred = head(patches)
    print(f"[bootstrap] in {tuple(patches.shape)} → out {tuple(pred.shape)}")
    print(f"[bootstrap] pred depth range: [{pred.min():.3f}, {pred.max():.3f}] (m)")
    print(f"[bootstrap] params: {sum(p.numel() for p in head.parameters()) / 1e3:.1f}K")

    # Test gt_per_patch_depth + loss
    H, W = 512, 512
    grid_h, grid_w = 32, 32
    gt_full = torch.rand(1, 8, H, W, device="cuda") * 5.0 + 0.5
    # Zero out a strip to test invalid masking.
    gt_full[:, :, :64, :] = 0
    patch_d, patch_v = gt_per_patch_depth(gt_full, grid_h, grid_w)
    print(f"[bootstrap] gt patch depth shape: {tuple(patch_d.shape)}, "
          f"valid count: {patch_v.float().sum().item():.0f}/{patch_v.numel()}")
    loss = bootstrap_depth_loss(pred, patch_d, patch_v)
    print(f"[bootstrap] loss (untrained): {loss.item():.4f}")
    loss.backward()
    print(f"[bootstrap] backward OK")
    print(f"[bootstrap] PASS")
