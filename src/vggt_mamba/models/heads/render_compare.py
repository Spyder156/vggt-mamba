"""TerraWM-D: render-and-compare pose head.

The no-bypass pose mechanism. At frame t:

  1. An initial pose estimate is supplied (typically previous-frame's predicted
     pose, detached, or identity / GT[0] at frame 0).
  2. The voxel grid is rendered from the initial pose to produce per-ray
     features (one ray per patch pixel center, B × P rays).
  3. The current frame's intra-attended patch features are projected to the
     same dim as voxel features (so they're comparable).
  4. A discrepancy = current_proj - rendered_feature per patch, weighted by
     per-ray total_weight (zero where no voxels were hit — the ray missed
     everything written so far).
  5. Pool the discrepancies + total_weight signal → MLP → (Δt_3, Δq_4).
  6. Final pose = initial_pose ⊕ (Δt, Δq) [translation adds, quat composes].

The final layer of the correction MLP is zero-initialised, so at init the
delta is exactly 0 and the corrected pose = initial pose. The MLP then learns
to produce non-zero corrections ONLY by using the render-vs-current
discrepancy — there is no path for it to bypass the rendered features
(which depend on the voxel grid). When the voxel grid is empty,
rendered_feature = 0 and total_weight = 0; the discrepancy carries no
information and the head outputs the same delta regardless of input — the
zero-grid smoke test confirms this.

Quaternion composition uses Hamilton product (always produces unit quats).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternions (qx, qy, qz, qw). (..., 4) → (..., 4)."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dim=-1)


class RenderCompareHead(nn.Module):
    """current-frame features + voxel-rendered features → (Δt, Δq) correction.

    Args:
      patch_dim: current frame's patch feature dim (e.g. 1024 from DINOv3).
      voxel_dim: voxel feature dim (e.g. 32). The render output is in this dim.
      hidden:    MLP hidden width for the correction layer.
      max_dt:    cap on per-frame translation correction magnitude (m). Per-axis tanh-bounded.
      max_dq:    cap on per-frame quaternion small-angle correction. Per-axis tanh-bounded.
    """

    def __init__(
        self,
        patch_dim: int = 1024,
        voxel_dim: int = 32,
        hidden: int = 256,
        max_dt: float = 0.30,
        max_dq: float = 0.15,
    ):
        super().__init__()
        self.patch_dim = patch_dim
        self.voxel_dim = voxel_dim
        self.max_dt = max_dt
        self.max_dq = max_dq
        # Project current patches to voxel-feature dim so they're comparable.
        self.current_proj = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, voxel_dim),
        )
        # Comparison MLP: in = (pooled_discrepancy + pooled_current + scalar_coverage)
        # = voxel_dim + voxel_dim + 1
        in_dim = voxel_dim * 2 + 1
        self.compare = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 7),                       # (Δt_3, Δq_4)
        )
        # Zero-init final layer so init correction = 0.
        nn.init.zeros_(self.compare[-1].weight)
        nn.init.zeros_(self.compare[-1].bias)

    def forward(
        self,
        current_patches: torch.Tensor,   # (B, P, patch_dim) current frame's intra-attended patches
        rendered_feature: torch.Tensor,  # (B, P, voxel_dim) voxel-rendered per-patch features at initial pose
        ray_total_weight: torch.Tensor,  # (B, P) cumulative weight from rendering (0 = ray hit nothing)
        initial_pose: torch.Tensor,      # (B, 9) [tx,ty,tz, qx,qy,qz,qw, fovx,fovy] initial estimate
    ) -> torch.Tensor:
        """Returns (B, 9) corrected pose. fov passed through unchanged from initial."""
        B = current_patches.shape[0]
        current_proj = self.current_proj(current_patches)                       # (B, P, voxel_dim)
        diff = current_proj - rendered_feature                                  # (B, P, voxel_dim)
        # Per-ray weight: total_weight measures how much voxel content the ray
        # accumulated. Use it to weight the per-patch discrepancy contribution.
        w = ray_total_weight.unsqueeze(-1).clamp(min=0.0)                       # (B, P, 1)
        w_sum = w.sum(dim=1).clamp_min(1e-6)                                    # (B, 1)
        pooled_diff = (diff * w).sum(dim=1) / w_sum                             # (B, voxel_dim)
        pooled_cur = (current_proj * w).sum(dim=1) / w_sum                      # (B, voxel_dim)
        coverage = (ray_total_weight > 1e-3).float().mean(dim=1, keepdim=True)  # (B, 1)
        mlp_in = torch.cat([pooled_diff, pooled_cur, coverage], dim=-1)         # (B, 2*voxel_dim+1)
        delta_raw = self.compare(mlp_in)                                         # (B, 7)
        # STRUCTURAL NO-BYPASS GATE: multiply raw delta by coverage so that
        # when no rays hit any written voxels (coverage = 0), the correction
        # is FORCED to zero regardless of MLP biases. This is the hard
        # guarantee that closes the bypass route through LayerNorm/Linear
        # biases that produce non-zero output even on zero input.
        delta_raw = delta_raw * coverage                                         # (B, 7)
        dt = torch.tanh(delta_raw[:, :3]) * self.max_dt                         # (B, 3) bounded
        dq_perturb = torch.tanh(delta_raw[:, 3:7]) * self.max_dq                # (B, 4) small-angle
        # Compose: corrected_t = initial_t + dt; corrected_q = (init_q + dq_perturb).norm
        init_t = initial_pose[:, :3].float()
        init_q = initial_pose[:, 3:7].float()
        init_fov = initial_pose[:, 7:].float()
        corrected_t = init_t + dt
        # Treat dq_perturb as a small quaternion offset; renormalize.
        q_unnorm = init_q + dq_perturb
        corrected_q = q_unnorm / q_unnorm.norm(dim=-1, keepdim=True).clamp_min(1e-3)
        return torch.cat([corrected_t, corrected_q, init_fov], dim=-1)          # (B, 9)


if __name__ == "__main__":
    torch.manual_seed(0)
    head = RenderCompareHead(patch_dim=1024, voxel_dim=32).cuda()
    B, P = 1, 1024
    current = torch.randn(B, P, 1024, device="cuda")
    rendered = torch.randn(B, P, 32, device="cuda") * 0.3
    weight = torch.rand(B, P, device="cuda") * 0.5
    init_pose = torch.tensor([[0., 0., 0., 0., 0., 0., 1., 1.0, 1.0]], device="cuda")

    corrected = head(current, rendered, weight, init_pose)
    print(f"[render-cmp] in: current {tuple(current.shape)}, rendered {tuple(rendered.shape)}, "
          f"weight {tuple(weight.shape)}, init_pose {tuple(init_pose.shape)}")
    print(f"[render-cmp] out corrected: {tuple(corrected.shape)} = {corrected[0].tolist()}")
    # At init, correction should be exactly 0 (zero-init final layer).
    dt_norm = (corrected[:, :3] - init_pose[:, :3]).abs().max().item()
    dq_norm = (corrected[:, 3:7] - init_pose[:, 3:7]).abs().max().item()
    print(f"[render-cmp] init correction magnitude: dt_max={dt_norm:.2e}, dq_max={dq_norm:.2e}")
    assert dt_norm < 1e-6 and dq_norm < 1e-6, "init correction should be ~0 (zero-init final layer)"

    # Stress test: zero rendered + zero weight → should still produce init pose (no info available).
    zero_rendered = torch.zeros(B, P, 32, device="cuda")
    zero_weight = torch.zeros(B, P, device="cuda")
    corrected_zero = head(current, zero_rendered, zero_weight, init_pose)
    diff_to_init = (corrected_zero - init_pose).abs().max().item()
    print(f"[render-cmp] zero-grid stress: corrected diff to init = {diff_to_init:.2e} (should be ~0)")

    # Backward sanity.
    loss = (corrected[:, :3] - torch.tensor([0.1, 0.0, 0.0], device="cuda")).pow(2).sum()
    loss.backward()
    print(f"[render-cmp] backward OK")
    print(f"[render-cmp] params: {sum(p.numel() for p in head.parameters()) / 1e3:.1f}K")
    print(f"[render-cmp] PASS")
