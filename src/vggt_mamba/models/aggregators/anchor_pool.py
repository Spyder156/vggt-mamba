"""Bounded FIFO anchor pool for feedforward re-grounding (Experiment 2).

The pool holds K_a (= 32 by default) anchors, each anchor is a (visual descriptor,
3D world position) pair. Anchors are written one batch per frame from the current
frame's "most distinctive" patches (top-N by feature L2 norm) and replaced FIFO.

Each frame, the pool is read via cross-attention from the current frame's patches:
high-confidence (patch, anchor) matches produce an aggregate "correction signal"
that a small MLP turns into (Δt, Δq), added to the coarse pose from the camera
head. This is the feedforward analog of loop closure — re-observing a previously-
seen scene region constrains the current pose.

State is fixed-size: descriptors (B, K_a, D) + positions (B, K_a, 3) + valid mask
(B, K_a) + write_idx (B,). Constant-memory pillar preserved.

Pre-registered consistency loss lives in losses/multitask.py and uses
project_points_to_pixels from eval/metrics.py — both verified by the projection
smoke test before this module was built.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AnchorState:
    """Per-batch anchor pool state. Pre-allocated for graph compatibility."""
    descriptors: torch.Tensor  # (B, K_a, D)
    positions: torch.Tensor    # (B, K_a, 3) in world frame, fp32
    valid: torch.Tensor        # (B, K_a) bool
    write_idx: torch.Tensor    # (B,) long


def quat_to_rot_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternion (qx, qy, qz, qw) → 3x3 rotation matrix.
    q: (..., 4)  →  R: (..., 3, 3)
    """
    qx, qy, qz, qw = q.unbind(-1)
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    R = torch.stack([
        torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
        torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
        torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
    ], dim=-2)
    return R


def cam9_to_pose_w_c(cam9: torch.Tensor) -> torch.Tensor:
    """[tx,ty,tz, qx,qy,qz,qw, fovx,fovy] → 4x4 world-from-camera transform.
    cam9: (..., 9)  →  T: (..., 4, 4)  in fp32.
    """
    t = cam9[..., :3].float()
    q = F.normalize(cam9[..., 3:7].float(), dim=-1)
    R = quat_to_rot_matrix(q)
    T = torch.zeros(*cam9.shape[:-1], 4, 4, device=cam9.device, dtype=torch.float32)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def backproject_patch_to_world(
    patch_pixel: torch.Tensor,   # (B, N, 2)
    patch_depth: torch.Tensor,   # (B, N)
    K_intrinsics: torch.Tensor,  # (B, 3, 3)
    pose_w_c: torch.Tensor,      # (B, 4, 4)
) -> torch.Tensor:
    """Pixel + depth + intrinsics + pose → world-frame 3D point. fp32."""
    B, N, _ = patch_pixel.shape
    px = patch_pixel.float()
    d = patch_depth.float().clamp_min(1e-3)                       # (B, N)
    K_inv = torch.linalg.inv(K_intrinsics.float())                 # (B, 3, 3)
    ones = torch.ones(B, N, 1, device=patch_pixel.device, dtype=torch.float32)
    px_h = torch.cat([px, ones], dim=-1)                           # (B, N, 3)
    # camera-frame ray: K^-1 @ pixel_h, scaled by depth
    P_cam = torch.einsum("bij,bnj->bni", K_inv, px_h) * d.unsqueeze(-1)
    # world-frame: R_wc @ P_cam + t_wc
    R_wc = pose_w_c[:, :3, :3]                                     # (B, 3, 3)
    t_wc = pose_w_c[:, :3, 3]                                      # (B, 3)
    P_world = torch.einsum("bij,bnj->bni", R_wc, P_cam) + t_wc.unsqueeze(-2)
    return P_world


class AnchorPool(nn.Module):
    """Feedforward re-grounding pool.

    Per frame:
      1. READ: cross-attention from current patches → anchor descriptors.
         Match scores = sigmoid(patch · anchor / sqrt(D)).
      2. CORRECT: pool matched (descriptor, position) by per-anchor max score;
         single MLP takes (coarse_pose, pooled_feature) → (Δt, Δq).
         corrected_pose = coarse_pose with t+Δt and quat ⊕ Δq (re-normalized).
      3. WRITE: top-n_writes patches by L2 norm → FIFO write to pool.

    All operations are differentiable. State is pre-allocated and fixed-size
    so CUDA graph capture remains valid (Speed-B compatible).

    The correction layer is the only learned component here.
    """

    def __init__(
        self,
        dim: int,
        n_anchors: int = 32,
        n_writes: int = 4,
        match_temp: float = 1.0,
        match_threshold: float = 0.5,
        correction_hidden: int = 128,
    ):
        super().__init__()
        self.dim = dim
        self.K_a = n_anchors
        self.n_writes = n_writes
        self.match_temp = match_temp
        self.match_threshold = match_threshold
        # Correction MLP: in = (coarse_pose_9, pooled_desc_D, pooled_pos_3, pool_filled_1) → (Δt_3, Δq_4)
        self.correction = nn.Sequential(
            nn.LayerNorm(9 + dim + 3 + 1),
            nn.Linear(9 + dim + 3 + 1, correction_hidden),
            nn.GELU(),
            nn.Linear(correction_hidden, correction_hidden // 2),
            nn.GELU(),
            nn.Linear(correction_hidden // 2, 7),
        )
        # Zero-init the final layer so initial correction = 0 (no-op at init).
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def init_state(self, batch_size: int, device: str = "cuda",
                   dtype: torch.dtype = torch.bfloat16) -> AnchorState:
        return AnchorState(
            descriptors=torch.zeros(batch_size, self.K_a, self.dim, device=device, dtype=dtype),
            positions=torch.zeros(batch_size, self.K_a, 3, device=device, dtype=torch.float32),
            valid=torch.zeros(batch_size, self.K_a, device=device, dtype=torch.bool),
            write_idx=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    def reset_state(self, state: AnchorState) -> None:
        """Zero in-place (preserves addresses for graph compatibility)."""
        state.descriptors.zero_()
        state.positions.zero_()
        state.valid.zero_()
        state.write_idx.zero_()

    def _compute_scores(
        self, patches: torch.Tensor, descriptors: torch.Tensor, valid: torch.Tensor,
    ) -> torch.Tensor:
        """Sigmoid match scores per (patch, anchor) pair.

        patches: (B, P, D),  descriptors: (B, K_a, D),  valid: (B, K_a) bool
        Returns: (B, P, K_a) scores in [0, 1]. Invalid anchors get score 0.
        """
        # Normalize for cosine-style scoring; stable across feature magnitudes.
        p_n = F.normalize(patches.float(), dim=-1)
        d_n = F.normalize(descriptors.float(), dim=-1)
        sim = torch.einsum("bpd,bad->bpa", p_n, d_n)                # (B, P, K_a) in [-1, 1]
        scores = torch.sigmoid(sim / max(self.match_temp, 1e-3))     # (B, P, K_a) in (0, 1)
        # Mask out invalid anchor slots.
        scores = scores * valid.unsqueeze(1).float()
        return scores

    def correct_pose(
        self,
        state: AnchorState,
        patches: torch.Tensor,                  # (B, P, D)
        coarse_pose: torch.Tensor,              # (B, 9)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read pool and apply correction MLP.

        Returns:
          corrected_pose: (B, 9)
          scores: (B, P, K_a) — for consistency loss, passed through unchanged.
        """
        B, P, D = patches.shape
        # Detach + clone state tensors at use time. detach() alone aliases the
        # underlying storage, which the FIFO write step then mutates in place —
        # autograd would catch that as a version mismatch even with no grad
        # connection. clone() gives an independent copy whose version is fresh.
        descriptors_d = state.descriptors.detach().clone()
        positions_d = state.positions.detach().clone()
        valid_d = state.valid.detach().clone()
        scores = self._compute_scores(patches, descriptors_d, valid_d)           # (B, P, K_a)
        # Per-anchor importance = max over patches of (patch, anchor) score.
        anchor_weight = scores.max(dim=1).values                                 # (B, K_a)
        # Weighted sum of descriptors + positions.
        weight_norm = anchor_weight.sum(dim=-1, keepdim=True).clamp_min(1e-6)    # (B, 1)
        w = (anchor_weight / weight_norm).unsqueeze(-1)                           # (B, K_a, 1)
        pooled_desc = (w * descriptors_d.float()).sum(dim=1)                      # (B, D)
        pooled_pos = (w * positions_d.float()).sum(dim=1)                         # (B, 3)
        # Pool-filled fraction as an explicit feature.
        pool_filled = (state.valid.float().sum(dim=-1, keepdim=True) / self.K_a)  # (B, 1)
        mlp_in = torch.cat([coarse_pose.float(), pooled_desc, pooled_pos, pool_filled], dim=-1)
        delta = self.correction(mlp_in)                                           # (B, 7)
        # Bound the correction magnitude. Per-frame pose correction shouldn't
        # exceed ~10 cm in translation or a small-angle rotation; an unbounded
        # MLP output (especially early in training) can flip the quaternion
        # through zero where normalize() blows up to NaN. tanh keeps it sane.
        dt = torch.tanh(delta[:, :3]) * 0.10                                      # ≤ 0.10 m
        dq = torch.tanh(delta[:, 3:7]) * 0.05                                     # small quat perturbation
        t = coarse_pose[:, :3].float() + dt
        # Compose with a safety floor on the norm so even if dq drives the quat
        # near zero, we don't divide by ~0.
        q_unnorm = coarse_pose[:, 3:7].float() + dq
        q = q_unnorm / q_unnorm.norm(dim=-1, keepdim=True).clamp_min(1e-3)
        fov = coarse_pose[:, 7:].float()
        corrected = torch.cat([t, q, fov], dim=-1)                                # (B, 9)
        return corrected, scores

    @torch.no_grad()
    def write(
        self,
        state: AnchorState,
        patches: torch.Tensor,            # (B, P, D)
        patch_pixel: torch.Tensor,        # (B, P, 2) static across frames
        patch_depth: torch.Tensor,        # (B, P)
        K_intrinsics: torch.Tensor,       # (B, 3, 3)
        pose_w_c: torch.Tensor,           # (B, 4, 4) — the CORRECTED pose
    ) -> None:
        """Write top-N patches into pool, FIFO. Mutates state in place."""
        B, P, D = patches.shape
        # Select top-N by patch L2 norm in feature space.
        norms = patches.float().norm(dim=-1)                                      # (B, P)
        top_idx = norms.topk(self.n_writes, dim=-1).indices                        # (B, N)
        sel_desc = torch.gather(patches, 1, top_idx.unsqueeze(-1).expand(-1, -1, D))  # (B, N, D)
        sel_pix = torch.gather(patch_pixel, 1, top_idx.unsqueeze(-1).expand(-1, -1, 2))
        sel_d = torch.gather(patch_depth, 1, top_idx)                              # (B, N)
        sel_world = backproject_patch_to_world(sel_pix, sel_d, K_intrinsics, pose_w_c)  # (B, N, 3)
        # FIFO write into K_a slots.
        for b in range(B):
            wi = int(state.write_idx[b].item())
            for k in range(self.n_writes):
                slot = (wi + k) % self.K_a
                state.descriptors[b, slot] = sel_desc[b, k].to(state.descriptors.dtype)
                state.positions[b, slot] = sel_world[b, k]
                state.valid[b, slot] = True
            state.write_idx[b] = (wi + self.n_writes) % self.K_a


def build_patch_pixel_grid(grid_h: int, grid_w: int, img_size: int,
                           device: str = "cuda") -> torch.Tensor:
    """Pre-compute pixel centers for each patch in the (grid_h, grid_w) layout.
    Returns: (P, 2) where P = grid_h * grid_w. Add batch dim externally.
    """
    patch_size_h = img_size / grid_h
    patch_size_w = img_size / grid_w
    ys = (torch.arange(grid_h, device=device, dtype=torch.float32) + 0.5) * patch_size_h
    xs = (torch.arange(grid_w, device=device, dtype=torch.float32) + 0.5) * patch_size_w
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pixel = torch.stack([xx.flatten(), yy.flatten()], dim=-1)                      # (P, 2) [u, v]
    return pixel


if __name__ == "__main__":
    # Smoke test: instantiate, init state, run one step.
    torch.manual_seed(0)
    B, P, D, K_a = 2, 64, 256, 8
    pool = AnchorPool(dim=D, n_anchors=K_a, n_writes=2).cuda()
    state = pool.init_state(batch_size=B, dtype=torch.float32)
    patches = torch.randn(B, P, D, device="cuda")
    coarse_pose = torch.cat([
        torch.zeros(B, 3, device="cuda"),
        torch.tensor([0., 0., 0., 1.], device="cuda").expand(B, 4).contiguous(),
        torch.tensor([1., 1.], device="cuda").expand(B, 2).contiguous(),
    ], dim=-1)
    # First call: pool empty → correction should be ~zero
    corrected, scores = pool.correct_pose(state, patches, coarse_pose)
    print(f"empty pool: ||corrected - coarse|| = {(corrected - coarse_pose).norm():.4e}")
    assert (corrected - coarse_pose).norm() < 1e-4, "empty pool should yield zero correction"

    # Write some anchors and re-check
    patch_pixel = torch.rand(B, P, 2, device="cuda") * 512
    patch_depth = torch.rand(B, P, device="cuda") * 4 + 0.5
    K_int = torch.tensor([[[500., 0., 256.], [0., 500., 256.], [0., 0., 1.]]],
                         device="cuda").expand(B, 3, 3).contiguous()
    pose = torch.eye(4, device="cuda").unsqueeze(0).expand(B, 4, 4).contiguous()
    pool.write(state, patches, patch_pixel, patch_depth, K_int, pose)
    print(f"after 1 write: valid count per batch = {state.valid.sum(dim=-1).tolist()}")

    corrected2, scores2 = pool.correct_pose(state, patches, coarse_pose)
    print(f"after write: ||corrected2 - coarse|| = {(corrected2 - coarse_pose).norm():.4e}")
    print(f"scores max: {scores2.max().item():.3f}")
    print(f"params: {sum(p.numel() for p in pool.parameters()) / 1e3:.1f}K")
