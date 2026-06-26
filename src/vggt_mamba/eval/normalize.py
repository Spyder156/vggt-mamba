"""Path A scene normalization (mirror of VGGT's training-time normalization).

At train time we explicitly DIVIDE the GT scene by mean-point-distance-from-cam-0
before computing loss. This teaches the model to predict in a scale-normalized
space (the only consistent choice across heterogeneous training data).

At eval time we Sim(3)-align predictions to GT (umeyama_sim3 in metrics.py) and
report shape errors after alignment.

Reference: vggt/training/train_utils/normalization.py
            ::normalize_camera_extrinsics_and_points_batch
"""
from __future__ import annotations

import torch


def back_project_depth_to_world(
    depth: torch.Tensor,                   # (B, T, H, W)
    valid: torch.Tensor,                   # (B, T, H, W) bool
    K: torch.Tensor,                       # (B, 3, 3)
    pose_w_c: torch.Tensor,                # (B, T, 4, 4)
) -> torch.Tensor:
    """Back-project per-pixel depth to world points using TUM intrinsics + GT pose.
    Invalid pixels become (0, 0, 0). Returns (B, T, H, W, 3) in world coords.
    """
    B, T, H, W = depth.shape
    device, dtype = depth.device, depth.dtype
    # Pixel grid
    js, is_ = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    fx = K[..., 0, 0].view(B, 1, 1, 1)
    fy = K[..., 1, 1].view(B, 1, 1, 1)
    cx = K[..., 0, 2].view(B, 1, 1, 1)
    cy = K[..., 1, 2].view(B, 1, 1, 1)
    z = depth                                                                     # (B, T, H, W)
    x_cam = (is_[None, None] - cx) * z / fx
    y_cam = (js[None, None] - cy) * z / fy
    P_cam = torch.stack([x_cam, y_cam, z], dim=-1)                                # (B, T, H, W, 3)

    # World-from-cam: P_world = R @ P_cam + t
    R = pose_w_c[..., :3, :3].unsqueeze(2).unsqueeze(2)                            # (B, T, 1, 1, 3, 3)
    t = pose_w_c[..., :3, 3].unsqueeze(2).unsqueeze(2)                             # (B, T, 1, 1, 3)
    P_world = torch.einsum("btijkl,btijl->btijk", R.expand(-1, -1, H, W, 3, 3), P_cam) + t
    P_world = P_world * valid.unsqueeze(-1).to(dtype)
    return P_world


def normalize_scene_by_mean_distance(
    pose_w_c: torch.Tensor,                # (B, T, 4, 4)
    depth: torch.Tensor,                   # (B, T, H, W)
    valid: torch.Tensor,                   # (B, T, H, W) bool
    K: torch.Tensor,                       # (B, 3, 3)
    eps: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VGGT-style scene normalization.

    Steps:
        1. Re-center so cam 0 is at world origin (pose_w_c[0] becomes identity).
        2. Compute mean Euclidean distance of valid 3D points from cam 0.
        3. Divide all camera translations and depths by that mean distance.

    Returns:
        normed_pose_w_c: (B, T, 4, 4)
        normed_depth:    (B, T, H, W)
        scale:           (B,)  — the divisor applied per batch element
    """
    B, T, H, W = depth.shape

    # Re-center: pose_w_c'[i] = pose_w_c[0]^-1 @ pose_w_c[i]
    pose0_inv = torch.linalg.inv(pose_w_c[:, 0])                                    # (B, 4, 4)
    centered_pose = torch.einsum("bij,btjk->btik", pose0_inv, pose_w_c)              # (B, T, 4, 4)

    # Back-project to world points in the new (cam-0-centered) frame
    world_pts = back_project_depth_to_world(depth, valid, K, centered_pose)         # (B, T, H, W, 3)
    dist = world_pts.norm(dim=-1)                                                   # (B, T, H, W)
    dist_sum = (dist * valid.to(dist.dtype)).sum(dim=[1, 2, 3])                     # (B,)
    valid_count = valid.to(dist.dtype).sum(dim=[1, 2, 3])                           # (B,)
    scale = (dist_sum / (valid_count + eps)).clamp(min=eps)                         # (B,)

    # Divide translations + depth by scale
    s = scale.view(B, 1, 1, 1)
    normed_pose = centered_pose.clone()
    normed_pose[..., :3, 3] = normed_pose[..., :3, 3] / scale.view(B, 1, 1)
    normed_depth = depth / s
    return normed_pose, normed_depth, scale


if __name__ == "__main__":
    # Round-trip-ish: scale factor for a synthetic scene at average-distance 2.0
    # should give scale ≈ 2.0 and normed scene at average-distance ≈ 1.0.
    torch.manual_seed(0)
    B, T, H, W = 1, 4, 16, 16
    # Random unit depth around mean 2.0 m
    depth = (1.5 + 1.0 * torch.rand(B, T, H, W))
    valid = torch.ones_like(depth, dtype=torch.bool)
    K = torch.tensor([[100.0, 0, 8], [0, 100.0, 8], [0, 0, 1]]).view(1, 3, 3)
    pose = torch.eye(4).view(1, 1, 4, 4).expand(B, T, -1, -1).contiguous()
    # add small jitter to non-zero frames
    pose[:, 1:, :3, 3] += 0.1 * torch.randn(B, T - 1, 3)
    pose_n, depth_n, scale = normalize_scene_by_mean_distance(pose, depth, valid, K)
    print(f"[normalize] input mean depth: {depth.mean().item():.3f}")
    print(f"[normalize] scale (per batch): {scale.tolist()}")
    print(f"[normalize] normed mean depth: {depth_n.mean().item():.3f}  (expect ~0.5–1.0)")
    print(f"[normalize] T_w_c[0] after normalize:\n{pose_n[0, 0]}")
