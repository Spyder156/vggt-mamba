"""Evaluation metrics and Phase 1 loss helpers.

Most metrics accept torch tensors or numpy arrays. Heavy implementations are
left as stubs and will be wired up alongside their respective phases.
"""

from __future__ import annotations

import numpy as np
import torch

ArrayLike = torch.Tensor | np.ndarray


def _to_numpy(x: ArrayLike) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


# ---------- Pointmap / 3D reconstruction ----------

def pointmap_chamfer_distance(pred: ArrayLike, gt: ArrayLike) -> float:
    """Symmetric mean Chamfer between two point clouds. CPU/numpy."""
    from scipy.spatial import cKDTree

    p, g = _to_numpy(pred), _to_numpy(gt)
    tp, tg = cKDTree(p), cKDTree(g)
    d_pg, _ = tg.query(p, k=1)
    d_gp, _ = tp.query(g, k=1)
    return float(0.5 * (d_pg.mean() + d_gp.mean()))


def chamfer_distance_torch(
    a: torch.Tensor, b: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """Differentiable symmetric Chamfer for batched point clouds.

    Args:
        a: (B, N, 3)
        b: (B, M, 3)

    Returns:
        scalar if reduction='mean', else (B,).
    """
    dists = torch.cdist(a, b, p=2)  # (B, N, M)
    d_ab = dists.min(dim=2).values  # (B, N)
    d_ba = dists.min(dim=1).values  # (B, M)
    per_batch = 0.5 * (d_ab.mean(dim=1) + d_ba.mean(dim=1))
    return per_batch.mean() if reduction == "mean" else per_batch


def normal_consistency(pred_normals: ArrayLike, gt_normals: ArrayLike) -> float:
    raise NotImplementedError("wire up after Phase 1 — see CUT3R repo for reference")


# ---------- Camera pose ----------

def absolute_translation_error(
    pred_poses: ArrayLike, gt_poses: ArrayLike, align: str = "sim3"
) -> float:
    raise NotImplementedError("wire up after Phase 3")


def relative_pose_error(
    pred_poses: ArrayLike, gt_poses: ArrayLike, delta: int = 1
) -> tuple[float, float]:
    raise NotImplementedError("wire up after Phase 3")


# ---------- Depth ----------

def depth_abs_rel(
    pred: ArrayLike, gt: ArrayLike, valid_mask: ArrayLike | None = None
) -> float:
    p, g = _to_numpy(pred), _to_numpy(gt)
    if valid_mask is None:
        valid_mask = g > 0
    m = _to_numpy(valid_mask).astype(bool)
    return float(np.mean(np.abs(p[m] - g[m]) / np.maximum(g[m], 1e-6)))


def depth_threshold_accuracy(
    pred: ArrayLike, gt: ArrayLike, valid_mask: ArrayLike | None = None, thresh: float = 1.25
) -> float:
    p, g = _to_numpy(pred), _to_numpy(gt)
    if valid_mask is None:
        valid_mask = g > 0
    m = _to_numpy(valid_mask).astype(bool)
    p, g = np.maximum(p[m], 1e-6), np.maximum(g[m], 1e-6)
    ratio = np.maximum(p / g, g / p)
    return float(np.mean(ratio < thresh))


# ---------- Phase 1 main metric: multi-view consistency ----------

def transform_points(P: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Apply 4x4 transform to points.

    Args:
        P: (..., 3, H, W)
        T: (..., 4, 4)
    """
    b_shape = P.shape[:-3]
    h, w = P.shape[-2:]
    flat = P.reshape(*b_shape, 3, h * w).transpose(-1, -2)  # (..., N, 3)
    R = T[..., :3, :3]
    t = T[..., :3, 3].unsqueeze(-2)
    out = flat @ R.transpose(-1, -2) + t
    return out.transpose(-1, -2).reshape(*b_shape, 3, h, w)


def sample_valid_points(
    pmap: torch.Tensor, valid: torch.Tensor, n_samples: int
) -> torch.Tensor:
    """Randomly subsample valid points from a pointmap.

    Args:
        pmap: (B, T, 3, H, W) in some coordinate frame
        valid: (B, T, H, W) bool
        n_samples: target points per (B, T) entry. If a frame has fewer valid
                   pixels we sample with replacement.

    Returns:
        (B, T, n_samples, 3) sampled points.
    """
    b, t, _, h, w = pmap.shape
    out = pmap.new_zeros(b, t, n_samples, 3)
    for bi in range(b):
        for ti in range(t):
            v = valid[bi, ti].flatten()
            pts = pmap[bi, ti].view(3, -1).T  # (H*W, 3)
            idx_valid = v.nonzero(as_tuple=False).squeeze(-1)
            if idx_valid.numel() == 0:
                continue
            replace = idx_valid.numel() < n_samples
            if replace:
                sel = idx_valid[
                    torch.randint(0, idx_valid.numel(), (n_samples,), device=pmap.device)
                ]
            else:
                sel = idx_valid[
                    torch.randperm(idx_valid.numel(), device=pmap.device)[:n_samples]
                ]
            out[bi, ti] = pts[sel]
    return out


def multi_view_consistency(
    pred_pointmaps_cam: torch.Tensor,
    valid: torch.Tensor,
    poses_w_c: torch.Tensor,
    n_samples: int = 2048,
) -> torch.Tensor:
    """Average pairwise Chamfer between frames in a window, in world coords.

    Args:
        pred_pointmaps_cam: (B, T, 3, H, W) in each frame's camera frame.
        valid: (B, T, H, W) bool.
        poses_w_c: (B, T, 4, 4) world-from-camera.
        n_samples: random points per frame.

    Returns:
        scalar — mean Chamfer over all (i,j) pairs with i<j.
    """
    pts_world = transform_points(pred_pointmaps_cam, poses_w_c)        # (B, T, 3, H, W)
    sampled = sample_valid_points(pts_world, valid, n_samples)         # (B, T, N, 3)

    t = sampled.shape[1]
    pair_losses = []
    for i in range(t):
        for j in range(i + 1, t):
            pair_losses.append(chamfer_distance_torch(sampled[:, i], sampled[:, j]))
    if not pair_losses:
        return sampled.new_zeros(())
    return torch.stack(pair_losses).mean()


if __name__ == "__main__":
    # Self-test: a perfect pointmap (= GT) should have ~0 multi-view chamfer.
    from vggt_mamba.data.tum_rgbd import unproject_depth_to_pointmap

    B, T, H, W = 1, 4, 32, 32
    K = torch.eye(3).unsqueeze(0).expand(B, 3, 3)
    depth = torch.rand(B, T, H, W) * 2 + 1
    valid = torch.ones_like(depth, dtype=torch.bool)
    poses = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, T, 4, 4).clone()
    pmap = unproject_depth_to_pointmap(depth, K.unsqueeze(1).expand(B, T, 3, 3))
    loss = multi_view_consistency(pmap, valid, poses, n_samples=512)
    print(f"[metrics] identity-pose multi-view chamfer (should be ~0): {loss.item():.6f}")

    pmap_noisy = pmap.clone()
    pmap_noisy[:, 2] += 0.1 * torch.randn_like(pmap_noisy[:, 2])
    loss_noisy = multi_view_consistency(pmap_noisy, valid, poses, n_samples=512)
    print(f"[metrics] perturbed multi-view chamfer (should be >0): {loss_noisy.item():.6f}")
