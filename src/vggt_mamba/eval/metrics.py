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

def project_points_to_pixels(
    world_points: torch.Tensor,
    pose_w_c: torch.Tensor,
    K: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pinhole-project 3D world points into camera pixels.

    Args:
        world_points: (..., N, 3) — points in world coordinates
        pose_w_c:     (..., 4, 4) — world-from-camera transform (camera origin
                      in world; rotation columns = camera basis in world).
                      Same convention as TUM's pose_w_c.
        K:            (..., 3, 3) — pinhole intrinsics at the same image
                      resolution the pixels will be compared against.

    Returns:
        pixels: (..., N, 2) projected pixel (u, v).
        in_front: (..., N) bool — True iff point is in front of the camera (Z>0).

    Convention: pose_w_c maps camera-frame points to world. To go world→camera
    we invert it. For a rigid SE(3) transform that's R.T applied to (P - t).
    """
    R_wc = pose_w_c[..., :3, :3]                    # (..., 3, 3)
    t_wc = pose_w_c[..., :3, 3]                     # (..., 3)
    # camera-from-world: rotate by R_wc^T, then subtract camera origin
    P_centered = world_points - t_wc.unsqueeze(-2)  # (..., N, 3)
    # Apply R_wc^T: (... 3 3) @ (... N 3)^T → (... 3 N) → transpose to (... N 3)
    P_cam = torch.einsum("...ij,...nj->...ni", R_wc.transpose(-2, -1), P_centered)
    z = P_cam[..., 2]                                # (..., N)
    in_front = z > 1e-6
    z_safe = z.clamp_min(1e-6)
    # Pinhole: pixel = K @ (P_cam / z)
    P_norm = P_cam / z_safe.unsqueeze(-1)            # (..., N, 3), homogeneous = (x/z, y/z, 1)
    # K @ P_norm^T → (..., 3, N) → transpose → (..., N, 3), take first 2 dims
    pixels_h = torch.einsum("...ij,...nj->...ni", K, P_norm)
    pixels = pixels_h[..., :2]
    return pixels, in_front


def umeyama_sim3(P: np.ndarray, Q: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Sim(3) alignment of point set P to Q (Umeyama 1991).

    Solves min_{s, R, t} || s*R @ P_i + t - Q_i ||^2. Standard for SLAM /
    visual-odometry trajectory evaluation, which only recovers pose up to a
    similarity transform.

    Args:
        P, Q: (N, 3) point sets, N >= 3.
    Returns:
        (s, R, t) where R is (3, 3), t is (3,), s is scalar.
    """
    assert P.shape == Q.shape and P.shape[1] == 3 and P.shape[0] >= 3
    n = P.shape[0]
    mu_p = P.mean(axis=0)
    mu_q = Q.mean(axis=0)
    Pc = P - mu_p
    Qc = Q - mu_q
    H = Pc.T @ Qc / n
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = (U @ S @ Vt).T
    var_p = (Pc ** 2).sum() / n
    s = (D * np.diag(S)).sum() / max(var_p, 1e-12)
    t = mu_q - s * (R @ mu_p)
    return float(s), R, t


def absolute_translation_error(
    pred_poses: ArrayLike, gt_poses: ArrayLike, align: str = "sim3"
) -> dict:
    """Absolute Trajectory Error after Sim(3) alignment.

    Args:
        pred_poses, gt_poses: (N, 4, 4) world-from-camera transforms.
        align: 'sim3' (default), 'se3', or 'none'.
    Returns dict with ate_rmse_m, ate_mean_m, ate_median_m and aligned trajectory.
    """
    p = _to_numpy(pred_poses)[..., :3, 3]   # (N, 3)
    g = _to_numpy(gt_poses)[..., :3, 3]
    if align == "sim3":
        s, R, t = umeyama_sim3(p, g)
        aligned = (s * (R @ p.T)).T + t
    elif align == "se3":
        # Force scale=1 — translation + rotation only.
        s, R, t = umeyama_sim3(p, g)
        aligned = ((R @ p.T)).T + t
    elif align == "none":
        aligned = p
    else:
        raise ValueError(f"unknown align mode {align!r}")
    err = np.linalg.norm(aligned - g, axis=1)
    return {
        "ate_rmse_m": float(np.sqrt(np.mean(err ** 2))),
        "ate_mean_m": float(err.mean()),
        "ate_median_m": float(np.median(err)),
        "n_frames": int(p.shape[0]),
        "align": align,
        "aligned_trajectory_m": aligned,
    }


def relative_pose_error(
    pred_poses: ArrayLike, gt_poses: ArrayLike, delta: int = 1
) -> dict:
    """Relative Pose Error (translation + rotation) over a window of `delta` frames.

    For each i: rel_pred = pred_i^-1 @ pred_{i+delta}; rel_gt analogously.
    Error transform = rel_gt^-1 @ rel_pred. Returns RMSE of translation (m)
    and rotation (rad).
    """
    p = _to_numpy(pred_poses)
    g = _to_numpy(gt_poses)
    assert p.shape == g.shape and p.shape[-2:] == (4, 4) and delta >= 1
    n = p.shape[0]
    e_trans, e_rot = [], []
    for i in range(n - delta):
        rel_pred = np.linalg.inv(p[i]) @ p[i + delta]
        rel_gt = np.linalg.inv(g[i]) @ g[i + delta]
        diff = np.linalg.inv(rel_gt) @ rel_pred
        e_trans.append(float(np.linalg.norm(diff[:3, 3])))
        cos = (np.trace(diff[:3, :3]) - 1.0) / 2.0
        e_rot.append(float(np.arccos(np.clip(cos, -1.0, 1.0))))
    et = np.asarray(e_trans)
    er = np.asarray(e_rot)
    return {
        "rpe_trans_rmse_m": float(np.sqrt(np.mean(et ** 2))),
        "rpe_rot_rmse_deg": float(np.degrees(np.sqrt(np.mean(er ** 2)))),
        "rpe_trans_mean_m": float(et.mean()),
        "rpe_rot_mean_deg": float(np.degrees(er.mean())),
        "delta_frames": int(delta),
        "n_pairs": int(len(et)),
    }


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
