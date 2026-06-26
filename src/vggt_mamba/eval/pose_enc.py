"""9-dim pose encoding ↔ SE(3) + intrinsics (mirrors VGGT's pose_enc convention).

Pose enc layout (per camera):
    [0:3]  translation t in cam-from-world frame (NOT camera world position)
    [3:7]  unit quaternion (qx, qy, qz, qw) of cam-from-world rotation
    [7:9]  field of view in radians (fov_h, fov_w)

Convention: OpenCV camera (x-right, y-down, z-forward), cam-from-world
extrinsics, principal point assumed at image center.

All tensors are float; we don't track gradients through the inversion path
during eval. At train time, use `world_from_cam_to_pose_enc` to build the
GT target from TUM's world-from-cam matrices.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------- quaternion <-> rotation matrix ----------

def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) → (..., 3, 3). Input order: (qx, qy, qz, qw)."""
    qx, qy, qz, qw = q.unbind(-1)
    norm = (qx * qx + qy * qy + qz * qz + qw * qw).clamp(min=1e-12)
    s = 2.0 / norm
    xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
    xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
    wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
    R = torch.stack([
        1 - (yy + zz), xy - wz, xz + wy,
        xy + wz, 1 - (xx + zz), yz - wx,
        xz - wy, yz + wx, 1 - (xx + yy),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


def mat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) → (..., 4) as (qx, qy, qz, qw). Numerically-stable branched form."""
    # Use the standard Shepperd / branched method.
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    trace = m00 + m11 + m22

    cond_trace = trace > 0
    cond_x = (~cond_trace) & (m00 > m11) & (m00 > m22)
    cond_y = (~cond_trace) & (~cond_x) & (m11 > m22)
    # else: cond_z

    eps = 1e-8
    s_trace = torch.sqrt(trace.clamp(min=eps) + 1.0) * 2
    qw_t = 0.25 * s_trace
    qx_t = (m21 - m12) / s_trace
    qy_t = (m02 - m20) / s_trace
    qz_t = (m10 - m01) / s_trace

    s_x = torch.sqrt((1 + m00 - m11 - m22).clamp(min=eps)) * 2
    qw_x = (m21 - m12) / s_x
    qx_x = 0.25 * s_x
    qy_x = (m01 + m10) / s_x
    qz_x = (m02 + m20) / s_x

    s_y = torch.sqrt((1 + m11 - m00 - m22).clamp(min=eps)) * 2
    qw_y = (m02 - m20) / s_y
    qx_y = (m01 + m10) / s_y
    qy_y = 0.25 * s_y
    qz_y = (m12 + m21) / s_y

    s_z = torch.sqrt((1 + m22 - m00 - m11).clamp(min=eps)) * 2
    qw_z = (m10 - m01) / s_z
    qx_z = (m02 + m20) / s_z
    qy_z = (m12 + m21) / s_z
    qz_z = 0.25 * s_z

    qx = torch.where(cond_trace, qx_t, torch.where(cond_x, qx_x, torch.where(cond_y, qx_y, qx_z)))
    qy = torch.where(cond_trace, qy_t, torch.where(cond_x, qy_x, torch.where(cond_y, qy_y, qy_z)))
    qz = torch.where(cond_trace, qz_t, torch.where(cond_x, qz_x, torch.where(cond_y, qz_y, qz_z)))
    qw = torch.where(cond_trace, qw_t, torch.where(cond_x, qw_x, torch.where(cond_y, qw_y, qw_z)))
    return F.normalize(torch.stack([qx, qy, qz, qw], dim=-1), dim=-1)


# ---------- pose enc encode / decode ----------

def world_from_cam_to_pose_enc(
    pose_w_c: torch.Tensor, fov_h: torch.Tensor, fov_w: torch.Tensor
) -> torch.Tensor:
    """TUM-style world-from-cam 4×4 + FoV → 9-d pose enc (cam-from-world flavor).

    Args:
        pose_w_c: (..., 4, 4) world-from-cam
        fov_h, fov_w: (..., ) field of view in radians
    Returns:
        (..., 9) pose encoding [t_cw, quat_cw, fov_h, fov_w]
    """
    # cam-from-world = inverse of world-from-cam
    R_wc = pose_w_c[..., :3, :3]
    t_wc = pose_w_c[..., :3, 3]
    R_cw = R_wc.transpose(-1, -2)
    t_cw = -torch.einsum("...ij,...j->...i", R_cw, t_wc)
    quat_cw = mat_to_quat(R_cw)
    fov = torch.stack([fov_h, fov_w], dim=-1)
    return torch.cat([t_cw, quat_cw, fov], dim=-1)


def pose_enc_to_world_from_cam(pose_enc: torch.Tensor) -> torch.Tensor:
    """9-d pose enc → (..., 4, 4) world-from-cam matrix.

    Inverts the cam-from-world encoding back to world-from-cam (which is what
    TUM provides and what most viz pipelines expect).
    """
    t_cw = pose_enc[..., :3]
    q = pose_enc[..., 3:7]
    R_cw = quat_to_mat(F.normalize(q, dim=-1))
    R_wc = R_cw.transpose(-1, -2)
    t_wc = -torch.einsum("...ij,...j->...i", R_wc, t_cw)
    T = torch.zeros(*pose_enc.shape[:-1], 4, 4, device=pose_enc.device, dtype=pose_enc.dtype)
    T[..., :3, :3] = R_wc
    T[..., :3, 3] = t_wc
    T[..., 3, 3] = 1.0
    return T


def pose_enc_to_intrinsics(
    pose_enc: torch.Tensor, image_size_hw: tuple[int, int]
) -> torch.Tensor:
    """Rebuild a 3×3 K from the FoV pair in pose_enc (PP at image center)."""
    H, W = image_size_hw
    fov_h = pose_enc[..., 7]
    fov_w = pose_enc[..., 8]
    fy = (H / 2.0) / torch.tan(fov_h / 2.0)
    fx = (W / 2.0) / torch.tan(fov_w / 2.0)
    K = torch.zeros(*pose_enc.shape[:-1], 3, 3, device=pose_enc.device, dtype=pose_enc.dtype)
    K[..., 0, 0] = fx
    K[..., 1, 1] = fy
    K[..., 0, 2] = W / 2
    K[..., 1, 2] = H / 2
    K[..., 2, 2] = 1.0
    return K


def fov_from_intrinsics(K: torch.Tensor, image_size_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """K + image size → (fov_h, fov_w) in radians."""
    H, W = image_size_hw
    fov_h = 2 * torch.atan((H / 2.0) / K[..., 1, 1])
    fov_w = 2 * torch.atan((W / 2.0) / K[..., 0, 0])
    return fov_h, fov_w


if __name__ == "__main__":
    # Round-trip sanity: pose_w_c -> enc -> pose_w_c should be identity.
    torch.manual_seed(0)
    R = torch.linalg.qr(torch.randn(3, 3))[0]
    if torch.det(R) < 0:
        R[:, 0] *= -1
    t = torch.randn(3)
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    fov = torch.tensor([math.radians(60), math.radians(80)])
    enc = world_from_cam_to_pose_enc(T.unsqueeze(0), fov[0:1], fov[1:2])
    T_back = pose_enc_to_world_from_cam(enc).squeeze(0)
    err = (T_back - T).abs().max()
    print(f"[pose_enc] round-trip max err: {err.item():.2e} (should be < 1e-5)")

    K_back = pose_enc_to_intrinsics(enc, (480, 640)).squeeze(0)
    print(f"[pose_enc] K from enc:\n{K_back}")
