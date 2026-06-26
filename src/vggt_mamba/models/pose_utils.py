"""Pose math used by TerraWM (delta cam supervision + external integrator).

Conventions:
  - pose_w_c: world-from-camera (TUM convention). The R columns are camera
    basis vectors expressed in world; t is the camera origin in world.
  - relative_motion[t]: motion from frame t-1 to frame t in CAMERA frame.
    Computed as T_{t-1}^{-1} @ T_t.  Frame 0's relative motion is identity.
  - 7-vector representation: [tx, ty, tz, qx, qy, qz, qw].
"""
from __future__ import annotations

import torch


def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion (qx, qy, qz, qw) → 3x3 rotation matrix.
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


def rot_to_quat(R: torch.Tensor) -> torch.Tensor:
    """3x3 rotation matrix → unit quaternion (qx, qy, qz, qw).

    Numerically stable branch-based formulation. Vectorized over leading dims.
    R: (..., 3, 3)  →  q: (..., 4)
    """
    *batch, _, _ = R.shape
    m = R.reshape(-1, 3, 3)
    m00, m01, m02 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    m10, m11, m12 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    m20, m21, m22 = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    tr = m00 + m11 + m22
    N = m.shape[0]
    out = torch.zeros(N, 4, device=R.device, dtype=R.dtype)

    cond0 = tr > 0
    cond1 = (~cond0) & (m00 > m11) & (m00 > m22)
    cond2 = (~cond0) & (~cond1) & (m11 > m22)
    cond3 = (~cond0) & (~cond1) & (~cond2)

    # Branch 0: tr > 0
    s0 = torch.sqrt(tr.clamp_min(0) + 1.0) * 2
    if cond0.any():
        s = s0[cond0]
        out[cond0, 3] = 0.25 * s
        out[cond0, 0] = (m21[cond0] - m12[cond0]) / s
        out[cond0, 1] = (m02[cond0] - m20[cond0]) / s
        out[cond0, 2] = (m10[cond0] - m01[cond0]) / s
    # Branch 1: m00 largest
    if cond1.any():
        s = torch.sqrt((1.0 + m00[cond1] - m11[cond1] - m22[cond1]).clamp_min(1e-12)) * 2
        out[cond1, 3] = (m21[cond1] - m12[cond1]) / s
        out[cond1, 0] = 0.25 * s
        out[cond1, 1] = (m01[cond1] + m10[cond1]) / s
        out[cond1, 2] = (m02[cond1] + m20[cond1]) / s
    # Branch 2: m11 largest
    if cond2.any():
        s = torch.sqrt((1.0 + m11[cond2] - m00[cond2] - m22[cond2]).clamp_min(1e-12)) * 2
        out[cond2, 3] = (m02[cond2] - m20[cond2]) / s
        out[cond2, 0] = (m01[cond2] + m10[cond2]) / s
        out[cond2, 1] = 0.25 * s
        out[cond2, 2] = (m12[cond2] + m21[cond2]) / s
    # Branch 3: m22 largest
    if cond3.any():
        s = torch.sqrt((1.0 + m22[cond3] - m00[cond3] - m11[cond3]).clamp_min(1e-12)) * 2
        out[cond3, 3] = (m10[cond3] - m01[cond3]) / s
        out[cond3, 0] = (m02[cond3] + m20[cond3]) / s
        out[cond3, 1] = (m12[cond3] + m21[cond3]) / s
        out[cond3, 2] = 0.25 * s
    # Normalize
    out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return out.reshape(*batch, 4)


def pose_w_c_to_T(t: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """(t, q) → 4x4 transform. t: (..., 3), q: (..., 4). Returns (..., 4, 4)."""
    R = quat_to_rot(q)
    *batch, _ = t.shape
    T = torch.zeros(*batch, 4, 4, device=t.device, dtype=t.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def T_to_pose_w_c(T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """4x4 transform → (t, q).  T: (..., 4, 4)."""
    return T[..., :3, 3], rot_to_quat(T[..., :3, :3])


def gt_relative_motion_from_abs_poses(poses_w_c: torch.Tensor) -> torch.Tensor:
    """(B, T, 4, 4) world-from-cam → (B, T, 7) per-frame relative motion.

    Output[b, t] is the camera-frame motion from frame t-1 to frame t,
    i.e., T_{t-1}^{-1} @ T_t in 7-vector form.  Frame 0's entry is identity
    (Δt=0, Δq=(0,0,0,1)).
    """
    B, T = poses_w_c.shape[:2]
    # Identity-shift the previous-pose stack.
    prev = poses_w_c.clone()
    prev[:, 0] = torch.eye(4, device=poses_w_c.device, dtype=poses_w_c.dtype)
    prev[:, 1:] = poses_w_c[:, :-1]
    # Inverse of SE(3): R^T, -R^T t. Avoids generic 4x4 inverse.
    R_prev = prev[..., :3, :3]
    t_prev = prev[..., :3, 3]
    inv_R = R_prev.transpose(-2, -1)
    inv_t = -torch.einsum("...ij,...j->...i", inv_R, t_prev)
    inv_T = torch.zeros_like(prev)
    inv_T[..., :3, :3] = inv_R
    inv_T[..., :3, 3] = inv_t
    inv_T[..., 3, 3] = 1.0
    # Relative motion
    rel_T = inv_T @ poses_w_c                                              # (B, T, 4, 4)
    # For frame 0, force exact identity (the construction above also yields
    # identity, but numerical noise is possible; this guarantees zero).
    eye = torch.eye(4, device=poses_w_c.device, dtype=poses_w_c.dtype).expand(B, 4, 4)
    rel_T[:, 0] = eye
    rel_t = rel_T[..., :3, 3]
    rel_q = rot_to_quat(rel_T[..., :3, :3])
    return torch.cat([rel_t, rel_q], dim=-1)                                # (B, T, 7)


def integrate_deltas_to_absolute(
    pred_deltas: torch.Tensor,      # (T, 7)  per-frame (Δt, Δq)
    initial_pose: torch.Tensor,     # (4, 4)  starting world-from-camera (e.g. identity or GT[0])
) -> torch.Tensor:
    """External integrator for inference: accumulate per-frame delta poses
    into a stream of absolute world-from-camera poses.

    T_world[0] = initial_pose
    T_world[t] = T_world[t-1] @ delta_T[t]  for t >= 1
    The delta at frame 0 is typically identity (and the model is supervised to
    output identity at frame 0). It is ignored here; we start from initial_pose.

    Returns: (T, 4, 4) absolute world-from-camera poses.
    """
    T = pred_deltas.shape[0]
    abs_T = torch.zeros(T, 4, 4, device=pred_deltas.device, dtype=pred_deltas.dtype)
    abs_T[0] = initial_pose
    for t in range(1, T):
        dt = pred_deltas[t, :3]
        dq = pred_deltas[t, 3:7] / pred_deltas[t, 3:7].norm().clamp_min(1e-12)
        delta_T = pose_w_c_to_T(dt, dq)
        abs_T[t] = abs_T[t - 1] @ delta_T
    return abs_T


if __name__ == "__main__":
    # Sanity tests.
    torch.manual_seed(0)
    # Roundtrip quat ↔ rot.
    q = torch.randn(5, 4); q = q / q.norm(dim=-1, keepdim=True)
    R = quat_to_rot(q)
    q2 = rot_to_quat(R)
    # Sign ambiguity: q and -q encode same rotation; canonicalize by qw≥0.
    q = q * q[..., 3:4].sign()
    q2 = q2 * q2[..., 3:4].sign()
    print(f"quat roundtrip max err: {(q - q2).abs().max():.2e}")

    # Identity for relative motion from identical poses.
    poses = torch.eye(4).expand(2, 8, 4, 4).contiguous()
    rel = gt_relative_motion_from_abs_poses(poses)
    print(f"identity relative-motion translation max: {rel[..., :3].abs().max():.2e}")
    print(f"identity relative-motion quat last comp min (should be 1): {rel[..., 6].min():.4f}")

    # Integrator round-trip: random deltas, integrate, check chain consistency.
    deltas = torch.zeros(5, 7); deltas[:, :3] = torch.randn(5, 3) * 0.1
    deltas[:, 3:7] = torch.tensor([0, 0, 0, 1.0]).expand(5, 4)
    abs_T = integrate_deltas_to_absolute(deltas, torch.eye(4))
    print(f"integrator: final position after 5 deltas: {abs_T[-1, :3, 3].tolist()}")
    print(f"  (expected sum of deltas: {deltas[1:, :3].sum(0).tolist()})")
