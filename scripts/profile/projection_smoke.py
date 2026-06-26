"""Smoke test for the consistency-loss projection math.

Verifies project_points_to_pixels against hand-computed projections, before any
training is launched against it. A sign error or convention flip in this
function would silently invalidate the entire experiment-2 anchor pool result.

Tests:
  1. Identity pose, point in front of camera → trivially-computable pixel.
  2. Translated camera (5cm right) → known pixel shift.
  3. Pose offset → expected pixel error ≈ (offset_m × focal_length / depth_m).
  4. TUM-realistic intrinsics + GT pose from a real frame → projection of a
     point at predicted depth should fall near the patch's pixel.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt_mamba.eval.metrics import project_points_to_pixels  # noqa: E402


def case(label):
    print(f"\n[case] {label}")


def main():
    # ---------- Case 1: identity pose ----------
    case("identity pose, point on optical axis at depth 5m")
    K = torch.tensor([[500., 0., 256.],
                      [0., 500., 256.],
                      [0., 0., 1.]])
    pose = torch.eye(4)
    P_world = torch.tensor([[0., 0., 5.]])   # point at (0, 0, 5)
    px, in_front = project_points_to_pixels(P_world, pose, K)
    expected = torch.tensor([[256., 256.]])  # cx, cy (point on optical axis)
    print(f"  projected: {px.tolist()}, expected: {expected.tolist()}")
    assert torch.allclose(px, expected, atol=1e-4), f"FAIL: {px} vs {expected}"
    assert bool(in_front[0]), f"in_front should be True"
    print("  PASS")

    # ---------- Case 2: point off-axis at known location ----------
    case("identity pose, point at (1, 0, 5) → x pixel should shift by fx*1/5 = 100")
    P_world = torch.tensor([[1., 0., 5.]])
    px, _ = project_points_to_pixels(P_world, pose, K)
    expected = torch.tensor([[256. + 500. * 1.0 / 5.0, 256.]])  # = 356, 256
    print(f"  projected: {px.tolist()}, expected: {expected.tolist()}")
    assert torch.allclose(px, expected, atol=1e-4), f"FAIL"
    print("  PASS")

    # ---------- Case 3: translated camera ----------
    case("camera translated +x by 1m, point at world (0, 0, 5)")
    pose_t = torch.eye(4)
    pose_t[0, 3] = 1.0   # camera origin at world (1, 0, 0)
    # world point (0,0,5): in camera frame it's (-1, 0, 5). Pixel: cx - fx/5 = 156
    P_world = torch.tensor([[0., 0., 5.]])
    px, _ = project_points_to_pixels(P_world, pose_t, K)
    expected = torch.tensor([[256. - 100., 256.]])
    print(f"  projected: {px.tolist()}, expected: {expected.tolist()}")
    assert torch.allclose(px, expected, atol=1e-4), f"FAIL"
    print("  PASS")

    # ---------- Case 4: rotated camera (90° about y) ----------
    case("camera rotated 90° about world-y (looking down +x), point at world (5, 0, 0)")
    # R_wc rotates camera-basis to world. Camera's +z axis (look direction) should
    # point along world +x after rotation. R_wc such that R_wc @ ẑ_cam = x̂_world.
    # 90° rotation about y: cos(90)=0, sin(90)=1
    # R_y(90) = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    pose_r = torch.eye(4)
    pose_r[:3, :3] = torch.tensor([[0., 0., 1.],
                                   [0., 1., 0.],
                                   [-1., 0., 0.]])
    P_world = torch.tensor([[5., 0., 0.]])
    px, _ = project_points_to_pixels(P_world, pose_r, K)
    # In camera frame: P_cam = R_wc^T @ (P_world - t_wc) = R_y(-90) @ (5,0,0)
    # R_y(-90) = [[0,0,-1],[0,1,0],[1,0,0]], so P_cam = (0, 0, 5).
    # Pixel: (256, 256).
    expected = torch.tensor([[256., 256.]])
    print(f"  projected: {px.tolist()}, expected: {expected.tolist()}")
    assert torch.allclose(px, expected, atol=1e-4), f"FAIL"
    print("  PASS")

    # ---------- Case 5: pose offset → expected pixel error ----------
    case("pose offset by 5cm in x → expected pixel error ≈ fx * 0.05 / depth_m")
    depth = 3.0
    P_world = torch.tensor([[0., 0., depth]])
    pose_gt = torch.eye(4)
    pose_drift = torch.eye(4)
    pose_drift[0, 3] = 0.05   # camera origin shifted by 5cm
    px_gt, _ = project_points_to_pixels(P_world, pose_gt, K)
    px_drift, _ = project_points_to_pixels(P_world, pose_drift, K)
    err = (px_gt - px_drift).abs()
    expected_err = 500. * 0.05 / depth   # ≈ 8.33 pixels
    print(f"  gt pixel:    {px_gt.tolist()}")
    print(f"  drift pixel: {px_drift.tolist()}")
    print(f"  pixel error: {err.tolist()}, expected ~{expected_err:.2f} px in x")
    assert abs(err[0, 0].item() - expected_err) < 0.1, f"FAIL: x error {err[0,0]}"
    assert err[0, 1].item() < 1e-4, f"FAIL: y error should be ~0"
    print("  PASS")

    # ---------- Case 6: TUM-realistic intrinsics (Freiburg3) at img_size=512 ----------
    case("TUM-Freiburg3 intrinsics at 512x512, depth 2m, 1cm offset")
    fx_orig, fy_orig = 535.4, 539.2
    cx_orig, cy_orig = 320.1, 247.6
    sx, sy = 512. / 640., 512. / 480.
    K_tum = torch.tensor([[fx_orig * sx, 0., cx_orig * sx],
                          [0., fy_orig * sy, cy_orig * sy],
                          [0., 0., 1.]])
    P_world = torch.tensor([[0., 0., 2.]])
    pose_gt = torch.eye(4)
    pose_drift = torch.eye(4)
    pose_drift[0, 3] = 0.01   # 1 cm
    px_gt, _ = project_points_to_pixels(P_world, pose_gt, K_tum)
    px_drift, _ = project_points_to_pixels(P_world, pose_drift, K_tum)
    err_x = (px_gt - px_drift)[0, 0].abs().item()
    expected = (fx_orig * sx) * 0.01 / 2.0   # ≈ 2.14 px
    print(f"  pixel error: {err_x:.3f} px, expected ~{expected:.3f} px")
    assert abs(err_x - expected) < 0.05, f"FAIL"
    print("  PASS")

    # ---------- Case 7: batched (B=2, N=4 points each) ----------
    case("batched inputs (B=2, N=4)")
    K_b = K.unsqueeze(0).expand(2, 3, 3)
    pose_b = torch.eye(4).unsqueeze(0).expand(2, 4, 4)
    Pw = torch.tensor([[[1., 0., 5.], [0., 1., 5.], [-1., 0., 5.], [0., 0., 10.]],
                       [[2., 0., 5.], [0., 0., 5.], [0., -2., 5.], [0., 0., 5.]]])
    px, in_front = project_points_to_pixels(Pw, pose_b, K_b)
    print(f"  shape: {tuple(px.shape)} (expect (2, 4, 2))")
    print(f"  in_front: {in_front.tolist()}")
    assert px.shape == (2, 4, 2), f"FAIL shape"
    # Spot check: batch 0, point 0 = (1, 0, 5) → (cx + fx/5, cy) = (356, 256)
    assert torch.allclose(px[0, 0], torch.tensor([356., 256.]), atol=1e-4)
    print("  PASS")

    # ---------- Case 8: behind-camera point ----------
    case("point behind camera (z<0) → in_front False")
    P_world = torch.tensor([[0., 0., -1.]])
    px, in_front = project_points_to_pixels(P_world, torch.eye(4), K)
    print(f"  in_front: {in_front.tolist()}, projected (sanity, ignore): {px.tolist()}")
    assert not bool(in_front[0]), f"FAIL: should be behind"
    print("  PASS")

    print("\n[smoke] all 8 cases PASS — projection math verified.")


if __name__ == "__main__":
    main()
