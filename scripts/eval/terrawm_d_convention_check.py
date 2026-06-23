"""Convention / data-pipeline sanity checks — NO model forward, NO learned heads.

Tests whether the data + coordinate transformations the pipeline performs are
internally consistent. Twelve separate viz, each saved as one PNG, every
intermediate value printed to stdout for cross-checking. If anything is
inconsistent here, the model has been getting wrong data the whole time and
no amount of architecture change can fix it.

Output: viz/output/terrawm_d_convention_check/
  01_rgb_loading.png
  02_depth_loading.png
  03_intrinsics_scaling.png
  04_tum_pose_convention.png
  05_quaternion_convention.png
  06_pose_w_c_to_T_roundtrip.png
  07_gt_relative_motion.png
  08_build_rays_from_pose.png
  09_backproject_vs_buildrays.png
  10_depth_direction.png
  11_voxel_grid_coords.png
  12_cam9_roundtrip.png

Plus stdout dumps for every check so the user can sanity-read the numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt_mamba.data.tum_rgbd import sync_sequence, intrinsics_for                 # noqa: E402
from vggt_mamba.models.aggregators.anchor_pool import (                              # noqa: E402
    cam9_to_pose_w_c, build_patch_pixel_grid,
)
from vggt_mamba.models.terrawm_d import _pose_T_to_cam9                              # noqa: E402
from vggt_mamba.models.pose_utils import (                                           # noqa: E402
    pose_w_c_to_T, T_to_pose_w_c, gt_relative_motion_from_abs_poses,
)
from vggt_mamba.models.voxel_grid import (                                          # noqa: E402
    VoxelGridConfig, init_voxel_state, world_to_grid_coords,
    trilinear_sample_grid, build_rays_from_pose,
    backproject_patches_to_world,
)


OUT_DIR = Path("viz/output/terrawm_d_convention_check")
SEQ = "rgbd_dataset_freiburg2_desk"
DATA_ROOT = Path("/workspace/datasets/tum_rgbd")
FRAME = 100
IMG_SIZE = 512                                                                       # dinov3 setting


def banner(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def check_01_rgb_loading():
    banner("01 — RGB loading")
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = sync_sequence(DATA_ROOT / SEQ)
    rec = recs[FRAME]
    raw = np.asarray(Image.open(rec.rgb_path))                                       # H, W, 3 uint8
    print(f"raw PNG: shape={raw.shape} dtype={raw.dtype}")
    print(f"raw value range: [{raw.min()}, {raw.max()}]")
    print(f"raw mean R,G,B: ({raw[..., 0].mean():.1f}, {raw[..., 1].mean():.1f}, {raw[..., 2].mean():.1f})")

    # Reproduce model-side load (terrawm_d_regrounding_stream.load_rgb)
    pil = Image.open(rec.rgb_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    model_in = np.asarray(pil, dtype=np.float32) / 255.0                            # H, W, 3 in [0,1]
    print(f"model input: shape={model_in.shape} dtype={model_in.dtype}")
    print(f"model input range: [{model_in.min():.3f}, {model_in.max():.3f}]")
    print(f"model input mean R,G,B: ({model_in[..., 0].mean():.3f}, "
           f"{model_in[..., 1].mean():.3f}, {model_in[..., 2].mean():.3f})")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(raw)
    axes[0].set_title(f"raw PNG ({raw.shape[1]}×{raw.shape[0]}, uint8)\n"
                       f"mean RGB: ({raw[..., 0].mean():.0f}, {raw[..., 1].mean():.0f}, {raw[..., 2].mean():.0f})")
    axes[0].axis("off")
    axes[1].imshow(model_in)
    axes[1].set_title(f"model input (resized to {IMG_SIZE}×{IMG_SIZE}, float ∈ [0,1])\n"
                       f"mean RGB: ({model_in[..., 0].mean():.2f}, {model_in[..., 1].mean():.2f}, {model_in[..., 2].mean():.2f})")
    axes[1].axis("off")
    # Difference (after upscaling model_in back to raw size for comparison)
    from PIL import Image as PIL_Image
    model_back = np.asarray(PIL_Image.fromarray(
        (model_in * 255).astype(np.uint8)).resize((raw.shape[1], raw.shape[0]))
    )
    diff = np.abs(raw.astype(np.int16) - model_back.astype(np.int16)).astype(np.uint8)
    axes[2].imshow(diff, vmin=0, vmax=50)
    axes[2].set_title(f"|raw − model_input_upscaled|\n"
                       f"max diff = {diff.max()}, mean = {diff.mean():.1f}")
    axes[2].axis("off")
    fig.suptitle(f"01 — RGB loading: model sees {IMG_SIZE}×{IMG_SIZE} from PNG {raw.shape[1]}×{raw.shape[0]}. "
                  f"Catches: BGR/RGB swap, value-range bug, channel-order bug.")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_rgb_loading.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def check_02_depth_loading():
    banner("02 — Depth loading")
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = sync_sequence(DATA_ROOT / SEQ)
    rec = recs[FRAME]
    raw_depth = np.asarray(Image.open(rec.depth_path))
    print(f"raw depth PNG: shape={raw_depth.shape} dtype={raw_depth.dtype}")
    print(f"raw depth range: [{raw_depth.min()}, {raw_depth.max()}] (uint16)")
    print(f"raw depth nonzero count: {(raw_depth > 0).sum()} / {raw_depth.size} "
           f"({(raw_depth > 0).mean()*100:.1f}%)")
    # TUM convention: divide by 5000 → meters
    depth_m = raw_depth.astype(np.float32) / 5000.0
    depth_m_valid = depth_m[depth_m > 0]
    print(f"after /5000 (TUM convention): range [{depth_m.min():.4f}, {depth_m.max():.4f}] m")
    print(f"valid depth range (excluding zeros): [{depth_m_valid.min():.4f}, {depth_m_valid.max():.4f}] m")
    print(f"valid depth median: {np.median(depth_m_valid):.4f} m")
    # Resize to model size
    pil_resized = Image.open(rec.depth_path).resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    depth_resized = np.asarray(pil_resized, dtype=np.float32) / 5000.0
    print(f"after resize to {IMG_SIZE}×{IMG_SIZE} (nearest): shape {depth_resized.shape}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    im = axes[0, 0].imshow(raw_depth, cmap="turbo")
    axes[0, 0].set_title(f"raw uint16 PNG\nrange [{raw_depth.min()}, {raw_depth.max()}]")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)

    im = axes[0, 1].imshow(depth_m, cmap="turbo", vmin=0, vmax=8)
    axes[0, 1].set_title(f"after /5000 → meters\n"
                          f"valid range [{depth_m_valid.min():.2f}, {depth_m_valid.max():.2f}] m")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    im = axes[1, 0].imshow(depth_resized, cmap="turbo", vmin=0, vmax=8)
    axes[1, 0].set_title(f"resized to {IMG_SIZE}² (nearest, /5000)\n"
                          f"range [{depth_resized.min():.2f}, {depth_resized.max():.2f}] m")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    # Histogram
    axes[1, 1].hist(depth_m_valid.flatten(), bins=100, color="tab:blue", alpha=0.7,
                     label=f"raw / 5000 ({raw_depth.shape[1]}×{raw_depth.shape[0]})")
    axes[1, 1].hist(depth_resized[depth_resized > 0].flatten(), bins=100,
                     color="tab:red", alpha=0.5, label=f"resized {IMG_SIZE}²")
    axes[1, 1].set_xlabel("depth (m)")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].set_title("histogram of valid depths\n"
                          "(catches /5000 wrong, treating 0 as 0m vs invalid)")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    fig.suptitle(f"02 — Depth loading. TUM stores depth as uint16, dividing by 5000 → meters. "
                  f"Catches: wrong divisor, zero-as-invalid handling, resize interpolation bug.")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_depth_loading.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def check_03_intrinsics_scaling():
    banner("03 — Intrinsics scaling")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fx, fy, cx, cy = intrinsics_for(SEQ)
    print(f"TUM-published intrinsics for {SEQ} (640×480):")
    print(f"  fx={fx:.4f}  fy={fy:.4f}  cx={cx:.4f}  cy={cy:.4f}")
    sx = IMG_SIZE / 640.0
    sy = IMG_SIZE / 480.0
    print(f"\nResize scale factors: sx={sx:.4f}  sy={sy:.4f}")
    fx_scaled = fx * sx
    fy_scaled = fy * sy
    cx_scaled = cx * sx
    cy_scaled = cy * sy
    print(f"\nScaled intrinsics for {IMG_SIZE}×{IMG_SIZE}:")
    print(f"  fx={fx_scaled:.4f}  fy={fy_scaled:.4f}  cx={cx_scaled:.4f}  cy={cy_scaled:.4f}")
    print(f"\nK matrix used by model:")
    K_used = np.array([[fx_scaled, 0, cx_scaled],
                        [0, fy_scaled, cy_scaled],
                        [0, 0, 1]])
    print(K_used)
    # Note: TUM 640×480 with cx≈319.5, cy≈239.5; principal-point-near-image-center.
    # After scale, cx_scaled should be near IMG_SIZE/2.
    print(f"\nSanity: cx_scaled should be near IMG_SIZE/2 = {IMG_SIZE/2:.1f} (got {cx_scaled:.1f})")
    print(f"        cy_scaled should be near IMG_SIZE/2 = {IMG_SIZE/2:.1f} (got {cy_scaled:.1f})")

    # Overlay patch pixel grid on a known image
    patch_pixel = build_patch_pixel_grid(32, 32, IMG_SIZE, device="cpu").numpy()      # (32*32, 2)
    print(f"\nPatch pixel grid: shape {patch_pixel.shape}")
    print(f"  x range: [{patch_pixel[:, 0].min():.1f}, {patch_pixel[:, 0].max():.1f}]")
    print(f"  y range: [{patch_pixel[:, 1].min():.1f}, {patch_pixel[:, 1].max():.1f}]")

    from PIL import Image
    recs = sync_sequence(DATA_ROOT / SEQ)
    pil = Image.open(recs[FRAME].rgb_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(pil)
    axes[0].scatter(cx_scaled, cy_scaled, color="red", s=200, marker="+", linewidths=3,
                     label=f"principal point ({cx_scaled:.0f}, {cy_scaled:.0f})")
    axes[0].axhline(cy_scaled, color="red", alpha=0.3)
    axes[0].axvline(cx_scaled, color="red", alpha=0.3)
    axes[0].set_title(f"principal point on image\nIMG_SIZE={IMG_SIZE}, image center={IMG_SIZE/2:.0f}")
    axes[0].set_xlim(0, IMG_SIZE); axes[0].set_ylim(IMG_SIZE, 0)
    axes[0].legend()

    axes[1].imshow(pil, alpha=0.5)
    axes[1].scatter(patch_pixel[:, 0], patch_pixel[:, 1], color="lime", s=4)
    axes[1].set_title(f"patch pixel grid (32×32 = 1024 points)\n"
                       f"x∈[{patch_pixel[:, 0].min():.0f},{patch_pixel[:, 0].max():.0f}], "
                       f"y∈[{patch_pixel[:, 1].min():.0f},{patch_pixel[:, 1].max():.0f}]")
    axes[1].set_xlim(0, IMG_SIZE); axes[1].set_ylim(IMG_SIZE, 0)
    fig.suptitle(f"03 — Intrinsics scaling. Catches: fx/fy swap, cx/cy swap, scale factor wrong, "
                  f"pixel grid origin (top-left vs bottom-left).")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_intrinsics_scaling.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def check_04_tum_pose_convention():
    banner("04 — TUM pose convention (raw)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    recs = sync_sequence(DATA_ROOT / SEQ)
    rec = recs[FRAME]
    pose = rec.pose_w_c                                                              # (4, 4)
    print(f"rec.pose_w_c (4x4) for frame {FRAME}:")
    print(pose)
    print(f"\ndet(R) = {np.linalg.det(pose[:3, :3]):.6f}  (should be +1)")
    print(f"R^T @ R - I (should be ~0):")
    print(pose[:3, :3].T @ pose[:3, :3] - np.eye(3))
    print(f"\ntranslation (t) = {pose[:3, 3]}")
    print(f"\nbottom row = {pose[3]}  (should be [0, 0, 0, 1])")
    print(f"\nTUM ground-truth file format:")
    print(f"  timestamp tx ty tz qx qy qz qw  (qx,qy,qz,qw — NOT wxyz)")
    print(f"  pose_w_c[:, 3] columns are world-from-camera direction vectors")
    print(f"  i.e. R columns = camera basis vectors expressed in world frame")

    R = pose[:3, :3]
    t = pose[:3, 3]
    # Camera basis vectors in world: R @ e_i for i = x, y, z
    cam_x = R @ np.array([1, 0, 0])
    cam_y = R @ np.array([0, 1, 0])
    cam_z = R @ np.array([0, 0, 1])
    print(f"\nCamera basis vectors in world frame (R columns):")
    print(f"  cam_x (right):   {cam_x}")
    print(f"  cam_y (down):    {cam_y}")
    print(f"  cam_z (forward): {cam_z}")
    print(f"\nIf TUM uses standard cam convention: +x right, +y down, +z forward.")

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    # Draw a few consecutive frames as camera frustums.
    for fi in range(FRAME, min(FRAME + 50, len(recs)), 10):
        p = recs[fi].pose_w_c
        R_i, t_i = p[:3, :3], p[:3, 3]
        scale = 0.3
        for vec, color, lbl in [(R_i[:, 0] * scale, "red", "cam_x"),
                                  (R_i[:, 1] * scale, "green", "cam_y"),
                                  (R_i[:, 2] * scale, "blue", "cam_z")]:
            ax.quiver(t_i[0], t_i[1], t_i[2], vec[0], vec[1], vec[2],
                       color=color, arrow_length_ratio=0.3, linewidth=2)
    # Single frame at FRAME for legend
    p = recs[FRAME].pose_w_c
    ax.scatter(p[0, 3], p[1, 3], p[2, 3], color="black", s=80, label=f"frame {FRAME}")
    ax.set_xlabel("World X"); ax.set_ylabel("World Y"); ax.set_zlabel("World Z")
    ax.set_title(f"Camera basis vectors in world frame\n"
                  f"red=cam_x, green=cam_y, blue=cam_z (every 10 frames)")
    ax.legend()

    # Print all relevant numbers in a text panel.
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    txt = (
        f"Frame {FRAME} pose_w_c (4×4):\n"
        f"{pose}\n\n"
        f"det(R) = {np.linalg.det(R):.6f}  (target +1)\n"
        f"R^T R deviation from I: max abs {np.max(np.abs(R.T @ R - np.eye(3))):.2e}\n\n"
        f"Translation t = {t}\n\n"
        f"Camera basis vectors in WORLD frame (R columns):\n"
        f"  cam_x = {cam_x}  (camera right-axis in world)\n"
        f"  cam_y = {cam_y}  (camera down-axis in world)\n"
        f"  cam_z = {cam_z}  (camera forward-axis in world)\n\n"
        f"Bottom row = {pose[3]}  (target [0, 0, 0, 1])\n\n"
        f"INTERPRETATION:\n"
        f"  pose_w_c maps points in camera-frame → world-frame:\n"
        f"      P_world = R · P_cam + t\n"
        f"  So t = position of camera origin in world,\n"
        f"     R columns = camera axes expressed in world frame.\n\n"
        f"TUM raw format: tx ty tz qx qy qz qw (xyzw, NOT wxyz)\n"
    )
    ax2.text(0.0, 1.0, txt, family="monospace", fontsize=8, verticalalignment="top",
              transform=ax2.transAxes)
    fig.suptitle(f"04 — TUM pose convention. Catches: w_c vs c_w mismatch, "
                  f"quat order, R transposed.")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_tum_pose_convention.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def check_05_quaternion_convention():
    banner("05 — Quaternion convention")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = sync_sequence(DATA_ROOT / SEQ)
    pose = recs[FRAME].pose_w_c
    R_from_tum = pose[:3, :3].copy()

    # Now: do we convert this same pose via the model's path and get the same R?
    # The model's path: pose_w_c → cam9 → cam9_to_pose_w_c (which is in anchor_pool).
    # cam9 format: tx,ty,tz, qx,qy,qz,qw, fovx,fovy
    # cam9_to_pose_w_c does: F.normalize(q) → quat_to_rot_matrix (Hamilton, xyzw)
    pose_t = torch.from_numpy(pose).float().unsqueeze(0)                             # (1, 4, 4)
    t, q = T_to_pose_w_c(pose_t)                                                     # uses pose_utils — this should match TUM convention
    print(f"T_to_pose_w_c output for frame {FRAME}:")
    print(f"  t = {t[0].numpy()}")
    print(f"  q = {q[0].numpy()}  (xyzw order)")
    print(f"  |q| = {q[0].norm().item():.6f}  (should be 1)")
    # Round-trip
    T_back = pose_w_c_to_T(t, q)
    print(f"\npose_w_c_to_T(t, q) reproduces:")
    print(T_back[0].numpy())
    err = np.max(np.abs(T_back[0].numpy() - pose))
    print(f"\nmax abs round-trip error: {err:.2e}")

    # cam9 path
    cam9 = torch.cat([t, q, torch.tensor([[1.0, 1.0]])], dim=-1)
    T_via_cam9 = cam9_to_pose_w_c(cam9)
    print(f"\ncam9_to_pose_w_c reproduces:")
    print(T_via_cam9[0].numpy())
    err_cam9 = np.max(np.abs(T_via_cam9[0].numpy() - pose))
    print(f"max abs cam9 round-trip error: {err_cam9:.2e}")

    R_via_pose_utils = T_back[0, :3, :3].numpy()
    R_via_cam9 = T_via_cam9[0, :3, :3].numpy()
    print(f"\nMax |R_from_TUM - R_via_pose_utils|: {np.max(np.abs(R_from_tum - R_via_pose_utils)):.2e}")
    print(f"Max |R_from_TUM - R_via_cam9|:       {np.max(np.abs(R_from_tum - R_via_cam9)):.2e}")

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis("off")
    txt = (
        f"05 — Quaternion convention round-trip @ frame {FRAME}\n"
        f"=" * 70 + "\n\n"
        f"TUM R (extracted from pose_w_c[:3,:3]):\n{R_from_tum}\n\n"
        f"Quaternion extracted via T_to_pose_w_c (xyzw): {q[0].numpy()}\n"
        f"|q| = {q[0].norm().item():.8f}  (target 1.0)\n\n"
        f"Pose round-trip via pose_w_c_to_T(t, q):\n{T_back[0].numpy()}\n"
        f"  max abs error vs TUM pose: {err:.2e}\n\n"
        f"Pose round-trip via cam9_to_pose_w_c (anchor_pool):\n{T_via_cam9[0].numpy()}\n"
        f"  max abs error vs TUM pose: {err_cam9:.2e}\n\n"
        f"R reproduction errors:\n"
        f"  pose_utils path: {np.max(np.abs(R_from_tum - R_via_pose_utils)):.2e}\n"
        f"  cam9 path:       {np.max(np.abs(R_from_tum - R_via_cam9)):.2e}\n\n"
        f"INTERPRETATION:\n"
        f"  Both errors should be ~1e-6. If one is large, that path has a\n"
        f"  quat convention bug (xyzw vs wxyz, Hamilton vs JPL, transposed R).\n"
    )
    ax.text(0.0, 1.0, txt, family="monospace", fontsize=9,
             verticalalignment="top", transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_quaternion_convention.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def check_06_pose_w_c_roundtrip():
    banner("06 — pose_w_c_to_T round-trip across many frames")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = sync_sequence(DATA_ROOT / SEQ)
    errs = []
    cam9_errs = []
    sample_idx = np.linspace(0, len(recs) - 1, 100).astype(int)
    for fi in sample_idx:
        pose = recs[fi].pose_w_c
        pose_t = torch.from_numpy(pose).float().unsqueeze(0)
        t, q = T_to_pose_w_c(pose_t)
        T_back = pose_w_c_to_T(t, q)
        errs.append(float(np.max(np.abs(T_back[0].numpy() - pose))))
        cam9 = torch.cat([t, q, torch.tensor([[1.0, 1.0]])], dim=-1)
        T_via_cam9 = cam9_to_pose_w_c(cam9)
        cam9_errs.append(float(np.max(np.abs(T_via_cam9[0].numpy() - pose))))

    errs = np.array(errs)
    cam9_errs = np.array(cam9_errs)
    print(f"pose_utils round-trip over {len(sample_idx)} frames:")
    print(f"  max err: {errs.max():.2e}  mean: {errs.mean():.2e}")
    print(f"cam9 round-trip:")
    print(f"  max err: {cam9_errs.max():.2e}  mean: {cam9_errs.mean():.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].plot(sample_idx, errs, "o-", color="tab:blue", markersize=3)
    axes[0].set_xlabel("frame")
    axes[0].set_ylabel("max abs round-trip error")
    axes[0].set_title(f"pose_utils round-trip: max {errs.max():.2e}")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3)
    axes[1].plot(sample_idx, cam9_errs, "o-", color="tab:red", markersize=3)
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("max abs round-trip error")
    axes[1].set_title(f"cam9 (anchor_pool) round-trip: max {cam9_errs.max():.2e}")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"06 — Pose encode/decode round-trip stability across the sequence")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_pose_w_c_to_T_roundtrip.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def check_07_gt_relative_motion():
    banner("07 — gt_relative_motion: what target the model is trained against")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = sync_sequence(DATA_ROOT / SEQ)
    # Take 8 consecutive frames
    poses = np.stack([r.pose_w_c for r in recs[FRAME:FRAME + 8]])
    poses_t = torch.from_numpy(poses).float().unsqueeze(0)                           # (1, 8, 4, 4)
    delta_7 = gt_relative_motion_from_abs_poses(poses_t)                             # (1, 8, 7)
    print(f"gt_relative_motion_from_abs_poses output shape: {delta_7.shape}")
    print(f"delta[0, t, :3] for t=0..7 (camera-frame translation deltas):")
    for t in range(8):
        print(f"  t={t}: t_delta = {delta_7[0, t, :3].numpy()}  |Δt|={float(delta_7[0, t, :3].norm()):.4f}m  "
               f"q = {delta_7[0, t, 3:7].numpy()}")

    # Cross-check: compute the world-frame translation delta directly
    world_t = poses[:, :3, 3]                                                        # (8, 3)
    world_dt = np.diff(world_t, axis=0)                                              # (7, 3)
    world_dt_mag = np.linalg.norm(world_dt, axis=-1)
    print(f"\nWORLD-frame translation deltas (poses[t][:3,3] - poses[t-1][:3,3]):")
    for t in range(7):
        print(f"  t={t+1}: dt_world = {world_dt[t]}  |dt_world|={world_dt_mag[t]:.4f}m")

    # Camera-frame delta is what gt_relative_motion returns. Confirm consistency.
    # Camera-frame delta from t-1 to t:
    #   T_delta = inv(pose[t-1]) @ pose[t]
    # Translation component = inv(pose[t-1]).R · (pose[t].t - pose[t-1].t)
    print(f"\nManual camera-frame delta translation (inv(prev) @ curr):")
    for t in range(1, 8):
        prev = poses[t - 1]
        curr = poses[t]
        T_delta = np.linalg.inv(prev) @ curr
        print(f"  t={t}: cam-frame dt = {T_delta[:3, 3]}  |dt|={np.linalg.norm(T_delta[:3, 3]):.4f}m")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(range(8), [float(delta_7[0, t, :3].norm()) for t in range(8)],
                     "o-", label="gt_relative_motion |Δt|")
    axes[0, 0].plot(range(1, 8), world_dt_mag, "x--", label="raw world Δt magnitude")
    axes[0, 0].set_xlabel("frame index (within window)")
    axes[0, 0].set_ylabel("|Δt| (m)")
    axes[0, 0].set_title("translation delta magnitudes\n(should be same magnitudes either way)")
    axes[0, 0].grid(alpha=0.3); axes[0, 0].legend()

    # Per-axis cam-frame vs world deltas
    for ax_idx in range(3):
        for t in range(1, 8):
            T_delta = np.linalg.inv(poses[t - 1]) @ poses[t]
            cam_dt = T_delta[:3, 3]
            world_dt_now = poses[t, :3, 3] - poses[t - 1, :3, 3]
            if t == 1:
                axes[0, 1].plot(t, cam_dt[0], "o", color="red", label="cam Δx" if ax_idx == 0 else None)
                axes[0, 1].plot(t, world_dt_now[0], "x", color="darkred", label="world Δx" if ax_idx == 0 else None)
    ax = axes[0, 1]
    for t in range(1, 8):
        T_delta = np.linalg.inv(poses[t - 1]) @ poses[t]
        cam_dt = T_delta[:3, 3]
        world_dt_now = poses[t, :3, 3] - poses[t - 1, :3, 3]
        for ax_idx, axn in enumerate("xyz"):
            color = ["red", "green", "blue"][ax_idx]
            ax.plot(t, cam_dt[ax_idx], "o", color=color)
            ax.plot(t, world_dt_now[ax_idx], "x", color=color)
    ax.set_title("per-axis Δt: o = cam-frame (model target), x = world-frame\n"
                  "red=x, green=y, blue=z")
    ax.set_xlabel("frame"); ax.set_ylabel("Δt (m)")
    ax.grid(alpha=0.3)

    # Text summary
    axes[1, 0].axis("off")
    summary = "gt_relative_motion (model's pose loss TARGET) at frames " + str(list(range(FRAME, FRAME + 8))) + "\n\n"
    for t in range(8):
        summary += f"  t={t}:  cam-frame Δt = {delta_7[0, t, :3].numpy()}\n"
    axes[1, 0].text(0.0, 1.0, summary, family="monospace", fontsize=8, verticalalignment="top",
                     transform=axes[1, 0].transAxes)

    axes[1, 1].axis("off")
    summary2 = "WORLD-frame Δt (just position differences in world)\n\n"
    for t in range(1, 8):
        wt = poses[t, :3, 3] - poses[t - 1, :3, 3]
        summary2 += f"  t={t}:  world Δt = {wt}\n"
    summary2 += "\nINTERPRETATION:\n"
    summary2 += "  Model is trained against the cam-frame deltas (top-left).\n"
    summary2 += "  These should have the same magnitudes as world-frame deltas\n"
    summary2 += "  but DIFFERENT per-axis values (because rotated into cam frame).\n"
    summary2 += "  If they look identical, the cam-frame conversion is missing.\n"
    axes[1, 1].text(0.0, 1.0, summary2, family="monospace", fontsize=8, verticalalignment="top",
                     transform=axes[1, 1].transAxes)
    fig.suptitle(f"07 — gt_relative_motion: what cam_delta_gt actually means")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "07_gt_relative_motion.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def check_08_build_rays():
    banner("08 — build_rays_from_pose (ray construction)")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    recs = sync_sequence(DATA_ROOT / SEQ)
    pose = recs[FRAME].pose_w_c
    pose_t = torch.from_numpy(pose).float().unsqueeze(0)
    fx, fy, cx, cy = intrinsics_for(SEQ)
    sx, sy = IMG_SIZE / 640.0, IMG_SIZE / 480.0
    K = torch.tensor([[[fx*sx, 0, cx*sx], [0, fy*sy, cy*sy], [0, 0, 1]]])

    # 5 chosen pixels: 4 corners + center
    pixels = torch.tensor([[[10., 10.],
                              [IMG_SIZE - 10., 10.],
                              [IMG_SIZE - 10., IMG_SIZE - 10.],
                              [10., IMG_SIZE - 10.],
                              [IMG_SIZE / 2., IMG_SIZE / 2.]]])                       # (1, 5, 2)
    ray_o, ray_d = build_rays_from_pose(pose_t, K, pixels)
    print(f"5-pixel test:")
    for i, name in enumerate(["top-left", "top-right", "bot-right", "bot-left", "center"]):
        print(f"  {name} pixel {pixels[0, i].tolist()}:")
        print(f"    ray_o = {ray_o[0, i].numpy()}")
        print(f"    ray_d = {ray_d[0, i].numpy()}  |d|={ray_d[0, i].norm().item():.6f}")
    # All ray origins should equal camera origin
    cam_origin = pose[:3, 3]
    print(f"\nCamera origin from pose: {cam_origin}")
    print(f"All ray origins should equal this — max deviation: "
           f"{float((ray_o[0] - torch.from_numpy(cam_origin).float()).abs().max()):.6f}")
    # Center ray should point in camera +z direction
    cam_forward = pose[:3, :3] @ np.array([0, 0, 1.0])
    print(f"\nCamera forward (R @ [0,0,1]): {cam_forward}")
    print(f"Center pixel ray_d:            {ray_d[0, 4].numpy()}")
    dot = float(np.dot(ray_d[0, 4].numpy(), cam_forward))
    print(f"  dot product (should be ~+1 if center pixel ray = forward): {dot:.6f}")

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    cam_pos = pose[:3, 3]
    for i, (name, color) in enumerate(zip(
        ["top-left", "top-right", "bot-right", "bot-left", "center"],
        ["red", "green", "blue", "orange", "purple"]
    )):
        o = ray_o[0, i].numpy()
        d = ray_d[0, i].numpy()
        end = o + d * 2.0
        ax.plot([o[0], end[0]], [o[1], end[1]], [o[2], end[2]],
                 color=color, label=name, linewidth=2)
    ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], color="black", s=100, label="camera origin")
    ax.set_xlabel("World X"); ax.set_ylabel("World Y"); ax.set_zlabel("World Z")
    ax.set_title("Rays in world frame (length 2m)")
    ax.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    txt = f"08 — build_rays_from_pose check\n\n"
    txt += f"camera position (pose[:3,3]) = {cam_pos}\n"
    txt += f"camera forward (R · [0,0,1]) = {cam_forward}\n"
    txt += f"center-pixel ray_d           = {ray_d[0, 4].numpy()}\n"
    txt += f"dot(center_ray, cam_forward) = {dot:.6f}  (target +1.0)\n\n"
    txt += "Per-pixel rays (origin should equal camera position):\n"
    for i, name in enumerate(["top-left", "top-right", "bot-right", "bot-left", "center"]):
        txt += f"  {name:10s} pix={pixels[0, i].numpy()}\n"
        txt += f"    ray_o = {ray_o[0, i].numpy()}\n"
        txt += f"    ray_d = {ray_d[0, i].numpy()}  |d|={ray_d[0, i].norm().item():.4f}\n"
    txt += f"\nINTERPRETATION:\n"
    txt += "  • All ray origins must equal camera position (otherwise origin bug)\n"
    txt += "  • Center-pixel ray must coincide with camera-forward (otherwise K or R bug)\n"
    txt += "  • All |d| must be 1.0 (rays should be unit-length)\n"
    ax2.text(0.0, 1.0, txt, family="monospace", fontsize=8, verticalalignment="top",
              transform=ax2.transAxes)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "08_build_rays_from_pose.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def check_09_backproject_vs_buildrays():
    banner("09 — backproject_patches_to_world vs build_rays_from_pose consistency")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = sync_sequence(DATA_ROOT / SEQ)
    pose = recs[FRAME].pose_w_c
    pose_t = torch.from_numpy(pose).float().unsqueeze(0)
    fx, fy, cx, cy = intrinsics_for(SEQ)
    sx, sy = IMG_SIZE / 640.0, IMG_SIZE / 480.0
    K = torch.tensor([[[fx*sx, 0, cx*sx], [0, fy*sy, cy*sy], [0, 0, 1]]])

    pixels = torch.tensor([[[10., 10.],
                              [IMG_SIZE - 10., 10.],
                              [IMG_SIZE - 10., IMG_SIZE - 10.],
                              [10., IMG_SIZE - 10.],
                              [IMG_SIZE / 2., IMG_SIZE / 2.]]])
    depth = torch.tensor([[1.5, 1.5, 1.5, 1.5, 1.5]])                                # (1, 5) — all 1.5m

    # Path A: build_rays_from_pose, then origin + depth · dir
    ray_o, ray_d = build_rays_from_pose(pose_t, K, pixels)
    pts_via_rays = ray_o + depth.unsqueeze(-1) * ray_d                               # (1, 5, 3)

    # Path B: backproject_patches_to_world
    pts_via_backproject = backproject_patches_to_world(pixels, depth, K, pose_t)     # (1, 5, 3)

    print("World points via two methods (should be identical):")
    print(f"{'pixel':10s}  {'via_rays':25s}  {'via_backproject':25s}  max_abs_diff")
    for i, name in enumerate(["top-left", "top-right", "bot-right", "bot-left", "center"]):
        a = pts_via_rays[0, i].numpy()
        b = pts_via_backproject[0, i].numpy()
        d = np.max(np.abs(a - b))
        print(f"  {name:10s}  {str(a):25s}  {str(b):25s}  {d:.2e}")
    max_diff_overall = float((pts_via_rays - pts_via_backproject).abs().max())
    print(f"\nOverall max abs diff: {max_diff_overall:.2e}  (target ~0)")

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis("off")
    txt = f"09 — backproject_patches_to_world vs build_rays_from_pose @ depth=1.5m\n"
    txt += "=" * 70 + "\n\n"
    txt += f"Camera pose origin: {pose[:3, 3]}\n\n"
    txt += f"{'pixel':12s}  {'via_rays (origin+d·dir)':32s}  {'via_backproject':32s}  diff\n"
    for i, name in enumerate(["top-left", "top-right", "bot-right", "bot-left", "center"]):
        a = pts_via_rays[0, i].numpy()
        b = pts_via_backproject[0, i].numpy()
        d = np.max(np.abs(a - b))
        txt += f"  {name:12s}  {str(np.round(a, 3)):32s}  {str(np.round(b, 3)):32s}  {d:.2e}\n"
    txt += f"\nMax overall diff: {max_diff_overall:.2e}\n\n"
    txt += "INTERPRETATION:\n"
    txt += "  • Same camera, same K, same pixel, same depth → SAME world point.\n"
    txt += "  • If diff is NOT ~0, write-time and render-time interpret\n"
    txt += "    coordinates differently. THIS WOULD EXPLAIN EVERYTHING:\n"
    txt += "      writes go to one set of world coords,\n"
    txt += "      renders read from another,\n"
    txt += "      mass piles up in wrong places.\n"
    txt += "  • Specifically: check whether build_rays normalizes dir before/after\n"
    txt += "    R rotation, which would affect dir norm but not (origin + d·dir)\n"
    txt += "    if depth is in 'meters along ray' vs 'meters along camera z'.\n"
    txt += "    THE DIFFERENCE BETWEEN THESE TWO DEPTH SEMANTICS IS A KNOWN BUG CLASS.\n"
    ax.text(0.0, 1.0, txt, family="monospace", fontsize=9, verticalalignment="top",
             transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "09_backproject_vs_buildrays.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def check_10_depth_direction():
    banner("10 — depth direction: along-ray vs along-camera-z")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = sync_sequence(DATA_ROOT / SEQ)
    pose = recs[FRAME].pose_w_c
    pose_t = torch.from_numpy(pose).float().unsqueeze(0)
    fx, fy, cx, cy = intrinsics_for(SEQ)
    sx, sy = IMG_SIZE / 640.0, IMG_SIZE / 480.0
    K = torch.tensor([[[fx*sx, 0, cx*sx], [0, fy*sy, cy*sy], [0, 0, 1]]])

    pixels = torch.tensor([[[10., 10.],
                              [IMG_SIZE - 10., 10.],
                              [IMG_SIZE / 2., IMG_SIZE / 2.]]])                       # (1, 3, 2) — 2 corners + center
    depth = torch.tensor([[1.5, 1.5, 1.5]])

    # Backproject to world
    pts_world = backproject_patches_to_world(pixels, depth, K, pose_t)               # (1, 3, 3)
    # Now project back: P_cam = R^T (P_world - t)
    R = pose_t[0, :3, :3]
    t = pose_t[0, :3, 3]
    pts_cam = torch.einsum("ij,nj->ni", R.T, pts_world[0] - t)                       # (3, 3)
    print(f"After backproject and re-project to camera frame:")
    for i, name in enumerate(["top-left (corner)", "top-right (corner)", "center"]):
        pc = pts_cam[i].numpy()
        print(f"  {name}: P_cam = {pc}")
        print(f"    z component = {pc[2]:.4f}  (this is depth-along-camera-z)")
        print(f"    |P_cam|     = {np.linalg.norm(pc):.4f}  (this is depth-along-ray)")
        ratio = np.linalg.norm(pc) / pc[2] if abs(pc[2]) > 1e-6 else np.nan
        print(f"    |P_cam| / z = {ratio:.4f}  (close to 1 only for near-center pixels)")
    print(f"\nIf the model treats depth as 'meters along ray' but the rendering integrates")
    print(f"depth as 'meters along z', that's a per-pixel scale mismatch that grows toward")
    print(f"image corners — a known bug class.")

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis("off")
    txt = "10 — depth direction check (along-ray vs along-camera-z)\n"
    txt += "=" * 70 + "\n\n"
    txt += "Backproject depth=1.5m at three pixels, then map back to camera frame.\n"
    txt += "If everything is consistent, this should reveal the depth semantics:\n\n"
    for i, name in enumerate(["top-left (corner)", "top-right (corner)", "center"]):
        pc = pts_cam[i].numpy()
        ratio = np.linalg.norm(pc) / pc[2] if abs(pc[2]) > 1e-6 else float('nan')
        txt += f"  {name}:\n"
        txt += f"    P_cam   = {pc}\n"
        txt += f"    z       = {pc[2]:.4f}  (depth along camera +z axis)\n"
        txt += f"    |P_cam| = {np.linalg.norm(pc):.4f}  (length along ray)\n"
        txt += f"    ratio   = |P_cam|/z = {ratio:.4f}\n\n"
    txt += "INTERPRETATION:\n"
    txt += "  • For center pixel, ratio ≈ 1 always.\n"
    txt += "  • For corner pixels, ratio > 1 (by amount that depends on FOV).\n"
    txt += "  • If z values are ALL 1.5 across pixels, depth is interpreted as 'z'.\n"
    txt += "  • If |P_cam| values are ALL 1.5, depth is 'along ray'.\n"
    txt += "  • The current backproject code uses cam_dir * depth where cam_dir is\n"
    txt += "    K_inv @ [u, v, 1] (NOT normalized). That means depth is along z, not\n"
    txt += "    along ray. Confirming this here.\n\n"
    txt += "  • If render_rays_volumetric uses normalized rays and treats t_vals as\n"
    txt += "    'along ray' distance, but writes used depth as 'along z' distance,\n"
    txt += "    there's a per-pixel mismatch that BREAKS THE WHOLE SYSTEM at non-center\n"
    txt += "    pixels.\n"
    ax.text(0.0, 1.0, txt, family="monospace", fontsize=9, verticalalignment="top",
             transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "10_depth_direction.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def check_11_voxel_grid_coords():
    banner("11 — voxel grid coordinate system")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bounds = (-4.0, -4.0, -2.0, 4.0, 4.0, 8.0)
    resolution = (64, 64, 64)
    cfg = VoxelGridConfig(bounds=bounds, resolution=resolution, feature_dim=2)
    state = init_voxel_state(cfg, batch_size=1, device="cuda", dtype=torch.float32)

    # Set a single voxel feature to a known value, sample at its world center, recover.
    KNOWN_IDX = (10, 20, 30)
    voxel_size = ((bounds[3]-bounds[0])/resolution[0],
                   (bounds[4]-bounds[1])/resolution[1],
                   (bounds[5]-bounds[2])/resolution[2])
    print(f"voxel_size = {voxel_size}")
    # Center of voxel at (ix, iy, iz):
    world_x = bounds[0] + (KNOWN_IDX[0] + 0.5) * voxel_size[0]
    world_y = bounds[1] + (KNOWN_IDX[1] + 0.5) * voxel_size[1]
    world_z = bounds[2] + (KNOWN_IDX[2] + 0.5) * voxel_size[2]
    print(f"\nChosen voxel: idx={KNOWN_IDX}")
    print(f"World-center of this voxel (computed manually): ({world_x:.4f}, {world_y:.4f}, {world_z:.4f})")

    KNOWN_VALUE = 42.0
    state.features[0, KNOWN_IDX[0], KNOWN_IDX[1], KNOWN_IDX[2], 0] = KNOWN_VALUE
    state.write_mass[0, KNOWN_IDX[0], KNOWN_IDX[1], KNOWN_IDX[2], 0] = 1.0

    # world_to_grid_coords for this point
    pt = torch.tensor([[[world_x, world_y, world_z]]]).cuda()                         # (1, 1, 3)
    gc = world_to_grid_coords(pt, cfg)                                                # (1, 1, 3) in continuous voxel coords
    print(f"\nworld_to_grid_coords output: {gc[0, 0].cpu().numpy()}")
    print(f"  expected: ({KNOWN_IDX[0] + 0.5}, {KNOWN_IDX[1] + 0.5}, {KNOWN_IDX[2] + 0.5})")
    diff = np.abs(gc[0, 0].cpu().numpy() - np.array([KNOWN_IDX[0] + 0.5, KNOWN_IDX[1] + 0.5, KNOWN_IDX[2] + 0.5]))
    print(f"  diff: {diff}  (should be ~0)")

    # Now sample
    sampled_feat, sampled_mass = trilinear_sample_grid(state, pt)
    print(f"\ntrilinear_sample_grid at voxel center:")
    print(f"  features[0, :] = {sampled_feat[0, 0].cpu().numpy()}  (expected first component near {KNOWN_VALUE})")
    print(f"  mass           = {sampled_mass[0, 0].cpu().numpy()}  (expected near 1.0)")

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis("off")
    txt = "11 — voxel grid coordinate system check\n"
    txt += "=" * 70 + "\n\n"
    txt += f"Voxel grid config:\n"
    txt += f"  bounds = {bounds}\n"
    txt += f"  resolution = {resolution}\n"
    txt += f"  voxel size = {voxel_size}\n\n"
    txt += f"Chosen voxel index: {KNOWN_IDX}\n"
    txt += f"Voxel center in world (manually computed):\n"
    txt += f"  ({world_x:.4f}, {world_y:.4f}, {world_z:.4f})\n\n"
    txt += f"Set features[{KNOWN_IDX}, 0] = {KNOWN_VALUE}, mass = 1.0\n\n"
    txt += f"world_to_grid_coords({world_x:.4f}, {world_y:.4f}, {world_z:.4f}):\n"
    txt += f"  output: {gc[0, 0].cpu().numpy()}\n"
    txt += f"  expected: ({KNOWN_IDX[0] + 0.5}, {KNOWN_IDX[1] + 0.5}, {KNOWN_IDX[2] + 0.5})\n"
    txt += f"  diff: {diff}  (target ~0)\n\n"
    txt += f"trilinear_sample_grid at voxel center:\n"
    txt += f"  features[:] = {sampled_feat[0, 0].cpu().numpy()}\n"
    txt += f"  mass        = {sampled_mass[0, 0].cpu().numpy()}\n"
    txt += f"\nINTERPRETATION:\n"
    txt += f"  • features[0] should be ≈ {KNOWN_VALUE} (we put it there).\n"
    txt += f"  • mass should be ≈ 1.0 (we put it there).\n"
    txt += f"  • If sampled feature is at index OTHER than [0] of voxel_feat_dim,\n"
    txt += f"    or close to 0, then world_to_grid_coords / trilinear_sample_grid\n"
    txt += f"    disagree on axis order.\n"
    txt += f"  • Specifically: F.grid_sample needs (z, y, x) ordering in 5D, and\n"
    txt += f"    voxel_grid.py permutes features (B, V_x, V_y, V_z, D) → (B, D, V_z, V_y, V_x).\n"
    txt += f"    A bug in this permute is catastrophic and silent.\n"
    ax.text(0.0, 1.0, txt, family="monospace", fontsize=9, verticalalignment="top",
             transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "11_voxel_grid_coords.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def check_12_cam9_roundtrip():
    banner("12 — cam9 roundtrip & delta semantics")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = sync_sequence(DATA_ROOT / SEQ)
    pose = recs[FRAME].pose_w_c
    pose_t = torch.from_numpy(pose).float().unsqueeze(0)
    fov = torch.tensor([[1.0, 1.0]])
    cam9 = _pose_T_to_cam9(pose_t, fov)
    print(f"frame {FRAME} pose_w_c → cam9 (model's encoding):")
    labels = ["tx", "ty", "tz", "qx", "qy", "qz", "qw", "fovx", "fovy"]
    for i, lbl in enumerate(labels):
        print(f"  {lbl} = {cam9[0, i].item():.6f}")
    pose_back = cam9_to_pose_w_c(cam9)
    print(f"\ncam9 → pose_w_c reproduces:")
    print(pose_back[0].numpy())
    err = float(np.max(np.abs(pose_back[0].numpy() - pose)))
    print(f"max abs error: {err:.2e}")

    # Two consecutive frames → "delta" via concat
    pose_t1 = torch.from_numpy(recs[FRAME].pose_w_c).float().unsqueeze(0)
    pose_t2 = torch.from_numpy(recs[FRAME + 1].pose_w_c).float().unsqueeze(0)
    cam9_t1 = _pose_T_to_cam9(pose_t1, fov)
    cam9_t2 = _pose_T_to_cam9(pose_t2, fov)
    print(f"\nFrame {FRAME}:   cam9 = {cam9_t1[0].numpy()}")
    print(f"Frame {FRAME+1}: cam9 = {cam9_t2[0].numpy()}")
    # The model's TRAINING TARGET for the pose head is gt_relative_motion (cam-frame delta).
    poses_stack = torch.stack([pose_t1, pose_t2], dim=1)                              # (1, 2, 4, 4)
    gt_delta = gt_relative_motion_from_abs_poses(poses_stack[0:1])                    # (1, 2, 7)
    print(f"\ngt_relative_motion_from_abs_poses (delta for t=1):")
    print(f"  delta_7[0, 1, :] = {gt_delta[0, 1].numpy()}")
    print(f"  cam-frame translation = {gt_delta[0, 1, :3].numpy()}")
    print(f"  cam-frame quaternion   = {gt_delta[0, 1, 3:7].numpy()}")
    # The model's PREDICTED pose head output is cam9-format delta:
    #   T_world_t = T_world_{t-1} @ cam9_to_pose_w_c(delta_9)
    # So delta_9 = cam9_to_pose_w_c(.) is a 4x4 representing "t in t-1's frame".
    # gt_relative_motion gives this same thing as a 7-vec (t, q).
    # The model loss compares pred_delta_9 (which is [t,q,fov]) to gt_delta_7 (just [t,q]).
    # Let's confirm by composing GT delta into world-frame and seeing if it equals pose_t2.
    gt_delta_T = cam9_to_pose_w_c(torch.cat([gt_delta[0, 1:2], fov], dim=-1))
    composed = pose_t1 @ gt_delta_T
    err_comp = float(np.max(np.abs(composed[0].numpy() - pose_t2[0].numpy())))
    print(f"\nCompose pose_t1 @ gt_delta (in cam9 format) and compare to pose_t2:")
    print(composed[0].numpy())
    print(f"max abs error vs pose_t2: {err_comp:.2e}  (target ~0)")

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis("off")
    txt = "12 — cam9 encoding + delta composition semantics\n"
    txt += "=" * 70 + "\n\n"
    txt += f"Frame {FRAME}: pose_w_c → cam9 = [\n"
    for i, lbl in enumerate(labels):
        txt += f"  {lbl} = {cam9[0, i].item():.6f}\n"
    txt += f"]\n\n"
    txt += f"cam9 → pose_w_c round-trip max abs error: {err:.2e}\n\n"
    txt += "—" * 70 + "\n\n"
    txt += f"gt_relative_motion (cam-frame delta from {FRAME} to {FRAME+1}):\n"
    txt += f"  Δt (cam-frame) = {gt_delta[0, 1, :3].numpy()}\n"
    txt += f"  Δq (cam-frame) = {gt_delta[0, 1, 3:7].numpy()}\n"
    txt += f"  |Δt| = {float(gt_delta[0, 1, :3].norm()):.4f} m\n\n"
    txt += f"Compose pose[{FRAME}] @ cam9_to_pose_w_c(Δ_cam) → does it equal pose[{FRAME+1}]?\n"
    txt += f"  max abs error: {err_comp:.2e}  (target ~0)\n\n"
    txt += "INTERPRETATION:\n"
    txt += "  • cam9 round-trip error must be ~1e-6 (encoder/decoder symmetric).\n"
    txt += "  • The composition equation must hold:\n"
    txt += "      pose[t] = pose[t-1] @ cam9_to_pose_w_c(gt_delta[t])\n"
    txt += "    If err_comp is large, the model is being trained against a delta\n"
    txt += "    target that DOES NOT compose back to ground truth. That would be\n"
    txt += "    a catastrophic loss-target bug.\n"
    ax.text(0.0, 1.0, txt, family="monospace", fontsize=9, verticalalignment="top",
             transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "12_cam9_roundtrip.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUT_DIR.absolute()}")
    print(f"Sequence: {SEQ}")
    print(f"Test frame: {FRAME}")
    print(f"Image size: {IMG_SIZE}")
    check_01_rgb_loading()
    check_02_depth_loading()
    check_03_intrinsics_scaling()
    check_04_tum_pose_convention()
    check_05_quaternion_convention()
    check_06_pose_w_c_roundtrip()
    check_07_gt_relative_motion()
    check_08_build_rays()
    check_09_backproject_vs_buildrays()
    check_10_depth_direction()
    check_11_voxel_grid_coords()
    check_12_cam9_roundtrip()
    print("\n" + "=" * 80)
    print(f"  ALL 12 CHECKS COMPLETE")
    print(f"  Outputs: {OUT_DIR.absolute()}/")
    print(f"  Files: 01_rgb_loading.png ... 12_cam9_roundtrip.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
