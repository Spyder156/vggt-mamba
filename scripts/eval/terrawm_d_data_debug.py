"""Comprehensive data/coordinate/convention debug viz harness.

After A1 (3% toward-GT) and A2 (17% toward-GT) BOTH FAILED, the iteration-
moves-perfectly-away-from-GT pattern strongly suggests a coordinate, sign,
or pose-frame convention bug somewhere in the pipeline. This script produces
a battery of comparative visualizations to make every stage of the pipeline
directly inspectable.

Outputs (all under viz/output/terrawm_d_data_debug/):

  trajectory_3d.png            - GT vs pred trajectory + voxel bounds box
  voxel_grid_f{K}.png          - xy/xz/yz projections + 3D scatter
  gradient_sanity_f{K}.png     - perturb GT pose by ±x±y±z, plot grad arrows
  loss_landscape_f{K}.png      - L_depth on (tx, ty) grid around GT pose

  frame_{K:04d}/depth_panel.png
                               - RGB, GT depth, bootstrap_d, rendered@GT-pose,
                                 rendered@pred-pose, depth mask
  frame_{K:04d}/per_patch_scatter.png
                               - bootstrap_d vs GT, rendered vs GT (per-patch)
  frame_{K:04d}/backprojection.png
                               - backprojected world points at GT vs pred pose
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt_mamba.data.tum_rgbd import sync_sequence, intrinsics_for                 # noqa: E402
from vggt_mamba.models.aggregators.anchor_pool import cam9_to_pose_w_c              # noqa: E402
from vggt_mamba.models.terrawm_d import build_terrawm_d, _pose_T_to_cam9             # noqa: E402
from vggt_mamba.models.voxel_grid import (                                          # noqa: E402
    VoxelGridState, backproject_patches_to_world, build_rays_from_pose,
    init_voxel_state, render_rays_volumetric, write_voxels_trilinear,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from terrawm_d_regrounding_stream import (                                          # noqa: E402
    load_model, load_rgb,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--seq", default="rgbd_dataset_freiburg2_desk")
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--out", type=Path,
                    default=Path("viz/output/terrawm_d_data_debug"))
    p.add_argument("--test-frames", type=int, nargs="+",
                    default=[10, 100, 500, 1000, 1500, 2000])
    p.add_argument("--n-frames", type=int, default=2100,
                    help="Cap streaming to N frames (must cover max test frame).")
    return p.parse_args()


def load_gt_depth_image(rec, img_size: int, depth_max_m: float = 8.0):
    from PIL import Image
    d = Image.open(rec.depth_path).resize((img_size, img_size), Image.NEAREST)
    d = np.asarray(d, dtype=np.float32) / 5000.0
    d = np.where((d > 0) & (d < depth_max_m), d, np.nan)
    return d


@torch.no_grad()
def stream_and_snapshot(model, recs, K, fov, gt_rel_T, test_frame_set):
    """Stream with the model; snapshot voxel state AND pred pose at each
    test frame. Also keep the running pred trajectory for the trajectory plot."""
    img_size = model.img_size
    voxel_state = model.init_voxel_state(1, "cuda", torch.float32)
    patch_pixel = model._patch_pixel_grid.cuda().unsqueeze(0).float()
    prev_pose_9 = torch.tensor([[0., 0, 0, 0, 0, 0, 1, 1.0, 1.0]],
                                device="cuda", dtype=torch.float32)
    pred_traj = []
    snaps = {}
    for i, rec in enumerate(recs):
        rgb = load_rgb(rec, img_size).cuda(non_blocking=True)
        initial_T = cam9_to_pose_w_c(prev_pose_9)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            patches = model._encode_frame(rgb)
            bootstrap_d = model.bootstrap_depth(patches).float()
            voxel_feat = model.patch_to_voxel(patches).float()
            wc = model.write_confidence(patches).float() if model.use_write_confidence else None
            ray_o1, ray_d1 = build_rays_from_pose(initial_T, K, patch_pixel)
            r1 = render_rays_volumetric(voxel_state, ray_o1, ray_d1,
                n_samples=model.n_render_samples,
                near=model.render_near, far=model.render_far)
            init9 = _pose_T_to_cam9(initial_T, fov)
            d9 = model.pose_head(patches, r1["feature"], r1["total_weight"], init9)
            dT = cam9_to_pose_w_c(d9)
            corrected_pose_T = (initial_T.float() @ dT).float()
            wp = backproject_patches_to_world(patch_pixel, bootstrap_d, K, corrected_pose_T)
            write_voxels_trilinear(voxel_state, wp, voxel_feat, weights=wc)
            prev_pose_9 = _pose_T_to_cam9(corrected_pose_T, fov).float()
        pred_traj.append(corrected_pose_T[0].detach().float().cpu().numpy().copy())
        if i in test_frame_set:
            snaps[i] = {
                "voxel_state": {
                    "features": voxel_state.features.detach().clone(),
                    "write_mass": voxel_state.write_mass.detach().clone(),
                    "cfg": voxel_state.cfg,
                },
                "pred_pose_T": corrected_pose_T[0].detach().float().cpu().numpy().copy(),
                "bootstrap_d": bootstrap_d[0].detach().float().cpu().numpy().copy(),
                "patches": patches.detach().clone(),
            }
        if (i + 1) % 200 == 0:
            print(f"[debug-viz] streamed {i+1}/{len(recs)}")
    return pred_traj, snaps


def restore_voxel_state(snap_dict):
    s = init_voxel_state(snap_dict["voxel_state"]["cfg"], 1, "cuda", torch.float32)
    s.features.copy_(snap_dict["voxel_state"]["features"])
    s.write_mass.copy_(snap_dict["voxel_state"]["write_mass"])
    return s


def viz_trajectory(out_dir: Path, pred_traj: list[np.ndarray],
                    gt_rel_T_np: np.ndarray, voxel_bounds: tuple[float, ...]):
    """A: 3D trajectory + voxel bounds box. Multi-panel: 3D + 3 projections."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    pred_t = np.stack([p[:3, 3] for p in pred_traj])           # (N, 3)
    gt_t = gt_rel_T_np[:, :3, 3]                                # (N, 3)
    n = min(len(pred_t), len(gt_t))
    pred_t, gt_t = pred_t[:n], gt_t[:n]
    x_lo, y_lo, z_lo, x_hi, y_hi, z_hi = voxel_bounds

    fig = plt.figure(figsize=(20, 16))
    # 3D plot
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.plot(gt_t[:, 0], gt_t[:, 1], gt_t[:, 2], "g-", label="GT trajectory", linewidth=1.5)
    ax.plot(pred_t[:, 0], pred_t[:, 1], pred_t[:, 2], "r-", label="Pred trajectory", linewidth=1.5)
    ax.scatter(*gt_t[0], color="green", s=100, marker="o", label="GT frame 0")
    ax.scatter(*gt_t[-1], color="darkgreen", s=100, marker="x", label=f"GT frame {n-1}")
    ax.scatter(*pred_t[-1], color="darkred", s=100, marker="x", label=f"Pred frame {n-1}")
    # Voxel bounds box edges
    edges = [
        ([x_lo, x_hi], [y_lo, y_lo], [z_lo, z_lo]),
        ([x_lo, x_hi], [y_hi, y_hi], [z_lo, z_lo]),
        ([x_lo, x_hi], [y_lo, y_lo], [z_hi, z_hi]),
        ([x_lo, x_hi], [y_hi, y_hi], [z_hi, z_hi]),
        ([x_lo, x_lo], [y_lo, y_hi], [z_lo, z_lo]),
        ([x_hi, x_hi], [y_lo, y_hi], [z_lo, z_lo]),
        ([x_lo, x_lo], [y_lo, y_hi], [z_hi, z_hi]),
        ([x_hi, x_hi], [y_lo, y_hi], [z_hi, z_hi]),
        ([x_lo, x_lo], [y_lo, y_lo], [z_lo, z_hi]),
        ([x_hi, x_hi], [y_lo, y_lo], [z_lo, z_hi]),
        ([x_lo, x_lo], [y_hi, y_hi], [z_lo, z_hi]),
        ([x_hi, x_hi], [y_hi, y_hi], [z_lo, z_hi]),
    ]
    for ex, ey, ez in edges:
        ax.plot(ex, ey, ez, "b-", alpha=0.3, linewidth=0.7)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("3D — GT (green) vs Pred (red) + voxel grid bounds (blue)")
    ax.legend()

    # 2D projections
    for k, (i1, i2, name) in enumerate([(0, 1, "xy (top-down)"), (0, 2, "xz (side)"),
                                          (1, 2, "yz (front)")]):
        ax = fig.add_subplot(2, 2, k + 2)
        ax.plot(gt_t[:, i1], gt_t[:, i2], "g-", label="GT", linewidth=1.5)
        ax.plot(pred_t[:, i1], pred_t[:, i2], "r-", label="Pred", linewidth=1.5)
        ax.scatter(gt_t[0, i1], gt_t[0, i2], color="green", s=80, marker="o")
        ax.scatter(gt_t[-1, i1], gt_t[-1, i2], color="darkgreen", s=80, marker="x")
        ax.scatter(pred_t[-1, i1], pred_t[-1, i2], color="darkred", s=80, marker="x")
        # Bounds rect
        bounds_box = {
            (0, 1): [(x_lo, x_hi, y_lo, y_hi)],
            (0, 2): [(x_lo, x_hi, z_lo, z_hi)],
            (1, 2): [(y_lo, y_hi, z_lo, z_hi)],
        }[(i1, i2)][0]
        a, b, c, d = bounds_box
        ax.plot([a, b, b, a, a], [c, c, d, d, c], "b-", alpha=0.4, linewidth=0.7)
        # Label every Nth GT frame for orientation
        step = max(1, n // 20)
        for fi in range(0, n, step):
            ax.annotate(str(fi), (gt_t[fi, i1], gt_t[fi, i2]),
                         fontsize=6, color="darkgreen", alpha=0.5)
        ax.set_xlabel(["X", "X", "Y"][k])
        ax.set_ylabel(["Y", "Z", "Z"][k])
        ax.set_title(f"{name} — GT vs Pred + voxel bounds")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Trajectory comparison: GT vs Pred, n={n} frames", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "trajectory_3d.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def viz_depth_panel(out_dir: Path, frame_idx: int, rgb_np: np.ndarray,
                     gt_depth_np: np.ndarray, bootstrap_d_patch: np.ndarray,
                     r_depth_gt: np.ndarray, r_mass_gt: np.ndarray,
                     r_depth_pred: np.ndarray, r_mass_pred: np.ndarray,
                     img_size: int, grid_h: int, grid_w: int,
                     gt_pose_t: np.ndarray, pred_pose_t: np.ndarray):
    """B: depth panel with consistent colormap across all depth views."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bootstrap_d_img = np.array(bootstrap_d_patch).reshape(grid_h, grid_w)
    bootstrap_d_img = np.kron(bootstrap_d_img, np.ones((img_size // grid_h, img_size // grid_w)))
    r_depth_gt_img = np.array(r_depth_gt).reshape(grid_h, grid_w)
    r_depth_gt_img = np.kron(r_depth_gt_img, np.ones((img_size // grid_h, img_size // grid_w)))
    r_depth_pred_img = np.array(r_depth_pred).reshape(grid_h, grid_w)
    r_depth_pred_img = np.kron(r_depth_pred_img, np.ones((img_size // grid_h, img_size // grid_w)))
    r_mass_pred_img = np.array(r_mass_pred).reshape(grid_h, grid_w)
    r_mass_pred_img = np.kron(r_mass_pred_img, np.ones((img_size // grid_h, img_size // grid_w)))

    # Consistent colormap range across all depth views.
    all_depths = np.concatenate([
        gt_depth_np[~np.isnan(gt_depth_np)],
        bootstrap_d_img.flatten(),
        r_depth_gt_img.flatten(),
        r_depth_pred_img.flatten(),
    ])
    vmin = float(np.nanpercentile(all_depths, 1))
    vmax = float(np.nanpercentile(all_depths, 99))

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes[0, 0].imshow(rgb_np)
    axes[0, 0].set_title(f"RGB (frame {frame_idx})")
    axes[0, 0].axis("off")

    im = axes[0, 1].imshow(gt_depth_np, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f"GT depth\nrange [{np.nanmin(gt_depth_np):.2f}, {np.nanmax(gt_depth_np):.2f}]m")
    axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    im = axes[0, 2].imshow(bootstrap_d_img, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[0, 2].set_title(f"bootstrap_d (encoder)\nrange [{bootstrap_d_img.min():.2f}, {bootstrap_d_img.max():.2f}]m")
    axes[0, 2].axis("off")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    im = axes[1, 0].imshow(r_depth_gt_img, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title(f"rendered_depth @ GT pose\nrange [{r_depth_gt_img.min():.2f}, {r_depth_gt_img.max():.2f}]m")
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    im = axes[1, 1].imshow(r_depth_pred_img, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[1, 1].set_title(f"rendered_depth @ pred pose\nrange [{r_depth_pred_img.min():.2f}, {r_depth_pred_img.max():.2f}]m")
    axes[1, 1].axis("off")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    im = axes[1, 2].imshow(r_mass_pred_img, cmap="gray")
    axes[1, 2].set_title(f"depth mass @ pred pose\n||pred-GT||={np.linalg.norm(pred_pose_t - gt_pose_t):.3f}m")
    axes[1, 2].axis("off")
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

    fig.suptitle(f"Frame {frame_idx} depth panel — same colormap range "
                  f"[{vmin:.2f}, {vmax:.2f}]m. Inversions/scale-bugs show as miscolored regions.",
                  fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / f"frame_{frame_idx:04d}/depth_panel.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def viz_per_patch_scatter(out_dir: Path, frame_idx: int,
                            bootstrap_d_patch: np.ndarray,
                            gt_depth_patch: np.ndarray,
                            r_depth_gt_patch: np.ndarray,
                            r_depth_pred_patch: np.ndarray):
    """C: per-patch depth scatter against GT depth. Off-diagonal = bug."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    valid = (gt_depth_patch > 0.1) & (gt_depth_patch < 8.0)
    gt_v = gt_depth_patch[valid]

    for ax, name, vals in [
        (axes[0], "bootstrap_d", bootstrap_d_patch[valid]),
        (axes[1], "rendered@GT-pose", r_depth_gt_patch[valid]),
        (axes[2], "rendered@pred-pose", r_depth_pred_patch[valid]),
    ]:
        lo, hi = 0.0, max(gt_v.max(), vals.max() if len(vals) else 1.0) * 1.05
        ax.scatter(gt_v, vals, s=4, alpha=0.4)
        ax.plot([lo, hi], [lo, hi], "g-", alpha=0.5, label="y=x (perfect)")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("GT depth (m)")
        ax.set_ylabel(f"{name} (m)")
        if len(vals) > 0:
            mean_err = float(np.mean(np.abs(vals - gt_v)))
            corr = float(np.corrcoef(vals, gt_v)[0, 1])
            ax.set_title(f"{name} vs GT\nmean |err|={mean_err:.2f}m  corr={corr:+.3f}")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(f"Per-patch depth scatter @ frame {frame_idx} — "
                  f"off-diagonal or negative correlation = bug", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / f"frame_{frame_idx:04d}/per_patch_scatter.png",
                 dpi=110, bbox_inches="tight")
    plt.close(fig)


def viz_backprojection(out_dir: Path, frame_idx: int,
                        world_pts_gt: np.ndarray, world_pts_pred: np.ndarray,
                        gt_pos: np.ndarray, pred_pos: np.ndarray,
                        voxel_bounds: tuple[float, ...]):
    """D: backprojected world points at GT vs pred pose. Same patches, two poses."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(18, 6))
    x_lo, y_lo, z_lo, x_hi, y_hi, z_hi = voxel_bounds
    for k, (i1, i2, name) in enumerate([(0, 1, "xy (top-down)"),
                                          (0, 2, "xz (side)"),
                                          (1, 2, "yz (front)")]):
        ax = fig.add_subplot(1, 3, k + 1)
        ax.scatter(world_pts_gt[:, i1], world_pts_gt[:, i2], s=4, alpha=0.5,
                    color="green", label="patches backprojected via GT pose")
        ax.scatter(world_pts_pred[:, i1], world_pts_pred[:, i2], s=4, alpha=0.5,
                    color="red", label="patches backprojected via pred pose")
        ax.scatter(gt_pos[i1], gt_pos[i2], color="darkgreen", s=120, marker="o",
                    label="GT camera")
        ax.scatter(pred_pos[i1], pred_pos[i2], color="darkred", s=120, marker="o",
                    label="pred camera")
        bounds_box = {
            (0, 1): (x_lo, x_hi, y_lo, y_hi),
            (0, 2): (x_lo, x_hi, z_lo, z_hi),
            (1, 2): (y_lo, y_hi, z_lo, z_hi),
        }[(i1, i2)]
        a, b, c, d = bounds_box
        ax.plot([a, b, b, a, a], [c, c, d, d, c], "b-", alpha=0.4, linewidth=0.7,
                 label="voxel bounds")
        ax.set_xlabel(["X", "X", "Y"][k])
        ax.set_ylabel(["Y", "Z", "Z"][k])
        ax.set_title(name)
        ax.grid(alpha=0.3); ax.legend(fontsize=7); ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle(f"Backprojection @ frame {frame_idx} — same patches + bootstrap_d, "
                  f"two poses. Pred should produce displaced points consistent with the "
                  f"camera offset.", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / f"frame_{frame_idx:04d}/backprojection.png",
                 dpi=110, bbox_inches="tight")
    plt.close(fig)


def viz_voxel_grid(out_dir: Path, frame_idx: int, voxel_state: VoxelGridState,
                    gt_traj_so_far: np.ndarray, pred_traj_so_far: np.ndarray):
    """E: voxel grid content projections (xy, xz, yz) + 3D scatter + camera trajs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    mass = voxel_state.write_mass[0, ..., 0].detach().float().cpu().numpy()  # (Vx, Vy, Vz)
    bounds = voxel_state.cfg.bounds
    x_lo, y_lo, z_lo, x_hi, y_hi, z_hi = bounds
    Vx, Vy, Vz = mass.shape

    fig = plt.figure(figsize=(20, 14))
    extents = {
        "xy": (x_lo, x_hi, y_lo, y_hi),
        "xz": (x_lo, x_hi, z_lo, z_hi),
        "yz": (y_lo, y_hi, z_lo, z_hi),
    }
    # XY projection (sum over z)
    ax = fig.add_subplot(2, 3, 1)
    proj_xy = mass.sum(axis=2)  # (Vx, Vy)
    im = ax.imshow(proj_xy.T, origin="lower",
                    extent=extents["xy"], cmap="viridis", aspect="equal")
    ax.plot(gt_traj_so_far[:, 0], gt_traj_so_far[:, 1], "g-", linewidth=1, label="GT cam")
    ax.plot(pred_traj_so_far[:, 0], pred_traj_so_far[:, 1], "r-", linewidth=1, label="pred cam")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_title("xy projection (sum over z)")
    plt.colorbar(im, ax=ax, fraction=0.046); ax.legend(fontsize=7)

    # XZ projection (sum over y)
    ax = fig.add_subplot(2, 3, 2)
    proj_xz = mass.sum(axis=1)
    im = ax.imshow(proj_xz.T, origin="lower",
                    extent=extents["xz"], cmap="viridis", aspect="equal")
    ax.plot(gt_traj_so_far[:, 0], gt_traj_so_far[:, 2], "g-", linewidth=1, label="GT cam")
    ax.plot(pred_traj_so_far[:, 0], pred_traj_so_far[:, 2], "r-", linewidth=1, label="pred cam")
    ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_title("xz projection (sum over y)")
    plt.colorbar(im, ax=ax, fraction=0.046); ax.legend(fontsize=7)

    # YZ projection (sum over x)
    ax = fig.add_subplot(2, 3, 3)
    proj_yz = mass.sum(axis=0)
    im = ax.imshow(proj_yz.T, origin="lower",
                    extent=extents["yz"], cmap="viridis", aspect="equal")
    ax.plot(gt_traj_so_far[:, 1], gt_traj_so_far[:, 2], "g-", linewidth=1, label="GT cam")
    ax.plot(pred_traj_so_far[:, 1], pred_traj_so_far[:, 2], "r-", linewidth=1, label="pred cam")
    ax.set_xlabel("Y"); ax.set_ylabel("Z"); ax.set_title("yz projection (sum over x)")
    plt.colorbar(im, ax=ax, fraction=0.046); ax.legend(fontsize=7)

    # 3D scatter of high-mass voxels
    ax = fig.add_subplot(2, 1, 2, projection="3d")
    thresh = max(float(np.percentile(mass[mass > 0], 90)) if (mass > 0).any() else 0.01, 0.01)
    high_mass_idx = np.where(mass > thresh)
    if len(high_mass_idx[0]) > 0:
        x_centers = np.linspace(x_lo, x_hi, Vx)
        y_centers = np.linspace(y_lo, y_hi, Vy)
        z_centers = np.linspace(z_lo, z_hi, Vz)
        xs = x_centers[high_mass_idx[0]]
        ys = y_centers[high_mass_idx[1]]
        zs = z_centers[high_mass_idx[2]]
        cs = mass[high_mass_idx]
        sample = np.random.choice(len(xs), min(5000, len(xs)), replace=False)
        ax.scatter(xs[sample], ys[sample], zs[sample], c=cs[sample], s=4, cmap="viridis", alpha=0.5)
    ax.plot(gt_traj_so_far[:, 0], gt_traj_so_far[:, 1], gt_traj_so_far[:, 2],
             "g-", linewidth=2, label="GT cam")
    ax.plot(pred_traj_so_far[:, 0], pred_traj_so_far[:, 1], pred_traj_so_far[:, 2],
             "r-", linewidth=2, label="pred cam")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(f"3D — high-mass voxels (top 10%) + trajectories so far")
    ax.legend()
    fig.suptitle(f"Voxel grid content @ frame {frame_idx} — "
                  f"total mass={float(mass.sum()):.0f}, n_nonzero={int((mass>0).sum())}",
                  fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / f"voxel_grid_f{frame_idx:04d}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def compute_grad_at_pose(model, voxel_state, pose_T, K, patch_pixel, bootstrap_d):
    """Compute ∇_pose L_depth(rendered, bootstrap_d). Returns (grad_t, L)."""
    pose = pose_T.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            ray_o, ray_d = build_rays_from_pose(pose, K, patch_pixel)
            r = render_rays_volumetric(voxel_state, ray_o, ray_d,
                n_samples=model.n_render_samples,
                near=model.render_near, far=model.render_far)
            r_depth = r["depth"].float()
            mass = r["total_weight"].float()
            valid = (mass > model.unwritten_mask_threshold).float()
            diff = (r_depth - bootstrap_d.detach()).abs()
            L = (diff * valid * mass).sum() / valid.sum().clamp_min(1.0)
        grad_pose, = torch.autograd.grad(L, pose)
    return (grad_pose[0, :3, 3].detach().float().cpu().numpy(),
             float(L.detach()))


def viz_gradient_sanity(out_dir: Path, frame_idx: int,
                         model, voxel_state, gt_T_torch, K, patch_pixel,
                         bootstrap_d):
    """F: perturb GT pose by ±x, ±y, ±z and 4 random; plot gradient direction
    as arrows. If gradient points OPPOSITE to perturbation (toward GT), good.
    If gradient points SAME as perturbation (away from GT), SIGN BUG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_pos = gt_T_torch[0, :3, 3].float().cpu().numpy()
    test_directions = [
        ("+x", np.array([1.0, 0, 0])),
        ("-x", np.array([-1.0, 0, 0])),
        ("+y", np.array([0, 1.0, 0])),
        ("-y", np.array([0, -1.0, 0])),
        ("+z", np.array([0, 0, 1.0])),
        ("-z", np.array([0, 0, -1.0])),
    ]
    rng = np.random.default_rng(0)
    for k in range(4):
        d = rng.standard_normal(3)
        d /= np.linalg.norm(d)
        test_directions.append((f"rand{k}", d))

    perturb_mag = 1.0  # 1m perturbation
    records = []
    for name, direction in test_directions:
        perturb = direction * perturb_mag
        pert_T = gt_T_torch.clone()
        pert_T[0, :3, 3] = pert_T[0, :3, 3] + torch.from_numpy(perturb.astype(np.float32)).cuda()
        grad_t, L = compute_grad_at_pose(model, voxel_state, pert_T, K, patch_pixel, bootstrap_d)
        # PREDICT: grad should point opposite to perturb (toward GT).
        # Compute cos(grad, -perturb): +1 means perfect (toward GT), -1 means flipped.
        grad_norm = np.linalg.norm(grad_t)
        cos_toward = float(np.dot(grad_t, -direction) / max(grad_norm, 1e-9))
        records.append({"name": name, "perturb": perturb, "grad_t": grad_t,
                         "cos_toward_GT": cos_toward, "L": L,
                         "grad_norm": float(grad_norm)})

    # Plot: 3 panels (xy, xz, yz). Each shows GT, and arrows from perturbed-pose → gradient direction.
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for k, (i1, i2, name) in enumerate([(0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz")]):
        ax = axes[k]
        ax.scatter(gt_pos[i1], gt_pos[i2], color="green", s=200, marker="*", label="GT pose")
        for rec in records:
            perturbed_pos = gt_pos + rec["perturb"]
            ax.scatter(perturbed_pos[i1], perturbed_pos[i2], color="orange", s=50,
                        marker="o", alpha=0.7)
            # Arrow from perturbed position in direction of (negative) gradient
            # — i.e., the direction the optimizer would move next.
            move_dir = -rec["grad_t"] / max(rec["grad_norm"], 1e-9)
            ax.annotate("", xy=(perturbed_pos[i1] + move_dir[i1] * 0.3,
                                  perturbed_pos[i2] + move_dir[i2] * 0.3),
                         xytext=(perturbed_pos[i1], perturbed_pos[i2]),
                         arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
            ax.text(perturbed_pos[i1] + 0.05, perturbed_pos[i2] + 0.05,
                     f"{rec['name']}: cos={rec['cos_toward_GT']:+.2f}", fontsize=7)
        ax.set_xlabel(["X", "X", "Y"][k])
        ax.set_ylabel(["Y", "Z", "Z"][k])
        ax.set_title(f"{name} — orange=perturbed pose, red arrow=−grad direction")
        ax.grid(alpha=0.3); ax.legend(); ax.set_aspect("equal", adjustable="datalim")

    cos_summary = ", ".join(f"{r['name']}={r['cos_toward_GT']:+.2f}" for r in records)
    fig.suptitle(f"Gradient sanity @ frame {frame_idx} — "
                  f"red arrow should point BACK TO GT (green star). "
                  f"cos=+1: perfect. cos=-1: SIGN BUG (gradient points AWAY).\n{cos_summary}",
                  fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / f"gradient_sanity_f{frame_idx:04d}.png",
                 dpi=110, bbox_inches="tight")
    plt.close(fig)


def viz_loss_landscape(out_dir: Path, frame_idx: int,
                        model, voxel_state, gt_T_torch, pred_pose_t,
                        K, patch_pixel, bootstrap_d):
    """G: L_depth(tx, ty) on a 21x21 grid around GT pose. Reveals multi-basin
    loss surfaces and the location of the global minimum relative to GT."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_pos = gt_T_torch[0, :3, 3].float().cpu().numpy()
    grid_range_m = 2.0
    n_grid = 21
    offsets = np.linspace(-grid_range_m, grid_range_m, n_grid)
    L_grid = np.zeros((n_grid, n_grid))
    for ix, dx in enumerate(offsets):
        for iy, dy in enumerate(offsets):
            pert_T = gt_T_torch.clone()
            pert_T[0, 0, 3] = pert_T[0, 0, 3] + dx
            pert_T[0, 1, 3] = pert_T[0, 1, 3] + dy
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ray_o, ray_d = build_rays_from_pose(pert_T, K, patch_pixel)
                    r = render_rays_volumetric(voxel_state, ray_o, ray_d,
                        n_samples=model.n_render_samples,
                        near=model.render_near, far=model.render_far)
                    r_depth = r["depth"].float()
                    mass = r["total_weight"].float()
                    valid = (mass > model.unwritten_mask_threshold).float()
                    diff = (r_depth - bootstrap_d.detach()).abs()
                    L = (diff * valid * mass).sum() / valid.sum().clamp_min(1.0)
            L_grid[ix, iy] = float(L)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(L_grid.T, origin="lower",
                    extent=(gt_pos[0] - grid_range_m, gt_pos[0] + grid_range_m,
                            gt_pos[1] - grid_range_m, gt_pos[1] + grid_range_m),
                    cmap="turbo", aspect="equal")
    ax.scatter(gt_pos[0], gt_pos[1], color="green", s=200, marker="*",
                label="GT pose", edgecolors="white", linewidths=1.5)
    ax.scatter(pred_pose_t[0], pred_pose_t[1], color="red", s=200, marker="x",
                label="pred pose", linewidths=2)
    # Mark the argmin
    argmin_ij = np.unravel_index(np.argmin(L_grid), L_grid.shape)
    argmin_pos = (gt_pos[0] + offsets[argmin_ij[0]], gt_pos[1] + offsets[argmin_ij[1]])
    ax.scatter(argmin_pos[0], argmin_pos[1], color="cyan", s=200, marker="o",
                label="argmin(L)", edgecolors="black", linewidths=1.5)
    plt.colorbar(im, ax=ax, label="L_depth")
    ax.set_xlabel("X (world)"); ax.set_ylabel("Y (world)")
    dist_argmin = np.linalg.norm(np.array(argmin_pos) - gt_pos[:2])
    ax.set_title(f"Loss landscape @ frame {frame_idx} — "
                  f"L_depth(tx, ty) around GT pose (z, rotation fixed at GT).\n"
                  f"Argmin {dist_argmin:.2f}m from GT — "
                  f"non-zero distance means the minimum isn't at GT (loss surface bias)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"loss_landscape_f{frame_idx:04d}.png",
                 dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.test_frames:
        (args.out / f"frame_{f:04d}").mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(args.ckpt, args.weights_root)
    img_size = {"vjepa": 384, "dinov2": 518, "dinov3": 512}[cfg["encoder"]]
    fx, fy, cx, cy = intrinsics_for(args.seq)
    sx, sy = img_size / 640.0, img_size / 480.0
    K = torch.tensor([[[fx*sx, 0, cx*sx], [0, fy*sy, cy*sy], [0, 0, 1]]], device="cuda")
    fov = torch.tensor([[1.0, 1.0]], device="cuda")
    recs = sync_sequence(args.data_root / args.seq)[:args.n_frames]
    print(f"[debug-viz] {args.seq}: streaming {len(recs)} frames")

    gt_poses = np.stack([r.pose_w_c for r in recs])
    P0_inv = np.linalg.inv(gt_poses[0])
    gt_rel = np.einsum("ij,njk->nik", P0_inv, gt_poses)
    gt_rel_T_np = gt_rel
    gt_rel_T = torch.from_numpy(gt_rel).float().cuda()

    # Stream + snapshot
    test_frame_set = set(args.test_frames)
    pred_traj, snaps = stream_and_snapshot(model, recs, K, fov, gt_rel_T, test_frame_set)
    voxel_bounds = tuple(cfg["model"]["voxel_bounds"])

    # === A. Trajectory ===
    print(f"[debug-viz] generating trajectory_3d.png")
    viz_trajectory(args.out, pred_traj, gt_rel_T_np, voxel_bounds)

    # === B, C, D, E, F, G per test frame ===
    patch_pixel = model._patch_pixel_grid.cuda().unsqueeze(0).float()
    for fi in args.test_frames:
        if fi not in snaps:
            continue
        print(f"[debug-viz] frame {fi} — depth panel, scatter, backprojection, voxel grid")
        snap = snaps[fi]
        voxel_state = restore_voxel_state(snap)
        gt_T = gt_rel_T[fi:fi+1]
        gt_pos = gt_T[0, :3, 3].float().cpu().numpy()
        pred_pos = snap["pred_pose_T"][:3, 3]
        bootstrap_d_patch = snap["bootstrap_d"]
        bootstrap_d_torch = torch.from_numpy(bootstrap_d_patch).unsqueeze(0).cuda()

        # GT depth (image res)
        rec = recs[fi]
        gt_depth_img = load_gt_depth_image(rec, img_size, cfg["data"]["depth_max_m"])
        rgb_pil = load_rgb(rec, img_size).squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Renders at GT and pred pose
        pred_T = torch.from_numpy(snap["pred_pose_T"]).unsqueeze(0).cuda()
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                ray_o_gt, ray_d_gt = build_rays_from_pose(gt_T, K, patch_pixel)
                r_gt = render_rays_volumetric(voxel_state, ray_o_gt, ray_d_gt,
                    n_samples=model.n_render_samples,
                    near=model.render_near, far=model.render_far)
                ray_o_pr, ray_d_pr = build_rays_from_pose(pred_T, K, patch_pixel)
                r_pr = render_rays_volumetric(voxel_state, ray_o_pr, ray_d_pr,
                    n_samples=model.n_render_samples,
                    near=model.render_near, far=model.render_far)
        r_depth_gt = r_gt["depth"][0].float().cpu().numpy()
        r_mass_gt = r_gt["total_weight"][0].float().cpu().numpy()
        r_depth_pr = r_pr["depth"][0].float().cpu().numpy()
        r_mass_pr = r_pr["total_weight"][0].float().cpu().numpy()

        # GT depth at patch resolution (for scatter)
        with torch.no_grad():
            gt_depth_t = torch.from_numpy(np.nan_to_num(gt_depth_img, nan=0)
                                          ).float().unsqueeze(0).unsqueeze(0).cuda()
            gt_depth_patch = F.adaptive_avg_pool2d(gt_depth_t,
                                                     (model.grid_h, model.grid_w)
                                                    ).squeeze().cpu().numpy().flatten()

        # B
        viz_depth_panel(args.out, fi, rgb_pil, gt_depth_img, bootstrap_d_patch,
                        r_depth_gt, r_mass_gt, r_depth_pr, r_mass_pr,
                        img_size, model.grid_h, model.grid_w,
                        gt_pos, pred_pos)
        # C
        viz_per_patch_scatter(args.out, fi, bootstrap_d_patch, gt_depth_patch,
                              r_depth_gt, r_depth_pr)
        # D — backproject patches at both poses
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                world_pts_gt = backproject_patches_to_world(
                    patch_pixel, bootstrap_d_torch, K, gt_T
                )[0].float().cpu().numpy()
                world_pts_pr = backproject_patches_to_world(
                    patch_pixel, bootstrap_d_torch, K, pred_T
                )[0].float().cpu().numpy()
        viz_backprojection(args.out, fi, world_pts_gt, world_pts_pr,
                            gt_pos, pred_pos, voxel_bounds)
        # E — voxel grid content
        viz_voxel_grid(args.out, fi, voxel_state,
                        gt_rel_T_np[:fi+1, :3, 3], np.stack([p[:3, 3] for p in pred_traj[:fi+1]]))
        # F, G — only on a couple of frames (heavier compute)
        if fi in [args.test_frames[2], args.test_frames[3]]:        # middle frames
            print(f"[debug-viz]   frame {fi} — gradient sanity + loss landscape (heavier)")
            viz_gradient_sanity(args.out, fi, model, voxel_state, gt_T, K, patch_pixel,
                                 bootstrap_d_torch)
            viz_loss_landscape(args.out, fi, model, voxel_state, gt_T, pred_pos,
                                K, patch_pixel, bootstrap_d_torch)
    print(f"\n[debug-viz] all artifacts saved under {args.out}/")


if __name__ == "__main__":
    main()
