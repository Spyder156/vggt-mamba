"""TerraWM-D occupancy diagnostic — GT-pose variant.

Same diagnostic as `terrawm_d_occupancy.py`, but writes use the GROUND-TRUTH
camera pose instead of the model-predicted pose. This isolates the voxel
write mechanism from the pose head's drift:

  - If GT-pose writes give a fully populated, scene-coherent grid:
    → architecture is sound, the pose head is the bottleneck.
  - If GT-pose writes are still sparse/incoherent:
    → write mechanism or bootstrap depth is sick (or the bounds are
       genuinely too small for the GT trajectory).

Bypasses the pose head entirely. Bootstrap depth runs as normal; the write
step uses the GT pose directly for backprojection. We don't bother running
the second render or the loss path — this is pure write diagnostic.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt_mamba.data.tum_rgbd import sync_sequence, intrinsics_for                 # noqa: E402
from vggt_mamba.models.terrawm_d import build_terrawm_d                            # noqa: E402
from vggt_mamba.models.voxel_grid import (                                          # noqa: E402
    backproject_patches_to_world, in_bounds_mask, world_to_grid_coords,
    write_voxels_trilinear,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--seq", default="rgbd_dataset_freiburg3_sitting_xyz")
    p.add_argument("--n-frames", type=int, default=601)
    p.add_argument("--snapshot-frames", type=int, nargs="+",
                   default=[50, 100, 200, 400, 600])
    p.add_argument("--out-dir", type=Path,
                   default=Path("viz/output/terrawm_d_occupancy_gt"))
    return p.parse_args()


def load_model(ckpt_path, weights_root):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_terrawm_d(
        cfg["encoder"], str(weights_root),
        n_intraframe_layers=cfg["model"]["n_intraframe_layers"],
        voxel_bounds=tuple(cfg["model"]["voxel_bounds"]),
        voxel_resolution=tuple(cfg["model"]["voxel_resolution"]),
        voxel_feature_dim=cfg["model"]["voxel_feature_dim"],
        n_render_samples=cfg["model"]["n_render_samples"],
        render_near=cfg["model"]["render_near"],
        render_far=cfg["model"]["render_far"],
        bootstrap_hidden=cfg["model"]["bootstrap_hidden"],
        bootstrap_max_depth=cfg["model"]["bootstrap_max_depth"],
        pose_head_hidden=cfg["model"]["pose_head_hidden"],
        pose_max_dt=cfg["model"]["pose_max_dt"],
        pose_max_dq=cfg["model"]["pose_max_dq"],
        unwritten_mask_threshold=cfg["model"]["unwritten_mask_threshold"],
        use_write_confidence=cfg["model"].get("use_write_confidence", False),
        write_confidence_hidden=cfg["model"].get("write_confidence_hidden", 64),
        differentiable_write_geometry=cfg["model"].get("differentiable_write_geometry", False),
    )
    model.load_state_dict(ckpt["model"], strict=False)
    return model.cuda().eval(), cfg


def load_rgb(rec, img_size):
    from PIL import Image
    img = Image.open(rec.rgb_path).convert("RGB").resize((img_size, img_size))
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0
                            ).permute(2, 0, 1).unsqueeze(0).contiguous()


@torch.no_grad()
def frame_step_gt_pose(model, rgb, gt_pose_T, voxel_state, K, fov):
    """Write into the voxel grid using bootstrap depth + GT pose. Skip pose
    head, skip the second render. Returns per-frame diagnostic stats.
    """
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        patches = model._encode_frame(rgb)
        # No pose head: use GT pose directly.
        bootstrap_d = model.bootstrap_depth(patches)                              # (B, P)
        # Voxel feature projection (same as in _frame_step).
        voxel_feat = model.patch_to_voxel(patches)                                # (B, P, voxel_dim)
    patch_pixel = model._patch_pixel(rgb.shape[0], rgb.device)                    # (B, P, 2)
    world_pts = backproject_patches_to_world(
        patch_pixel, bootstrap_d.float(), K, gt_pose_T,
    )                                                                              # (B, P, 3)
    # Stats: how many patches are in-bounds, where they project.
    gc = world_to_grid_coords(world_pts, voxel_state.cfg)
    inb = in_bounds_mask(gc, voxel_state.cfg).float()
    cam_t = gt_pose_T[0, :3, 3].cpu().numpy()
    stats = {
        "cam_pos": cam_t,
        "patch_depth_p50": float(bootstrap_d.median()),
        "patch_depth_p95": float(torch.quantile(bootstrap_d.float(), 0.95)),
        "world_z_p50": float(world_pts[0, :, 2].median()),
        "world_z_p95": float(torch.quantile(world_pts[0, :, 2], 0.95)),
        "world_z_p05": float(torch.quantile(world_pts[0, :, 2], 0.05)),
        "frac_in_bounds": float(inb.mean()),
    }
    # Do the write.
    write_voxels_trilinear(voxel_state, world_pts, voxel_feat.float())
    return stats


def snapshot_grid(voxel_state):
    return voxel_state.write_mass.detach().clone().cpu().squeeze(-1).squeeze(0).numpy()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model, cfg = load_model(args.ckpt, args.weights_root)
    img_size = {"vjepa": 384, "dinov2": 518, "dinov3": 512}[cfg["encoder"]]
    fx, fy, cx, cy = intrinsics_for(args.seq)
    sx, sy = img_size / 640.0, img_size / 480.0
    K = torch.tensor([[[fx * sx, 0., cx * sx], [0., fy * sy, cy * sy], [0., 0., 1.]]],
                     device="cuda")
    fov = torch.tensor([[1.0, 1.0]], device="cuda")
    recs = sync_sequence(args.data_root / args.seq)[:args.n_frames]

    voxel_bounds = tuple(cfg["model"]["voxel_bounds"])
    voxel_res = tuple(cfg["model"]["voxel_resolution"])
    print(f"[d-occ-gt] bounds={voxel_bounds}  res={voxel_res}")
    print(f"[d-occ-gt] streaming {len(recs)} frames continuously, GT poses")

    # Relativize GT poses to frame 0 (matches training convention: pose[0] = identity).
    gt_poses_w_c = np.stack([r.pose_w_c for r in recs])                          # (T, 4, 4)
    P0 = gt_poses_w_c[0]
    P0_inv = np.linalg.inv(P0)
    gt_rel = np.einsum("ij,njk->nik", P0_inv, gt_poses_w_c)                       # (T, 4, 4)
    print(f"[d-occ-gt] GT trajectory: x range {gt_rel[:, 0, 3].min():.2f}/{gt_rel[:, 0, 3].max():.2f}  "
          f"y range {gt_rel[:, 1, 3].min():.2f}/{gt_rel[:, 1, 3].max():.2f}  "
          f"z range {gt_rel[:, 2, 3].min():.2f}/{gt_rel[:, 2, 3].max():.2f}")

    voxel_state = model.init_voxel_state(batch_size=1, device="cuda", dtype=torch.float32)
    snapshots = {}
    per_frame = []
    snapshot_set = set(args.snapshot_frames)

    t0 = time.perf_counter()
    for i, rec in enumerate(recs):
        rgb = load_rgb(rec, img_size).cuda(non_blocking=True)
        gt_pose_T = torch.from_numpy(gt_rel[i]).float().unsqueeze(0).cuda()      # (1, 4, 4)
        stats = frame_step_gt_pose(model, rgb, gt_pose_T, voxel_state, K, fov)
        stats["frame"] = i
        stats["voxel_mass_total"] = float(voxel_state.write_mass.sum())
        per_frame.append(stats)
        if i in snapshot_set:
            snapshots[i] = snapshot_grid(voxel_state)
            print(f"[d-occ-gt]   f={i:4d}  mass={stats['voxel_mass_total']:.1f}  "
                  f"depth p50={stats['patch_depth_p50']:.2f}m p95={stats['patch_depth_p95']:.2f}m  "
                  f"world_z p05={stats['world_z_p05']:+.2f}m p95={stats['world_z_p95']:+.2f}m  "
                  f"frac_inb={stats['frac_in_bounds']:.2f}")
    dt = time.perf_counter() - t0
    print(f"[d-occ-gt] streamed {len(recs)} frames in {dt:.1f}s")

    # === Per-frame trajectory plot ===
    frames = np.array([s["frame"] for s in per_frame])
    bootstrap_d_p50 = np.array([s["patch_depth_p50"] for s in per_frame])
    bootstrap_d_p95 = np.array([s["patch_depth_p95"] for s in per_frame])
    world_z_p05 = np.array([s["world_z_p05"] for s in per_frame])
    world_z_p95 = np.array([s["world_z_p95"] for s in per_frame])
    frac_inb = np.array([s["frac_in_bounds"] for s in per_frame])
    cam_pos = np.stack([s["cam_pos"] for s in per_frame])
    mass_total = np.array([s["voxel_mass_total"] for s in per_frame])

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes[0, 0].plot(frames, bootstrap_d_p50, label="p50", color="tab:blue")
    axes[0, 0].plot(frames, bootstrap_d_p95, label="p95", color="tab:orange")
    axes[0, 0].axhline(cfg["model"]["bootstrap_max_depth"], color="red", linestyle=":",
                       label=f"max={cfg['model']['bootstrap_max_depth']}m")
    axes[0, 0].set_title("bootstrap depth per frame (m)")
    axes[0, 0].set_xlabel("frame"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
    axes[0, 1].plot(frames, world_z_p05, label="world-z p05", color="tab:cyan")
    axes[0, 1].plot(frames, world_z_p95, label="world-z p95", color="tab:orange")
    axes[0, 1].axhline(voxel_bounds[2], color="red", linestyle=":", label=f"z_min={voxel_bounds[2]}")
    axes[0, 1].axhline(voxel_bounds[5], color="red", linestyle=":", label=f"z_max={voxel_bounds[5]}")
    axes[0, 1].set_title("world-z of written patches (GT pose) per frame")
    axes[0, 1].set_xlabel("frame"); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
    axes[1, 0].plot(frames, frac_inb, color="tab:green")
    axes[1, 0].set_title("fraction of patches in-bounds per frame (GT pose)")
    axes[1, 0].set_xlabel("frame"); axes[1, 0].set_ylim(-0.05, 1.05); axes[1, 0].grid(alpha=0.3)
    axes[1, 1].plot(cam_pos[:, 0], cam_pos[:, 2], color="tab:purple", linewidth=0.8)
    axes[1, 1].scatter(cam_pos[0, 0], cam_pos[0, 2], color="green", s=40, zorder=5, label="start")
    axes[1, 1].scatter(cam_pos[-1, 0], cam_pos[-1, 2], color="red", s=40, zorder=5, label="end")
    axes[1, 1].axvline(voxel_bounds[0], color="red", linestyle=":")
    axes[1, 1].axvline(voxel_bounds[3], color="red", linestyle=":")
    axes[1, 1].axhline(voxel_bounds[2], color="red", linestyle=":")
    axes[1, 1].axhline(voxel_bounds[5], color="red", linestyle=":")
    axes[1, 1].set_title("GT camera trajectory (xz plane) vs grid bounds")
    axes[1, 1].set_xlabel("x (m)"); axes[1, 1].set_ylabel("z (m)")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3); axes[1, 1].set_aspect("equal")
    axes[2, 0].plot(frames, mass_total, color="tab:red")
    axes[2, 0].set_title("total voxel mass accumulated vs frame (GT pose)")
    axes[2, 0].set_xlabel("frame"); axes[2, 0].grid(alpha=0.3)
    rate = np.gradient(mass_total)
    axes[2, 1].plot(frames, rate, color="tab:cyan")
    axes[2, 1].set_title("mass write rate (Δmass / frame)")
    axes[2, 1].set_xlabel("frame"); axes[2, 1].grid(alpha=0.3)
    fig.suptitle(f"TerraWM-D per-frame write diagnostics — GT POSE — {args.seq}", fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out_dir / "per_frame_diagnostics_gt.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[d-occ-gt]   saved per_frame_diagnostics_gt.png")

    # === Occupancy heatmaps per snapshot ===
    n_snaps = len(snapshots)
    snap_frames_sorted = sorted(snapshots.keys())
    fig, axes = plt.subplots(n_snaps, 6, figsize=(22, 3.5 * n_snaps))
    if n_snaps == 1:
        axes = axes[None, :]
    for row, fnum in enumerate(snap_frames_sorted):
        mass = snapshots[fnum]
        n_nonzero = int((mass > 0).sum())
        total_mass = float(mass.sum())
        m_xy = mass.sum(axis=2); m_xz = mass.sum(axis=1); m_yz = mass.sum(axis=0)
        m_x = mass.sum(axis=(1, 2)); m_y = mass.sum(axis=(0, 2)); m_z = mass.sum(axis=(0, 1))
        v_x, v_y, v_z = mass.shape
        axes[row, 0].imshow(m_xy.T, origin="lower", cmap="hot", aspect="auto",
                            extent=[voxel_bounds[0], voxel_bounds[3], voxel_bounds[1], voxel_bounds[4]])
        axes[row, 0].set_title(f"f={fnum}  xy (sum-z)\nmass={total_mass:.0f}  nonzero={n_nonzero}/{v_x*v_y*v_z}", fontsize=8)
        axes[row, 0].set_xlabel("x (m)"); axes[row, 0].set_ylabel("y (m)")
        axes[row, 1].imshow(m_xz.T, origin="lower", cmap="hot", aspect="auto",
                            extent=[voxel_bounds[0], voxel_bounds[3], voxel_bounds[2], voxel_bounds[5]])
        axes[row, 1].set_title(f"f={fnum}  xz (sum-y)", fontsize=8)
        axes[row, 1].set_xlabel("x (m)"); axes[row, 1].set_ylabel("z (m)")
        axes[row, 2].imshow(m_yz.T, origin="lower", cmap="hot", aspect="auto",
                            extent=[voxel_bounds[1], voxel_bounds[4], voxel_bounds[2], voxel_bounds[5]])
        axes[row, 2].set_title(f"f={fnum}  yz (sum-x)", fontsize=8)
        axes[row, 2].set_xlabel("y (m)"); axes[row, 2].set_ylabel("z (m)")
        x_pos = np.linspace(voxel_bounds[0], voxel_bounds[3], v_x)
        y_pos = np.linspace(voxel_bounds[1], voxel_bounds[4], v_y)
        z_pos = np.linspace(voxel_bounds[2], voxel_bounds[5], v_z)
        axes[row, 3].bar(x_pos, m_x, width=(voxel_bounds[3]-voxel_bounds[0])/v_x, color="tab:blue")
        axes[row, 3].set_title(f"f={fnum}  marginal x", fontsize=8); axes[row, 3].set_xlabel("x")
        axes[row, 4].bar(y_pos, m_y, width=(voxel_bounds[4]-voxel_bounds[1])/v_y, color="tab:blue")
        axes[row, 4].set_title(f"f={fnum}  marginal y", fontsize=8); axes[row, 4].set_xlabel("y")
        axes[row, 5].bar(z_pos, m_z, width=(voxel_bounds[5]-voxel_bounds[2])/v_z, color="tab:blue")
        axes[row, 5].set_title(f"f={fnum}  marginal z", fontsize=8); axes[row, 5].set_xlabel("z")
    fig.suptitle("Voxel occupancy with GT pose — same panels as the predicted-pose run for comparison",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(args.out_dir / "occupancy_marginals_gt.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[d-occ-gt]   saved occupancy_marginals_gt.png")

    # === Summary JSON ===
    summary = {
        "ckpt": str(args.ckpt),
        "seq": args.seq,
        "n_frames": len(recs),
        "voxel_bounds": list(voxel_bounds),
        "voxel_resolution": list(voxel_res),
        "snapshot_frames": snap_frames_sorted,
        "snapshots": {
            int(f): {
                "total_mass": float(snapshots[f].sum()),
                "n_nonzero_voxels": int((snapshots[f] > 0).sum()),
                "total_voxels": int(snapshots[f].size),
                "fill_fraction": float((snapshots[f] > 0).mean()),
            } for f in snap_frames_sorted
        },
        "per_frame_summary": {
            "bootstrap_depth_p50_mean": float(np.mean(bootstrap_d_p50)),
            "bootstrap_depth_p95_mean": float(np.mean(bootstrap_d_p95)),
            "world_z_p05_mean": float(np.mean(world_z_p05)),
            "world_z_p95_mean": float(np.mean(world_z_p95)),
            "frac_in_bounds_mean": float(np.mean(frac_inb)),
            "frac_in_bounds_min": float(np.min(frac_inb)),
            "frac_in_bounds_max": float(np.max(frac_inb)),
            "gt_cam_x_range_m": [float(cam_pos[:, 0].min()), float(cam_pos[:, 0].max())],
            "gt_cam_y_range_m": [float(cam_pos[:, 1].min()), float(cam_pos[:, 1].max())],
            "gt_cam_z_range_m": [float(cam_pos[:, 2].min()), float(cam_pos[:, 2].max())],
            "gt_cam_bbox_diag_m": float(np.linalg.norm(cam_pos.max(0) - cam_pos.min(0))),
        },
    }
    print(f"\n[d-occ-gt] === GT-POSE DIAGNOSIS ===")
    last_summary = summary["snapshots"][snap_frames_sorted[-1]]
    print(f"  in-bounds fraction (mean):                   {summary['per_frame_summary']['frac_in_bounds_mean']:.3f}")
    print(f"  in-bounds fraction (worst frame):            {summary['per_frame_summary']['frac_in_bounds_min']:.3f}")
    print(f"  world-z p95 mean (GT pose):                  {summary['per_frame_summary']['world_z_p95_mean']:+.3f} m")
    print(f"  final-snapshot fill_fraction:                {last_summary['fill_fraction']:.4f}  ({last_summary['n_nonzero_voxels']}/{last_summary['total_voxels']})")
    print(f"  GT cam bbox diagonal:                        {summary['per_frame_summary']['gt_cam_bbox_diag_m']:.3f} m")
    print(f"  Compare to predicted-pose run: cam_z reached -8.77m, fill_fraction = 0.102")
    print(f"  If GT-pose fill_fraction >> predicted-pose => pose head is the bottleneck")
    print(f"  If GT-pose fill_fraction ≈ predicted-pose => write mechanism or bootstrap depth is the issue")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[d-occ-gt] saved {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
