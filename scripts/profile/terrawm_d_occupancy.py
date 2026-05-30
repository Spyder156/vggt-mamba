"""TerraWM-D voxel-grid occupancy diagnostic.

Question: the continuous run's 20220 mass spreads across only ~8% of 64^3
voxels and renders nearly empty. Is that because writes are OOB-clipped at
the grid boundary (=> bounds fix would help), or because writes are sparse
even well inside the bounds (=> write mechanism / bootstrap depth is sick,
bounds fix changes nothing)?

This script streams the sequence continuously, snapshots voxel state +
per-frame write-position statistics at checkpoints, and dumps:

  1. Marginal mass projections (sum along each axis) -> 3 heatmaps showing
     the spatial distribution of writes.
  2. Per-axis 1D mass histograms (sum mass over the other 2 axes). If mass
     piles up at the boundary index (0 or V-1), writes are clipping there.
     If mass is interior with sparse coverage, write mechanism is sick.
  3. Per-frame write-position trajectory: for each frame, where is the
     camera, what depth is bootstrap predicting, what fraction of patches
     project to in-bounds world coords, and what's the mean world-z of
     written points.

Pure diagnostic. No training. ~15 minutes for 600 frames.
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
)
from vggt_mamba.models.aggregators.anchor_pool import cam9_to_pose_w_c              # noqa: E402


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
                   default=Path("viz/output/terrawm_d_occupancy"))
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
    )
    model.load_state_dict(ckpt["model"], strict=False)
    return model.cuda().eval(), cfg


def load_rgb(rec, img_size):
    from PIL import Image
    img = Image.open(rec.rgb_path).convert("RGB").resize((img_size, img_size))
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0
                            ).permute(2, 0, 1).unsqueeze(0).contiguous()


@torch.no_grad()
def per_frame_write_stats(model, rgb, voxel_state, prev_pose, K, fov):
    """Mimic streaming_forward but call _frame_step directly so we can read
    bootstrap_depth_patch (streaming_forward drops it). prev_pose is the
    previous ABSOLUTE pose; returns stats + the new absolute pose for the
    next iteration's prev_pose.
    """
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        patches = model._encode_frame(rgb)
        initial_T = cam9_to_pose_w_c(prev_pose)
        step_out = model._frame_step(patches, voxel_state, initial_T, K, fov)
    # Step output now exposes the new absolute pose via "corrected_pose_T".
    # "camera" is the predicted DELTA — not the absolute pose.
    pose_T = step_out["corrected_pose_T"].float()                         # (1, 4, 4) abs at this frame
    # Convert 4x4 -> 9-vec for the caller's feedback loop.
    from vggt_mamba.models.terrawm_d import _pose_T_to_cam9                # noqa: E402
    new_abs_pose_9 = _pose_T_to_cam9(pose_T, fov)                          # (1, 9)
    patch_depth = step_out["bootstrap_depth_patch"].float()               # (1, P)
    patch_pixel = model._patch_pixel_grid.cuda().unsqueeze(0).float()     # (1, P, 2)
    P_world = backproject_patches_to_world(patch_pixel, patch_depth, K, pose_T)
    # In-bounds check.
    cfg_v = voxel_state.cfg
    gc = world_to_grid_coords(P_world, cfg_v)                              # (1, P, 3)
    inb = in_bounds_mask(gc, cfg_v).float()                                # (1, P) bool->float
    cam_t = pose_T[0, :3, 3].cpu().numpy()                                 # (3,) camera world pos
    stats = {
        "cam_pos": cam_t,
        "patch_depth_mean": float(patch_depth.mean()),
        "patch_depth_p50": float(patch_depth.median()),
        "patch_depth_p95": float(torch.quantile(patch_depth, 0.95)),
        "patch_depth_max": float(patch_depth.max()),
        "world_z_mean": float(P_world[0, :, 2].mean()),
        "world_z_p50": float(P_world[0, :, 2].median()),
        "world_z_p95": float(torch.quantile(P_world[0, :, 2], 0.95)),
        "frac_in_bounds": float(inb.mean()),
        "gc_x_mean": float(gc[0, :, 0].mean()),
        "gc_y_mean": float(gc[0, :, 1].mean()),
        "gc_z_mean": float(gc[0, :, 2].mean()),
    }
    return stats, new_abs_pose_9


def snapshot_grid(voxel_state):
    return voxel_state.write_mass.detach().clone().cpu().squeeze(-1).squeeze(0).numpy()  # (V_x, V_y, V_z)


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
    voxel_size = [(voxel_bounds[i+3] - voxel_bounds[i]) / voxel_res[i] for i in range(3)]
    print(f"[d-occ] bounds={voxel_bounds}  res={voxel_res}  voxel_size={voxel_size}")
    print(f"[d-occ] streaming {len(recs)} frames continuously")

    voxel_state = model.init_voxel_state(batch_size=1, device="cuda", dtype=torch.float32)
    prev_pose = torch.tensor([[0., 0., 0., 0., 0., 0., 1., 1.0, 1.0]],
                              device="cuda", dtype=torch.float32)

    snapshots = {}                                                # f -> (V_x, V_y, V_z) mass
    per_frame = []                                                # list of stats dicts
    snapshot_set = set(args.snapshot_frames)
    t0 = time.perf_counter()
    for i, rec in enumerate(recs):
        rgb = load_rgb(rec, img_size).cuda(non_blocking=True)
        stats, corrected = per_frame_write_stats(model, rgb, voxel_state, prev_pose, K, fov)
        stats["frame"] = i
        stats["voxel_mass_total"] = float(voxel_state.write_mass.sum())
        per_frame.append(stats)
        prev_pose = corrected.float()
        if i in snapshot_set:
            snapshots[i] = snapshot_grid(voxel_state)
            print(f"[d-occ]   f={i:4d}  mass={stats['voxel_mass_total']:.1f}  "
                  f"depth p50={stats['patch_depth_p50']:.2f}m p95={stats['patch_depth_p95']:.2f}m  "
                  f"world_z p50={stats['world_z_p50']:+.2f}m p95={stats['world_z_p95']:+.2f}m  "
                  f"frac_inb={stats['frac_in_bounds']:.2f}")
    dt = time.perf_counter() - t0
    print(f"[d-occ] streamed {len(recs)} frames in {dt:.1f}s")

    # === Per-frame trajectory plots ===
    frames = np.array([s["frame"] for s in per_frame])
    bootstrap_d_p50 = np.array([s["patch_depth_p50"] for s in per_frame])
    bootstrap_d_p95 = np.array([s["patch_depth_p95"] for s in per_frame])
    world_z_p50 = np.array([s["world_z_p50"] for s in per_frame])
    world_z_p95 = np.array([s["world_z_p95"] for s in per_frame])
    frac_inb = np.array([s["frac_in_bounds"] for s in per_frame])
    cam_pos = np.stack([s["cam_pos"] for s in per_frame])           # (T, 3)
    mass_total = np.array([s["voxel_mass_total"] for s in per_frame])

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    # (0,0) bootstrap depth percentiles vs frame.
    axes[0, 0].plot(frames, bootstrap_d_p50, label="p50", color="tab:blue")
    axes[0, 0].plot(frames, bootstrap_d_p95, label="p95", color="tab:orange")
    axes[0, 0].axhline(cfg["model"]["bootstrap_max_depth"], color="red", linestyle=":",
                       label=f"max={cfg['model']['bootstrap_max_depth']}m")
    axes[0, 0].set_title("bootstrap depth per frame (m)")
    axes[0, 0].set_xlabel("frame"); axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)
    # (0,1) world-z of written points percentiles vs frame.
    axes[0, 1].plot(frames, world_z_p50, label="world-z p50", color="tab:blue")
    axes[0, 1].plot(frames, world_z_p95, label="world-z p95", color="tab:orange")
    axes[0, 1].axhline(voxel_bounds[2], color="red", linestyle=":", label=f"z_min={voxel_bounds[2]}")
    axes[0, 1].axhline(voxel_bounds[5], color="red", linestyle=":", label=f"z_max={voxel_bounds[5]}")
    axes[0, 1].set_title("world-z of written patches per frame (m)")
    axes[0, 1].set_xlabel("frame"); axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)
    # (1,0) fraction of patches landing in-bounds.
    axes[1, 0].plot(frames, frac_inb, color="tab:green")
    axes[1, 0].set_title("fraction of patches landing in-bounds per frame")
    axes[1, 0].set_xlabel("frame"); axes[1, 0].set_ylim(-0.05, 1.05); axes[1, 0].grid(alpha=0.3)
    # (1,1) camera trajectory in xz plane.
    axes[1, 1].plot(cam_pos[:, 0], cam_pos[:, 2], color="tab:purple", linewidth=0.8)
    axes[1, 1].scatter(cam_pos[0, 0], cam_pos[0, 2], color="green", s=40, zorder=5, label="start")
    axes[1, 1].scatter(cam_pos[-1, 0], cam_pos[-1, 2], color="red", s=40, zorder=5, label="end")
    # Bounds box in xz.
    axes[1, 1].axvline(voxel_bounds[0], color="red", linestyle=":")
    axes[1, 1].axvline(voxel_bounds[3], color="red", linestyle=":")
    axes[1, 1].axhline(voxel_bounds[2], color="red", linestyle=":")
    axes[1, 1].axhline(voxel_bounds[5], color="red", linestyle=":")
    axes[1, 1].set_title("camera trajectory (xz plane) vs grid bounds")
    axes[1, 1].set_xlabel("x (m)"); axes[1, 1].set_ylabel("z (m)")
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3); axes[1, 1].set_aspect("equal")
    # (2,0) total mass vs frame.
    axes[2, 0].plot(frames, mass_total, color="tab:red")
    axes[2, 0].set_title("total voxel mass accumulated vs frame")
    axes[2, 0].set_xlabel("frame"); axes[2, 0].grid(alpha=0.3)
    # (2,1) effective fill rate (mass / frame).
    rate = np.gradient(mass_total)
    axes[2, 1].plot(frames, rate, color="tab:cyan")
    axes[2, 1].set_title("mass write rate (Δmass / frame)")
    axes[2, 1].set_xlabel("frame"); axes[2, 1].grid(alpha=0.3)

    fig.suptitle(f"TerraWM-D per-frame write diagnostics — {args.seq}", fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out_dir / "per_frame_diagnostics.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[d-occ]   saved per_frame_diagnostics.png")

    # === Occupancy heatmaps per snapshot ===
    # For each snapshot, plot 3 marginal projections (sum along axis) + 3 1D histograms.
    n_snaps = len(snapshots)
    snap_frames_sorted = sorted(snapshots.keys())
    fig, axes = plt.subplots(n_snaps, 6, figsize=(22, 3.5 * n_snaps))
    if n_snaps == 1:
        axes = axes[None, :]
    for row, fnum in enumerate(snap_frames_sorted):
        mass = snapshots[fnum]                                              # (V_x, V_y, V_z)
        n_nonzero = int((mass > 0).sum())
        total_mass = float(mass.sum())
        # Marginal projections (sum along each axis).
        m_xy = mass.sum(axis=2)                                             # (V_x, V_y) view from +z
        m_xz = mass.sum(axis=1)                                             # (V_x, V_z) view from +y
        m_yz = mass.sum(axis=0)                                             # (V_y, V_z) view from +x
        # 1D marginal histograms (mass summed over the other two axes).
        m_x = mass.sum(axis=(1, 2))                                         # (V_x,)
        m_y = mass.sum(axis=(0, 2))                                         # (V_y,)
        m_z = mass.sum(axis=(0, 1))                                         # (V_z,)
        # Plot.
        v_x, v_y, v_z = mass.shape
        axes[row, 0].imshow(m_xy.T, origin="lower", cmap="hot", aspect="auto",
                            extent=[voxel_bounds[0], voxel_bounds[3], voxel_bounds[1], voxel_bounds[4]])
        axes[row, 0].set_title(f"f={fnum}  xy projection (sum over z)\nmass={total_mass:.0f}  nonzero={n_nonzero}/{v_x*v_y*v_z}", fontsize=8)
        axes[row, 0].set_xlabel("x (m)"); axes[row, 0].set_ylabel("y (m)")
        axes[row, 1].imshow(m_xz.T, origin="lower", cmap="hot", aspect="auto",
                            extent=[voxel_bounds[0], voxel_bounds[3], voxel_bounds[2], voxel_bounds[5]])
        axes[row, 1].set_title(f"f={fnum}  xz projection (sum over y)", fontsize=8)
        axes[row, 1].set_xlabel("x (m)"); axes[row, 1].set_ylabel("z (m)")
        axes[row, 2].imshow(m_yz.T, origin="lower", cmap="hot", aspect="auto",
                            extent=[voxel_bounds[1], voxel_bounds[4], voxel_bounds[2], voxel_bounds[5]])
        axes[row, 2].set_title(f"f={fnum}  yz projection (sum over x)", fontsize=8)
        axes[row, 2].set_xlabel("y (m)"); axes[row, 2].set_ylabel("z (m)")
        # 1D histograms (mass vs axis position).
        x_pos = np.linspace(voxel_bounds[0], voxel_bounds[3], v_x)
        y_pos = np.linspace(voxel_bounds[1], voxel_bounds[4], v_y)
        z_pos = np.linspace(voxel_bounds[2], voxel_bounds[5], v_z)
        axes[row, 3].bar(x_pos, m_x, width=(voxel_bounds[3]-voxel_bounds[0])/v_x, color="tab:blue")
        axes[row, 3].set_title(f"f={fnum}  mass marginal along x", fontsize=8)
        axes[row, 3].set_xlabel("x (m)"); axes[row, 3].axvline(voxel_bounds[0], color="red", linestyle=":")
        axes[row, 3].axvline(voxel_bounds[3], color="red", linestyle=":")
        axes[row, 4].bar(y_pos, m_y, width=(voxel_bounds[4]-voxel_bounds[1])/v_y, color="tab:blue")
        axes[row, 4].set_title(f"f={fnum}  mass marginal along y", fontsize=8)
        axes[row, 4].set_xlabel("y (m)"); axes[row, 4].axvline(voxel_bounds[1], color="red", linestyle=":")
        axes[row, 4].axvline(voxel_bounds[4], color="red", linestyle=":")
        axes[row, 5].bar(z_pos, m_z, width=(voxel_bounds[5]-voxel_bounds[2])/v_z, color="tab:blue")
        axes[row, 5].set_title(f"f={fnum}  mass marginal along z", fontsize=8)
        axes[row, 5].set_xlabel("z (m)"); axes[row, 5].axvline(voxel_bounds[2], color="red", linestyle=":")
        axes[row, 5].axvline(voxel_bounds[5], color="red", linestyle=":")

    fig.suptitle(
        "Voxel occupancy — left 3: spatial projections (where mass is)  "
        "right 3: 1D marginals along each axis (boundary pile-up = OOB clipping)\n"
        "If right-3 histograms have spikes at the red dotted lines (bounds), writes are clipping. "
        "If interior is sparse without boundary spikes, write mechanism is sick.",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(args.out_dir / "occupancy_marginals.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[d-occ]   saved occupancy_marginals.png")

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
                "mass_p95_voxel": float(np.quantile(snapshots[f][snapshots[f] > 0], 0.95))
                                  if (snapshots[f] > 0).any() else 0.0,
                # Per-axis edge concentration (mass in outer 10% of axis index range).
                "edge_frac_x": float((snapshots[f].sum(axis=(1, 2))[:max(1, voxel_res[0]//10)].sum()
                                      + snapshots[f].sum(axis=(1, 2))[-max(1, voxel_res[0]//10):].sum())
                                      / max(1e-9, snapshots[f].sum())),
                "edge_frac_y": float((snapshots[f].sum(axis=(0, 2))[:max(1, voxel_res[1]//10)].sum()
                                      + snapshots[f].sum(axis=(0, 2))[-max(1, voxel_res[1]//10):].sum())
                                      / max(1e-9, snapshots[f].sum())),
                "edge_frac_z": float((snapshots[f].sum(axis=(0, 1))[:max(1, voxel_res[2]//10)].sum()
                                      + snapshots[f].sum(axis=(0, 1))[-max(1, voxel_res[2]//10):].sum())
                                      / max(1e-9, snapshots[f].sum())),
            } for f in snap_frames_sorted
        },
        "per_frame_summary": {
            "bootstrap_depth_p50_mean": float(np.mean(bootstrap_d_p50)),
            "bootstrap_depth_p95_mean": float(np.mean(bootstrap_d_p95)),
            "world_z_p50_mean": float(np.mean(world_z_p50)),
            "world_z_p95_mean": float(np.mean(world_z_p95)),
            "frac_in_bounds_mean": float(np.mean(frac_inb)),
            "frac_in_bounds_min": float(np.min(frac_inb)),
            "cam_x_range_m": [float(cam_pos[:, 0].min()), float(cam_pos[:, 0].max())],
            "cam_y_range_m": [float(cam_pos[:, 1].min()), float(cam_pos[:, 1].max())],
            "cam_z_range_m": [float(cam_pos[:, 2].min()), float(cam_pos[:, 2].max())],
        },
    }
    print(f"\n[d-occ] === DIAGNOSIS ===")
    print(f"  in-bounds fraction (mean over frames):       {summary['per_frame_summary']['frac_in_bounds_mean']:.3f}")
    print(f"  in-bounds fraction (worst frame):            {summary['per_frame_summary']['frac_in_bounds_min']:.3f}")
    print(f"  world-z p95 mean over frames:                {summary['per_frame_summary']['world_z_p95_mean']:+.3f} m   (grid z: {voxel_bounds[2]} to {voxel_bounds[5]})")
    print(f"  bootstrap depth p95 mean over frames:        {summary['per_frame_summary']['bootstrap_depth_p95_mean']:.3f} m")
    last_snap = snapshots[snap_frames_sorted[-1]]
    last_summary = summary["snapshots"][snap_frames_sorted[-1]]
    print(f"  final-snapshot fill_fraction:                {last_summary['fill_fraction']:.4f}  ({last_summary['n_nonzero_voxels']}/{last_summary['total_voxels']})")
    print(f"  final-snapshot edge mass fraction  x/y/z:    {last_summary['edge_frac_x']:.3f} / {last_summary['edge_frac_y']:.3f} / {last_summary['edge_frac_z']:.3f}")
    print(f"  (edge_frac high => OOB clipping; edge_frac low + low fill => write mechanism sick)")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[d-occ] saved {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
