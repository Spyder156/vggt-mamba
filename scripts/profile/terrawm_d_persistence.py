"""TerraWM-D persistence test — distinguishes 'persistent scene memory' from
'short-horizon fusion buffer'.

The original ablation (reset at frame 200, measure frames 200-249) confirmed
the voxel grid is load-bearing but couldn't distinguish persistent-vs-buffer
because post-reset late mean was 0.000m — could mean either (a) the buffer
is full again by then or (b) the new camera views don't overlap with the
wiped early writes anyway.

This test sweeps the reset point R across {none, 100, 200, 300, 400, 500}
and measures Δt-divergence at a single late evaluation frame K=600.

  - PERSISTENT scene: divergence grows monotonically as R drops earlier
    (more useful writes wiped). Curve: decreasing in R.
  - BUFFER (~100-frame window): divergence at frame K is roughly the same
    for all R ≤ K - 100 because the buffer is always "full" of the most
    recent 100 writes. Curve: flat for R ≤ 500.

Also captures voxel state snapshots at frame K for each reset point, so we
can render the SAME camera pose from each grid and visually compare
(if they look identical → buffer; if early-reset renders are degraded → persistent).
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

from vggt_mamba.data.tum_rgbd import sync_sequence, intrinsics_for                # noqa: E402
from vggt_mamba.models.terrawm_d import build_terrawm_d                           # noqa: E402
from vggt_mamba.models.voxel_grid import (                                         # noqa: E402
    build_rays_from_pose, init_voxel_state, render_rays_volumetric, reset_voxel_state,
)
from vggt_mamba.models.aggregators.anchor_pool import cam9_to_pose_w_c              # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--seq", default="rgbd_dataset_freiburg3_sitting_xyz")
    p.add_argument("--reset-points", type=int, nargs="+", default=[100, 200, 300, 400, 500])
    p.add_argument("--eval-frame", type=int, default=600)
    p.add_argument("--out-dir", type=Path,
                   default=Path("viz/output/terrawm_d_persistence"))
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


def snapshot_voxel_state(state):
    """Deep-copy voxel state (features + write_mass) to CPU for later rendering."""
    return {
        "features": state.features.detach().clone().cpu(),
        "write_mass": state.write_mass.detach().clone().cpu(),
    }


def restore_voxel_state(model, snapshot, device="cuda"):
    state = model.init_voxel_state(batch_size=1, device=device, dtype=torch.float32)
    state.features.copy_(snapshot["features"].to(device))
    state.write_mass.copy_(snapshot["write_mass"].to(device))
    return state


@torch.no_grad()
def stream_one_run(model, recs, K, fov, reset_frame: int | None, snapshot_at: int):
    """Stream sequence; return (per-frame pose array, voxel state snapshot at snapshot_at)."""
    img_size = model.img_size
    voxel_state = model.init_voxel_state(batch_size=1, device="cuda", dtype=torch.float32)
    prev_pose = torch.tensor([[0., 0., 0., 0., 0., 0., 1., 1.0, 1.0]],
                              device="cuda", dtype=torch.float32)
    poses = []
    snap = None
    for i, rec in enumerate(recs):
        if reset_frame is not None and i == reset_frame:
            reset_voxel_state(voxel_state)
        rgb = load_rgb(rec, img_size).cuda(non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out, corrected = model.streaming_forward(rgb, voxel_state, prev_pose, K, fov=fov)
        poses.append(corrected[0].float().cpu().numpy())
        prev_pose = corrected.float()
        if i == snapshot_at:
            snap = snapshot_voxel_state(voxel_state)
    return np.stack(poses), snap


@torch.no_grad()
def render_at_pose(model, voxel_state, pose_9, K, img_size: int) -> dict:
    """Render the voxel grid at a given pose; return depth + mass at full resolution
    (using per-patch rays + bilinear upsample, same as the model's dense output)."""
    pose_T = cam9_to_pose_w_c(pose_9.cuda())
    P = model.n_patches
    patch_pixel = model._patch_pixel_grid.cuda().unsqueeze(0)             # (1, P, 2)
    ray_o, ray_d = build_rays_from_pose(pose_T, K, patch_pixel)
    out = render_rays_volumetric(
        voxel_state, ray_o, ray_d,
        n_samples=model.n_render_samples, near=model.render_near, far=model.render_far,
    )
    patch_depth = out["depth"]                                            # (1, P)
    patch_mass = out["total_weight"]                                      # (1, P)
    import torch.nn.functional as F
    H = W = img_size
    dense_d = F.interpolate(
        patch_depth.view(1, 1, model.grid_h, model.grid_w),
        size=(H, W), mode="bilinear", align_corners=True
    ).squeeze().cpu().numpy()
    dense_m = F.interpolate(
        patch_mass.view(1, 1, model.grid_h, model.grid_w),
        size=(H, W), mode="bilinear", align_corners=True
    ).squeeze().cpu().numpy()
    return {"depth": dense_d, "mass": dense_m}


def orbit_poses(center_xyz: np.ndarray, radius: float, height: float, n: int = 8) -> np.ndarray:
    """Generate n camera poses orbiting `center_xyz` at the given radius, looking inward.
    Returns (n, 9) cam9 vectors."""
    out = []
    for i in range(n):
        theta = 2 * np.pi * i / n
        cam_pos = center_xyz + np.array([radius * np.cos(theta), height, radius * np.sin(theta)])
        # Look-at: z axis points from camera to center; build a right-handed frame.
        z = center_xyz - cam_pos
        z = z / np.linalg.norm(z)
        up = np.array([0.0, 1.0, 0.0])
        x = np.cross(up, z); x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z], axis=1)                                    # columns = camera basis in world
        # cam9: [tx,ty,tz, qx,qy,qz,qw, fovx,fovy]
        from scipy.spatial.transform import Rotation as Rot
        q = Rot.from_matrix(R).as_quat()                                   # (qx, qy, qz, qw)
        out.append(np.concatenate([cam_pos, q, [1.0, 1.0]]))
    return np.stack(out).astype(np.float32)


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
    recs = sync_sequence(args.data_root / args.seq)
    print(f"[d-persist] {args.seq}: {len(recs)} frames available, "
          f"streaming to frame {args.eval_frame}, "
          f"reset points {args.reset_points}")
    n_stream = args.eval_frame + 1
    recs = recs[:n_stream]

    # Run "continuous" (no reset) first; serves as the reference.
    all_runs = {}  # R_label -> {'poses': (T, 9), 'snapshot': dict}
    R_to_run = [("continuous", None)] + [(f"reset_{R}", R) for R in args.reset_points]
    for label, R in R_to_run:
        t0 = time.perf_counter()
        poses, snap = stream_one_run(model, recs, K, fov, reset_frame=R, snapshot_at=args.eval_frame)
        dt = time.perf_counter() - t0
        all_runs[label] = {"poses": poses, "snapshot": snap}
        print(f"[d-persist]   {label:15s}  streamed in {dt:.1f}s  "
              f"voxel-mass at frame {args.eval_frame}: {snap['write_mass'].sum().item():.1f}")

    # === Persistence curve ===
    ref_poses = all_runs["continuous"]["poses"]
    eval_idx = args.eval_frame
    R_vals = []
    divergences = []
    for label, R in R_to_run[1:]:                                          # skip continuous
        poses = all_runs[label]["poses"]
        # Mean Δt-magnitude difference over a window around the eval frame.
        # Use frames [eval_frame-20, eval_frame] to smooth over noise.
        win_lo = max(R + 50, eval_idx - 20)
        win_hi = eval_idx + 1
        if win_hi <= win_lo:
            div = float(np.linalg.norm(ref_poses[eval_idx, :3] - poses[eval_idx, :3]))
        else:
            diff = ref_poses[win_lo:win_hi, :3] - poses[win_lo:win_hi, :3]
            div = float(np.linalg.norm(diff, axis=-1).mean())
        R_vals.append(R)
        divergences.append(div)
        print(f"[d-persist]   R={R:3d}: mean Δt-diff at frame {eval_idx} = {div:.5f} m")

    # Plot persistence curve.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(R_vals, divergences, "o-", linewidth=2, markersize=8, color="tab:blue")
    ax.set_xlabel("reset frame R  (early reset = more pre-reset writes wiped)")
    ax.set_ylabel(f"mean Δt-diff vs continuous, at frame {eval_idx} (m)")
    ax.set_title(
        "TerraWM-D persistence sweep\n"
        "PERSISTENT scene: divergence decreases monotonically with R (early reset → more divergence)\n"
        "BUFFER (~100-frame window): flat curve (early writes don't matter)"
    )
    ax.grid(alpha=0.3)
    # Annotate the buffer-window expectation line.
    ax.axhline(y=0.02, color="gray", linestyle=":", alpha=0.6, label="0.02 m (negligible threshold)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "persistence_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[d-persist]   saved persistence_curve.png")

    # === Same-pose render comparison ===
    # Render the EVAL FRAME's pose (from continuous run, which is best-trained) from each
    # voxel-state snapshot. If all renders look the same → buffer; if early-reset renders
    # are degraded → persistent.
    eval_pose_9 = torch.tensor(all_runs["continuous"]["poses"][eval_idx]).unsqueeze(0)
    renders = {}
    for label, _ in R_to_run:
        snap = all_runs[label]["snapshot"]
        state = restore_voxel_state(model, snap)
        renders[label] = render_at_pose(model, state, eval_pose_9, K, img_size)

    # Plot side-by-side depth + mass.
    n_runs = len(R_to_run)
    fig, axes = plt.subplots(2, n_runs, figsize=(3 * n_runs, 6))
    depth_min, depth_max = 0.0, 8.0
    for col, (label, _) in enumerate(R_to_run):
        d = renders[label]["depth"]
        m = renders[label]["mass"]
        axes[0, col].imshow(d, cmap="viridis", vmin=depth_min, vmax=depth_max)
        axes[0, col].set_title(label, fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].imshow(m, cmap="hot", vmin=0, vmax=1.5)
        axes[1, col].axis("off")
    axes[0, 0].set_ylabel("rendered depth (m)", fontsize=10)
    axes[1, 0].set_ylabel("total ray weight (coverage)", fontsize=10)
    fig.suptitle(
        f"Same-pose render comparison @ frame {eval_idx} pose, from each voxel-grid snapshot\n"
        "If renders look identical → grid is a buffer (early writes don't affect frame-{eval_idx} render)\n"
        "If early-reset (R=100) is degraded vs continuous → grid is persistent (early writes still influence)",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(args.out_dir / "same_pose_renders.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[d-persist]   saved same_pose_renders.png")

    # === Novel-view orbit (continuous run only) ===
    # Take the continuous-run voxel state, render from 8 novel poses orbiting the scene
    # center. Visual proof of "the grid is a 3D scene" (or evidence it isn't).
    snap = all_runs["continuous"]["snapshot"]
    state = restore_voxel_state(model, snap)
    # Center = the average of the continuous-run camera trajectory (rough scene center proxy).
    cam_traj = all_runs["continuous"]["poses"][:eval_idx + 1, :3]
    center = cam_traj.mean(axis=0).astype(np.float32)
    # Orbit at radius 1.5 m around center, at a height equal to the trajectory's mean y.
    novel_poses = orbit_poses(center, radius=1.5, height=center[1], n=8)
    novel_renders = []
    for i in range(8):
        p9 = torch.tensor(novel_poses[i:i+1])
        novel_renders.append(render_at_pose(model, state, p9, K, img_size))
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for i in range(8):
        r = i // 4; c = i % 4
        axes[r, c].imshow(novel_renders[i]["depth"], cmap="viridis", vmin=0, vmax=6)
        axes[r, c].set_title(f"orbit angle {i * 45}°", fontsize=9)
        axes[r, c].axis("off")
    fig.suptitle(
        f"Novel-view depth orbit from continuous-run voxel grid at frame {eval_idx}\n"
        "8 cameras orbiting the scene center; if the grid is a coherent 3D scene, "
        "renders show structure that looks like 'depth from this new angle'",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(args.out_dir / "novel_view_orbit.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[d-persist]   saved novel_view_orbit.png")

    # === Write summary JSON ===
    summary = {
        "ckpt": str(args.ckpt),
        "seq": args.seq,
        "eval_frame": eval_idx,
        "reset_points": args.reset_points,
        "divergences_at_eval_frame": dict(zip(R_vals, divergences)),
        "voxel_mass_at_eval_frame": {
            label: float(all_runs[label]["snapshot"]["write_mass"].sum())
            for label, _ in R_to_run
        },
    }
    # Verdict heuristic on persistence curve.
    div_arr = np.array(divergences)
    if len(div_arr) >= 3:
        # Persistent if monotonic decreasing AND spread > 0.05 m.
        spread = float(div_arr.max() - div_arr.min())
        slope = float(np.polyfit(R_vals, divergences, 1)[0])               # m per frame
        is_decreasing = slope < 0
        verdict = "PERSISTENT" if (is_decreasing and spread > 0.05) else "BUFFER"
        summary["spread_m"] = spread
        summary["slope_m_per_frame"] = slope
        summary["verdict"] = verdict
        print(f"[d-persist] persistence curve spread: {spread:.5f} m   "
              f"slope: {slope:+.2e} m/frame")
        print(f"[d-persist] VERDICT: {verdict}")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[d-persist] saved {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
