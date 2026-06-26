"""TerraWM-D GT-vs-pred render check — disambiguates write-coverage vs pose-drift.

The long-horizon collapse on fr1/room shows render coverage crashing around
f≈575-590. Two candidate mechanisms with very different fixes:

  - WRITE-COVERAGE: writes are too directional (~forward only at bootstrap
    depth), so when the camera looks in a different direction the rays hit
    unwritten directions. Fix: multi-direction / multi-depth write strategy.
  - POSE-DRIFT: predicted camera has drifted off the mapped region; the grid
    still has content in the directions writes were placed, just not where
    the now-drifted camera is looking. Fix: re-grounding / recovery mechanism.

The disambiguating test: at the frames where coverage crashes, render the
grid from the GT camera pose (where the writes ACTUALLY came from).

  - GT-pose render full (~1) + pred-pose render empty (~0) → POSE-DRIFT.
    Camera has left the map. Write-strategy fix wouldn't help.
  - GT-pose render also sparse → WRITE-COVERAGE. The grid genuinely lacks
    content in those directions regardless of viewing pose. Write-strategy
    fix would help.

Cheap to run: stream once, snapshot voxel state at chosen frames, then for
each snapshot compare renders from pred-pose vs GT-pose.
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
from vggt_mamba.models.aggregators.anchor_pool import cam9_to_pose_w_c              # noqa: E402
from vggt_mamba.models.terrawm_d import build_terrawm_d, _pose_T_to_cam9             # noqa: E402
from vggt_mamba.models.voxel_grid import (                                          # noqa: E402
    backproject_patches_to_world, build_rays_from_pose, init_voxel_state,
    render_rays_volumetric, write_voxels_trilinear,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--seq", default="rgbd_dataset_freiburg1_room")
    p.add_argument("--n-frames", type=int, default=1100)
    p.add_argument("--test-frames", type=int, nargs="+",
                   default=[100, 300, 500, 575, 600, 700, 900, 1050])
    p.add_argument("--out-dir", type=Path,
                   default=Path("viz/output/terrawm_d_gt_vs_pred_render"))
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
        pose_gate_mode=cfg["model"].get("pose_gate_mode", "coverage"),
    )
    model.load_state_dict(ckpt["model"], strict=False)
    return model.cuda().eval(), cfg


def load_rgb(rec, img_size):
    from PIL import Image
    img = Image.open(rec.rgb_path).convert("RGB").resize((img_size, img_size))
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0
                            ).permute(2, 0, 1).unsqueeze(0).contiguous()


def render_stats(voxel_state, pose_T, model, K, fov):
    """Render the grid from a fixed pose; return coverage/feat_norm/depth-mean."""
    patch_pixel = model._patch_pixel_grid.cuda().unsqueeze(0)                    # (1, P, 2)
    ray_o, ray_d = build_rays_from_pose(pose_T, K, patch_pixel)
    out = render_rays_volumetric(
        voxel_state, ray_o, ray_d,
        n_samples=model.n_render_samples, near=model.render_near, far=model.render_far,
    )
    coverage = float((out["total_weight"] > 1e-3).float().mean())
    mean_weight = float(out["total_weight"].mean())
    feat_norm = float(out["feature"].norm(dim=-1).mean())
    depth_mean = float(out["depth"].mean())
    depth_std = float(out["depth"].std())
    return {"coverage": coverage, "mean_weight": mean_weight,
            "feat_norm": feat_norm, "depth_mean": depth_mean, "depth_std": depth_std,
            "patch_total_weight": out["total_weight"][0].cpu().numpy(),
            "patch_depth": out["depth"][0].cpu().numpy()}


@torch.no_grad()
def stream_and_snapshot(model, recs, K, fov, gt_rel_T, test_frame_set):
    """Stream with pred poses; at each frame in test_frame_set, snapshot voxel_state
    and record (pred_pose, gt_pose, pose_displacement)."""
    img_size = model.img_size
    voxel_state = model.init_voxel_state(batch_size=1, device="cuda", dtype=torch.float32)
    prev_pose_9 = torch.tensor([[0., 0., 0., 0., 0., 0., 1., 1.0, 1.0]],
                                device="cuda", dtype=torch.float32)
    patch_pixel = model._patch_pixel_grid.cuda().unsqueeze(0).float()
    snapshots = {}
    pred_traj = []

    for i, rec in enumerate(recs):
        rgb = load_rgb(rec, img_size).cuda(non_blocking=True)
        initial_T = cam9_to_pose_w_c(prev_pose_9)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            patches = model._encode_frame(rgb)
            bootstrap_d = model.bootstrap_depth(patches).float()
            voxel_feat = model.patch_to_voxel(patches).float()
            write_w = model.write_confidence(patches).float() if model.use_write_confidence else None
            ray_o1, ray_d1 = build_rays_from_pose(initial_T, K, patch_pixel)
            render1 = render_rays_volumetric(
                voxel_state, ray_o1, ray_d1,
                n_samples=model.n_render_samples, near=model.render_near, far=model.render_far,
            )
            initial_pose_9 = _pose_T_to_cam9(initial_T, fov)
            # Mirror _frame_step's gate mode.
            if model.pose_gate_mode == "grid_mass":
                mass_total = voxel_state.write_mass.sum().detach()
                mass_gate = torch.sigmoid((mass_total - 1e3) / 1e2)
                external_gate = mass_gate.expand(1).unsqueeze(-1).to(initial_pose_9.dtype)
            else:
                external_gate = None
            delta_pose_9 = model.pose_head(patches, render1["feature"], render1["total_weight"],
                                            initial_pose_9, external_gate=external_gate)

        delta_pose_T = cam9_to_pose_w_c(delta_pose_9)
        corrected_pose_T = initial_T.float() @ delta_pose_T

        if i in test_frame_set:
            pred_pose_T_cpu = corrected_pose_T[0].cpu().numpy().copy()
            gt_pose_T_cpu = gt_rel_T[i].cpu().numpy().copy()
            disp = float(np.linalg.norm(pred_pose_T_cpu[:3, 3] - gt_pose_T_cpu[:3, 3]))
            snapshots[i] = {
                "features": voxel_state.features.detach().clone().cpu(),
                "write_mass": voxel_state.write_mass.detach().clone().cpu(),
                "pred_pose_T": pred_pose_T_cpu,
                "gt_pose_T": gt_pose_T_cpu,
                "displacement_m": disp,
            }
            print(f"[d-renderchk]   snap f={i:4d}  mass={voxel_state.write_mass.sum().item():.0f}  "
                  f"displacement = {disp:.4f}m")

        world_pts = backproject_patches_to_world(patch_pixel, bootstrap_d, K, corrected_pose_T.detach())
        write_voxels_trilinear(voxel_state, world_pts, voxel_feat, weights=write_w)

        new_abs_9 = _pose_T_to_cam9(corrected_pose_T, fov)
        prev_pose_9 = new_abs_9.float()
        pred_traj.append(corrected_pose_T[0, :3, 3].cpu().numpy())
        if (i + 1) % 200 == 0:
            print(f"[d-renderchk]   streamed {i + 1}/{len(recs)}")
    return snapshots, np.stack(pred_traj)


def restore_voxel_state(model, snap, device="cuda"):
    state = model.init_voxel_state(batch_size=1, device=device, dtype=torch.float32)
    state.features.copy_(snap["features"].to(device))
    state.write_mass.copy_(snap["write_mass"].to(device))
    return state


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
    print(f"[d-renderchk] {args.seq}: streaming {len(recs)} frames; test frames {args.test_frames}")

    # Relativize GT to frame 0.
    gt_poses_w_c = np.stack([r.pose_w_c for r in recs])
    P0_inv = np.linalg.inv(gt_poses_w_c[0])
    gt_rel = np.einsum("ij,njk->nik", P0_inv, gt_poses_w_c)
    gt_rel_T = torch.from_numpy(gt_rel).float().cuda()

    test_frame_set = set(args.test_frames)
    t0 = time.perf_counter()
    snapshots, pred_traj = stream_and_snapshot(model, recs, K, fov, gt_rel_T, test_frame_set)
    print(f"[d-renderchk] streamed in {time.perf_counter() - t0:.1f}s")

    # === Compare renders from pred-pose vs GT-pose for each snapshot ===
    print(f"\n[d-renderchk] === RENDER COMPARISON (pred-pose vs GT-pose) ===")
    print(f"{'frame':>6s} {'disp':>8s} {'mass':>10s}  "
          f"{'pred_cov':>9s} {'gt_cov':>8s}  "
          f"{'pred_fn':>10s} {'gt_fn':>10s}  "
          f"{'pred_d':>8s} {'gt_d':>8s}")
    rows = []
    for fnum in sorted(snapshots.keys()):
        snap = snapshots[fnum]
        state = restore_voxel_state(model, snap)
        pred_pose_T = torch.from_numpy(snap["pred_pose_T"]).float().cuda().unsqueeze(0)
        gt_pose_T = torch.from_numpy(snap["gt_pose_T"]).float().cuda().unsqueeze(0)
        pred_stats = render_stats(state, pred_pose_T, model, K, fov)
        gt_stats = render_stats(state, gt_pose_T, model, K, fov)
        mass = float(snap["write_mass"].sum())
        print(f"{fnum:6d} {snap['displacement_m']:8.4f} {mass:10.0f}  "
              f"{pred_stats['coverage']:9.3f} {gt_stats['coverage']:8.3f}  "
              f"{pred_stats['feat_norm']:10.3f} {gt_stats['feat_norm']:10.3f}  "
              f"{pred_stats['depth_mean']:8.3f} {gt_stats['depth_mean']:8.3f}")
        rows.append({
            "frame": fnum,
            "displacement_m": snap["displacement_m"],
            "mass_total": mass,
            "pred_coverage": pred_stats["coverage"],
            "gt_coverage": gt_stats["coverage"],
            "pred_feat_norm": pred_stats["feat_norm"],
            "gt_feat_norm": gt_stats["feat_norm"],
            "pred_depth_mean": pred_stats["depth_mean"],
            "gt_depth_mean": gt_stats["depth_mean"],
        })

    # === Verdict ===
    print(f"\n[d-renderchk] === MECHANISM VERDICT ===")
    # Find the latest frame where pred coverage is high (≥ 0.5) — model still in mapped region.
    # Find the first frame where pred coverage is low (< 0.2) — model has drifted.
    drifted = [r for r in rows if r["pred_coverage"] < 0.2]
    if not drifted:
        verdict = "NO_COVERAGE_CRASH_OBSERVED in test frames; pick later test frames"
    else:
        first_drift = drifted[0]
        gt_cov_at_drift = first_drift["gt_coverage"]
        disp_at_drift = first_drift["displacement_m"]
        if gt_cov_at_drift > 0.5:
            verdict = f"POSE-DRIFT (f={first_drift['frame']}): GT-pose render coverage {gt_cov_at_drift:.2f} " \
                       f"while pred-pose render coverage {first_drift['pred_coverage']:.2f}. " \
                       f"The grid HAS content; the predicted camera has drifted off it ({disp_at_drift:.2f}m). " \
                       f"FIX: re-grounding mechanism (option 2/3); option 1 won't help."
        elif gt_cov_at_drift < 0.3:
            verdict = f"WRITE-COVERAGE (f={first_drift['frame']}): GT-pose render coverage {gt_cov_at_drift:.2f} " \
                       f"(grid is also sparse from GT viewpoint). " \
                       f"FIX: multi-direction / multi-depth write strategy (option 1)."
        else:
            verdict = f"MIXED (f={first_drift['frame']}): GT-pose render {gt_cov_at_drift:.2f}, " \
                       f"pred-pose render {first_drift['pred_coverage']:.2f}. " \
                       f"Both contribute; fixes from both options likely needed."
    print(f"  {verdict}")

    # === Visualizations ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    frames = [r["frame"] for r in rows]
    pred_covs = [r["pred_coverage"] for r in rows]
    gt_covs = [r["gt_coverage"] for r in rows]
    disps = [r["displacement_m"] for r in rows]
    pred_fns = [r["pred_feat_norm"] for r in rows]
    gt_fns = [r["gt_feat_norm"] for r in rows]
    mass = [r["mass_total"] for r in rows]

    axes[0, 0].plot(frames, pred_covs, "o-", color="tab:red", label="pred-pose render")
    axes[0, 0].plot(frames, gt_covs, "o-", color="tab:green", label="GT-pose render")
    axes[0, 0].set_xlabel("frame"); axes[0, 0].set_ylabel("render coverage")
    axes[0, 0].set_title("Render coverage at snapshot frames")
    axes[0, 0].set_ylim(-0.05, 1.05); axes[0, 0].grid(alpha=0.3); axes[0, 0].legend()

    axes[0, 1].plot(frames, pred_fns, "o-", color="tab:red", label="pred-pose render")
    axes[0, 1].plot(frames, gt_fns, "o-", color="tab:green", label="GT-pose render")
    axes[0, 1].set_xlabel("frame"); axes[0, 1].set_ylabel("rendered feature norm")
    axes[0, 1].set_title("Rendered feature norm at snapshot frames")
    axes[0, 1].grid(alpha=0.3); axes[0, 1].legend()

    axes[1, 0].plot(frames, disps, "o-", color="tab:purple")
    axes[1, 0].set_xlabel("frame"); axes[1, 0].set_ylabel("|pred_pos - gt_pos| (m)")
    axes[1, 0].set_title("Predicted-pose displacement from GT")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(frames, mass, "o-", color="tab:blue")
    axes[1, 1].set_xlabel("frame"); axes[1, 1].set_ylabel("voxel mass total")
    axes[1, 1].set_title("Voxel mass total (sanity check — should grow)")
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle(f"GT-vs-pred render check — {args.seq}\n"
                  f"verdict: {verdict.split('.')[0]}", fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out_dir / "gt_vs_pred_render.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[d-renderchk] saved gt_vs_pred_render.png")

    summary = {
        "ckpt": str(args.ckpt),
        "seq": args.seq,
        "rows": rows,
        "verdict": verdict,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[d-renderchk] saved {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
