"""Diagnostic confirmation test for the mass-saturation hypothesis.

The data_debug viz showed write_mass grows unbounded (2M total by frame 2000),
saturating the volumetric render so rendered_depth collapses to ~near_plane
regardless of pose. This script runs the SAME stream + viz but clamps
write_mass to a cap after every write, testing whether saturation is the
load-bearing bug.

Cap chosen to keep per-sample alpha meaningful:
    alpha_i = 1 - exp(-mass · dt),  dt = (8.0 - 0.1) / 64 ≈ 0.123
    For alpha ≈ 0.3-0.5: mass ≈ 3-6
    --max-voxel-mass default 5.0

If the hypothesis is right:
  - rendered_depth at late frames is no longer constant
  - per-patch scatter rendered@GT-pose vs GT is positively correlated again
  - loss landscape argmin moves toward GT (small distance from GT pose)
  - gradient sanity cos values are mostly positive

If wrong, the rendered depth still degenerates and the failure is somewhere
deeper than mass saturation.
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
    backproject_patches_to_world, build_rays_from_pose,
    init_voxel_state, render_rays_volumetric, write_voxels_trilinear,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from terrawm_d_regrounding_stream import load_model, load_rgb                       # noqa: E402
from terrawm_d_data_debug import (                                                  # noqa: E402
    load_gt_depth_image, viz_depth_panel, viz_per_patch_scatter,
    viz_voxel_grid, viz_gradient_sanity, viz_loss_landscape,
    viz_trajectory, viz_backprojection,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--seq", default="rgbd_dataset_freiburg2_desk")
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--out", type=Path,
                    default=Path("viz/output/terrawm_d_masscap_debug"))
    p.add_argument("--test-frames", type=int, nargs="+",
                    default=[10, 100, 500, 1000, 1500, 2000])
    p.add_argument("--n-frames", type=int, default=2100)
    p.add_argument("--max-voxel-mass", type=float, default=5.0,
                    help="Clamp write_mass to this max after every write.")
    return p.parse_args()


@torch.no_grad()
def stream_with_masscap(model, recs, K, fov, gt_rel_T, test_frame_set,
                         max_voxel_mass: float):
    img_size = model.img_size
    voxel_state = model.init_voxel_state(1, "cuda", torch.float32)
    patch_pixel = model._patch_pixel_grid.cuda().unsqueeze(0).float()
    prev_pose_9 = torch.tensor([[0., 0, 0, 0, 0, 0, 1, 1.0, 1.0]],
                                device="cuda", dtype=torch.float32)
    pred_traj = []
    snaps = {}
    mass_log = []
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
            # === THE HYPOTHESIS TEST: clamp mass after the write ===
            voxel_state.write_mass.clamp_(max=max_voxel_mass)
            prev_pose_9 = _pose_T_to_cam9(corrected_pose_T, fov).float()
        pred_traj.append(corrected_pose_T[0].detach().float().cpu().numpy().copy())
        mass_log.append({
            "frame": i,
            "total_mass": float(voxel_state.write_mass.sum()),
            "nonzero_voxels": int((voxel_state.write_mass > 0).sum()),
            "max_voxel_mass": float(voxel_state.write_mass.max()),
        })
        if i in test_frame_set:
            snaps[i] = {
                "voxel_state": {
                    "features": voxel_state.features.detach().clone(),
                    "write_mass": voxel_state.write_mass.detach().clone(),
                    "cfg": voxel_state.cfg,
                },
                "pred_pose_T": corrected_pose_T[0].detach().float().cpu().numpy().copy(),
                "bootstrap_d": bootstrap_d[0].detach().float().cpu().numpy().copy(),
            }
        if (i + 1) % 200 == 0:
            print(f"[masscap] streamed {i+1}/{len(recs)}  "
                   f"total_mass={mass_log[-1]['total_mass']:.0f}  "
                   f"nonzero={mass_log[-1]['nonzero_voxels']}  "
                   f"max={mass_log[-1]['max_voxel_mass']:.2f}")
    return pred_traj, snaps, mass_log


def restore_voxel_state(snap_dict):
    s = init_voxel_state(snap_dict["voxel_state"]["cfg"], 1, "cuda", torch.float32)
    s.features.copy_(snap_dict["voxel_state"]["features"])
    s.write_mass.copy_(snap_dict["voxel_state"]["write_mass"])
    return s


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
    print(f"[masscap] {args.seq}: streaming {len(recs)} frames  "
           f"max_voxel_mass={args.max_voxel_mass}")

    gt_poses = np.stack([r.pose_w_c for r in recs])
    P0_inv = np.linalg.inv(gt_poses[0])
    gt_rel = np.einsum("ij,njk->nik", P0_inv, gt_poses)
    gt_rel_T_np = gt_rel
    gt_rel_T = torch.from_numpy(gt_rel).float().cuda()

    test_frame_set = set(args.test_frames)
    pred_traj, snaps, mass_log = stream_with_masscap(
        model, recs, K, fov, gt_rel_T, test_frame_set, args.max_voxel_mass,
    )
    voxel_bounds = tuple(cfg["model"]["voxel_bounds"])

    # === A. trajectory ===
    print(f"[masscap] trajectory_3d.png")
    viz_trajectory(args.out, pred_traj, gt_rel_T_np, voxel_bounds)

    # === Mass growth curve (NEW — directly shows the cap working) ===
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    frames = [m["frame"] for m in mass_log]
    axes[0].plot(frames, [m["total_mass"] for m in mass_log])
    axes[0].set_xlabel("frame"); axes[0].set_ylabel("total mass")
    axes[0].set_title(f"Total mass (cap={args.max_voxel_mass}/voxel)")
    axes[0].grid(alpha=0.3)
    axes[1].plot(frames, [m["nonzero_voxels"] for m in mass_log])
    axes[1].set_xlabel("frame"); axes[1].set_ylabel("nonzero voxels")
    axes[1].set_title("Nonzero voxel count")
    axes[1].grid(alpha=0.3)
    axes[2].plot(frames, [m["max_voxel_mass"] for m in mass_log])
    axes[2].axhline(args.max_voxel_mass, color="red", linestyle="--",
                     label=f"cap={args.max_voxel_mass}")
    axes[2].set_xlabel("frame"); axes[2].set_ylabel("max voxel mass")
    axes[2].set_title("Max single-voxel mass (should saturate at cap)")
    axes[2].grid(alpha=0.3); axes[2].legend()
    plt.tight_layout()
    plt.savefig(args.out / "mass_growth.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # === Per-frame viz ===
    patch_pixel = model._patch_pixel_grid.cuda().unsqueeze(0).float()
    for fi in args.test_frames:
        if fi not in snaps:
            continue
        print(f"[masscap] frame {fi}")
        snap = snaps[fi]
        voxel_state = restore_voxel_state(snap)
        gt_T = gt_rel_T[fi:fi+1]
        gt_pos = gt_T[0, :3, 3].float().cpu().numpy()
        pred_pos = snap["pred_pose_T"][:3, 3]
        bootstrap_d_patch = snap["bootstrap_d"]
        bootstrap_d_torch = torch.from_numpy(bootstrap_d_patch).unsqueeze(0).cuda()

        rec = recs[fi]
        gt_depth_img = load_gt_depth_image(rec, img_size, cfg["data"]["depth_max_m"])
        rgb_pil = load_rgb(rec, img_size).squeeze(0).permute(1, 2, 0).cpu().numpy()

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

        with torch.no_grad():
            gt_depth_t = torch.from_numpy(np.nan_to_num(gt_depth_img, nan=0)
                                          ).float().unsqueeze(0).unsqueeze(0).cuda()
            gt_depth_patch = F.adaptive_avg_pool2d(gt_depth_t,
                                                     (model.grid_h, model.grid_w)
                                                    ).squeeze().cpu().numpy().flatten()

        viz_depth_panel(args.out, fi, rgb_pil, gt_depth_img, bootstrap_d_patch,
                        r_depth_gt, r_mass_gt, r_depth_pr, r_mass_pr,
                        img_size, model.grid_h, model.grid_w,
                        gt_pos, pred_pos)
        viz_per_patch_scatter(args.out, fi, bootstrap_d_patch, gt_depth_patch,
                              r_depth_gt, r_depth_pr)
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
        viz_voxel_grid(args.out, fi, voxel_state,
                        gt_rel_T_np[:fi+1, :3, 3], np.stack([p[:3, 3] for p in pred_traj[:fi+1]]))
        if fi in [args.test_frames[2], args.test_frames[3]]:
            print(f"[masscap]   gradient sanity + loss landscape")
            viz_gradient_sanity(args.out, fi, model, voxel_state, gt_T, K, patch_pixel,
                                 bootstrap_d_torch)
            viz_loss_landscape(args.out, fi, model, voxel_state, gt_T, pred_pos,
                                K, patch_pixel, bootstrap_d_torch)
    print(f"\n[masscap] artifacts saved under {args.out}/")


if __name__ == "__main__":
    main()
