"""Depth-based refinement on the CLEAN grid — the one remaining fork-decider.

Photometric refinement on the clean grid moved only 52% of fires toward GT
(see project_regrounding_correction_weak.md). The bottleneck is the photometric
signal itself, not grid pollution.

This script tests whether a DIFFERENT signal class — geometric depth — carries
the corrective information that color lacks. Same 44 fires from the
fr2/desk run, same clean (GT-pose-built) grid, same refinement structure
(K=10 steps, LR=0.05, bounded). Only the loss changes:

    L = L1(rendered_depth, gt_depth_patch)   masked on depth_mask

Where rendered_depth is the volumetric-rendered expected depth from the
voxel grid at the current refining pose, and gt_depth_patch is the GT per-
patch depth from the dataset (cleanest signal — removes one source of noise
relative to using bootstrap_d which is itself learned).

Decision fork after running:
  - Depth toward-GT > 70%  →  geometric signal carries corrective info.
                              Next move: depth-consistency loss term in
                              the long training run (signal lives in loss,
                              not in a separate corrector).
  - Depth toward-GT ≈ 52%  →  no hand-crafted gradient signal corrects
                              fine pose in this representation. The
                              correction has to be LEARNED end-to-end
                              (proposal-head route), not engineered.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt_mamba.data.tum_rgbd import sync_sequence, intrinsics_for                 # noqa: E402
from vggt_mamba.models.voxel_grid import (                                          # noqa: E402
    backproject_patches_to_world, build_rays_from_pose,
    render_rays_volumetric, write_voxels_trilinear,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from terrawm_d_regrounding_stream import (                                          # noqa: E402
    load_model, load_rgb, svd_project_to_so3,
)
from terrawm_d_regrounding_clean_grid import (                                      # noqa: E402
    snapshot_voxel_state, restore_voxel_state,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--seq", default="rgbd_dataset_freiburg2_desk")
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--regrounding-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k-refine", type=int, default=10)
    p.add_argument("--refine-lr", type=float, default=0.05)
    p.add_argument("--max-step-translation-m", type=float, default=0.10)
    p.add_argument("--max-total-jump-m", type=float, default=2.0)
    p.add_argument("--depth-max-m", type=float, default=8.0)
    return p.parse_args()


def load_gt_depth_patch(rec, img_size, grid_h, grid_w, depth_max_m=8.0):
    """Load GT depth, downsample to patch resolution, mask invalid."""
    from PIL import Image
    d = Image.open(rec.depth_path).resize((img_size, img_size), Image.NEAREST)
    d = np.asarray(d, dtype=np.float32) / 5000.0                                # TUM standard
    d = np.where((d > 0) & (d < depth_max_m), d, 0.0)
    d_t = torch.from_numpy(d).unsqueeze(0).unsqueeze(0).cuda()                  # (1, 1, H, W)
    d_patch = F.adaptive_avg_pool2d(d_t, (grid_h, grid_w)).squeeze(1).squeeze(0)  # (gh, gw)
    valid_patch = (d_patch > 1e-3).float()                                       # (gh, gw)
    return d_patch.flatten().unsqueeze(0), valid_patch.flatten().unsqueeze(0)   # (1, P), (1, P)


def compute_depth_mismatch(model, voxel_state, pose_T, K, patch_pixel, gt_depth_patch, gt_valid_patch):
    """Render depth from grid at given pose; L1 vs GT per-patch depth, masked on
    depth_mask AND GT valid. Returns (loss tensor, scalar diagnostic)."""
    ray_o, ray_d = build_rays_from_pose(pose_T, K, patch_pixel)
    r = render_rays_volumetric(
        voxel_state, ray_o, ray_d,
        n_samples=model.n_render_samples, near=model.render_near, far=model.render_far,
    )
    rendered_depth = r["depth"]                                                  # (1, P)
    ray_total_w = r["total_weight"]                                              # (1, P)
    depth_mask = (ray_total_w > model.unwritten_mask_threshold).float() * gt_valid_patch
    diff = (rendered_depth - gt_depth_patch).abs()                               # (1, P)
    masked_sum = (diff * depth_mask).sum()
    loss = masked_sum / depth_mask.sum().clamp_min(1.0)
    return loss, float(loss.detach())


def refine_pose_depth(
    model, voxel_state, initial_pose_T, K, patch_pixel, gt_depth_patch, gt_valid_patch,
    k_refine, lr, max_step_translation_m, max_total_jump_m,
):
    """K gradient steps of depth L1 w.r.t. pose. Mirrors refine_pose_photometric."""
    pose = initial_pose_T.detach().clone().requires_grad_(True)
    initial_t = initial_pose_T[:, :3, 3].detach().clone()
    with torch.enable_grad():
        for _ in range(k_refine):
            L_depth, _ = compute_depth_mismatch(
                model, voxel_state, pose, K, patch_pixel, gt_depth_patch, gt_valid_patch,
            )
            grad_pose, = torch.autograd.grad(L_depth, pose, create_graph=False)
            with torch.no_grad():
                step = -lr * grad_pose
                t_step = step[:, :3, 3]
                t_norm = t_step.norm(dim=-1, keepdim=True)
                t_scale = (max_step_translation_m / t_norm.clamp_min(1e-6)).clamp_max(1.0)
                step[:, :3, 3] = t_step * t_scale
                pose = (pose + step).detach()
                pose[:, :3, :3] = svd_project_to_so3(pose[0, :3, :3]).unsqueeze(0)
                pose.requires_grad_(True)
    refined_t = pose[:, :3, 3].detach()
    total_jump = float((refined_t - initial_t).norm())
    with torch.no_grad():
        _, depth_after = compute_depth_mismatch(
            model, voxel_state, pose, K, patch_pixel, gt_depth_patch, gt_valid_patch,
        )
    if total_jump > max_total_jump_m:
        return initial_pose_T.detach(), depth_after, "rejected_excessive_jump", total_jump
    return pose.detach(), depth_after, "applied", total_jump


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model, cfg = load_model(args.ckpt, args.weights_root)
    img_size = {"vjepa": 384, "dinov2": 518, "dinov3": 512}[cfg["encoder"]]
    fx, fy, cx, cy = intrinsics_for(args.seq)
    sx, sy = img_size / 640.0, img_size / 480.0
    K = torch.tensor([[[fx*sx, 0, cx*sx], [0, fy*sy, cy*sy], [0, 0, 1]]], device="cuda")
    recs = sync_sequence(args.data_root / args.seq)
    print(f"[depth-clean] {args.seq}: streaming {len(recs)} GT-pose frames + replaying depth refinement")

    fires = json.loads((args.regrounding_dir / "fires.json").read_text())
    fire_frames = sorted({f["frame"] for f in fires})
    fire_frame_set = set(fire_frames)

    gt_poses = np.stack([r.pose_w_c for r in recs])
    P0_inv = np.linalg.inv(gt_poses[0])
    gt_rel = np.einsum("ij,njk->nik", P0_inv, gt_poses)
    gt_rel_T = torch.from_numpy(gt_rel).float().cuda()
    patch_pixel = model._patch_pixel_grid.cuda().unsqueeze(0).float()
    voxel_state = model.init_voxel_state(1, "cuda", torch.float32)
    snapshots = {}

    # Build clean grid using GT poses + snapshot at fire frames.
    for i, rec in enumerate(recs):
        rgb = load_rgb(rec, img_size).cuda(non_blocking=True)
        gt_T = gt_rel_T[i:i+1]
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                patches = model._encode_frame(rgb)
                bootstrap_d = model.bootstrap_depth(patches).float()
                voxel_feat = model.patch_to_voxel(patches).float()
                wc = model.write_confidence(patches).float() if model.use_write_confidence else None
                wp = backproject_patches_to_world(patch_pixel, bootstrap_d, K, gt_T)
                write_voxels_trilinear(voxel_state, wp, voxel_feat, weights=wc)
        if i in fire_frame_set:
            snapshots[i] = snapshot_voxel_state(voxel_state)
        if (i + 1) % 200 == 0:
            print(f"[depth-clean] streamed {i+1}/{len(recs)} GT-pose frames")
    print(f"[depth-clean] snapshotted {len(snapshots)} clean-grid states")

    # Replay depth refinement at each fire frame.
    per_fire = []
    for fire in fires:
        t = fire["frame"]
        if t not in snapshots:
            continue
        if not fire.get("fired"):
            continue
        clean_state = restore_voxel_state(model, snapshots[t])
        gt_d_patch, gt_v_patch = load_gt_depth_patch(
            recs[t], img_size, model.grid_h, model.grid_w, args.depth_max_m,
        )
        if gt_v_patch.sum() < 10:                                                # too few valid pixels
            continue
        initial_T = torch.eye(4, device="cuda").unsqueeze(0).float()
        initial_T[0, :3, 3] = torch.tensor(fire["initial_pose_t"], device="cuda")
        gt_pos = np.array(fire["gt_pos"])
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            refined_T, depth_after, status, total_jump = refine_pose_depth(
                model, clean_state, initial_T, K, patch_pixel, gt_d_patch, gt_v_patch,
                k_refine=args.k_refine, lr=args.refine_lr,
                max_step_translation_m=args.max_step_translation_m,
                max_total_jump_m=args.max_total_jump_m,
            )
        refined_t = refined_T[0, :3, 3].float().cpu().numpy()
        displacement_clean = float(np.linalg.norm(refined_t - gt_pos))
        per_fire.append({
            "frame": int(t),
            "status": status,
            "displacement_before_m": float(fire["displacement_before_m"]),
            "displacement_after_clean_depth_m": displacement_clean,
            "depth_loss_after": depth_after,
            "total_jump_m": total_jump,
        })

    # Aggregate + verdict.
    deltas = np.array([f["displacement_after_clean_depth_m"] - f["displacement_before_m"]
                        for f in per_fire])
    toward_gt = sum(1 for d in deltas if d < 0)
    mean_delta = float(deltas.mean()) if len(deltas) else float("nan")
    median_delta = float(np.median(deltas)) if len(deltas) else float("nan")

    print(f"\n[depth-clean] === DEPTH CLEAN-GRID DISAMBIGUATION RESULT ===")
    print(f"  n fires replayed: {len(per_fire)}")
    print(f"  Δ displacement (refined−initial):")
    print(f"    mean   = {mean_delta:+.4f} m")
    print(f"    median = {median_delta:+.4f} m")
    print(f"  Toward GT (refined closer to GT than initial): "
           f"{toward_gt}/{len(per_fire)} = {toward_gt/max(len(per_fire),1):.2%}")
    print(f"\n  REFERENCE (from project_regrounding_correction_weak):")
    print(f"    Photometric clean-grid toward-GT: 23/44 = 52%")
    print(f"\n[depth-clean] === DECISION FORK ===")
    pct = toward_gt / max(len(per_fire), 1)
    if pct > 0.70:
        verdict = (f"DEPTH WORKS ({pct:.0%} > 70%): geometric signal carries corrective info "
                    f"that color lacks. Add a depth-consistency loss term to the long training run.")
    elif pct > 0.60:
        verdict = (f"DEPTH HELPS, NOT SUFFICIENT ({pct:.0%}): geometric signal contributes but "
                    f"alone isn't enough. Long run should include both photometric + depth loss "
                    f"terms; either signal alone caps below 70%.")
    elif pct > 0.55:
        verdict = (f"DEPTH MARGINAL ({pct:.0%}): geometric signal slightly better than chance, "
                    f"not enough to qualify as 'carries corrective info'. Lean toward learned-"
                    f"proposal route; depth loss helps but won't carry the long run alone.")
    else:
        verdict = (f"DEPTH SAME AS PHOTOMETRIC ({pct:.0%} ≈ 52%): no hand-crafted gradient signal "
                    f"corrects fine pose in this representation. The correction has to be LEARNED "
                    f"end-to-end (proposal-head route), not engineered. Long run should focus on "
                    f"the training signal that produces good per-frame pose directly.")
    print(f"  {verdict}")

    args.out.write_text(json.dumps({
        "seq": args.seq,
        "n_fires_replayed": len(per_fire),
        "mean_delta_disp_m": mean_delta,
        "median_delta_disp_m": median_delta,
        "toward_gt_frac": toward_gt / max(len(per_fire), 1),
        "toward_gt_n": toward_gt,
        "reference_photometric_clean_grid_toward_gt_frac": 23/44,
        "verdict": verdict,
        "per_fire": per_fire,
    }, indent=2))
    print(f"\n[depth-clean] saved {args.out}")


if __name__ == "__main__":
    main()
