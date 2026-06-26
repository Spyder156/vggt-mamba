"""Depth-iteration stream — A1 (per-frame) vs A2 (triggered) vs baseline.

Tests the iterate-at-inference hypothesis with the validated mechanism
(gradient of depth-mismatch w.r.t. pose, the signal the depth-clean-grid
diagnostic measured at 66% toward GT). Three modes:

  --mode baseline   — no iteration, pose head's one-shot output only.
  --mode a1_per_frame — K=3 depth-refinement steps EVERY frame, starting from
                         pose head's output. Tests "per-frame inner loop"
                         (operates near 0.3m drift, ~51% direction-reliability
                         per β_disp data — weakest regime).
  --mode a2_triggered — K=10 depth-refinement steps ONLY when photometric mm
                         exceeds threshold + cooldown. The trigger uses the
                         FPR=0 photometric signal we already validated; the
                         CORRECTION swaps from photometric to depth (the
                         never-tested combination). Operates near 1-2m drift,
                         ~57-60% direction-reliability — the regime where the
                         signal is strongest.

Refinement target: bootstrap_d (the encoder's per-patch depth prediction).
This is the INFERENCE-TIME-AVAILABLE signal — NOT GT depth. The diagnostic
that produced 66% used GT depth on the clean grid; using bootstrap_d on the
streamed grid is the realistic deployable version.

Per-fire / per-frame log includes:
  - pose displacement BEFORE refinement (vs GT)
  - pose displacement AFTER refinement (vs GT)
  - This enables co-guard 1: does refinement move TOWARD GT, not just reduce
    the depth-mismatch proxy. Photometric re-grounding passed co-guard 2 (88%
    mm reduction) but failed co-guard 1 (41% toward GT) — depth refinement has
    to clear the same bar that one failed.
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
from vggt_mamba.models.aggregators.anchor_pool import cam9_to_pose_w_c              # noqa: E402
from vggt_mamba.models.terrawm_d import build_terrawm_d, _pose_T_to_cam9             # noqa: E402
from vggt_mamba.models.voxel_grid import (                                          # noqa: E402
    backproject_patches_to_world, build_rays_from_pose,
    render_rays_volumetric, write_voxels_trilinear,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from terrawm_d_regrounding_stream import (                                          # noqa: E402
    load_model, load_rgb, svd_project_to_so3, compute_photo_mismatch,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--seq", default="rgbd_dataset_freiburg2_desk")
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=["baseline", "a1_per_frame", "a2_triggered"],
                    required=True)
    # A1 / A2 hyperparameters
    p.add_argument("--k-refine-a1", type=int, default=3)
    p.add_argument("--k-refine-a2", type=int, default=10)
    p.add_argument("--refine-lr", type=float, default=0.05)
    p.add_argument("--max-step-translation-m", type=float, default=0.10)
    p.add_argument("--max-total-jump-m", type=float, default=2.0)
    # A2 trigger (re-use re-grounding's photometric trigger — FPR=0 known)
    p.add_argument("--trigger-threshold", type=float, default=0.334)
    p.add_argument("--cooldown-frames", type=int, default=30)
    return p.parse_args()


def compute_depth_mismatch_loss(model, voxel_state, pose_T, K, patch_pixel, bootstrap_d):
    """L1 between rendered depth and bootstrap_d, mass-weighted, masked.
    Returns (loss tensor, scalar diagnostic). Used both as refinement objective
    AND as the value to log."""
    ray_o, ray_d = build_rays_from_pose(pose_T, K, patch_pixel)
    r = render_rays_volumetric(
        voxel_state, ray_o, ray_d,
        n_samples=model.n_render_samples, near=model.render_near, far=model.render_far,
    )
    r_depth = r["depth"].float()
    mass = r["total_weight"].float()
    valid = (mass > model.unwritten_mask_threshold).float()
    diff = (r_depth - bootstrap_d.detach()).abs()
    denom = valid.sum().clamp_min(1.0)
    L = (diff * valid * mass).sum() / denom
    return L, float(L.detach())


def refine_pose_depth(model, voxel_state, initial_pose_T, K, patch_pixel,
                       bootstrap_d, k_refine, lr, max_step_translation_m, max_total_jump_m):
    """K gradient steps of depth L1 (vs bootstrap_d) w.r.t. pose. Mirrors
    refine_pose_photometric structure exactly so they're directly comparable."""
    pose = initial_pose_T.detach().clone().requires_grad_(True)
    initial_t = initial_pose_T[:, :3, 3].detach().clone()
    with torch.enable_grad():
        for _ in range(k_refine):
            L, _ = compute_depth_mismatch_loss(model, voxel_state, pose, K,
                                                 patch_pixel, bootstrap_d)
            grad_pose, = torch.autograd.grad(L, pose, create_graph=False)
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
        _, depth_after = compute_depth_mismatch_loss(model, voxel_state, pose, K,
                                                       patch_pixel, bootstrap_d)
    if total_jump > max_total_jump_m:
        return initial_pose_T.detach(), depth_after, "rejected_excessive_jump", total_jump
    return pose.detach(), depth_after, "applied", total_jump


@torch.no_grad()
def stream(model, cfg, recs, K, fov, gt_rel_T, args):
    img_size = model.img_size
    voxel_state = model.init_voxel_state(1, "cuda", torch.float32)
    prev_pose_9 = torch.tensor([[0., 0, 0, 0, 0, 0, 1, 1.0, 1.0]],
                                device="cuda", dtype=torch.float32)
    patch_pixel = model._patch_pixel_grid.cuda().unsqueeze(0).float()
    events = []
    displacements = []
    cooldown_until = -1

    for i, rec in enumerate(recs):
        rgb = load_rgb(rec, img_size).cuda(non_blocking=True)
        initial_T = cam9_to_pose_w_c(prev_pose_9)
        # Normal model forward — get the pose head's one-shot estimate.
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            patches = model._encode_frame(rgb)
            bootstrap_d = model.bootstrap_depth(patches).float()
            voxel_feat = model.patch_to_voxel(patches).float()
            wc = model.write_confidence(patches).float() if model.use_write_confidence else None
            ray_o1, ray_d1 = build_rays_from_pose(initial_T, K, patch_pixel)
            r1 = render_rays_volumetric(voxel_state, ray_o1, ray_d1,
                n_samples=model.n_render_samples, near=model.render_near, far=model.render_far)
            init9 = _pose_T_to_cam9(initial_T, fov)
            d9 = model.pose_head(patches, r1["feature"], r1["total_weight"], init9)
            dT = cam9_to_pose_w_c(d9)
            corrected_pose_T = (initial_T.float() @ dT).float()

        gt_pos = gt_rel_T[i, :3, 3].float().cpu().numpy()
        pose_head_pos = corrected_pose_T[0, :3, 3].detach().float().cpu().numpy()
        disp_before = float(np.linalg.norm(pose_head_pos - gt_pos))
        final_pose_T = corrected_pose_T

        # === MODE-SPECIFIC ITERATION LOGIC ===
        if args.mode == "baseline":
            status = "baseline"

        elif args.mode == "a1_per_frame":
            # Refine every frame, K=3 steps. Operates in small-drift regime
            # (per-frame motion ~5-30mm; β_disp predicts 36-52% direction).
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                refined_T, depth_after, status, total_jump = refine_pose_depth(
                    model, voxel_state, corrected_pose_T, K, patch_pixel, bootstrap_d,
                    k_refine=args.k_refine_a1, lr=args.refine_lr,
                    max_step_translation_m=args.max_step_translation_m,
                    max_total_jump_m=args.max_total_jump_m,
                )
            if status == "applied":
                final_pose_T = refined_T
            refined_pos = final_pose_T[0, :3, 3].detach().float().cpu().numpy()
            disp_after = float(np.linalg.norm(refined_pos - gt_pos))
            events.append({
                "frame": int(i), "kind": "per_frame_iter",
                "status": status,
                "disp_before_m": disp_before, "disp_after_m": disp_after,
                "total_jump_m": total_jump,
                "toward_gt": bool(disp_after < disp_before),    # CO-GUARD 1
            })

        elif args.mode == "a2_triggered":
            # Photometric trigger (FPR=0 from prior re-grounding work), depth
            # correction (the never-tested combination, validated at 66% on
            # clean-grid diagnostic).
            with torch.no_grad():
                _, mm_rel_photo = compute_photo_mismatch(
                    model, voxel_state, corrected_pose_T, K, patch_pixel, rgb,
                )
            if mm_rel_photo >= args.trigger_threshold and i >= cooldown_until:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    refined_T, depth_after, status, total_jump = refine_pose_depth(
                        model, voxel_state, corrected_pose_T, K, patch_pixel, bootstrap_d,
                        k_refine=args.k_refine_a2, lr=args.refine_lr,
                        max_step_translation_m=args.max_step_translation_m,
                        max_total_jump_m=args.max_total_jump_m,
                    )
                if status == "applied":
                    final_pose_T = refined_T
                refined_pos = final_pose_T[0, :3, 3].detach().float().cpu().numpy()
                disp_after = float(np.linalg.norm(refined_pos - gt_pos))
                events.append({
                    "frame": int(i), "kind": "fire",
                    "status": status,
                    "mm_rel_photo_trigger": float(mm_rel_photo),
                    "disp_before_m": disp_before, "disp_after_m": disp_after,
                    "total_jump_m": total_jump,
                    "toward_gt": bool(disp_after < disp_before),    # CO-GUARD 1
                })
                cooldown_until = i + args.cooldown_frames

        # === Write + carry forward ===
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            wp = backproject_patches_to_world(patch_pixel, bootstrap_d, K, final_pose_T.detach())
            write_voxels_trilinear(voxel_state, wp, voxel_feat, weights=wc)
            prev_pose_9 = _pose_T_to_cam9(final_pose_T, fov).float()

        final_pos = final_pose_T[0, :3, 3].detach().float().cpu().numpy()
        displacements.append(float(np.linalg.norm(final_pos - gt_pos)))
        if (i + 1) % 200 == 0:
            print(f"[{args.mode}] f={i+1}  disp={displacements[-1]:.2f}m  events_so_far={len(events)}")
    return np.array(displacements), events


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model, cfg = load_model(args.ckpt, args.weights_root)
    img_size = {"vjepa": 384, "dinov2": 518, "dinov3": 512}[cfg["encoder"]]
    fx, fy, cx, cy = intrinsics_for(args.seq)
    sx, sy = img_size / 640.0, img_size / 480.0
    K = torch.tensor([[[fx*sx, 0, cx*sx], [0, fy*sy, cy*sy], [0, 0, 1]]], device="cuda")
    fov = torch.tensor([[1.0, 1.0]], device="cuda")
    recs = sync_sequence(args.data_root / args.seq)
    print(f"[depth-iter] {args.seq}: {len(recs)} frames  mode={args.mode}")

    gt_poses = np.stack([r.pose_w_c for r in recs])
    P0_inv = np.linalg.inv(gt_poses[0])
    gt_rel = np.einsum("ij,njk->nik", P0_inv, gt_poses)
    gt_rel_T = torch.from_numpy(gt_rel).float().cuda()

    disp, events = stream(model, cfg, recs, K, fov, gt_rel_T, args)
    np.savez(args.out / "per_frame.npz", displacement=disp)
    (args.out / "events.json").write_text(json.dumps(events, indent=2))
    (args.out / "config.json").write_text(json.dumps({
        "ckpt": str(args.ckpt), "seq": args.seq, "mode": args.mode,
        "trigger_threshold": args.trigger_threshold,
        "cooldown_frames": args.cooldown_frames,
        "k_refine_a1": args.k_refine_a1, "k_refine_a2": args.k_refine_a2,
        "refine_lr": args.refine_lr,
        "max_step_translation_m": args.max_step_translation_m,
        "max_total_jump_m": args.max_total_jump_m,
        "n_frames": len(recs), "n_events": len(events),
    }, indent=2))
    print(f"[depth-iter] saved {args.out}/  n_events={len(events)}  "
           f"mean_disp={float(disp.mean()):.3f}m")


if __name__ == "__main__":
    main()
