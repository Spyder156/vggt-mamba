"""Mismatch-triggered re-grounding — streaming inference with photometric
gradient-based pose refinement.

When the trigger fires (photometric mismatch_rel ≥ threshold AND not in
cooldown), takes K gradient steps of photometric loss w.r.t. the pose tensor,
replacing the per-frame corrected_pose_T with the refined one. The write
happens from the refined camera so new writes land where they should.

Bypass guards (per the design doc):
  - Inference-only: this script never runs at training time. Model.forward()
    is unchanged.
  - Gradient flows strictly into the pose tensor, never into model weights.
    Single-tensor grad via torch.autograd.grad(L_photo, pose).
  - Bounded per-step + total step magnitudes (max_step_t = 0.10m per step,
    max_jump = 2.0m total; reject and revert if exceeded).
  - Cooldown ≥ 30 frames between fires.
  - Per-fire log: status, mm_rel before/after, displacement before/after,
    so the gate-test can read co-guards from disk.

Two output modes:
  --baseline:   stream WITHOUT re-grounding (model.forward() equivalent).
                Trigger threshold is still EVALUATED but never applied —
                we log it as if it had fired so the gate-test can pair
                with the re-grounding run.
  default (no --baseline): stream WITH re-grounding applied at every fire
                outside the cooldown window.

Saves per-frame trajectory + per-fire log to enable the paired baseline-vs-RG
gate test (matched post-fire frames).
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--seq", default="rgbd_dataset_freiburg2_desk")
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory for per-frame.npz and fires.json")
    p.add_argument("--baseline", action="store_true",
                   help="Stream WITHOUT re-grounding (only log fires; don't apply).")
    # Locked hyperparameters from the design doc.
    p.add_argument("--trigger-threshold", type=float, default=0.334)
    p.add_argument("--cooldown-frames", type=int, default=30)
    p.add_argument("--k-refine", type=int, default=10)
    p.add_argument("--refine-lr", type=float, default=0.05)
    p.add_argument("--max-step-translation-m", type=float, default=0.10)
    p.add_argument("--max-total-jump-m", type=float, default=2.0)
    return p.parse_args()


def load_model(ckpt_path, weights_root):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    m = build_terrawm_d(
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
        use_photometric=True, photometric_hidden=64,
        photometric_pose_gradient=cfg["model"].get("photometric_pose_gradient", True),
    )
    m.load_state_dict(ckpt["model"], strict=False)
    return m.cuda().eval(), cfg


def load_rgb(rec, img_size):
    from PIL import Image
    img = Image.open(rec.rgb_path).convert("RGB").resize((img_size, img_size))
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0
                            ).permute(2, 0, 1).unsqueeze(0).contiguous()


def svd_project_to_so3(R: torch.Tensor) -> torch.Tensor:
    """Project a 3×3 matrix onto SO(3) via SVD. Used to re-orthonormalize the
    rotation block after a gradient step. Det forced positive (handle reflection).
    Computed in fp32 (autocast disabled) because linalg ops are not bf16-supported."""
    orig_dtype = R.dtype
    with torch.amp.autocast(device_type="cuda", enabled=False):
        R32 = R.float()
        U, _, Vt = torch.linalg.svd(R32)
        det = torch.det(U @ Vt)
        D = torch.eye(3, device=R.device, dtype=torch.float32)
        D[2, 2] = torch.sign(det)
        out = U @ D @ Vt
    return out.to(orig_dtype)


def compute_photo_mismatch(model, voxel_state, pose_T, K, patch_pixel, rgb_target,
                            ray_total_w_override=None):
    """Compute photometric mismatch_rel at a given pose. Used both as a trigger
    signal (no grad) and as the loss for refinement (with grad on pose)."""
    ray_o, ray_d = build_rays_from_pose(pose_T, K, patch_pixel)
    r = render_rays_volumetric(
        voxel_state, ray_o, ray_d,
        n_samples=model.n_render_samples, near=model.render_near, far=model.render_far,
    )
    rendered_feat = r["feature"]                                                # (B, P, voxel_dim)
    ray_total_w = r["total_weight"] if ray_total_w_override is None else ray_total_w_override
    patch_rgb_pred = model.color_head(rendered_feat)                            # (B, P, 3) in [0, 1]
    rgb_tgt_patch = F.adaptive_avg_pool2d(
        rgb_target.float(), (model.grid_h, model.grid_w)
    ).flatten(2).transpose(1, 2)                                                 # (B, P, 3)
    # Weighted L1 in [0, 1] color space — same form as terrawm_d_loss photometric_l1.
    w = ray_total_w.unsqueeze(-1).clamp(min=0.0)                                # (B, P, 1)
    w_sum = w.sum(dim=1).clamp_min(1e-6)                                        # (B, 1)
    pooled_diff = ((patch_rgb_pred - rgb_tgt_patch) * w).sum(dim=1) / w_sum
    pooled_cur = (rgb_tgt_patch * w).sum(dim=1) / w_sum
    # mismatch_rel (scalar diagnostic) — float; mismatch_l1 (tensor) drives backward.
    photo_diff_per_patch = (patch_rgb_pred - rgb_tgt_patch).norm(dim=-1)        # (B, P)
    coverage_mask = ray_total_w > model.unwritten_mask_threshold                # (B, P) bool
    masked_sum = (photo_diff_per_patch * coverage_mask.float()).sum()
    mismatch_l1 = masked_sum / coverage_mask.float().sum().clamp_min(1.0)
    return mismatch_l1, float(pooled_diff.norm()) / max(float(pooled_cur.norm()), 1e-6)


def refine_pose_photometric(
    model, voxel_state, initial_pose_T, K, patch_pixel, rgb_target,
    k_refine, lr, max_step_translation_m, max_total_jump_m,
):
    """Take k_refine gradient steps of photometric L1 w.r.t. pose.
    Returns (refined_pose_T, mm_rel_after, status_str, total_jump_m).
    Status: "applied" if refinement converged within bounds; "rejected_excessive_jump"
    if total jump > max_total_jump_m (revert to initial).
    """
    pose = initial_pose_T.detach().clone().requires_grad_(True)
    initial_t = initial_pose_T[:, :3, 3].detach().clone()
    mm_rel_trajectory = []
    with torch.enable_grad():
        for _ in range(k_refine):
            L_photo, mm_rel_now = compute_photo_mismatch(
                model, voxel_state, pose, K, patch_pixel, rgb_target,
            )
            mm_rel_trajectory.append(mm_rel_now)
            grad_pose, = torch.autograd.grad(L_photo, pose, create_graph=False)
            with torch.no_grad():
                step = -lr * grad_pose                                          # (B, 4, 4)
                # Bound translation step.
                t_step = step[:, :3, 3]
                t_norm = t_step.norm(dim=-1, keepdim=True)
                t_scale = (max_step_translation_m / t_norm.clamp_min(1e-6)).clamp_max(1.0)
                step[:, :3, 3] = t_step * t_scale
                # No rotation clip for MVP — bounded by tanh in pose_head, photo grad is small.
                pose = (pose + step).detach()
                # Re-orthonormalize rotation block.
                pose[:, :3, :3] = svd_project_to_so3(pose[0, :3, :3]).unsqueeze(0)
                pose.requires_grad_(True)
    refined_t = pose[:, :3, 3].detach()
    total_jump = float((refined_t - initial_t).norm())
    # Final mm_rel after refinement.
    with torch.no_grad():
        _, mm_rel_after = compute_photo_mismatch(
            model, voxel_state, pose, K, patch_pixel, rgb_target,
        )
    if total_jump > max_total_jump_m:
        return initial_pose_T.detach(), mm_rel_after, "rejected_excessive_jump", total_jump
    return pose.detach(), mm_rel_after, "applied", total_jump


@torch.no_grad()
def stream(model, cfg, recs, K, fov, gt_rel_T, args):
    img_size = model.img_size
    voxel_state = model.init_voxel_state(1, "cuda", torch.float32)
    prev_pose_9 = torch.tensor([[0., 0, 0, 0, 0, 0, 1, 1.0, 1.0]],
                                device="cuda", dtype=torch.float32)
    patch_pixel = model._patch_pixel_grid.cuda().unsqueeze(0).float()

    fires = []                                                                   # per-fire log
    displacements = []
    mm_rels_at_pose_head_output = []
    cooldown_until = -1

    for i, rec in enumerate(recs):
        rgb = load_rgb(rec, img_size).cuda(non_blocking=True)
        initial_T = cam9_to_pose_w_c(prev_pose_9)
        # === Encode + render at initial pose + pose-head step (the model's normal path) ===
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            patches = model._encode_frame(rgb)
            bootstrap_d = model.bootstrap_depth(patches).float()
            voxel_feat = model.patch_to_voxel(patches).float()
            wc = model.write_confidence(patches).float() if model.use_write_confidence else None
            ray_o1, ray_d1 = build_rays_from_pose(initial_T, K, patch_pixel)
            r1 = render_rays_volumetric(voxel_state, ray_o1, ray_d1,
                n_samples=model.n_render_samples, near=model.render_near, far=model.render_far)
            initial_pose_9 = _pose_T_to_cam9(initial_T, fov)
            d9 = model.pose_head(patches, r1["feature"], r1["total_weight"], initial_pose_9)
            dT = cam9_to_pose_w_c(d9)
            corrected_pose_T = (initial_T.float() @ dT).float()

        # === Compute photometric mismatch at the pose-head's output (the trigger signal) ===
        with torch.no_grad():
            _, mm_rel_trigger = compute_photo_mismatch(
                model, voxel_state, corrected_pose_T, K, patch_pixel, rgb,
            )

        gt_pos = gt_rel_T[i, :3, 3].float().cpu().numpy()
        initial_pred_pos = corrected_pose_T[0, :3, 3].detach().float().cpu().numpy()
        displacement_before = float(np.linalg.norm(initial_pred_pos - gt_pos))

        # === Trigger decision + refinement (or skip if --baseline) ===
        fired = False
        applied = False
        mm_rel_after = mm_rel_trigger
        displacement_after = displacement_before
        status = "none"
        total_jump = 0.0
        if mm_rel_trigger >= args.trigger_threshold and i >= cooldown_until:
            fired = True
            if not args.baseline:
                # Refine.
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    refined_T, mm_rel_after, status, total_jump = refine_pose_photometric(
                        model, voxel_state, corrected_pose_T, K, patch_pixel, rgb,
                        k_refine=args.k_refine, lr=args.refine_lr,
                        max_step_translation_m=args.max_step_translation_m,
                        max_total_jump_m=args.max_total_jump_m,
                    )
                if status == "applied":
                    applied = True
                    corrected_pose_T = refined_T
                    displacement_after = float(np.linalg.norm(
                        refined_T[0, :3, 3].detach().float().cpu().numpy() - gt_pos))
                cooldown_until = i + args.cooldown_frames
            else:
                status = "logged_only_baseline"
                cooldown_until = i + args.cooldown_frames
            fires.append({
                "frame": int(i),
                "fired": True, "applied": bool(applied),
                "status": status,
                "mm_rel_before": float(mm_rel_trigger),
                "mm_rel_after": float(mm_rel_after),
                "displacement_before_m": displacement_before,
                "displacement_after_m": displacement_after,
                "total_jump_m": total_jump,
                "gt_pos": [float(x) for x in gt_pos.tolist()],
                "initial_pose_t": [float(x) for x in initial_pred_pos.tolist()],
                "final_pose_t": [float(x) for x in corrected_pose_T[0, :3, 3].detach().float().cpu().numpy().tolist()],
            })

        # === Write at the (possibly refined) corrected pose ===
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            wp = backproject_patches_to_world(patch_pixel, bootstrap_d, K, corrected_pose_T.detach())
            write_voxels_trilinear(voxel_state, wp, voxel_feat, weights=wc)
            prev_pose_9 = _pose_T_to_cam9(corrected_pose_T, fov).float()

        displacements.append(displacement_after)
        mm_rels_at_pose_head_output.append(float(mm_rel_trigger))
        if (i + 1) % 200 == 0:
            mode = "BASELINE" if args.baseline else "RG"
            print(f"[{mode}] f={i+1}  disp={displacement_after:.2f}m  mm_rel={mm_rel_trigger:.3f}  "
                  f"fires_so_far={len(fires)}")

    return (np.array(displacements), np.array(mm_rels_at_pose_head_output), fires)


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
    print(f"[rg-stream] {args.seq}: streaming {len(recs)} frames  "
          f"mode={'BASELINE' if args.baseline else 'RE-GROUNDING'}  "
          f"threshold={args.trigger_threshold} cooldown={args.cooldown_frames} K={args.k_refine}")

    gt_poses = np.stack([r.pose_w_c for r in recs])
    P0_inv = np.linalg.inv(gt_poses[0])
    gt_rel = np.einsum("ij,njk->nik", P0_inv, gt_poses)
    gt_rel_T = torch.from_numpy(gt_rel).float().cuda()

    disp, mm_rel, fires = stream(model, cfg, recs, K, fov, gt_rel_T, args)
    np.savez(args.out / "per_frame.npz", displacement=disp, mismatch_rel=mm_rel)
    (args.out / "fires.json").write_text(json.dumps(fires, indent=2))
    (args.out / "config.json").write_text(json.dumps({
        "ckpt": str(args.ckpt), "seq": args.seq, "mode": "baseline" if args.baseline else "regrounding",
        "trigger_threshold": args.trigger_threshold,
        "cooldown_frames": args.cooldown_frames,
        "k_refine": args.k_refine, "refine_lr": args.refine_lr,
        "max_step_translation_m": args.max_step_translation_m,
        "max_total_jump_m": args.max_total_jump_m,
        "n_frames": len(recs), "n_fires": len(fires),
        "n_applied": sum(1 for f in fires if f.get("applied")),
    }, indent=2))
    print(f"[rg-stream] saved {args.out}/  "
          f"n_fires={len(fires)}  n_applied={sum(1 for f in fires if f.get('applied'))}")


if __name__ == "__main__":
    main()
