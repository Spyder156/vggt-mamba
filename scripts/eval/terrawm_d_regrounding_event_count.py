"""Step 0 of re-grounding gate: verify the held-out sequence has ≥3 contiguous
drift events with photometric mismatch crossing the locked trigger threshold
0.334. Without this, the gate test can't distinguish "one event handled" from
"many events handled" — fr1/room has only one event.

Streams the photometric ckpt on a candidate held-out sequence. Records, per
frame: displacement (vs GT), photometric mismatch_rel, trigger status. Then
counts contiguous "trigger fires" runs and contiguous "drift events" (disp≥1m).
Pre-registered pass: ≥ 3 contiguous events with at least one fire each.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

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
    p.add_argument("--trigger-threshold", type=float, default=0.334)
    p.add_argument("--drift-threshold-m", type=float, default=1.0)
    p.add_argument("--out", type=Path,
                   default=Path("viz/output/terrawm_d_regrounding_event_count/diag.json"))
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


@torch.no_grad()
def stream_with_trigger(model, recs, K, fov, gt_rel_T):
    img_size = model.img_size
    voxel_state = model.init_voxel_state(1, "cuda", torch.float32)
    prev_pose_9 = torch.tensor([[0., 0, 0, 0, 0, 0, 1, 1.0, 1.0]],
                                device="cuda", dtype=torch.float32)
    patch_pixel = model._patch_pixel_grid.cuda().unsqueeze(0).float()
    displacements, mm_rels, coverages = [], [], []

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
                n_samples=model.n_render_samples, near=model.render_near, far=model.render_far)
            rendered_feat = r1["feature"]
            ray_total_w = r1["total_weight"]
            # Photometric mismatch at the (drifted) initial pose, as the trigger sees it.
            patch_rgb_pred = model.color_head(rendered_feat)
            rgb_tgt_patch = torch.nn.functional.adaptive_avg_pool2d(
                rgb.float(), (model.grid_h, model.grid_w)).flatten(2).transpose(1, 2)
            w = ray_total_w.unsqueeze(-1).clamp(min=0.0)
            w_sum = w.sum(dim=1).clamp_min(1e-6)
            pooled_diff = ((patch_rgb_pred - rgb_tgt_patch) * w).sum(dim=1) / w_sum
            pooled_cur = (rgb_tgt_patch * w).sum(dim=1) / w_sum
            mm_l2 = float(pooled_diff.norm())
            mm_rel = mm_l2 / max(float(pooled_cur.norm()), 1e-6)
            coverage = float((ray_total_w > 1e-3).float().mean())
            # Continue streaming with pose head's output.
            initial_pose_9 = _pose_T_to_cam9(initial_T, fov)
            d9 = model.pose_head(patches, rendered_feat, ray_total_w, initial_pose_9)
            dT = cam9_to_pose_w_c(d9)
            cT = initial_T.float() @ dT
        pred_pos = cT[0, :3, 3].float().cpu().numpy()
        gt_pos = gt_rel_T[i, :3, 3].cpu().numpy()
        displacement = float(np.linalg.norm(pred_pos - gt_pos))
        displacements.append(displacement)
        mm_rels.append(mm_rel)
        coverages.append(coverage)
        wp = backproject_patches_to_world(patch_pixel, bootstrap_d, K, cT.detach())
        write_voxels_trilinear(voxel_state, wp, voxel_feat, weights=wc)
        prev_pose_9 = _pose_T_to_cam9(cT, fov).float()
        if (i + 1) % 200 == 0:
            print(f"[step0]   f={i+1}  disp={displacement:.2f}m  mm_rel={mm_rel:.3f}  cov={coverage:.2f}")
    return (np.array(displacements), np.array(mm_rels), np.array(coverages))


def count_events(mask, min_len=3):
    """Count contiguous runs of True with length ≥ min_len; return list of (start, end)."""
    runs = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            s = i
            while i < n and mask[i]:
                i += 1
            if i - s >= min_len:
                runs.append((s, i))
        else:
            i += 1
    return runs


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model, cfg = load_model(args.ckpt, args.weights_root)
    img_size = {"vjepa": 384, "dinov2": 518, "dinov3": 512}[cfg["encoder"]]
    fx, fy, cx, cy = intrinsics_for(args.seq)
    sx, sy = img_size / 640.0, img_size / 480.0
    K = torch.tensor([[[fx*sx, 0, cx*sx], [0, fy*sy, cy*sy], [0, 0, 1]]], device="cuda")
    fov = torch.tensor([[1.0, 1.0]], device="cuda")
    recs = sync_sequence(args.data_root / args.seq)
    print(f"[step0] {args.seq}: streaming {len(recs)} frames")
    gt_poses = np.stack([r.pose_w_c for r in recs])
    P0_inv = np.linalg.inv(gt_poses[0])
    gt_rel = np.einsum("ij,njk->nik", P0_inv, gt_poses)
    gt_rel_T = torch.from_numpy(gt_rel).float().cuda()
    disp, mm_rel, cov = stream_with_trigger(model, recs, K, fov, gt_rel_T)

    # Drift events: contiguous frames with disp ≥ threshold, ≥ 5 frames.
    drift_runs = count_events(disp >= args.drift_threshold_m, min_len=5)
    # Trigger fires: contiguous frames with mm_rel ≥ threshold, ≥ 1 frame.
    fire_runs = count_events(mm_rel >= args.trigger_threshold, min_len=1)
    # For each drift event, does it contain at least one trigger fire?
    drift_events_with_fire = []
    for s, e in drift_runs:
        fires_in = [r for r in fire_runs if r[0] < e and r[1] > s]
        drift_events_with_fire.append({
            "start": int(s), "end": int(e), "len": int(e - s),
            "n_fires": int(len(fires_in)),
            "max_disp": float(disp[s:e].max()),
            "max_mm_rel": float(mm_rel[s:e].max()),
        })

    n_events_with_fire = sum(1 for d in drift_events_with_fire if d["n_fires"] > 0)
    # Frames between events (recovery periods).
    recovery_durations = []
    for prev, cur in zip(drift_runs[:-1], drift_runs[1:]):
        recovery_durations.append(cur[0] - prev[1])

    print(f"\n[step0] === EVENT STRUCTURE ===")
    print(f"  Drift events (disp ≥ {args.drift_threshold_m}m, ≥5 frames): {len(drift_runs)}")
    print(f"  Events containing ≥1 trigger fire:                       {n_events_with_fire}")
    print(f"  Recovery periods between events:                          {len(recovery_durations)}")
    if recovery_durations:
        print(f"    median recovery duration: {int(np.median(recovery_durations))} frames")
    print(f"  Total trigger fires (contiguous runs):                    {len(fire_runs)}")
    print(f"\n[step0] Per-event detail:")
    for j, d in enumerate(drift_events_with_fire[:10]):
        print(f"  event {j+1}: frames {d['start']}-{d['end']}  ({d['len']}f)  "
               f"max_disp={d['max_disp']:.2f}m  fires={d['n_fires']}  max_mm={d['max_mm_rel']:.3f}")

    PASS = n_events_with_fire >= 3
    print(f"\n[step0] === VERDICT ===")
    if PASS:
        print(f"  PASS: {n_events_with_fire} drift events with trigger fires, "
               f"sufficient to constrain cooldown + per-event ΔATE statistics.")
    else:
        print(f"  FAIL: only {n_events_with_fire} drift event(s) with trigger fires. "
               f"Need ≥3. Fall back to concatenated fr2/desk + fr2/xyz, or pick another seq.")

    args.out.write_text(json.dumps({
        "seq": args.seq, "n_frames": len(recs),
        "n_drift_events": len(drift_runs),
        "n_events_with_fire": n_events_with_fire,
        "n_fire_runs": len(fire_runs),
        "median_recovery_duration": int(np.median(recovery_durations)) if recovery_durations else None,
        "drift_events": drift_events_with_fire,
        "verdict": "PASS" if PASS else "FAIL",
    }, indent=2))
    np.savez(args.out.parent / "per_frame.npz", displacement=disp, mismatch_rel=mm_rel, coverage=cov)
    print(f"[step0] saved {args.out.parent}/")


if __name__ == "__main__":
    main()
