"""Query-readout pre-launch β_disp test — gates the 10h training run.

THE PROPOSAL: replace render-and-compare (featural, drift-blind in self-similar
scenes per the photometric diagnostic) with a query-style readout that uses
∇_pose L_depth as its correction signal. The depth-clean-grid diagnostic
validated this gradient at 52% (streamed) / 66% (clean) toward-GT for re-
grounding refinement.

THE QUESTION THIS SCRIPT ANSWERS (cheaply, ~15 min, no training):
  Does ∇_pose L_depth, computed against the FIREWALLED ckpt's grid at known-
  drift perturbations from GT pose, point back toward GT with high enough
  reliability to drive a pose-head MLP?

  Gate is keyed on the STREAMED grid at 2.0m perturbation — that's the
  collapse-magnitude condition the head will actually face at inference. The
  clean grid is reported as secondary (upper-bound on the signal quality).

DESIGN:
  - Stream fr1/room twice: once with the firewalled model (streamed grid),
    once with GT poses (clean grid). Snapshot the voxel state at sampled
    test frames.
  - At each test frame × grid type × perturbation magnitude:
      Take GT pose. Perturb translation by m · d for random unit vectors d.
      Compute g = ∇_pose L_depth at the perturbed pose.
      Record: cos(g_t, -d) > 0 → did gradient point toward GT?
              ||g_t|| → signal strength.
  - Aggregate per (grid_type, magnitude, channel): toward-GT fraction +
    signal strength + 95% CI.

PASS (LOCKED — gates 10h training launch):
  STREAMED-grid toward-GT @ 2.0m translation ≥ 60%, 95% CI excludes 50%
  AND streamed signal strength ||g_t|| @ 2.0m ≥ 2× @ 0.3m (signal grows with
  drift — flat signal means structurally drift-blind regardless of direction).

PARTIAL: streamed passes @ 0.3m but not @ 2.0m → locally-good-globally-blind.
  Don't launch.

FAIL: streamed < 50% at any magnitude → switch to Option 2 (render-free
  occupancy query).
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
    VoxelGridState, backproject_patches_to_world, build_rays_from_pose,
    init_voxel_state, render_rays_volumetric, write_voxels_trilinear,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--seq", default="rgbd_dataset_freiburg1_room")
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--n-test-frames", type=int, default=50,
                   help="Number of test frames sampled evenly across the seq.")
    p.add_argument("--k-perturbations", type=int, default=10,
                   help="Random direction perturbations per (frame, magnitude).")
    p.add_argument("--t-magnitudes", type=float, nargs="+", default=[0.3, 1.0, 2.0],
                   help="Translation perturbation magnitudes (m). Gate is on 2.0m.")
    p.add_argument("--r-magnitudes", type=float, nargs="+", default=[0.05, 0.15, 0.3],
                   help="Rotation perturbation magnitudes (rad).")
    p.add_argument("--out", type=Path,
                   default=Path("viz/output/terrawm_d_query_grad_drift_test/verdict.json"))
    p.add_argument("--seed", type=int, default=0)
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
        use_photometric=cfg["model"].get("use_photometric", False),
        photometric_hidden=cfg["model"].get("photometric_hidden", 64),
        photometric_pose_gradient=cfg["model"].get("photometric_pose_gradient", False),
    )
    m.load_state_dict(ckpt["model"], strict=False)
    return m.cuda().eval(), cfg


def load_rgb(rec, img_size):
    from PIL import Image
    img = Image.open(rec.rgb_path).convert("RGB").resize((img_size, img_size))
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0
                            ).permute(2, 0, 1).unsqueeze(0).contiguous()


def snapshot_voxel(state: VoxelGridState):
    return {"features": state.features.detach().clone(),
            "write_mass": state.write_mass.detach().clone(),
            "cfg": state.cfg}


def restore_voxel(model, snap):
    s = init_voxel_state(snap["cfg"], 1, "cuda", torch.float32)
    s.features.copy_(snap["features"])
    s.write_mass.copy_(snap["write_mass"])
    return s


@torch.no_grad()
def stream_and_snapshot(model, recs, K, fov, test_frame_set, use_gt_poses: bool,
                         gt_rel_T):
    """Stream the sequence; snapshot voxel state at each test frame. If
    use_gt_poses=True (clean grid), writes happen at GT poses. Otherwise
    streaming with the model's actual predictions (the streamed grid)."""
    img_size = model.img_size
    voxel_state = model.init_voxel_state(1, "cuda", torch.float32)
    patch_pixel = model._patch_pixel_grid.cuda().unsqueeze(0).float()
    prev_pose_9 = torch.tensor([[0., 0, 0, 0, 0, 0, 1, 1.0, 1.0]],
                                device="cuda", dtype=torch.float32)
    snaps = {}
    for i, rec in enumerate(recs):
        rgb = load_rgb(rec, img_size).cuda(non_blocking=True)
        if use_gt_poses:
            pose_T = gt_rel_T[i:i+1]
        else:
            initial_T = cam9_to_pose_w_c(prev_pose_9)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                patches = model._encode_frame(rgb)
                ray_o1, ray_d1 = build_rays_from_pose(initial_T, K, patch_pixel)
                r1 = render_rays_volumetric(voxel_state, ray_o1, ray_d1,
                    n_samples=model.n_render_samples, near=model.render_near,
                    far=model.render_far)
                init9 = _pose_T_to_cam9(initial_T, fov)
                d9 = model.pose_head(patches, r1["feature"], r1["total_weight"], init9)
                dT = cam9_to_pose_w_c(d9)
                pose_T = (initial_T.float() @ dT).float()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            patches = model._encode_frame(rgb)
            bootstrap_d = model.bootstrap_depth(patches).float()
            voxel_feat = model.patch_to_voxel(patches).float()
            wc = model.write_confidence(patches).float() if model.use_write_confidence else None
            wp = backproject_patches_to_world(patch_pixel, bootstrap_d, K, pose_T.detach())
            write_voxels_trilinear(voxel_state, wp, voxel_feat, weights=wc)
            if not use_gt_poses:
                prev_pose_9 = _pose_T_to_cam9(pose_T, fov).float()
        if i in test_frame_set:
            snaps[i] = snapshot_voxel(voxel_state)
        if (i + 1) % 200 == 0:
            mode = "CLEAN" if use_gt_poses else "STREAMED"
            print(f"[grad-drift][{mode}] streamed {i+1}/{len(recs)}")
    return snaps


def compute_grad_at_pose(model, voxel_state, pose_T, K, patch_pixel,
                          bootstrap_d, current_patches):
    """Compute ∇_pose L_depth at the given (perturbed) pose. Returns the
    translation gradient (B, 3) and rotation gradient (B, 3, 3), plus the
    loss value as a scalar."""
    pose = pose_T.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            ray_o, ray_d = build_rays_from_pose(pose, K, patch_pixel)
            r = render_rays_volumetric(
                voxel_state, ray_o, ray_d,
                n_samples=model.n_render_samples,
                near=model.render_near, far=model.render_far,
            )
            r_depth = r["depth"].float()
            mass = r["total_weight"].float()
            valid = (mass > model.unwritten_mask_threshold).float()
            diff = (r_depth - bootstrap_d.detach()).abs()
            denom = valid.sum().clamp_min(1.0)
            L_depth = ((diff * valid * mass).sum() / denom)
        grad_pose, = torch.autograd.grad(L_depth, pose)
    return (grad_pose[:, :3, 3].detach().float().cpu().numpy()[0],     # (3,)
            grad_pose[:, :3, :3].detach().float().cpu().numpy()[0],    # (3, 3)
            float(L_depth.detach()))


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """(3,) axis-angle (axis * angle) → (3, 3) rotation matrix via Rodrigues."""
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1e-9:
        return np.eye(3)
    axis = axis_angle / angle
    K = np.array([[0, -axis[2], axis[1]],
                   [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    model, cfg = load_model(args.ckpt, args.weights_root)
    img_size = {"vjepa": 384, "dinov2": 518, "dinov3": 512}[cfg["encoder"]]
    fx, fy, cx, cy = intrinsics_for(args.seq)
    sx, sy = img_size / 640.0, img_size / 480.0
    K = torch.tensor([[[fx*sx, 0, cx*sx], [0, fy*sy, cy*sy], [0, 0, 1]]],
                      device="cuda")
    fov = torch.tensor([[1.0, 1.0]], device="cuda")
    recs = sync_sequence(args.data_root / args.seq)
    print(f"[grad-drift] {args.seq}: {len(recs)} frames")

    gt_poses = np.stack([r.pose_w_c for r in recs])
    P0_inv = np.linalg.inv(gt_poses[0])
    gt_rel = np.einsum("ij,njk->nik", P0_inv, gt_poses)
    gt_rel_T = torch.from_numpy(gt_rel).float().cuda()

    test_frames = sorted(set(
        int(x) for x in np.linspace(50, len(recs) - 50, args.n_test_frames).astype(int)
    ))
    test_frame_set = set(test_frames)
    print(f"[grad-drift] test frames: {len(test_frames)} sampled evenly")

    # === STREAM TWICE: clean grid (GT poses) AND streamed grid (pred poses) ===
    print(f"[grad-drift] streaming STREAMED grid (firewalled model's predictions)...")
    snaps_streamed = stream_and_snapshot(model, recs, K, fov, test_frame_set,
                                           use_gt_poses=False, gt_rel_T=gt_rel_T)
    print(f"[grad-drift] streaming CLEAN grid (GT poses)...")
    snaps_clean = stream_and_snapshot(model, recs, K, fov, test_frame_set,
                                        use_gt_poses=True, gt_rel_T=gt_rel_T)

    # === RUN GRADIENT TEST AT EACH TEST FRAME × MAGNITUDE × DIRECTION ===
    results = {grid: {
        "translation": {m: {"toward_gt": [], "signal_strength": []} for m in args.t_magnitudes},
        "rotation":    {m: {"toward_gt": [], "signal_strength": []} for m in args.r_magnitudes},
    } for grid in ["streamed", "clean"]}

    patch_pixel = model._patch_pixel_grid.cuda().unsqueeze(0).float()
    for grid_kind, snaps in [("streamed", snaps_streamed), ("clean", snaps_clean)]:
        print(f"\n[grad-drift] testing {grid_kind.upper()} grid at {len(snaps)} frames")
        for ti, frame in enumerate(test_frames):
            if frame not in snaps:
                continue
            voxel_state = restore_voxel(model, snaps[frame])
            rgb = load_rgb(recs[frame], img_size).cuda()
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    patches = model._encode_frame(rgb)
                    bootstrap_d = model.bootstrap_depth(patches).float()
            gt_T = gt_rel_T[frame:frame+1]

            for m in args.t_magnitudes:
                for _ in range(args.k_perturbations):
                    direction = rng.standard_normal(3)
                    direction = direction / max(np.linalg.norm(direction), 1e-9)
                    perturb_t = (m * direction).astype(np.float32)
                    pert_T = gt_T.clone()
                    pert_T[0, :3, 3] = pert_T[0, :3, 3] + torch.from_numpy(perturb_t).cuda()
                    grad_t, _, _ = compute_grad_at_pose(
                        model, voxel_state, pert_T, K, patch_pixel,
                        bootstrap_d, patches,
                    )
                    # Cos similarity: gradient should oppose the perturbation direction
                    grad_norm = float(np.linalg.norm(grad_t))
                    cos = float(np.dot(grad_t, -direction) / max(grad_norm * 1.0, 1e-9))
                    results[grid_kind]["translation"][m]["toward_gt"].append(cos > 0)
                    results[grid_kind]["translation"][m]["signal_strength"].append(grad_norm)

            for r_mag in args.r_magnitudes:
                for _ in range(args.k_perturbations):
                    axis = rng.standard_normal(3)
                    axis = axis / max(np.linalg.norm(axis), 1e-9)
                    perturb_aa = (r_mag * axis).astype(np.float64)
                    R_pert = axis_angle_to_matrix(perturb_aa)
                    pert_T = gt_T.clone()
                    pert_T[0, :3, :3] = pert_T[0, :3, :3] @ torch.from_numpy(
                        R_pert.astype(np.float32)).cuda()
                    _, grad_R, _ = compute_grad_at_pose(
                        model, voxel_state, pert_T, K, patch_pixel,
                        bootstrap_d, patches,
                    )
                    # Rotation gradient → tangent vector via skew-symmetric part
                    # grad_R is (3,3). Tangent at identity rotation: (R - R^T) / 2 → axis-angle-like.
                    grad_R_skew = (grad_R - grad_R.T) / 2.0
                    grad_axis = np.array([grad_R_skew[2, 1], grad_R_skew[0, 2], grad_R_skew[1, 0]])
                    grad_axis_norm = float(np.linalg.norm(grad_axis))
                    cos = float(np.dot(grad_axis, -axis) / max(grad_axis_norm, 1e-9))
                    results[grid_kind]["rotation"][r_mag]["toward_gt"].append(cos > 0)
                    results[grid_kind]["rotation"][r_mag]["signal_strength"].append(grad_axis_norm)
            if (ti + 1) % 10 == 0:
                print(f"[grad-drift][{grid_kind}] tested {ti+1}/{len(test_frames)} frames")

    # === AGGREGATE + VERDICT ===
    def summarize(records):
        toward = np.array(records["toward_gt"], dtype=bool)
        strength = np.array(records["signal_strength"], dtype=np.float64)
        n = len(toward)
        if n == 0:
            return {"n": 0, "toward_gt_frac": None, "ci95_low": None,
                     "ci95_high": None, "signal_strength_mean": None}
        p = float(toward.mean())
        # Wilson 95% CI for a Bernoulli p
        z = 1.96
        denom = 1 + z**2/n
        center = (p + z**2/(2*n)) / denom
        margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
        return {"n": n, "toward_gt_frac": p,
                 "ci95_low": float(center - margin), "ci95_high": float(center + margin),
                 "signal_strength_mean": float(strength.mean())}

    print("\n[grad-drift] === RESULTS ===")
    summary = {}
    for grid_kind in ["streamed", "clean"]:
        summary[grid_kind] = {"translation": {}, "rotation": {}}
        print(f"\n  Grid: {grid_kind.upper()}")
        for channel in ["translation", "rotation"]:
            mags = args.t_magnitudes if channel == "translation" else args.r_magnitudes
            print(f"    {channel.upper()}:")
            for m in mags:
                s = summarize(results[grid_kind][channel][m])
                summary[grid_kind][channel][str(m)] = s
                if s["n"] > 0:
                    unit = "m" if channel == "translation" else "rad"
                    print(f"      @ {m:.2f}{unit}: toward-GT = {s['toward_gt_frac']:.1%} "
                           f"(95% CI [{s['ci95_low']:.1%}, {s['ci95_high']:.1%}], "
                           f"n={s['n']}, ||g||={s['signal_strength_mean']:.4f})")

    # === PASS / PARTIAL / FAIL keyed on STREAMED grid @ 2.0m translation ===
    print("\n[grad-drift] === VERDICT (gated on STREAMED grid @ 2.0m translation) ===")
    streamed_t = summary["streamed"]["translation"]
    s_2m = streamed_t.get("2.0")
    s_03m = streamed_t.get("0.3")
    if s_2m is None or s_2m["n"] == 0:
        verdict = "INSUFFICIENT_DATA"
        bucket = "FAIL"
    else:
        ci_excludes_50 = s_2m["ci95_low"] > 0.5
        signal_grows = (s_2m["signal_strength_mean"] >= 2.0 * s_03m["signal_strength_mean"]
                         if s_03m and s_03m["n"] > 0 else False)
        if s_2m["toward_gt_frac"] >= 0.60 and ci_excludes_50 and signal_grows:
            bucket = "PASS"
            verdict = (f"PASS — streamed @ 2.0m: {s_2m['toward_gt_frac']:.1%} toward-GT "
                        f"(CI [{s_2m['ci95_low']:.1%}, {s_2m['ci95_high']:.1%}], excludes 50%), "
                        f"signal grows {s_2m['signal_strength_mean']/s_03m['signal_strength_mean']:.2f}× "
                        f"from 0.3m to 2.0m. Build QueryReadoutHead (Option 1, detached/first-order), "
                        f"warm-start from this ckpt, freeze grid-write, launch 40k train.")
        elif s_2m["toward_gt_frac"] >= 0.55 or (s_03m and s_03m["toward_gt_frac"] >= 0.65):
            bucket = "PARTIAL"
            verdict = (f"PARTIAL — streamed @ 2.0m: {s_2m['toward_gt_frac']:.1%} toward-GT. "
                        f"Signal exists but weak at collapse-magnitude. Locally-good-globally-blind "
                        f"is possible. Don't launch unless willing to bet the head MLP can extract "
                        f"more from the signal than its raw direction.")
        else:
            bucket = "FAIL"
            verdict = (f"FAIL — streamed @ 2.0m: {s_2m['toward_gt_frac']:.1%} toward-GT (≤ 60%). "
                        f"Gradient signal is drift-blind at collapse magnitudes. Switch to Option 2 "
                        f"(render-free occupancy query) and run its own β_disp gate.")

    print(f"  {verdict}")
    print(f"  [bucket: {bucket}]")

    out = {
        "ckpt": str(args.ckpt), "seq": args.seq,
        "n_test_frames": len(test_frames), "k_perturbations": args.k_perturbations,
        "summary": summary, "verdict": verdict, "bucket": bucket,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n[grad-drift] saved {args.out}")


if __name__ == "__main__":
    main()
