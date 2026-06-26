"""TerraWM-D drift-blindness check — validates the re-grounding trigger choice.

The GT-vs-pred render check (terrawm_d_gt_vs_pred_render.py) showed:
  - Displacement from GT is large from early in the run (2.76m at f=300)
  - Coverage stays ~1.0 through f=700 anyway
  - Coverage only crashes at f≈900-1050 when pred camera finally lands in
    unmapped space

This raises a critical question: IS THE 2m DRIFT VISIBLE to the pose head?

The pose head reads:
  diff = current_proj - rendered_feature       # per-patch feature mismatch
  pooled_diff = (diff * w).sum / w.sum         # weighted by ray weights

If the pose head's input signal (the mismatch) is HIGH while displacement is
high → drift IS visible to the channel; the pose head just isn't acting on it
(loss or training issue).

If the mismatch is LOW while displacement is high → drift-BLIND: the channel
cannot see the error. The geometric render-vs-current channel is too weak to
detect drift in locally self-similar scenes. PHOTOMETRIC LOSS IS THE FIX
(makes drift visible via color), and the re-grounding trigger must be
mismatch-based (not coverage-based, since coverage is high while drift is
already severe).

Cheap: streams fr1/room once, records per-frame mismatch + displacement +
coverage. ~3 minutes.
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
    backproject_patches_to_world, build_rays_from_pose,
    render_rays_volumetric, write_voxels_trilinear,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--seq", default="rgbd_dataset_freiburg1_room")
    p.add_argument("--n-frames", type=int, default=1100)
    p.add_argument("--out-dir", type=Path,
                   default=Path("viz/output/terrawm_d_drift_blind_check"))
    # When set, run the regression on PHOTOMETRIC mismatch (color head output
    # vs current frame's RGB) instead of geometric (current_proj vs rendered
    # feat). This is the pre-registered gate for the photometric retrain:
    # β_disp_photo must come back POSITIVE (CI excludes 0) to validate that
    # photometric cures the inversion. Requires a ckpt trained with
    # use_photometric=True.
    p.add_argument("--photometric", action="store_true",
                   help="Use photometric mismatch (rgb_pred vs rgb_target) instead of geometric.")
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
        use_photometric=cfg["model"].get("use_photometric", False),
        photometric_hidden=cfg["model"].get("photometric_hidden", 64),
        photometric_pose_gradient=cfg["model"].get("photometric_pose_gradient", False),
    )
    model.load_state_dict(ckpt["model"], strict=False)
    return model.cuda().eval(), cfg


def load_rgb(rec, img_size):
    from PIL import Image
    img = Image.open(rec.rgb_path).convert("RGB").resize((img_size, img_size))
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0
                            ).permute(2, 0, 1).unsqueeze(0).contiguous()


@torch.no_grad()
def stream_with_mismatch(model, recs, K, fov, gt_rel_T, photometric: bool = False):
    """Stream with pred poses. At each frame, compute the mismatch signal +
    displacement-from-GT. `photometric=False` (default) computes the geometric
    mismatch the pose head reads internally:
        pooled_diff_geo = (current_proj - rendered_feat) weighted by ray_total_w
    `photometric=True` computes photometric mismatch instead:
        pooled_diff_photo = (rgb_pred_patch - rgb_target_patch) weighted by ray_total_w
    The gate post-photometric-retrain requires β_disp_photo > +1.0 (positive),
    opposite to geometric β_disp_geom = -2.18.

    Two scalar mismatch metrics regardless of mode:
      - mismatch_l2: ‖pooled_diff‖ (absolute magnitude).
      - mismatch_rel: ‖pooled_diff‖ / ‖pooled_cur‖ (normalized).
    """
    if photometric:
        assert getattr(model, "use_photometric", False) and hasattr(model, "color_head"), \
            "--photometric requires a ckpt trained with use_photometric=True (ColorHead missing)."
    img_size = model.img_size
    voxel_state = model.init_voxel_state(batch_size=1, device="cuda", dtype=torch.float32)
    prev_pose_9 = torch.tensor([[0., 0., 0., 0., 0., 0., 1., 1.0, 1.0]],
                                device="cuda", dtype=torch.float32)
    patch_pixel = model._patch_pixel_grid.cuda().unsqueeze(0).float()

    stats = {
        "displacement": [], "coverage": [], "mismatch_l2": [], "mismatch_rel": [],
        "grid_mass": [], "pred_pos": [], "pooled_cur_norm": [], "pooled_diff_norm": [],
    }

    for i, rec in enumerate(recs):
        rgb = load_rgb(rec, img_size).cuda(non_blocking=True)
        initial_T = cam9_to_pose_w_c(prev_pose_9)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            patches = model._encode_frame(rgb)
            bootstrap_d = model.bootstrap_depth(patches).float()
            voxel_feat = model.patch_to_voxel(patches).float()
            write_w = model.write_confidence(patches).float() if model.use_write_confidence else None
            # First render — pose head's input.
            ray_o1, ray_d1 = build_rays_from_pose(initial_T, K, patch_pixel)
            render1 = render_rays_volumetric(
                voxel_state, ray_o1, ray_d1,
                n_samples=model.n_render_samples, near=model.render_near, far=model.render_far,
            )
            rendered_feat = render1["feature"]                                  # (B, P, voxel_dim)
            ray_total_w = render1["total_weight"]                               # (B, P)
            if photometric:
                # PHOTOMETRIC mismatch: compare ColorHead's predicted patch RGB
                # against current frame's patch-pooled RGB. Same weighting as
                # geometric (ray_total_w) so the two regressions are comparable.
                patch_rgb_pred = model.color_head(rendered_feat)                # (B, P, 3) ∈ [0, 1]
                rgb_target_patch = torch.nn.functional.adaptive_avg_pool2d(
                    rgb.float(), (model.grid_h, model.grid_w)
                ).flatten(2).transpose(1, 2)                                    # (B, P, 3)
                w = ray_total_w.unsqueeze(-1).clamp(min=0.0)                    # (B, P, 1)
                w_sum = w.sum(dim=1).clamp_min(1e-6)
                pooled_diff = ((patch_rgb_pred - rgb_target_patch) * w).sum(dim=1) / w_sum
                pooled_cur = (rgb_target_patch * w).sum(dim=1) / w_sum
                # Also keep geometric pooled_cur for reference (norm only).
                current_proj_geo = model.pose_head.current_proj(patches)        # (B, P, voxel_dim)
            else:
                # GEOMETRIC mismatch — the EXACT pooled_diff the pose head sees.
                current_proj = model.pose_head.current_proj(patches)            # (B, P, voxel_dim)
                diff = current_proj - rendered_feat
                w = ray_total_w.unsqueeze(-1).clamp(min=0.0)                    # (B, P, 1)
                w_sum = w.sum(dim=1).clamp_min(1e-6)
                pooled_diff = (diff * w).sum(dim=1) / w_sum                     # (B, voxel_dim)
                pooled_cur = (current_proj * w).sum(dim=1) / w_sum

            initial_pose_9 = _pose_T_to_cam9(initial_T, fov)
            if model.pose_gate_mode == "grid_mass":
                mass_total = voxel_state.write_mass.sum().detach()
                mass_gate = torch.sigmoid((mass_total - 1e3) / 1e2)
                external_gate = mass_gate.expand(1).unsqueeze(-1).to(initial_pose_9.dtype)
            else:
                external_gate = None
            delta_pose_9 = model.pose_head(patches, rendered_feat, ray_total_w,
                                            initial_pose_9, external_gate=external_gate)

        # Record metrics.
        pooled_diff_norm = float(pooled_diff.norm())
        pooled_cur_norm = float(pooled_cur.norm())
        mismatch_rel = pooled_diff_norm / max(pooled_cur_norm, 1e-6)
        coverage = float((ray_total_w > 1e-3).float().mean())

        # Composed pose for trajectory.
        delta_pose_T = cam9_to_pose_w_c(delta_pose_9)
        corrected_pose_T = initial_T.float() @ delta_pose_T
        pred_pos = corrected_pose_T[0, :3, 3].cpu().numpy()
        gt_pos = gt_rel_T[i, :3, 3].cpu().numpy()
        displacement = float(np.linalg.norm(pred_pos - gt_pos))

        stats["displacement"].append(displacement)
        stats["coverage"].append(coverage)
        stats["mismatch_l2"].append(pooled_diff_norm)
        stats["mismatch_rel"].append(mismatch_rel)
        stats["grid_mass"].append(float(voxel_state.write_mass.sum()))
        stats["pred_pos"].append(pred_pos.copy())
        stats["pooled_cur_norm"].append(pooled_cur_norm)
        stats["pooled_diff_norm"].append(pooled_diff_norm)

        # Write step.
        world_pts = backproject_patches_to_world(patch_pixel, bootstrap_d, K, corrected_pose_T.detach())
        write_voxels_trilinear(voxel_state, world_pts, voxel_feat, weights=write_w)
        new_abs_9 = _pose_T_to_cam9(corrected_pose_T, fov)
        prev_pose_9 = new_abs_9.float()

        if (i + 1) % 200 == 0:
            print(f"[d-blind]   f={i+1}  disp={displacement:.2f}m  cov={coverage:.2f}  "
                  f"mismatch_l2={pooled_diff_norm:.3f}  mismatch_rel={mismatch_rel:.3f}")

    for k in stats:
        if k != "pred_pos":
            stats[k] = np.array(stats[k])
    return stats


def main():
    args = parse_args()
    # Suffix the default output dir with mode so geometric and photometric
    # runs don't overwrite each other.
    if args.photometric and str(args.out_dir).endswith("terrawm_d_drift_blind_check"):
        args.out_dir = args.out_dir.with_name(args.out_dir.name + "_photo")
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
    print(f"[d-blind] {args.seq}: streaming {len(recs)} frames, computing mismatch + displacement")

    gt_poses_w_c = np.stack([r.pose_w_c for r in recs])
    P0_inv = np.linalg.inv(gt_poses_w_c[0])
    gt_rel = np.einsum("ij,njk->nik", P0_inv, gt_poses_w_c)
    gt_rel_T = torch.from_numpy(gt_rel).float().cuda()

    mode = "photometric" if args.photometric else "geometric"
    print(f"[d-blind] mismatch mode: {mode.upper()}")
    t0 = time.perf_counter()
    stats = stream_with_mismatch(model, recs, K, fov, gt_rel_T, photometric=args.photometric)
    print(f"[d-blind] streamed in {time.perf_counter() - t0:.1f}s")

    # === Drift-blindness verdict ===
    # Look at the window where displacement is HIGH (≥1m) and coverage is HIGH
    # (≥0.9). Is mismatch high (drift visible) or low (drift-blind)?
    disp = stats["displacement"]
    cov = stats["coverage"]
    mm_l2 = stats["mismatch_l2"]
    mm_rel = stats["mismatch_rel"]

    # Tight coverage match: both bins at coverage > 0.9, AND we explicitly
    # report per-bin mean coverage as the coverage-confound guard.
    high_disp_high_cov = (disp >= 1.0) & (cov > 0.9)
    low_disp_high_cov = (disp < 0.5) & (cov > 0.9)

    print(f"\n[d-blind] === DRIFT-BLINDNESS ANALYSIS ===")
    print(f"  Frames with displacement ≥ 1.0m AND coverage > 0.9: {int(high_disp_high_cov.sum())}/{len(disp)}")
    print(f"  Frames with displacement < 0.5m AND coverage > 0.9: {int(low_disp_high_cov.sum())}/{len(disp)}")

    verdict = "INSUFFICIENT_DATA"
    coverage_match_ok = False
    rel_ratio = None
    l2_ratio = None
    rel_p = None
    beta_disp_rel = None
    se_disp_rel = None
    ci_disp_rel = (None, None)
    beta_disp_rel_std = None
    beta_cov_rel = None
    beta_disp_l2 = None
    beta_cov_l2 = None
    n_pts = 0
    strict_hi = np.array([], dtype=bool)
    strict_lo = np.array([], dtype=bool)
    strict_rel_ratio = None
    strict_rel_p = None
    if high_disp_high_cov.sum() > 10 and low_disp_high_cov.sum() > 10:
        from scipy import stats as scistats

        # === Coverage-confound guard ===
        cov_hi_mean = float(cov[high_disp_high_cov].mean())
        cov_lo_mean = float(cov[low_disp_high_cov].mean())
        cov_diff = abs(cov_hi_mean - cov_lo_mean)
        cov_t, cov_p = scistats.ttest_ind(cov[high_disp_high_cov], cov[low_disp_high_cov], equal_var=False)
        print(f"\n  --- Coverage-confound guard ---")
        print(f"  Mean coverage in HIGH-drift bin: {cov_hi_mean:.4f}")
        print(f"  Mean coverage in LOW-drift  bin: {cov_lo_mean:.4f}")
        print(f"  |Δcoverage| = {cov_diff:.4f}   Welch's t={cov_t:+.2f}  p={cov_p:.4f}")
        coverage_match_ok = (cov_diff <= 0.02) and (cov_p > 0.01)
        if not coverage_match_ok:
            print(f"  WARNING: bins are NOT coverage-matched — drift-bin comparison is confounded.")
        else:
            print(f"  OK: bins are coverage-matched (|Δ|≤0.02 AND p>0.01).")

        # === PRIMARY: mismatch_rel (normalized — what the head effectively reads) ===
        rel_hi = mm_rel[high_disp_high_cov]
        rel_lo = mm_rel[low_disp_high_cov]
        rel_ratio = float(rel_hi.mean() / max(rel_lo.mean(), 1e-6))
        rel_t, rel_p = scistats.ttest_ind(rel_hi, rel_lo, equal_var=False)
        print(f"\n  --- PRIMARY: mismatch_rel = ||pooled_diff|| / ||pooled_cur|| ---")
        print(f"  Rel mismatch when drift HIGH (≥1m):  mean={rel_hi.mean():.4f}  median={np.median(rel_hi):.4f}")
        print(f"  Rel mismatch when drift LOW (<0.5m): mean={rel_lo.mean():.4f}  median={np.median(rel_lo):.4f}")
        print(f"  Ratio (high/low): {rel_ratio:.3f}   Welch's t={rel_t:+.2f}  p={rel_p:.4f}")

        # === SECONDARY: mismatch_l2 (absolute, susceptible to feature-norm confounds) ===
        l2_hi = mm_l2[high_disp_high_cov]
        l2_lo = mm_l2[low_disp_high_cov]
        l2_ratio = float(l2_hi.mean() / max(l2_lo.mean(), 1e-6))
        l2_t, l2_p = scistats.ttest_ind(l2_hi, l2_lo, equal_var=False)
        print(f"\n  --- SECONDARY: mismatch_l2 = ||pooled_diff|| (absolute) ---")
        print(f"  L2 mismatch when drift HIGH: mean={l2_hi.mean():.3f}")
        print(f"  L2 mismatch when drift LOW:  mean={l2_lo.mean():.3f}")
        print(f"  Ratio (high/low): {l2_ratio:.3f}   Welch's t={l2_t:+.2f}  p={l2_p:.4f}")

        # === Coverage-controlled regression (partial correlation) ===
        # Even when bins fail strict coverage-match, we can factor coverage out
        # by regressing mismatch_rel on (displacement, coverage). β_disp is the
        # drift effect with coverage held constant — it directly answers the
        # drift-blindness question.
        # Restrict to coverage > 0.5 to avoid the catastrophic-pose-loss tail
        # that has both 0 mismatch and 0 coverage (which would dominate as a
        # collinear cluster).
        mask = cov > 0.5
        X = np.column_stack([disp[mask], cov[mask], np.ones(int(mask.sum()))])
        y_rel = mm_rel[mask]
        beta_rel, *_ = np.linalg.lstsq(X, y_rel, rcond=None)
        beta_disp_rel, beta_cov_rel, intercept_rel = float(beta_rel[0]), float(beta_rel[1]), float(beta_rel[2])
        # OLS standard error on β_disp for the displacement coefficient.
        n_pts = int(mask.sum())
        resid = y_rel - X @ beta_rel
        sigma2 = float((resid ** 2).sum() / max(n_pts - 3, 1))
        XtX_inv = np.linalg.inv(X.T @ X)
        se_disp_rel = float(np.sqrt(sigma2 * XtX_inv[0, 0]))
        ci_disp_rel = (beta_disp_rel - 1.96 * se_disp_rel, beta_disp_rel + 1.96 * se_disp_rel)
        # Standardised: β_disp · std(disp) / std(rel) = the change in σ-units of
        # mismatch per σ-unit of displacement, factoring coverage out.
        std_ratio_rel = float(disp[mask].std()) / float(mm_rel[mask].std() + 1e-9)
        beta_disp_rel_std = beta_disp_rel * std_ratio_rel

        # Same for mismatch_l2.
        y_l2 = mm_l2[mask]
        beta_l2, *_ = np.linalg.lstsq(X, y_l2, rcond=None)
        beta_disp_l2, beta_cov_l2, intercept_l2 = float(beta_l2[0]), float(beta_l2[1]), float(beta_l2[2])

        # Strict-coverage sanity check: re-run mean comparison restricted to
        # coverage > 0.99 (tighter — both bins should have ~identical coverage).
        strict_hi = (disp >= 1.0) & (cov > 0.99)
        strict_lo = (disp < 0.5) & (cov > 0.99)
        strict_ok = strict_hi.sum() > 10 and strict_lo.sum() > 10
        strict_rel_ratio = None
        strict_rel_p = None
        if strict_ok:
            strict_rel_hi = mm_rel[strict_hi]
            strict_rel_lo = mm_rel[strict_lo]
            strict_rel_ratio = float(strict_rel_hi.mean() / max(strict_rel_lo.mean(), 1e-6))
            _, strict_rel_p = scistats.ttest_ind(strict_rel_hi, strict_rel_lo, equal_var=False)

        print(f"\n  --- Coverage-controlled regression (cov > 0.5, n={n_pts}) ---")
        print(f"  mismatch_rel = β_disp · disp + β_cov · cov + c")
        print(f"  β_disp = {beta_disp_rel:+.4f}  ± {se_disp_rel:.4f} (95% CI [{ci_disp_rel[0]:+.4f}, {ci_disp_rel[1]:+.4f}])")
        print(f"  β_cov  = {beta_cov_rel:+.4f}    c = {intercept_rel:+.4f}")
        print(f"  Standardised β_disp (σ-units): {beta_disp_rel_std:+.4f}")
        print(f"  Same regression on mismatch_l2: β_disp_l2 = {beta_disp_l2:+.3f}  β_cov_l2 = {beta_cov_l2:+.3f}")
        if strict_ok:
            print(f"\n  --- Strict-coverage sanity (cov > 0.99) ---")
            print(f"  HIGH-drift bin: n={int(strict_hi.sum())}  rel mismatch mean={strict_rel_hi.mean():.4f}")
            print(f"  LOW-drift bin:  n={int(strict_lo.sum())}  rel mismatch mean={strict_rel_lo.mean():.4f}")
            print(f"  Ratio (high/low): {strict_rel_ratio:.3f}   p={strict_rel_p:.4f}")
        else:
            print(f"  --- Strict-coverage sanity: insufficient samples (hi={int(strict_hi.sum())}, lo={int(strict_lo.sum())})")

        # === Verdict — keyed on β_disp (coverage-controlled), backed by strict-coverage sanity ===
        # Verdict text is mode-dependent: for GEOMETRIC, β_disp < 0 = drift-blind (BAD);
        # for PHOTOMETRIC, β_disp > 0 = signal-cured (GOOD, the pre-registered gate).
        ci_excludes_zero = (ci_disp_rel[0] > 0) or (ci_disp_rel[1] < 0)
        channel = "PHOTOMETRIC" if args.photometric else "GEOMETRIC"
        if not coverage_match_ok and not ci_excludes_zero:
            verdict = (f"COVERAGE-CONFOUND + INSIGNIFICANT ({channel}) — bins differ by {cov_diff:.3f} in coverage AND "
                       f"the coverage-controlled β_disp = {beta_disp_rel:+.4f} has CI {ci_disp_rel} crossing 0.")
        elif args.photometric:
            # GATE for photometric retrain — pre-registered:
            #   β_disp_photo ≥ +1.0, CI excludes 0  →  CURE CONFIRMED
            #   +0.3 ≤ β_disp_photo < +1.0          →  WEAK signal, partial cure
            #   β_disp_photo < +0.3 or ≤ 0         →  FAIL — photometric did not cure inversion
            if beta_disp_rel >= 1.0 and ci_excludes_zero:
                verdict = (f"GATE PASS — PHOTOMETRIC CURES INVERSION (coverage-controlled): β_disp_photo = "
                           f"{beta_disp_rel:+.4f} (95% CI {ci_disp_rel}), strict-cov ratio "
                           f"{strict_rel_ratio if strict_ok else 'NA'}. Photometric mismatch INCREASES "
                           f"with drift at fixed coverage — opposite sign from geometric β_disp_geom = -2.18. "
                           f"Re-grounding mechanism can be built on photometric mismatch.")
            elif 0.3 <= beta_disp_rel < 1.0 and ci_excludes_zero:
                verdict = (f"GATE PARTIAL — weak positive signal (β_disp_photo = {beta_disp_rel:+.4f}, "
                           f"95% CI {ci_disp_rel}). Photometric direction is correct but magnitude is weak. "
                           f"Re-grounding viable with caution; consider fusing with another signal.")
            else:
                verdict = (f"GATE FAIL — PHOTOMETRIC DID NOT CURE INVERSION: β_disp_photo = "
                           f"{beta_disp_rel:+.4f} (95% CI {ci_disp_rel}). Re-grounding cannot be built "
                           f"on photometric mismatch. Recovery mechanism needs rethinking (external "
                           f"loop closure / multi-view consistency / different feature).")
        else:
            # GEOMETRIC mode — the inversion diagnostic.
            if beta_disp_rel < -1.0 and ci_excludes_zero:
                verdict = (f"DRIFT-BLIND CONFIRMED (GEOMETRIC, coverage-controlled): β_disp = "
                           f"{beta_disp_rel:+.4f} (95% CI {ci_disp_rel}), strict-cov ratio "
                           f"{strict_rel_ratio if strict_ok else 'NA'}. Geometric channel actively "
                           f"misreports drift. Photometric is the primary fix.")
            elif abs(beta_disp_rel) < 0.5 and not ci_excludes_zero:
                verdict = (f"DRIFT-BLIND CONFIRMED (GEOMETRIC, flat): β_disp = {beta_disp_rel:+.4f} "
                           f"(95% CI {ci_disp_rel}). Geometric channel cannot see drift. Photometric is "
                           f"the primary fix.")
            elif beta_disp_rel > 1.0 and ci_excludes_zero:
                verdict = (f"GEOMETRIC SIGNAL PRESENT: β_disp = {beta_disp_rel:+.4f} "
                           f"(95% CI {ci_disp_rel}). Drift IS visible. Head is failing to act on it.")
            else:
                verdict = (f"AMBIGUOUS (GEOMETRIC): β_disp = {beta_disp_rel:+.4f} (95% CI {ci_disp_rel}).")
        if not coverage_match_ok:
            verdict = "[coverage-confound noted — keying on partial regression β_disp instead of bin means] " + verdict

    print(f"\n[d-blind] === VERDICT ===")
    print(f"  {verdict}")

    # Persist per-frame arrays for re-analysis without re-running.
    np.savez(args.out_dir / "per_frame.npz",
              displacement=disp, coverage=cov, mismatch_l2=mm_l2, mismatch_rel=mm_rel,
              grid_mass=stats["grid_mass"], pooled_cur_norm=stats["pooled_cur_norm"],
              pooled_diff_norm=stats["pooled_diff_norm"])

    # === Plots ===
    frames = np.arange(len(disp))

    # 1. Per-frame time series.
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    axes[0].plot(frames, disp, color="tab:purple", linewidth=0.8)
    axes[0].set_ylabel("displacement from GT (m)")
    axes[0].set_title("Displacement, coverage, mismatch over frames — drift-blindness check")
    axes[0].grid(alpha=0.3); axes[0].axhline(1.0, color="red", linestyle="--", alpha=0.4, label="1m threshold")
    axes[0].legend()
    axes[1].plot(frames, cov, color="tab:blue", linewidth=0.8)
    axes[1].set_ylabel("render coverage"); axes[1].set_ylim(-0.05, 1.05); axes[1].grid(alpha=0.3)
    axes[2].plot(frames, mm_l2, color="tab:red", linewidth=0.8)
    axes[2].set_ylabel("mismatch L2 (pose-head input)")
    axes[2].set_title("Mismatch_L2 — the signal the pose head reads (current_proj vs rendered_feat)")
    axes[2].grid(alpha=0.3)
    axes[3].plot(frames, mm_rel, color="tab:orange", linewidth=0.8)
    axes[3].set_ylabel("mismatch relative (||diff||/||cur||)")
    axes[3].set_xlabel("frame")
    axes[3].grid(alpha=0.3)
    fig.suptitle(f"{args.seq} — verdict: {verdict.split('—')[0].strip()}", fontsize=10)
    plt.tight_layout()
    plt.savefig(args.out_dir / "drift_blind_timeseries.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 2. Mismatch_rel (PRIMARY) vs displacement scatter, colored by coverage.
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sc = axes[0].scatter(disp, mm_rel, c=cov, cmap="viridis", s=8, alpha=0.6)
    plt.colorbar(sc, ax=axes[0], label="render coverage")
    axes[0].set_xlabel("displacement from GT (m)")
    axes[0].set_ylabel("mismatch_rel = ||diff||/||cur||  [PRIMARY]")
    axes[0].set_title("Mismatch_REL vs displacement (PRIMARY — keys the verdict)\n"
                       "Drift-blind: horizontal cloud. Drift-visible: rising trend.")
    axes[0].axvline(1.0, color="red", linestyle="--", alpha=0.4, label="1m threshold")
    axes[0].grid(alpha=0.3); axes[0].legend()
    sc2 = axes[1].scatter(disp, mm_l2, c=cov, cmap="viridis", s=8, alpha=0.6)
    plt.colorbar(sc2, ax=axes[1], label="render coverage")
    axes[1].set_xlabel("displacement from GT (m)")
    axes[1].set_ylabel("mismatch_l2 = ||diff||  [secondary]")
    axes[1].set_title("Mismatch_L2 vs displacement (secondary — feature-norm confounded)")
    axes[1].axvline(1.0, color="red", linestyle="--", alpha=0.4, label="1m threshold")
    axes[1].grid(alpha=0.3); axes[1].legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "mismatch_vs_displacement.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 3. Joint displacement-vs-coverage scatter — the diagnostic insight.
    fig, ax = plt.subplots(figsize=(11, 8))
    sc = ax.scatter(disp, cov, c=mm_rel, cmap="plasma", s=8, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="mismatch_rel")
    ax.axvline(1.0, color="red", linestyle="--", alpha=0.4, label="1m drift threshold")
    ax.axhline(0.9, color="orange", linestyle="--", alpha=0.4, label="coverage > 0.9 threshold")
    ax.set_xlabel("displacement from GT (m)")
    ax.set_ylabel("render coverage")
    ax.set_title("Joint distribution: displacement vs coverage (colored by mismatch_rel).\n"
                  "Top-right quadrant (high drift AND high coverage) is the drift-blind regime.\n"
                  "If color in that quadrant is uniform → drift-blind. If darker on the right → signal present.")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "displacement_vs_coverage.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "ckpt": str(args.ckpt),
        "seq": args.seq,
        "n_frames": len(recs),
        "mismatch_mode": "photometric" if args.photometric else "geometric",
        "verdict": verdict,
        "verdict_keyed_on": "partial_regression_beta_disp (mismatch_rel ~ disp + cov)",
        "coverage_match_ok": bool(coverage_match_ok),
        "rel_ratio": rel_ratio,
        "rel_p": float(rel_p) if rel_p is not None else None,
        "l2_ratio": l2_ratio,
        "partial_regression": {
            "beta_disp_rel": beta_disp_rel,
            "beta_disp_rel_se": se_disp_rel,
            "beta_disp_rel_ci_95": list(ci_disp_rel),
            "beta_disp_rel_std_units": beta_disp_rel_std,
            "beta_cov_rel": beta_cov_rel,
            "beta_disp_l2": beta_disp_l2,
            "beta_cov_l2": beta_cov_l2,
            "n_samples": n_pts,
        },
        "strict_coverage_sanity": {
            "n_high_drift": int(strict_hi.sum()),
            "n_low_drift": int(strict_lo.sum()),
            "rel_ratio": strict_rel_ratio,
            "rel_p": float(strict_rel_p) if strict_rel_p is not None else None,
        },
        "displacement_stats": {
            "mean": float(disp.mean()), "median": float(np.median(disp)),
            "max": float(disp.max()),
        },
        "coverage_stats": {
            "mean": float(cov.mean()), "min": float(cov.min()),
            "frac_above_09": float((cov > 0.9).mean()),
            "high_drift_bin_mean_cov": float(cov[high_disp_high_cov].mean()) if high_disp_high_cov.sum() > 0 else None,
            "low_drift_bin_mean_cov": float(cov[low_disp_high_cov].mean()) if low_disp_high_cov.sum() > 0 else None,
        },
        "mismatch_stats": {
            "l2_mean": float(mm_l2.mean()),
            "l2_at_high_drift_mean": float(mm_l2[high_disp_high_cov].mean()) if high_disp_high_cov.sum() > 0 else None,
            "l2_at_low_drift_mean": float(mm_l2[low_disp_high_cov].mean()) if low_disp_high_cov.sum() > 0 else None,
            "rel_at_high_drift_mean": float(mm_rel[high_disp_high_cov].mean()) if high_disp_high_cov.sum() > 0 else None,
            "rel_at_low_drift_mean": float(mm_rel[low_disp_high_cov].mean()) if low_disp_high_cov.sum() > 0 else None,
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[d-blind] saved {args.out_dir}/")


if __name__ == "__main__":
    main()
