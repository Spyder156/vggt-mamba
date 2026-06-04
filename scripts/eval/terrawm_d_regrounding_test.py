"""TerraWM-D re-grounding test — the world-model thesis test.

The premise: a persistent scene map should let the pose head re-ground when
the camera revisits a previously-seen region. If the architecture does what
it claims, drift should be BOUNDED — ATE should grow when the camera is in
unseen territory and SHRINK (or stop growing) when it revisits mapped regions.

This script:
  1. Streams a long-horizon revisit sequence (fr3/long_office_household).
  2. Sim3-aligns pred trajectory to GT once globally.
  3. For each frame, computes per-frame ATE = ||pred_aligned[i] - gt[i]||.
  4. Detects revisit events: frame i is a revisit if min_{j < i - gap} |gt[j] - gt[i]| < r_revisit.
  5. Compares ATE growth-rate at revisit windows vs non-revisit windows.

Pre-registered verdict:
  - PRIMARY: mean ATE-change in window AFTER revisits < mean change in non-revisit windows.
    Statistically lower → re-grounding works (drift slows or reverses at revisits).
    Same → re-grounding doesn't work (ATE grows monotonically regardless).
  - SECONDARY: long-horizon ATE growth slope. If bounded (sub-linear), persistence
    is providing some correction. If linear, no correction.

Run on the pure1 ckpt (best calibration to date, right-shaped trajectories).
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
from vggt_mamba.eval.metrics import umeyama_sim3                                    # noqa: E402
from vggt_mamba.models.aggregators.anchor_pool import cam9_to_pose_w_c              # noqa: E402
from vggt_mamba.models.terrawm_d import build_terrawm_d                             # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--seq", default="rgbd_dataset_freiburg3_long_office_household")
    p.add_argument("--n-frames", type=int, default=1500,
                   help="long-horizon needed for revisits to matter")
    p.add_argument("--r-revisit", type=float, default=0.3,
                   help="distance (m) below which frame is considered a revisit")
    p.add_argument("--gap-frames", type=int, default=100,
                   help="minimum frame gap for a revisit to count (filters consecutive frames)")
    p.add_argument("--window-after", type=int, default=20,
                   help="frames after a revisit over which to measure ATE change")
    p.add_argument("--window-before", type=int, default=20,
                   help="frames before a revisit, baseline ATE-change measurement")
    p.add_argument("--out-dir", type=Path,
                   default=Path("viz/output/terrawm_d_regrounding"))
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
    )
    model.load_state_dict(ckpt["model"], strict=False)
    return model.cuda().eval(), cfg


def load_rgb(rec, img_size):
    from PIL import Image
    img = Image.open(rec.rgb_path).convert("RGB").resize((img_size, img_size))
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0
                            ).permute(2, 0, 1).unsqueeze(0).contiguous()


@torch.no_grad()
def stream_sequence(model, recs, K, fov) -> np.ndarray:
    img_size = model.img_size
    voxel_state = model.init_voxel_state(batch_size=1, device="cuda", dtype=torch.float32)
    prev_pose_9 = torch.tensor([[0., 0., 0., 0., 0., 0., 1., 1.0, 1.0]],
                                device="cuda", dtype=torch.float32)
    pred_poses = []
    for i, rec in enumerate(recs):
        rgb = load_rgb(rec, img_size).cuda(non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out, new_abs_9 = model.streaming_forward(rgb, voxel_state, prev_pose_9, K, fov=fov)
        new_abs_T = cam9_to_pose_w_c(new_abs_9)
        pred_poses.append(new_abs_T[0].float().cpu().numpy())
        prev_pose_9 = new_abs_9.float()
        if (i + 1) % 200 == 0:
            print(f"[d-reground]   streamed {i + 1}/{len(recs)} frames")
    return np.stack(pred_poses)


def relativize_to_frame0(poses: np.ndarray) -> np.ndarray:
    P0_inv = np.linalg.inv(poses[0])
    return np.einsum("ij,njk->nik", P0_inv, poses)


def detect_revisits(gt_t: np.ndarray, r_revisit: float, gap_frames: int) -> np.ndarray:
    """For each frame i, return min distance to gt_t[j] for j in [0, i - gap_frames].
    A frame is a 'revisit candidate' iff this distance < r_revisit.
    """
    T = gt_t.shape[0]
    min_dist = np.full(T, np.inf)
    for i in range(gap_frames + 1, T):
        old = gt_t[: i - gap_frames]                                # (i - gap, 3)
        d = np.linalg.norm(old - gt_t[i], axis=-1)
        min_dist[i] = float(d.min())
    return min_dist


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
    print(f"[d-reground] {args.seq}: streaming {len(recs)} frames")
    t0 = time.perf_counter()
    pred_abs = stream_sequence(model, recs, K, fov)                                # (T, 4, 4)
    dt = time.perf_counter() - t0
    print(f"[d-reground] streamed in {dt:.1f}s")

    gt_abs_raw = np.stack([r.pose_w_c for r in recs])                              # (T, 4, 4)
    gt_abs = relativize_to_frame0(gt_abs_raw)
    pred_t = pred_abs[:, :3, 3]
    gt_t = gt_abs[:, :3, 3]

    # === Sim(3) global alignment ===
    # Single Sim3 over the WHOLE trajectory — best-fit. Per-frame ATE then shows
    # local deviation from the best-overall-alignment. If re-grounding works,
    # post-revisit frames should have lower local deviation than pre-revisit.
    s, R, t = umeyama_sim3(pred_t, gt_t)
    pred_aligned = (s * (R @ pred_t.T)).T + t
    per_frame_ate = np.linalg.norm(pred_aligned - gt_t, axis=-1)                   # (T,)
    print(f"[d-reground] Sim3 alignment: scale={s:.4f}")
    print(f"[d-reground] per-frame ATE: mean={per_frame_ate.mean():.4f}m  "
          f"median={np.median(per_frame_ate):.4f}m  "
          f"max={per_frame_ate.max():.4f}m  "
          f"final={per_frame_ate[-1]:.4f}m")

    # === Revisit detection ===
    min_dist_to_past = detect_revisits(gt_t, args.r_revisit, args.gap_frames)
    revisit_mask = min_dist_to_past < args.r_revisit                                # (T,) bool
    revisit_frames = np.where(revisit_mask)[0]
    print(f"[d-reground] {len(revisit_frames)} revisit frames detected "
          f"(r<{args.r_revisit}m, gap>{args.gap_frames})")
    if len(revisit_frames) == 0:
        print(f"[d-reground] WARNING: no revisits detected; nothing to test")

    # === Re-grounding signal ===
    # For each revisit event, compute ATE_change = ATE[i + window_after] - ATE[i].
    # Compare to a baseline of "non-revisit" frames: frames where the camera is
    # far from any earlier point.
    re_grounding_signal = []                                                       # at revisits
    baseline_signal = []                                                            # at non-revisits
    far_frames = np.where(min_dist_to_past > 2 * args.r_revisit)[0]
    far_frames = far_frames[far_frames > args.window_before]
    far_frames = far_frames[far_frames < len(per_frame_ate) - args.window_after]
    for i in revisit_frames:
        if i < args.window_before or i + args.window_after >= len(per_frame_ate):
            continue
        ate_before = per_frame_ate[i - args.window_before:i].mean()
        ate_after = per_frame_ate[i:i + args.window_after].mean()
        re_grounding_signal.append(float(ate_after - ate_before))
    for i in far_frames:
        ate_before = per_frame_ate[i - args.window_before:i].mean()
        ate_after = per_frame_ate[i:i + args.window_after].mean()
        baseline_signal.append(float(ate_after - ate_before))

    re_grounding_signal = np.array(re_grounding_signal)
    baseline_signal = np.array(baseline_signal)
    print()
    print(f"[d-reground] === RE-GROUNDING SIGNAL ===")
    print(f"  N revisits used:   {len(re_grounding_signal)}")
    print(f"  N baselines used:  {len(baseline_signal)}")
    print(f"  ATE-change after REVISIT (mean):  {re_grounding_signal.mean():+.5f} m")
    print(f"  ATE-change after REVISIT (median): {np.median(re_grounding_signal):+.5f} m")
    print(f"  ATE-change in BASELINE (mean):    {baseline_signal.mean():+.5f} m")
    print(f"  ATE-change in BASELINE (median):  {np.median(baseline_signal):+.5f} m")
    print(f"  Difference (revisit - baseline):   {re_grounding_signal.mean() - baseline_signal.mean():+.5f} m")
    print(f"  (negative = re-grounding works; revisits slow drift more than random)")

    # Welch's t-test for statistical significance.
    from scipy import stats
    if len(re_grounding_signal) > 1 and len(baseline_signal) > 1:
        t_stat, p_val = stats.ttest_ind(re_grounding_signal, baseline_signal, equal_var=False)
        print(f"  t-statistic:  {t_stat:+.3f}   p-value: {p_val:.4f}")
        signif = "SIGNIFICANT" if p_val < 0.05 else "not significant"
        sign_dir = "re-grounding shows" if re_grounding_signal.mean() < baseline_signal.mean() else "no effect"
        print(f"  Two-sided: {signif}  ({sign_dir})")
    else:
        p_val = float("nan")
        signif = "INSUFFICIENT_DATA"

    # === Long-horizon ATE growth ===
    # If re-grounding works on revisit-rich sequences, ATE should grow sub-linearly.
    # Fit log-log to per_frame_ate(t) and report the slope.
    valid_idx = np.arange(20, len(per_frame_ate))                                  # skip warmup
    log_t = np.log(valid_idx.astype(float))
    log_ate = np.log(np.clip(per_frame_ate[valid_idx], 1e-6, None))
    slope, intercept = np.polyfit(log_t, log_ate, 1)
    print(f"\n[d-reground] === LONG-HORIZON ATE GROWTH ===")
    print(f"  log-log slope: {slope:+.3f}   intercept: {intercept:+.3f}")
    print(f"  (slope=1: linear drift; slope=0.5: random walk; slope=0: bounded)")
    if slope < 0.3:
        growth_verdict = "BOUNDED  (persistence is suppressing drift)"
    elif slope < 0.7:
        growth_verdict = "RANDOM-WALK  (drift unbounded but sub-linear)"
    else:
        growth_verdict = "LINEAR-DRIFT  (no drift suppression)"
    print(f"  Verdict: {growth_verdict}")

    # === Visualizations ===

    # 1. Per-frame ATE over time, with revisit events marked.
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(per_frame_ate, color="tab:blue", linewidth=0.8, label="per-frame ATE")
    if len(revisit_frames):
        rev_y = per_frame_ate[revisit_frames]
        axes[0].scatter(revisit_frames, rev_y, color="red", s=8, alpha=0.6,
                        label=f"revisit events (n={len(revisit_frames)})")
    axes[0].set_ylabel("ATE per frame (m)")
    axes[0].set_title(f"Per-frame ATE over time — {args.seq}  "
                       f"({len(recs)} frames, Sim3-aligned, s={s:.3f})\n"
                       f"Revisit definition: min distance to gt[j], j < i-{args.gap_frames}, "
                       f"distance < {args.r_revisit}m")
    axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].plot(min_dist_to_past, color="tab:green", linewidth=0.8)
    axes[1].axhline(args.r_revisit, color="red", linestyle="--", alpha=0.6,
                     label=f"revisit threshold {args.r_revisit}m")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("min dist to past (m)")
    axes[1].grid(alpha=0.3); axes[1].legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "ate_over_time.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[d-reground] saved ate_over_time.png")

    # 2. Re-grounding signal histogram.
    fig, ax = plt.subplots(figsize=(11, 5))
    if len(re_grounding_signal) and len(baseline_signal):
        bins = np.linspace(
            min(re_grounding_signal.min(), baseline_signal.min()),
            max(re_grounding_signal.max(), baseline_signal.max()),
            40,
        )
        ax.hist(baseline_signal, bins=bins, alpha=0.5, color="gray",
                label=f"baseline (no revisit, mean={baseline_signal.mean():+.4f})")
        ax.hist(re_grounding_signal, bins=bins, alpha=0.5, color="tab:red",
                label=f"revisit event (mean={re_grounding_signal.mean():+.4f})")
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel("ATE change in window after event (m)")
        ax.set_ylabel("count")
        title_extra = f"p={p_val:.3f}" if not np.isnan(p_val) else "insufficient data"
        ax.set_title(f"Re-grounding signal: ATE change AT revisits vs OFF revisits  ({title_extra})\n"
                      f"Re-grounding works iff RED histogram is shifted LEFT of GRAY (less ATE growth at revisits)")
        ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "regrounding_signal_histogram.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[d-reground] saved regrounding_signal_histogram.png")

    # 3. XZ trajectory plot with revisit events.
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.plot(gt_t[:, 0], gt_t[:, 2], "o-", color="green", markersize=2, label="GT")
    ax.plot(pred_aligned[:, 0], pred_aligned[:, 2], "o-", color="red", markersize=2,
            label=f"pred (Sim3-aligned, ATE_RMSE={np.sqrt((per_frame_ate**2).mean()):.3f}m)")
    if len(revisit_frames):
        ax.scatter(gt_t[revisit_frames, 0], gt_t[revisit_frames, 2],
                   color="blue", s=20, alpha=0.3, label="revisit frames (on GT)")
    ax.scatter(gt_t[0, 0], gt_t[0, 2], color="black", s=100, marker="s", zorder=5, label="start")
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ax.set_title(f"Trajectory + revisit events — {args.seq}")
    ax.legend(); ax.grid(alpha=0.3); ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(args.out_dir / "trajectory_with_revisits.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[d-reground] saved trajectory_with_revisits.png")

    # === Verdict ===
    print(f"\n[d-reground] === FINAL VERDICT ===")
    sig_drop = (re_grounding_signal.mean() < baseline_signal.mean()) if len(re_grounding_signal) else False
    sig_signif = (p_val < 0.05) if not np.isnan(p_val) else False
    if sig_drop and sig_signif and slope < 0.5:
        verdict = "RE-GROUNDING WORKS  (revisits significantly slow drift, ATE bounded)"
    elif sig_drop and sig_signif:
        verdict = "RE-GROUNDING PARTIAL  (revisits slow drift, but overall growth still linear)"
    elif slope < 0.5:
        verdict = "DRIFT BOUNDED  (overall sub-linear growth, but no revisit-specific signal)"
    else:
        verdict = "NO RE-GROUNDING  (drift grows ~linearly, revisits no different from baseline)"
    print(f"  {verdict}")

    summary = {
        "ckpt": str(args.ckpt),
        "seq": args.seq,
        "n_frames": len(recs),
        "sim3_scale": float(s),
        "per_frame_ate": {
            "mean": float(per_frame_ate.mean()),
            "median": float(np.median(per_frame_ate)),
            "max": float(per_frame_ate.max()),
            "final": float(per_frame_ate[-1]),
        },
        "n_revisits": int(revisit_mask.sum()),
        "re_grounding_signal": {
            "n_revisits_used": int(len(re_grounding_signal)),
            "n_baselines_used": int(len(baseline_signal)),
            "mean_revisit": float(re_grounding_signal.mean()) if len(re_grounding_signal) else None,
            "mean_baseline": float(baseline_signal.mean()) if len(baseline_signal) else None,
            "difference": float(re_grounding_signal.mean() - baseline_signal.mean())
                if len(re_grounding_signal) and len(baseline_signal) else None,
            "p_value": float(p_val) if not np.isnan(p_val) else None,
        },
        "long_horizon_growth": {
            "log_log_slope": float(slope),
            "verdict": growth_verdict,
        },
        "final_verdict": verdict,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[d-reground] saved {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
