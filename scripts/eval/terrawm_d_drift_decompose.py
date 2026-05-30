"""TerraWM-D drift cause decomposition.

We've established the drift shape (RPE growth slope ~0.8 = systematic bias,
not random walk, not chaos). This script decomposes the CAUSE into:

  1. Rotation-vs-translation: integrate pred translations in GT rotations.
     If the resulting bbox ≈ GT bbox → rotation drift is the cause; translations
     in camera frame are roughly fine.  If still 3-6× → translations have their
     own systematic component independent of rotation.

  2. Mean residual per axis (translation): mean(pred_delta_t - gt_delta_t).
     Non-zero → directional bias.

  3. Scale regression: regress |pred_delta| vs |gt_delta|.
     Slope > ~1.2 → scale bias (model over-strides).

  4. Per-axis correlation: corr(pred_delta_xyz, gt_delta_xyz).
     Low correlation → predictions barely track GT regardless of bias.

Pure numpy on existing data. Re-streams sequences once to get fresh predictions.
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
from vggt_mamba.models.terrawm_d import build_terrawm_d                             # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--seqs", nargs="+", default=[
        "rgbd_dataset_freiburg3_sitting_xyz",
        "rgbd_dataset_freiburg1_room",
        "rgbd_dataset_freiburg3_walking_xyz",
    ])
    p.add_argument("--n-frames", type=int, default=601)
    p.add_argument("--out-dir", type=Path, default=Path("viz/output/terrawm_d_drift_decompose"))
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


@torch.no_grad()
def stream_sequence(model, recs, K, fov) -> np.ndarray:
    img_size = model.img_size
    voxel_state = model.init_voxel_state(batch_size=1, device="cuda", dtype=torch.float32)
    prev_pose_9 = torch.tensor([[0., 0., 0., 0., 0., 0., 1., 1.0, 1.0]],
                                device="cuda", dtype=torch.float32)
    pred_poses = []
    for rec in recs:
        rgb = load_rgb(rec, img_size).cuda(non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out, new_abs_9 = model.streaming_forward(rgb, voxel_state, prev_pose_9, K, fov=fov)
        new_abs_T = cam9_to_pose_w_c(new_abs_9)
        pred_poses.append(new_abs_T[0].float().cpu().numpy())
        prev_pose_9 = new_abs_9.float()
    return np.stack(pred_poses)


def relativize(poses: np.ndarray) -> np.ndarray:
    return np.einsum("ij,njk->nik", np.linalg.inv(poses[0]), poses)


def per_frame_deltas(abs_poses: np.ndarray) -> np.ndarray:
    """(T, 4, 4) absolute → (T-1, 4, 4) relative motion in PREVIOUS-frame's frame.
    delta[i] = abs[i]^-1 @ abs[i+1]   (camera-frame motion from i to i+1)
    """
    T = abs_poses.shape[0]
    deltas = np.zeros((T - 1, 4, 4))
    for i in range(T - 1):
        deltas[i] = np.linalg.inv(abs_poses[i]) @ abs_poses[i + 1]
    return deltas


def integrate_translations_in_gt_rotations(pred_deltas: np.ndarray,
                                            gt_abs: np.ndarray) -> np.ndarray:
    """For each frame transition, use the PREDICTED translation delta but rotate
    it by the GT rotation at that frame. This isolates rotation drift from
    translation drift.

    Returns: (T, 3) integrated positions.
    """
    T = gt_abs.shape[0]
    positions = np.zeros((T, 3))
    for i in range(T - 1):
        # GT rotation at frame i (world-from-camera).
        R_gt = gt_abs[i, :3, :3]
        # Predicted translation in camera frame.
        dt_cam = pred_deltas[i, :3, 3]
        # Translate the position using GT rotation to project camera-frame delta to world.
        positions[i + 1] = positions[i] + R_gt @ dt_cam
    return positions


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model, cfg = load_model(args.ckpt, args.weights_root)
    img_size = {"vjepa": 384, "dinov2": 518, "dinov3": 512}[cfg["encoder"]]
    print(f"[d-decomp] ckpt: {args.ckpt}")
    print(f"[d-decomp] seqs: {args.seqs}")

    results = {}
    for seq in args.seqs:
        seq_dir = args.data_root / seq
        if not seq_dir.exists():
            continue
        fx, fy, cx, cy = intrinsics_for(seq)
        sx, sy = img_size / 640.0, img_size / 480.0
        K = torch.tensor([[[fx * sx, 0., cx * sx], [0., fy * sy, cy * sy], [0., 0., 1.]]],
                         device="cuda")
        fov = torch.tensor([[1.0, 1.0]], device="cuda")
        recs = sync_sequence(seq_dir)[:args.n_frames]
        pred_abs = stream_sequence(model, recs, K, fov)
        gt_abs_raw = np.stack([r.pose_w_c for r in recs])
        gt_abs = relativize(gt_abs_raw)

        pred_deltas = per_frame_deltas(pred_abs)                                  # (T-1, 4, 4)
        gt_deltas = per_frame_deltas(gt_abs)
        pred_dt = pred_deltas[:, :3, 3]                                            # (T-1, 3)
        gt_dt = gt_deltas[:, :3, 3]

        # --- Test 1: Rotation-vs-translation isolation ---
        # Integrate pred translations in GT rotations.
        pred_t_in_gt_rot = integrate_translations_in_gt_rotations(pred_deltas, gt_abs)
        gt_t = gt_abs[:, :3, 3]
        pred_t = pred_abs[:, :3, 3]
        gt_bbox = float(np.linalg.norm(gt_t.max(0) - gt_t.min(0)))
        pred_bbox = float(np.linalg.norm(pred_t.max(0) - pred_t.min(0)))
        pred_t_in_gt_rot_bbox = float(np.linalg.norm(pred_t_in_gt_rot.max(0) - pred_t_in_gt_rot.min(0)))

        # --- Test 2: Mean residual per axis ---
        residual = pred_dt - gt_dt                                                 # (T-1, 3)
        mean_residual = residual.mean(axis=0)                                      # (3,)
        std_residual = residual.std(axis=0)

        # --- Test 3: Scale regression ---
        pred_mag = np.linalg.norm(pred_dt, axis=-1)
        gt_mag = np.linalg.norm(gt_dt, axis=-1)
        # OLS through origin: slope = sum(pred*gt) / sum(gt^2).
        slope_mag = float(np.sum(pred_mag * gt_mag) / max(np.sum(gt_mag ** 2), 1e-9))
        # Also OLS with intercept (sklearn-style).
        coeffs = np.polyfit(gt_mag, pred_mag, 1)                                   # [slope, intercept]
        slope_with_intercept, intercept_mag = float(coeffs[0]), float(coeffs[1])

        # --- Test 4: Per-axis correlation ---
        corr_per_axis = []
        for ax in range(3):
            p, g = pred_dt[:, ax], gt_dt[:, ax]
            if g.std() > 1e-9 and p.std() > 1e-9:
                corr = float(np.corrcoef(p, g)[0, 1])
            else:
                corr = float("nan")
            corr_per_axis.append(corr)

        results[seq] = {
            "gt_bbox_m": gt_bbox,
            "pred_bbox_m": pred_bbox,
            "pred_t_in_gt_rot_bbox_m": pred_t_in_gt_rot_bbox,
            "rotation_isolation": {
                "bbox_pred_full": pred_bbox,
                "bbox_pred_t_in_gt_rot": pred_t_in_gt_rot_bbox,
                "bbox_gt": gt_bbox,
                "ratio_full_over_gt": pred_bbox / max(gt_bbox, 1e-9),
                "ratio_t_in_gt_rot_over_gt": pred_t_in_gt_rot_bbox / max(gt_bbox, 1e-9),
            },
            "mean_residual_per_axis_m": mean_residual.tolist(),
            "std_residual_per_axis_m": std_residual.tolist(),
            "residual_l2_mean_m": float(np.linalg.norm(residual, axis=-1).mean()),
            "scale_slope_through_origin": slope_mag,
            "scale_slope_with_intercept": slope_with_intercept,
            "scale_intercept_m": intercept_mag,
            "per_axis_correlation": corr_per_axis,
            "pred_dt_mean_mag_m": float(pred_mag.mean()),
            "gt_dt_mean_mag_m": float(gt_mag.mean()),
            "pred_t": pred_t.tolist(),
            "gt_t": gt_t.tolist(),
            "pred_t_in_gt_rot": pred_t_in_gt_rot.tolist(),
            "pred_dt": pred_dt.tolist(),
            "gt_dt": gt_dt.tolist(),
        }

        print(f"\n[d-decomp] === {seq} ===")
        print(f"[d-decomp]   GT bbox:                       {gt_bbox:.3f} m")
        print(f"[d-decomp]   Pred bbox (full):              {pred_bbox:.3f} m  ({pred_bbox/gt_bbox:.2f}×)")
        print(f"[d-decomp]   Pred-t in GT-rot bbox:         {pred_t_in_gt_rot_bbox:.3f} m  ({pred_t_in_gt_rot_bbox/gt_bbox:.2f}×)")
        print(f"[d-decomp]   Mean per-axis residual (m):    x={mean_residual[0]:+.5f}  y={mean_residual[1]:+.5f}  z={mean_residual[2]:+.5f}")
        print(f"[d-decomp]   Std per-axis residual (m):     x={std_residual[0]:.5f}  y={std_residual[1]:.5f}  z={std_residual[2]:.5f}")
        print(f"[d-decomp]   GT |Δt| mean:                  {gt_mag.mean():.5f} m")
        print(f"[d-decomp]   Pred |Δt| mean:                {pred_mag.mean():.5f} m  ({pred_mag.mean()/gt_mag.mean():.2f}× GT magnitude)")
        print(f"[d-decomp]   Scale slope (through 0):       {slope_mag:.3f}")
        print(f"[d-decomp]   Scale slope (with intercept):  {slope_with_intercept:.3f}  (intercept {intercept_mag:.5f} m)")
        print(f"[d-decomp]   Per-axis correlation pred~gt:  x={corr_per_axis[0]:+.3f}  y={corr_per_axis[1]:+.3f}  z={corr_per_axis[2]:+.3f}")

    # ===== Visualizations =====

    # 1. Rotation-isolation: 3 trajectories per seq.
    n_seq = len(results)
    fig, axes = plt.subplots(1, n_seq, figsize=(5 * n_seq, 5))
    if n_seq == 1: axes = [axes]
    for ax, (seq, r) in zip(axes, results.items()):
        gt_t = np.array(r["gt_t"])
        pred_t = np.array(r["pred_t"])
        pred_gt_rot = np.array(r["pred_t_in_gt_rot"])
        ax.plot(gt_t[:, 0], gt_t[:, 2], "o-", color="green", markersize=2, label="GT")
        ax.plot(pred_t[:, 0], pred_t[:, 2], "o-", color="red", markersize=2,
                label=f"pred full ({r['rotation_isolation']['ratio_full_over_gt']:.1f}× GT)")
        ax.plot(pred_gt_rot[:, 0], pred_gt_rot[:, 2], "o-", color="blue", markersize=2,
                label=f"pred-t in GT-rot ({r['rotation_isolation']['ratio_t_in_gt_rot_over_gt']:.1f}× GT)")
        ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
        ax.set_title(seq.replace("rgbd_dataset_", ""))
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_aspect("equal")
    fig.suptitle("Rotation-vs-translation isolation\n"
                  "If blue ≈ green → rotation drift dominates; pred translations OK in cam frame.\n"
                  "If blue still huge → translations themselves systematic.",
                  fontsize=10)
    plt.tight_layout()
    plt.savefig(args.out_dir / "rotation_isolation.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 2. Scale regression scatter: |pred_delta| vs |gt_delta|.
    fig, axes = plt.subplots(1, n_seq, figsize=(5 * n_seq, 5))
    if n_seq == 1: axes = [axes]
    for ax, (seq, r) in zip(axes, results.items()):
        pred_dt = np.array(r["pred_dt"])
        gt_dt = np.array(r["gt_dt"])
        pm = np.linalg.norm(pred_dt, axis=-1)
        gm = np.linalg.norm(gt_dt, axis=-1)
        ax.scatter(gm, pm, s=4, alpha=0.4, color="tab:blue")
        # y=x reference
        m = max(gm.max(), pm.max())
        ax.plot([0, m], [0, m], "k--", linewidth=1, alpha=0.5, label="y=x")
        # Fit line
        slope = r["scale_slope_through_origin"]
        ax.plot([0, m], [0, slope * m], "r-", linewidth=2,
                label=f"slope={slope:.3f}")
        ax.set_xlabel("|GT Δt| (m/frame)"); ax.set_ylabel("|pred Δt| (m/frame)")
        ax.set_title(f"{seq.replace('rgbd_dataset_', '')}  "
                      f"scale={slope:.2f}×  "
                      f"corr_xyz=({r['per_axis_correlation'][0]:.2f},{r['per_axis_correlation'][1]:.2f},{r['per_axis_correlation'][2]:.2f})")
        ax.set_xlim(0, m); ax.set_ylim(0, m); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("Scale regression: |pred Δt| vs |gt Δt|\n"
                  "slope=1: magnitude-matched.  slope>1: over-strides.  slope<1: under-strides.",
                  fontsize=10)
    plt.tight_layout()
    plt.savefig(args.out_dir / "scale_regression.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 3. Per-axis residual histograms.
    fig, axes = plt.subplots(n_seq, 3, figsize=(15, 4 * n_seq))
    if n_seq == 1: axes = axes[None, :]
    axis_names = ["x", "y", "z"]
    for row, (seq, r) in enumerate(results.items()):
        pred_dt = np.array(r["pred_dt"])
        gt_dt = np.array(r["gt_dt"])
        residual = pred_dt - gt_dt
        for col, ax in enumerate(axes[row]):
            res = residual[:, col]
            mean = res.mean()
            ax.hist(res, bins=50, color="tab:blue", alpha=0.7)
            ax.axvline(0, color="black", linestyle="--", alpha=0.6, label="0")
            ax.axvline(mean, color="red", linestyle="-", linewidth=2, label=f"mean={mean:+.4f}")
            ax.set_title(f"{seq.replace('rgbd_dataset_', '')}  axis {axis_names[col]}  "
                          f"(corr_pred_gt={r['per_axis_correlation'][col]:+.2f})")
            ax.set_xlabel("pred_Δ - gt_Δ (m)"); ax.set_ylabel("count")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Per-axis residual distribution (pred_delta - gt_delta).\n"
                  "Mean ≠ 0 → directional bias on that axis.",
                  fontsize=10)
    plt.tight_layout()
    plt.savefig(args.out_dir / "residual_histograms.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ===== Summary JSON (strip arrays) =====
    summary = {
        "ckpt": str(args.ckpt),
        "results": {
            seq: {k: v for k, v in r.items() if k not in ("pred_t", "gt_t", "pred_t_in_gt_rot", "pred_dt", "gt_dt")}
            for seq, r in results.items()
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ===== Verdict per seq =====
    print(f"\n[d-decomp] === VERDICTS ===")
    for seq, r in results.items():
        rot_iso = r["rotation_isolation"]
        bias_xyz = r["mean_residual_per_axis_m"]
        scale = r["scale_slope_through_origin"]
        corrs = r["per_axis_correlation"]
        # Rotation diagnosis
        if rot_iso["ratio_t_in_gt_rot_over_gt"] < 1.5 and rot_iso["ratio_full_over_gt"] > 2.0:
            rot_verdict = "ROTATION-DOMINANT  (fixing rotation alone would mostly fix trajectory)"
        elif rot_iso["ratio_t_in_gt_rot_over_gt"] > 2.0:
            rot_verdict = "TRANSLATION-DOMINANT  (translation deltas themselves systematic)"
        else:
            rot_verdict = "MIXED"
        # Scale diagnosis
        if scale > 1.3:
            scale_verdict = f"SCALE-BIAS  pred is {scale:.1f}× larger than GT per frame"
        elif scale < 0.7:
            scale_verdict = f"UNDER-STRIDE  pred is {scale:.1f}× smaller than GT per frame"
        else:
            scale_verdict = f"SCALE-NEUTRAL  (slope {scale:.2f})"
        # Direction diagnosis
        bias_mag = float(np.linalg.norm(bias_xyz))
        gt_mag_mean = r["gt_dt_mean_mag_m"]
        if bias_mag > 0.3 * gt_mag_mean:
            dir_verdict = f"DIRECTIONAL-BIAS  (mean residual {bias_mag:.4f} m vs GT |dt| mean {gt_mag_mean:.4f} m)"
        else:
            dir_verdict = "DIRECTION-NEUTRAL"
        # Correlation diagnosis
        worst_corr = min(corrs)
        if worst_corr < 0.2:
            corr_verdict = f"WEAK-TRACKING  (worst axis corr {worst_corr:+.2f})"
        elif worst_corr > 0.7:
            corr_verdict = f"STRONG-TRACKING  (worst axis corr {worst_corr:+.2f})"
        else:
            corr_verdict = f"PARTIAL-TRACKING  (worst axis corr {worst_corr:+.2f})"
        print(f"  {seq}:")
        print(f"    {rot_verdict}")
        print(f"    {scale_verdict}")
        print(f"    {dir_verdict}")
        print(f"    {corr_verdict}")

    print(f"\n[d-decomp] saved {args.out_dir}/")


if __name__ == "__main__":
    main()
