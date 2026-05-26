"""Test 1: RPE growth pattern O(√δ) vs O(δ).

Diagnostic for whether the per-frame translation error in the streaming
camera head is unbiased random noise (random-walk drift, RPE_trans(δ) ~ √δ)
or systematic bias (RPE_trans(δ) ~ δ). The slope of log(RPE_trans) vs log(δ)
distinguishes them. A delta-prediction head can plausibly fix systematic bias
(O(δ)) but does not address random-walk drift (O(√δ)), which requires a
re-grounding mechanism (scene anchors / loop closure).

Runs streaming inference once per sequence (under Speed-B graphs) to get
the full trajectory, then computes RPE at many δ values from the cached
trajectory — no additional compute per δ.
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

from vggt_mamba.data.tum_rgbd import sync_sequence, _quat_to_rot, intrinsics_for  # noqa: E402
from vggt_mamba.eval.metrics import relative_pose_error             # noqa: E402
from vggt_mamba.models.terrawm import build_terrawm               # noqa: E402


DELTAS = [1, 2, 5, 10, 20, 50, 100, 200, 500]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--seqs", nargs="+", required=True)
    p.add_argument("--out", type=Path, default=Path("viz/output/rpe_growth.json"))
    p.add_argument("--plot", type=Path, default=Path("viz/output/rpe_growth.png"))
    return p.parse_args()


def load_model(ckpt_path, weights_root):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_terrawm(
        cfg["encoder"], str(weights_root),
        n_intraframe_layers=cfg["model"]["n_intraframe_layers"],
        n_summary_tokens=cfg["model"]["n_summary_tokens"],
        n_summary_dynamic=cfg["model"].get("n_summary_dynamic"),
        n_xfm_layers=cfg["model"]["n_xfm_layers"],
        d_state=cfg["model"]["d_state"],
        bidirectional=False,
        aggregator_name="mamba",
        track_enabled=cfg["model"]["track_enabled"],
        max_frames=ckpt["model"]["frame_embed"].shape[1],
        dense_residual_to_patches=cfg["model"].get("dense_residual_to_patches", True),
        predict_next_latent=cfg["model"].get("predict_next_latent", False),
        ema_momentum=cfg["model"].get("ema_momentum", 0.99),
        cross_frame_target=cfg["model"].get("cross_frame_target", "summary"),
        use_anchor_pool=cfg["model"].get("use_anchor_pool", False),
        n_anchors=cfg["model"].get("n_anchors", 32),
        n_anchor_writes=cfg["model"].get("n_anchor_writes", 4),
        anchor_match_threshold=cfg["model"].get("anchor_match_threshold", 0.5),
        delta_pose=cfg["model"].get("delta_pose", cfg["model"].get("terrawm", False)),
        motion_enc_freqs=cfg["model"].get("motion_enc_freqs", cfg["model"].get("terrawm_motion_freqs", 64)),
    )
    model.load_state_dict(ckpt["model"], strict=False)
    return model.cuda().eval(), cfg


def load_rgb(rec, img_size):
    from PIL import Image
    img = Image.open(rec.rgb_path).convert("RGB").resize((img_size, img_size))
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0
                            ).permute(2, 0, 1).unsqueeze(0).contiguous()


def cam9_to_pose(c):
    t = c[:3]
    q = c[3:7]; q = q / max(np.linalg.norm(q), 1e-12)
    R = _quat_to_rot(q[0], q[1], q[2], q[3])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def stream_trajectory(model, cfg, data_root, seq):
    recs = sync_sequence(data_root / seq)
    img_size = {"vjepa": 384, "dinov2": 518, "dinov3": 512}[cfg["encoder"]]
    state = model.init_streaming_state(use_cuda_graphs=True)
    anchor_state = None
    K_intrinsics = None
    if getattr(model, "use_anchor_pool", False):
        anchor_state = model.init_anchor_state(batch_size=1, dtype=torch.bfloat16, device="cuda")
        fx, fy, cx, cy = intrinsics_for(seq)
        sx, sy = img_size / 640.0, img_size / 480.0
        K_intrinsics = torch.tensor(
            [[[fx * sx, 0., cx * sx], [0., fy * sy, cy * sy], [0., 0., 1.]]],
            device="cuda",
        )
    terrawm = getattr(model, "delta_pose", False)
    pred, gt = [], []
    raw_cam9 = []
    t0 = time.perf_counter()
    for i, r in enumerate(recs):
        rgb = load_rgb(r, img_size).cuda(non_blocking=True)
        out, state = model.streaming_forward(
            rgb, state, frame_idx=i,
            anchor_state=anchor_state, K_intrinsics=K_intrinsics,
        )
        cam9_v = out["camera"][0, 0].float().cpu().numpy()
        if terrawm:
            raw_cam9.append(cam9_v)
        else:
            pred.append(cam9_to_pose(cam9_v))
        gt.append(r.pose_w_c)
    dt = time.perf_counter() - t0
    if terrawm:
        from vggt_mamba.models.pose_utils import integrate_deltas_to_absolute
        deltas_t = torch.tensor(np.stack(raw_cam9), dtype=torch.float32)
        initial = torch.tensor(recs[0].pose_w_c, dtype=torch.float32)
        abs_T = integrate_deltas_to_absolute(deltas_t[:, :7], initial)
        pred = abs_T.numpy()
    else:
        pred = np.stack(pred)
    return pred, np.stack(gt), dt, len(recs)


def compute_rpe_curve(pred, gt, deltas):
    """Return list of (delta, rpe_trans_rmse_m, rpe_rot_rmse_deg, n_pairs)."""
    out = []
    for d in deltas:
        if d >= len(pred):
            continue
        r = relative_pose_error(pred, gt, delta=d)
        out.append((d, r["rpe_trans_rmse_m"], r["rpe_rot_rmse_deg"], r["n_pairs"]))
    return out


def fit_slope(deltas, vals):
    """Log-log linear fit. Returns (slope, intercept) such that vals ≈ exp(intercept) * delta^slope."""
    ld = np.log(np.asarray(deltas, dtype=float))
    lv = np.log(np.asarray(vals, dtype=float))
    slope, intercept = np.polyfit(ld, lv, 1)
    return float(slope), float(intercept)


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model, cfg = load_model(args.ckpt, args.weights_root)

    print(f"[rpe-growth] ckpt={args.ckpt.name}")
    print(f"[rpe-growth] using Speed-B graphs for streaming")

    per_seq = {}
    for seq in args.seqs:
        print(f"\n[rpe-growth] {seq}")
        pred, gt, dt, n = stream_trajectory(model, cfg, args.data_root, seq)
        print(f"  streamed {n} frames in {dt:.1f}s ({n/dt:.1f} FPS)")
        curve = compute_rpe_curve(pred, gt, DELTAS)
        ds = [c[0] for c in curve]
        ts = [c[1] for c in curve]
        rs = [c[2] for c in curve]
        ts_slope, ts_intercept = fit_slope(ds, ts)
        rs_slope, rs_intercept = fit_slope(ds, rs)
        per_seq[seq] = {
            "n_frames": n,
            "curve": [{"delta": d, "trans_rmse_m": t, "rot_rmse_deg": r, "n_pairs": np_}
                      for (d, t, r, np_) in curve],
            "trans_logslope": ts_slope,
            "trans_intercept": ts_intercept,
            "rot_logslope": rs_slope,
            "rot_intercept": rs_intercept,
        }
        print(f"  RPE_trans curve:")
        for (d, t, r, n_) in curve:
            print(f"    δ={d:4d}  trans={t:.4f} m  rot={r:.2f}°  pairs={n_}")
        print(f"  trans log-log slope: {ts_slope:+.3f}  "
              f"(0.5 = random-walk drift, 1.0 = systematic bias)")
        print(f"  rot   log-log slope: {rs_slope:+.3f}")

    # Aggregate slope: average per-sequence slope (each sequence is one observation).
    seq_trans_slopes = [v["trans_logslope"] for v in per_seq.values()]
    seq_rot_slopes = [v["rot_logslope"] for v in per_seq.values()]
    agg = {
        "trans_logslope_mean": float(np.mean(seq_trans_slopes)),
        "trans_logslope_std": float(np.std(seq_trans_slopes)),
        "rot_logslope_mean": float(np.mean(seq_rot_slopes)),
        "rot_logslope_std": float(np.std(seq_rot_slopes)),
        "n_sequences": len(per_seq),
    }
    print(f"\n[rpe-growth] aggregate trans log-slope: {agg['trans_logslope_mean']:+.3f} "
          f"± {agg['trans_logslope_std']:.3f} (over {agg['n_sequences']} seqs)")
    print(f"[rpe-growth] aggregate rot   log-slope: {agg['rot_logslope_mean']:+.3f} "
          f"± {agg['rot_logslope_std']:.3f}")

    # Interpret.
    s = agg["trans_logslope_mean"]
    if abs(s - 0.5) < 0.15:
        verdict = "RANDOM_WALK_DRIFT (≈√δ)"
    elif abs(s - 1.0) < 0.15:
        verdict = "SYSTEMATIC_BIAS (≈δ)"
    elif 0.5 < s < 1.0:
        verdict = "MIXED (between √δ and δ)"
    elif s < 0.35:
        verdict = "SUB_RANDOM_WALK (negative correlation across frames)"
    else:
        verdict = f"ANOMALOUS (slope {s:.2f})"
    print(f"[rpe-growth] translation drift verdict: {verdict}")

    args.out.write_text(json.dumps({
        "ckpt": str(args.ckpt), "per_sequence": per_seq,
        "aggregate": agg, "verdict": verdict,
    }, indent=2))
    print(f"[rpe-growth] saved JSON -> {args.out}")

    # Plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for seq, v in per_seq.items():
            ds = [c["delta"] for c in v["curve"]]
            ts = [c["trans_rmse_m"] for c in v["curve"]]
            rs = [c["rot_rmse_deg"] for c in v["curve"]]
            short = seq.replace("rgbd_dataset_freiburg3_", "")
            axes[0].loglog(ds, ts, "o-", label=f"{short} (slope {v['trans_logslope']:+.2f})")
            axes[1].loglog(ds, rs, "o-", label=f"{short} (slope {v['rot_logslope']:+.2f})")
        # Reference lines.
        if per_seq:
            first = next(iter(per_seq.values()))
            ds = [c["delta"] for c in first["curve"]]
            ts0 = first["curve"][0]["trans_rmse_m"]
            sqrt_ref = [ts0 * (d / ds[0]) ** 0.5 for d in ds]
            lin_ref = [ts0 * (d / ds[0]) for d in ds]
            axes[0].loglog(ds, sqrt_ref, "k--", alpha=0.5, label="∝√δ (random walk)")
            axes[0].loglog(ds, lin_ref, "k:", alpha=0.5, label="∝δ (systematic)")
        axes[0].set_xlabel("δ (frame gap)"); axes[0].set_ylabel("RPE trans RMSE (m)")
        axes[0].set_title(f"Translation RPE growth · agg slope {agg['trans_logslope_mean']:+.2f} → {verdict}")
        axes[0].legend(fontsize=8); axes[0].grid(True, which="both", alpha=0.3)
        axes[1].set_xlabel("δ (frame gap)"); axes[1].set_ylabel("RPE rot RMSE (°)")
        axes[1].set_title(f"Rotation RPE growth · agg slope {agg['rot_logslope_mean']:+.2f}")
        axes[1].legend(fontsize=8); axes[1].grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[rpe-growth] saved plot -> {args.plot}")
    except Exception as e:
        print(f"[rpe-growth] plot failed: {e}")


if __name__ == "__main__":
    main()
