"""Test 2: per-frame translation error vs frame index.

Diagnostic question: does the per-frame predictor get worse as the streaming
sequence progresses (state degrading) or stay constant-quality (state fine,
drift is pure accumulation of constant-magnitude noise)?

Measures per-frame RPE δ=1 (||rel_pred_translation(t, t+1) - rel_gt_translation(t, t+1)||)
at every frame, then fits a linear regression of that quantity vs frame index.

Reads:
  - regression slope ~ 0 (within CI) → state-quality is stable over the stream;
    drift is pure accumulation of per-frame noise (no degradation).
  - slope significantly > 0  → per-frame predictor degrades over time; state is
    losing fidelity. Complicates the drift story — re-grounding may need to
    refresh state quality, not just add an anchor.

Also produces the Sim3-aligned per-frame residual as a secondary plot — that
one is expected to grow even with constant per-frame noise (random walk),
so it's not the main diagnostic but is useful context.
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

from vggt_mamba.data.tum_rgbd import sync_sequence, _quat_to_rot   # noqa: E402
from vggt_mamba.eval.metrics import umeyama_sim3                    # noqa: E402
from vggt_mamba.models.geomamba import build_geomamba               # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--seqs", nargs="+", required=True)
    p.add_argument("--out", type=Path, default=Path("viz/output/per_frame_error.json"))
    p.add_argument("--plot", type=Path, default=Path("viz/output/per_frame_error.png"))
    p.add_argument("--smooth-window", type=int, default=25,
                   help="rolling-mean window for plot readability")
    return p.parse_args()


def load_model(ckpt_path, weights_root):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_geomamba(
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
    pred, gt = [], []
    t0 = time.perf_counter()
    for i, r in enumerate(recs):
        rgb = load_rgb(r, img_size).cuda(non_blocking=True)
        out, state = model.streaming_forward(rgb, state, frame_idx=i)
        pred.append(cam9_to_pose(out["camera"][0, 0].float().cpu().numpy()))
        gt.append(r.pose_w_c)
    return np.stack(pred), np.stack(gt), time.perf_counter() - t0


def per_frame_delta_error(pred_poses, gt_poses):
    """For each consecutive frame pair (t, t+1), return ||rel_pred_trans - rel_gt_trans||.
    This is RPE δ=1 per-pair, not aggregated to RMSE."""
    n = len(pred_poses)
    err = np.zeros(n - 1)
    for i in range(n - 1):
        rel_pred = np.linalg.inv(pred_poses[i]) @ pred_poses[i + 1]
        rel_gt = np.linalg.inv(gt_poses[i]) @ gt_poses[i + 1]
        diff = np.linalg.inv(rel_gt) @ rel_pred
        err[i] = np.linalg.norm(diff[:3, 3])
    return err


def sim3_aligned_per_frame_residual(pred_poses, gt_poses):
    """Sim3-align the whole predicted trajectory to GT, then per-frame ||aligned - gt||."""
    p = pred_poses[..., :3, 3]
    g = gt_poses[..., :3, 3]
    s, R, t = umeyama_sim3(p, g)
    aligned = (s * (R @ p.T)).T + t
    return np.linalg.norm(aligned - g, axis=1)


def fit_linear_slope(y):
    """OLS slope of y vs index. Returns (slope_per_frame, intercept, p_value_approx)."""
    n = len(y)
    x = np.arange(n, dtype=float)
    if n < 3:
        return 0.0, float(y[0] if n else 0.0), 1.0
    slope, intercept = np.polyfit(x, y, 1)
    # Crude p-value via residual variance + slope SE.
    pred = slope * x + intercept
    residuals = y - pred
    sigma2 = (residuals ** 2).sum() / max(n - 2, 1)
    sx2 = ((x - x.mean()) ** 2).sum()
    se_slope = np.sqrt(sigma2 / max(sx2, 1e-12))
    t_stat = slope / max(se_slope, 1e-12)
    # Approximate two-sided p-value using normal CDF (n is large enough).
    from math import erf, sqrt
    p_value = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2))))
    return float(slope), float(intercept), float(p_value)


def rolling_mean(a, w):
    if w <= 1:
        return a
    pad = w // 2
    a_pad = np.concatenate([np.full(pad, a[0]), a, np.full(pad, a[-1])])
    kernel = np.ones(w) / w
    return np.convolve(a_pad, kernel, mode="valid")[:len(a)]


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model, cfg = load_model(args.ckpt, args.weights_root)

    print(f"[per-frame] ckpt={args.ckpt.name}")
    print(f"[per-frame] using Speed-B graphs")

    per_seq = {}
    for seq in args.seqs:
        print(f"\n[per-frame] {seq}")
        pred, gt, dt = stream_trajectory(model, cfg, args.data_root, seq)
        n = len(pred)
        print(f"  streamed {n} frames in {dt:.1f}s ({n/dt:.1f} FPS)")

        per_frame_err = per_frame_delta_error(pred, gt)
        sim3_resid = sim3_aligned_per_frame_residual(pred, gt)

        slope_dt, intercept_dt, p_dt = fit_linear_slope(per_frame_err)
        slope_sim3, intercept_sim3, p_sim3 = fit_linear_slope(sim3_resid)

        # Compare first 25% vs last 25% of per-frame error (robust to noise).
        q = n // 4
        first_q_mean = float(per_frame_err[:q].mean())
        last_q_mean = float(per_frame_err[-q:].mean())

        per_seq[seq] = {
            "n_frames": n,
            "per_frame_delta_err_slope_m_per_frame": slope_dt,
            "per_frame_delta_err_intercept_m": intercept_dt,
            "per_frame_delta_err_p_value": p_dt,
            "per_frame_delta_err_first_quartile_mean": first_q_mean,
            "per_frame_delta_err_last_quartile_mean": last_q_mean,
            "per_frame_delta_err_grand_mean": float(per_frame_err.mean()),
            "sim3_residual_slope_m_per_frame": slope_sim3,
            "sim3_residual_intercept_m": intercept_sim3,
            "sim3_residual_p_value": p_sim3,
            "_per_frame_err": per_frame_err.tolist(),  # for plot
            "_sim3_resid": sim3_resid.tolist(),
        }
        print(f"  per-frame δ=1 trans error:")
        print(f"    grand mean:        {per_frame_err.mean():.4f} m")
        print(f"    first 25% mean:    {first_q_mean:.4f} m")
        print(f"    last  25% mean:    {last_q_mean:.4f} m")
        print(f"    last/first ratio:  {last_q_mean / max(first_q_mean, 1e-9):.2f}x")
        print(f"    OLS slope:         {slope_dt:+.3e} m/frame  (p={p_dt:.3f})")
        # Projected change over the full sequence.
        proj = slope_dt * n
        print(f"    projected change over {n} frames: {proj:+.3f} m")
        print(f"  Sim3-aligned residual:")
        print(f"    grand mean:        {sim3_resid.mean():.4f} m")
        print(f"    OLS slope:         {slope_sim3:+.3e} m/frame")

    # Aggregate verdict on the PER-FRAME δ=1 slope (the diagnostic we care about).
    slopes = [v["per_frame_delta_err_slope_m_per_frame"] for v in per_seq.values()]
    p_values = [v["per_frame_delta_err_p_value"] for v in per_seq.values()]
    last_first_ratios = [
        v["per_frame_delta_err_last_quartile_mean"]
        / max(v["per_frame_delta_err_first_quartile_mean"], 1e-9)
        for v in per_seq.values()
    ]
    agg = {
        "mean_slope_m_per_frame": float(np.mean(slopes)),
        "min_p_value": float(np.min(p_values)),
        "mean_last_first_ratio": float(np.mean(last_first_ratios)),
        "n_sequences": len(per_seq),
    }
    print(f"\n[per-frame] aggregate per-frame δ=1 error slope: "
          f"{agg['mean_slope_m_per_frame']:+.3e} m/frame")
    print(f"[per-frame] mean (last-quartile / first-quartile) ratio: "
          f"{agg['mean_last_first_ratio']:.2f}x  (1.0 = stable, >1 = degrading)")
    print(f"[per-frame] min p-value across seqs: {agg['min_p_value']:.3f}")

    # Verdict.
    if agg["mean_last_first_ratio"] < 1.10 and agg["min_p_value"] > 0.05:
        verdict = "STATE_STABLE (per-frame error flat across stream; drift is pure accumulation)"
    elif agg["mean_last_first_ratio"] > 1.30 and agg["min_p_value"] < 0.05:
        verdict = "STATE_DEGRADING (per-frame error grows significantly over stream)"
    else:
        verdict = "AMBIGUOUS (small/inconsistent trend — re-examine plot)"
    print(f"[per-frame] verdict: {verdict}")

    # Save JSON (without the heavy arrays for readability).
    payload = {
        "ckpt": str(args.ckpt),
        "per_sequence_summary": {
            seq: {k: v for k, v in d.items() if not k.startswith("_")}
            for seq, d in per_seq.items()
        },
        "aggregate": agg,
        "verdict": verdict,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"[per-frame] saved JSON -> {args.out}")

    # Plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n_seqs = len(per_seq)
        fig, axes = plt.subplots(2, n_seqs, figsize=(5 * n_seqs, 8), squeeze=False)
        for col, (seq, d) in enumerate(per_seq.items()):
            short = seq.replace("rgbd_dataset_freiburg3_", "")
            per_frame = np.asarray(d["_per_frame_err"])
            sim3 = np.asarray(d["_sim3_resid"])
            n = len(per_frame)
            x = np.arange(n)
            smoothed = rolling_mean(per_frame, args.smooth_window)
            ax = axes[0, col]
            x_pf = np.arange(len(per_frame))   # per-frame error has n-1 points
            ax.plot(x_pf, per_frame, color="lightgray", linewidth=0.5, label="raw")
            ax.plot(x_pf, smoothed, color="tab:blue", linewidth=1.5, label=f"rolling mean (w={args.smooth_window})")
            slope = d["per_frame_delta_err_slope_m_per_frame"]
            intercept = d["per_frame_delta_err_intercept_m"]
            ax.plot(x_pf, slope * x_pf + intercept, "r--", linewidth=1.5,
                    label=f"OLS: {slope:+.2e} m/frame (p={d['per_frame_delta_err_p_value']:.3f})")
            ax.set_title(f"{short}\nper-frame δ=1 trans error")
            ax.set_xlabel("frame"); ax.set_ylabel("trans error (m)")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)

            ax2 = axes[1, col]
            sm2 = rolling_mean(sim3, args.smooth_window)
            ax2.plot(x[:len(sim3)], sim3, color="lightgray", linewidth=0.5)
            ax2.plot(x[:len(sim3)], sm2, color="tab:green", linewidth=1.5)
            ax2.set_title(f"{short}\nSim3-aligned residual (secondary)")
            ax2.set_xlabel("frame"); ax2.set_ylabel("aligned residual (m)")
            ax2.grid(alpha=0.3)
        fig.suptitle(f"Test 2 — per-frame error vs frame · {verdict}", fontsize=12)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[per-frame] saved plot -> {args.plot}")
    except Exception as e:
        print(f"[per-frame] plot failed: {e}")


if __name__ == "__main__":
    main()
