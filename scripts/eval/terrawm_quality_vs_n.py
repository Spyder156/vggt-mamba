"""TerraWM quality-vs-N — the existential experiment.

Loads both trained TerraWM checkpoints (Mamba aggregator + Attention
aggregator), runs each on every EVAL sequence × multiple start frames ×
N ∈ {4, 8, 16, 32, 64, 128, 256}. Aggregates per (model, N).

The headline plot the paper lives or dies on:
  Abs-Rel vs N for both aggregators, with N=training visibly marked.
  - If Mamba's curve stays flat past training N while attention's climbs,
    the architectural claim is alive.
  - If both stay flat or both climb together, the claim is dead.

Outputs:
  viz/output/phase3_qvn_eval/results.json
  viz/output/phase3_qvn_eval/quality_vs_n.png
  viz/output/phase3_qvn_eval/per_frame_at_largeN.png
  viz/output/phase3_qvn_eval/table.md
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt_mamba.data.tum_rgbd import EVAL_SEQS, intrinsics_for, sync_sequence  # noqa: E402
from vggt_mamba.eval.metrics import multi_view_consistency                      # noqa: E402
from vggt_mamba.models.terrawm import build_terrawm                            # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--ckpt-mamba", type=Path,
                   default=Path("experiments/phase3_qvn/dinov3_mamba/ckpt_002000.pt"))
    p.add_argument("--ckpt-attention", type=Path,
                   default=Path("experiments/phase3_qvn/dinov3_attention/ckpt_002000.pt"))
    p.add_argument("--n-list", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 256])
    p.add_argument("--frame-stride", type=int, default=10)
    p.add_argument("--starts-per-seq", type=int, default=4)
    p.add_argument("--start-gap", type=int, default=150)
    p.add_argument("--training-n", type=int, default=4,
                   help="N used at training time, marked on the plot")
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).resolve().parents[2] / "viz/output/phase3_qvn_eval")
    return p.parse_args()


def extend_frame_embed(model, t: int) -> None:
    cur = model.frame_embed.shape[1]
    if t <= cur:
        return
    extra = t - cur
    pad = torch.zeros(
        1, extra, 1, model.frame_embed.shape[-1],
        device=model.frame_embed.device, dtype=model.frame_embed.dtype,
    )
    model.frame_embed = torch.nn.Parameter(torch.cat([model.frame_embed.data, pad], dim=1))


def load_model(ckpt_path: Path, weights_root: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    agg = cfg.get("aggregator", "mamba")
    model = build_terrawm(
        cfg["encoder"], str(weights_root),
        n_intraframe_layers=cfg["model"]["n_intraframe_layers"],
        n_summary_tokens=cfg["model"]["n_summary_tokens"],
        n_xfm_layers=cfg["model"]["n_xfm_layers"],
        d_state=cfg["model"]["d_state"],
        bidirectional=cfg["model"]["bidirectional"],
        aggregator_name=agg,
        track_enabled=cfg["model"]["track_enabled"],
        # Match the ckpt's frame_embed size; extend later for inference at large N.
        max_frames=ckpt["model"]["frame_embed"].shape[1],
    )
    msg = model.load_state_dict(ckpt["model"], strict=False)
    non_enc = [k for k in msg.missing_keys if not k.startswith("encoder.")]
    print(f"[load:{agg}] step={ckpt['step']}  missing(non-encoder)={len(non_enc)}  "
          f"unexpected={len(msg.unexpected_keys)}")
    return model.cuda().eval(), cfg, agg


def build_window(records, start: int, n: int, frame_stride: int, seq_name: str,
                 img_size: int, depth_max_m: float = 8.0) -> dict:
    from PIL import Image
    idxs = [start + k * frame_stride for k in range(n)]
    window = [records[i] for i in idxs]
    rgbs, depths, poses = [], [], []
    for r in window:
        img = Image.open(r.rgb_path).convert("RGB").resize((img_size, img_size))
        rgbs.append(np.asarray(img, dtype=np.float32) / 255.0)
        d = np.asarray(
            Image.open(r.depth_path).resize((img_size, img_size), Image.NEAREST),
            dtype=np.float32,
        ) / 5000.0
        depths.append(d)
        poses.append(r.pose_w_c.astype(np.float32))
    rgb = np.stack(rgbs); depth = np.stack(depths); poses_w_c = np.stack(poses)
    valid = (depth > 0.01) & (depth < depth_max_m)
    fx, fy, cx, cy = intrinsics_for(seq_name)
    sx = img_size / 640.0
    sy = img_size / 480.0
    K = np.array([[fx*sx, 0, cx*sx], [0, fy*sy, cy*sy], [0, 0, 1]], dtype=np.float32)
    return {
        "rgb": torch.from_numpy(rgb).permute(0, 3, 1, 2).unsqueeze(0).contiguous(),
        "depth": torch.from_numpy(depth).unsqueeze(0),
        "valid": torch.from_numpy(valid).unsqueeze(0),
        "poses_w_c": torch.from_numpy(poses_w_c).unsqueeze(0),
        "K": torch.from_numpy(K).unsqueeze(0),
    }


def metrics_one(model, batch) -> dict | None:
    rgb = batch["rgb"].cuda(non_blocking=True)
    valid = batch["valid"].cuda(non_blocking=True)
    poses = batch["poses_w_c"].cuda(non_blocking=True)
    depth_gt = batch["depth"].cuda(non_blocking=True)

    try:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            preds = model(rgb)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_vram = torch.cuda.max_memory_allocated() / 1e6
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        return None
    except Exception:
        traceback.print_exc()
        gc.collect(); torch.cuda.empty_cache()
        return None

    pmap = preds["pointmap"].float()
    pred_d = pmap[0, :, 2].cpu().numpy()
    gt = depth_gt[0].cpu().numpy()
    v = valid[0].cpu().numpy()

    abs_rel_pf, delta_pf = [], []
    for t in range(pred_d.shape[0]):
        m = v[t]
        if m.sum() == 0:
            abs_rel_pf.append(np.nan); delta_pf.append(np.nan); continue
        p = np.maximum(pred_d[t][m], 1e-6)
        g = np.maximum(gt[t][m], 1e-6)
        abs_rel_pf.append(float(np.mean(np.abs(p - g) / g)))
        ratio = np.maximum(p / g, g / p)
        delta_pf.append(float(np.mean(ratio < 1.25)))
    mvc = float(multi_view_consistency(pmap, valid, poses, n_samples=1024))
    return {
        "abs_rel_per_frame": abs_rel_pf,
        "delta_per_frame": delta_pf,
        "abs_rel_mean": float(np.nanmean(abs_rel_pf)),
        "delta_mean": float(np.nanmean(delta_pf)),
        "mvc": mvc,
        "time_ms": elapsed * 1000,
        "peak_vram_mb": peak_vram,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[bench] loading eval sequences...")
    seqs = []
    for name in EVAL_SEQS:
        d = args.data_root / name
        if not d.exists():
            continue
        recs = sync_sequence(d)
        seqs.append((name, recs))
        print(f"  {name}: {len(recs)} frames")

    results: dict[str, list[dict]] = {"mamba": [], "attention": []}
    for ckpt_path, agg_expected in [(args.ckpt_mamba, "mamba"),
                                     (args.ckpt_attention, "attention")]:
        print(f"\n[bench] === {agg_expected} ({ckpt_path}) ===")
        model, cfg, agg = load_model(ckpt_path, args.weights_root)
        img_size = {"vjepa": 384, "dinov2": 518, "dinov3": 512}[cfg["encoder"]]

        for n in args.n_list:
            extend_frame_embed(model, n)
            span = (n - 1) * args.frame_stride + 1
            for seq_name, recs in seqs:
                max_start = len(recs) - span
                if max_start < 0:
                    continue
                starts = list(range(0, max_start + 1, args.start_gap))[:args.starts_per_seq]
                for start in starts:
                    batch = build_window(recs, start, n, args.frame_stride, seq_name, img_size)
                    m = metrics_one(model, batch)
                    if m is None:
                        print(f"  [{agg} N={n:3d} {seq_name[-25:]:>25s} start={start:4d}]  OOM/err  SKIP")
                        continue
                    m.update({"agg": agg, "n": n, "seq": seq_name, "start": start})
                    results[agg].append(m)
                    print(f"  [{agg} N={n:3d} {seq_name[-25:]:>25s} start={start:4d}]  "
                          f"abs_rel={m['abs_rel_mean']:.4f} δ={m['delta_mean']:.3f} "
                          f"mvc={m['mvc']:.4f}  vram={m['peak_vram_mb']:.0f}MB")
        del model
        gc.collect(); torch.cuda.empty_cache()

    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2))

    # Aggregate per (agg, N).
    summary: dict[str, dict[int, dict]] = {"mamba": {}, "attention": {}}
    for agg in ("mamba", "attention"):
        for n in args.n_list:
            rows = [r for r in results[agg] if r["n"] == n]
            if not rows:
                continue
            summary[agg][n] = {
                "n_windows": len(rows),
                "abs_rel_mean": float(np.mean([r["abs_rel_mean"] for r in rows])),
                "abs_rel_std":  float(np.std([r["abs_rel_mean"] for r in rows])),
                "delta_mean":   float(np.mean([r["delta_mean"] for r in rows])),
                "delta_std":    float(np.std([r["delta_mean"] for r in rows])),
                "mvc_mean":     float(np.mean([r["mvc"] for r in rows])),
                "time_ms":      float(np.mean([r["time_ms"] for r in rows])),
            }

    # Markdown table.
    lines = ["| N | model | n_win | Abs-Rel mean ± std | δ<1.25 mean ± std | MVC | time ms |",
             "|---|---|---|---|---|---|---|"]
    for n in args.n_list:
        for agg in ("mamba", "attention"):
            s = summary[agg].get(n)
            if s is None:
                continue
            lines.append(
                f"| {n} | {agg} | {s['n_windows']} | "
                f"{s['abs_rel_mean']:.4f} ± {s['abs_rel_std']:.4f} | "
                f"{s['delta_mean']:.3f} ± {s['delta_std']:.3f} | "
                f"{s['mvc_mean']:.4f} | {s['time_ms']:.0f} |"
            )
    md = "\n".join(lines)
    (args.out_dir / "table.md").write_text(md + "\n")
    print("\n" + md)

    # Plot 1: quality vs N
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    colors = {"mamba": "tab:blue", "attention": "tab:orange"}
    metric_specs = [
        ("abs_rel", "Abs-Rel (lower better)"),
        ("delta", "δ<1.25 (higher better)"),
        ("mvc", "Multi-view Chamfer (lower better)"),
    ]
    for ax, (key, title) in zip(axes, metric_specs):
        for agg in ("mamba", "attention"):
            ns, means, stds = [], [], []
            for n in args.n_list:
                s = summary[agg].get(n)
                if s is None:
                    continue
                ns.append(n)
                if key == "mvc":
                    means.append(s["mvc_mean"]); stds.append(0)
                else:
                    means.append(s[f"{key}_mean"]); stds.append(s[f"{key}_std"])
            ax.errorbar(ns, means, yerr=stds, fmt="o-", color=colors[agg],
                        label=agg, capsize=4, markersize=8, linewidth=2)
        ax.axvline(args.training_n, color="red", linestyle="--", alpha=0.5,
                   label=f"trained at N={args.training_n}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("frames N (log scale)")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("TerraWM quality vs sequence length — extrapolation test\n"
                 "trained at N=4, evaluated up to N=256 (64× extrapolation)", fontsize=12)
    plt.tight_layout()
    plt.savefig(args.out_dir / "quality_vs_n.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: per-frame Abs-Rel slope at the largest two N values
    big_ns = sorted([n for n in args.n_list if summary["mamba"].get(n) and
                     summary["attention"].get(n)])[-2:]
    if big_ns:
        fig, axes = plt.subplots(1, len(big_ns), figsize=(7 * len(big_ns), 5), squeeze=False)
        for col, n in enumerate(big_ns):
            ax = axes[0, col]
            for agg in ("mamba", "attention"):
                rows = [r for r in results[agg] if r["n"] == n]
                if not rows:
                    continue
                # Average per-frame curves across windows
                per_frame_arrs = np.array([r["abs_rel_per_frame"] for r in rows])
                mean_curve = np.nanmean(per_frame_arrs, axis=0)
                std_curve = np.nanstd(per_frame_arrs, axis=0)
                t = np.arange(len(mean_curve))
                ax.plot(t, mean_curve, "o-", color=colors[agg], label=agg, markersize=3)
                ax.fill_between(t, mean_curve - std_curve, mean_curve + std_curve,
                                color=colors[agg], alpha=0.15)
                # Linear fit slope
                valid = ~np.isnan(mean_curve)
                if valid.sum() >= 2:
                    slope, intercept = np.polyfit(t[valid], mean_curve[valid], 1)
                    ax.plot(t, intercept + slope * t, "--", color=colors[agg],
                            alpha=0.6, label=f"{agg} slope={slope:+.2e}")
            ax.set_title(f"N = {n}")
            ax.set_xlabel("frame position in window")
            ax.set_ylabel("Abs-Rel (mean ± std across windows)")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
        fig.suptitle("Per-frame Abs-Rel inside the window — does state help later frames?",
                     fontsize=11)
        plt.tight_layout()
        plt.savefig(args.out_dir / "per_frame_at_largeN.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"\n[bench] saved plots → {args.out_dir}")


if __name__ == "__main__":
    main()
