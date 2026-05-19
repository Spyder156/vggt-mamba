"""Aggregated multi-window eval — confirms the Phase 2.5 result isn't a fluke.

Loops over:
  - all 5 EVAL sequences
  - multiple start frames per sequence (every 100 frames, up to 5 starts)
  - N ∈ {4, 16, 32, 64}
  - both aggregators (attention, Mamba), each loaded once

Aggregates per (model, N): mean / median / std of Abs-Rel, δ<1.25, MVC, time.

Output:
  - viz/output/phase2_bench/results.json
  - viz/output/phase2_bench/summary.png — curves with std bars
  - viz/output/phase2_bench/table.md — markdown table for paper-style reporting
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt_mamba.data.tum_rgbd import EVAL_SEQS, TUMRGBDDataset, sync_sequence, intrinsics_for  # noqa: E402
from vggt_mamba.eval.metrics import multi_view_consistency                                       # noqa: E402
from vggt_mamba.models.mini3r import build_mini3r                                                 # noqa: E402


DEFAULT_CKPTS = {
    "attention": "experiments/phase2_real_multiview/dinov3_attention/ckpt_004000.pt",
    "mamba":     "experiments/phase2_real_multiview/dinov3_mamba/ckpt_004000.pt",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--ckpt-attention", type=Path, default=Path(DEFAULT_CKPTS["attention"]))
    p.add_argument("--ckpt-mamba", type=Path, default=Path(DEFAULT_CKPTS["mamba"]))
    p.add_argument("--n-list", type=int, nargs="+", default=[4, 16, 32, 64])
    p.add_argument("--frame-stride", type=int, default=10)
    p.add_argument("--starts-per-seq", type=int, default=5,
                   help="max number of start frames per sequence")
    p.add_argument("--start-gap", type=int, default=100,
                   help="frames between successive start positions")
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).resolve().parents[2] / "viz/output/phase2_bench")
    return p.parse_args()


def load_model(ckpt_path: Path, weights_root: Path, aggregator: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_mini3r(
        cfg["encoder"], str(weights_root),
        aggregator_name=aggregator,
        n_xfm_layers=cfg["model"]["n_xfm_layers"],
        d_state=cfg.get("d_state", 128),
    )
    msg = model.load_state_dict(ckpt["model"], strict=False)
    non_enc = [k for k in msg.missing_keys if not k.startswith("encoder.")]
    print(f"[load:{aggregator}] step={ckpt['step']} "
          f"missing(non-encoder)={len(non_enc)} unexpected={len(msg.unexpected_keys)}")
    return model.cuda().eval(), cfg


def build_window(records, start: int, n: int, frame_stride: int, seq_name: str, img_size: int,
                 depth_max_m: float = 8.0) -> dict:
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
    rgb = np.stack(rgbs)
    depth = np.stack(depths)
    poses_w_c = np.stack(poses)
    valid = (depth > 0.01) & (depth < depth_max_m)

    fx, fy, cx, cy = intrinsics_for(seq_name)
    sx = img_size / 640.0
    sy = img_size / 480.0
    K = np.array([[fx * sx, 0, cx * sx],
                  [0, fy * sy, cy * sy],
                  [0, 0, 1]], dtype=np.float32)

    return {
        "rgb": torch.from_numpy(rgb).permute(0, 3, 1, 2).unsqueeze(0).contiguous(),
        "depth": torch.from_numpy(depth).unsqueeze(0),
        "valid": torch.from_numpy(valid).unsqueeze(0),
        "poses_w_c": torch.from_numpy(poses_w_c).unsqueeze(0),
        "K": torch.from_numpy(K).unsqueeze(0),
    }


def metrics_one(model, batch) -> dict:
    rgb = batch["rgb"].cuda(non_blocking=True)
    valid = batch["valid"].cuda(non_blocking=True)
    poses = batch["poses_w_c"].cuda(non_blocking=True)
    depth_gt = batch["depth"].cuda(non_blocking=True)

    # warmup once (kernels, allocator) per (model, N) is enough — we'll re-warm only
    # when N changes, but it's cheap to just do it per window.
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        _ = model(rgb)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        pmap = model(rgb)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    pmap_fp32 = pmap.float()
    pred_depth = pmap_fp32[0, :, 2].cpu().numpy()
    gt = depth_gt[0].cpu().numpy()
    v = valid[0].cpu().numpy()

    abs_rel_pf, delta_pf = [], []
    for t in range(pred_depth.shape[0]):
        m = v[t]
        if m.sum() == 0:
            abs_rel_pf.append(np.nan); delta_pf.append(np.nan); continue
        p = np.maximum(pred_depth[t][m], 1e-6)
        g = np.maximum(gt[t][m], 1e-6)
        abs_rel_pf.append(float(np.mean(np.abs(p - g) / g)))
        ratio = np.maximum(p / g, g / p)
        delta_pf.append(float(np.mean(ratio < 1.25)))
    mvc = float(multi_view_consistency(pmap_fp32, valid, poses, n_samples=2048))

    return {
        "abs_rel_mean": float(np.nanmean(abs_rel_pf)),
        "delta_mean": float(np.nanmean(delta_pf)),
        "mvc": mvc,
        "time_ms": elapsed * 1000,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1e6,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load all eval sequences once.
    print("[bench] loading eval sequences…")
    seqs = []
    for name in EVAL_SEQS:
        seq_dir = args.data_root / name
        if not seq_dir.exists():
            continue
        recs = sync_sequence(seq_dir)
        seqs.append((name, recs))
        print(f"  {name}: {len(recs)} frames")

    # Run each model across all windows.
    results: dict[str, list[dict]] = {"attention": [], "mamba": []}
    for agg, ckpt in [("attention", args.ckpt_attention), ("mamba", args.ckpt_mamba)]:
        print(f"\n[bench] === {agg} ({ckpt}) ===")
        model, cfg = load_model(ckpt, args.weights_root, agg)
        img_size = {"vjepa": 384, "dinov2": 518, "dinov3": 512}[cfg["encoder"]]

        for n in args.n_list:
            span = (n - 1) * args.frame_stride + 1
            for seq_name, recs in seqs:
                max_start = len(recs) - span
                if max_start < 0:
                    continue
                starts = list(range(0, max_start + 1, args.start_gap))[:args.starts_per_seq]
                for start in starts:
                    batch = build_window(recs, start, n, args.frame_stride, seq_name, img_size)
                    m = metrics_one(model, batch)
                    m.update({"agg": agg, "n": n, "seq": seq_name, "start": start})
                    results[agg].append(m)
                    print(f"  [{agg} N={n:3d} {seq_name[-25:]:>25s} start={start:4d}]  "
                          f"abs_rel={m['abs_rel_mean']:.4f} δ={m['delta_mean']:.3f} "
                          f"mvc={m['mvc']:.4f}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2))

    # Aggregate per (agg, N).
    summary = {}
    for agg in ("attention", "mamba"):
        summary[agg] = {}
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
                "mvc_std":      float(np.std([r["mvc"] for r in rows])),
                "time_ms_mean": float(np.mean([r["time_ms"] for r in rows])),
            }

    # Markdown table.
    lines = ["| N | model | n_win | Abs-Rel mean ± std | δ<1.25 mean ± std | MVC mean ± std | time ms |",
             "|---|---|---|---|---|---|---|"]
    for n in args.n_list:
        for agg in ("attention", "mamba"):
            s = summary[agg].get(n)
            if s is None:
                continue
            lines.append(
                f"| {n} | {agg} | {s['n_windows']} | "
                f"{s['abs_rel_mean']:.4f} ± {s['abs_rel_std']:.4f} | "
                f"{s['delta_mean']:.3f} ± {s['delta_std']:.3f} | "
                f"{s['mvc_mean']:.4f} ± {s['mvc_std']:.4f} | "
                f"{s['time_ms_mean']:.0f} |"
            )
    table_md = "\n".join(lines)
    (args.out_dir / "table.md").write_text(table_md + "\n")
    print("\n" + table_md)

    # Plot.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"attention": "tab:orange", "mamba": "tab:blue"}
    metrics_plot = [("abs_rel", "Abs-Rel (lower better)", axes[0, 0]),
                    ("delta", "δ<1.25 (higher better)", axes[0, 1]),
                    ("mvc", "Multi-view Chamfer (lower better)", axes[1, 0]),
                    ("time_ms", "Forward time (ms)", axes[1, 1])]
    for key, title, ax in metrics_plot:
        for agg in ("attention", "mamba"):
            ns, means, stds = [], [], []
            for n in args.n_list:
                s = summary[agg].get(n)
                if s is None:
                    continue
                ns.append(n)
                if key == "time_ms":
                    means.append(s["time_ms_mean"]); stds.append(0)
                else:
                    means.append(s[f"{key}_mean"]); stds.append(s[f"{key}_std"])
            ax.errorbar(ns, means, yerr=stds, fmt="o-", color=colors[agg],
                        label=agg, capsize=4, markersize=8, linewidth=2)
        ax.set_xlabel("frames N"); ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
        if key == "time_ms":
            ax.set_yscale("log")
    fig.suptitle("Phase 2.5 — aggregated multi-window eval on all 5 EVAL sequences",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out_dir / "summary.png", dpi=120, bbox_inches="tight")
    print(f"\n[bench] saved {args.out_dir / 'summary.png'}")
    print(f"[bench] json   → {args.out_dir / 'results.json'}")
    print(f"[bench] table  → {args.out_dir / 'table.md'}")


if __name__ == "__main__":
    main()
