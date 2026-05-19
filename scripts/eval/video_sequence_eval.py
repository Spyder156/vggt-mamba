"""Phase 2.5 — real video sequence eval, attention vs Mamba.

Load both trained Mini-3R checkpoints (DINOv3 + attention from Phase 1,
DINOv3 + Mamba from Phase 2). Run each on a real TUM video at several
sequence lengths N. Measure:
  - per-frame depth Abs-Rel error vs Kinect GT
  - per-frame δ<1.25 inlier ratio
  - multi-view consistency (Chamfer in world coords)
  - per-frame wall time and peak VRAM at each N

Dump:
  - viz/output/phase2_video_eval/strips/{attn,mamba}_N{N}.png
      RGB / GT depth / pred depth strip, all frames stacked vertically
  - viz/output/phase2_video_eval/curves.png
      Quality + speed vs N, both models overlaid
  - viz/output/phase2_video_eval/results.json

Usage:
    ./docker/run.sh python scripts/eval/video_sequence_eval.py
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

from vggt_mamba.data.tum_rgbd import TUMRGBDDataset, unproject_depth_to_pointmap   # noqa: E402
from vggt_mamba.eval.metrics import multi_view_consistency                          # noqa: E402
from vggt_mamba.models.mini3r import build_mini3r                                    # noqa: E402


DEFAULT_CKPTS = {
    "attention": "experiments/phase1_tokenizer_probe/dinov3/ckpt_004000.pt",
    "mamba":     "experiments/phase2_mamba_swap/dinov3_mamba/ckpt_004000.pt",
}
DEFAULT_SEQ = "rgbd_dataset_freiburg3_long_office_household"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--seq", default=DEFAULT_SEQ)
    p.add_argument("--n-list", type=int, nargs="+", default=[4, 8, 16, 24])
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--frame-stride", type=int, default=10,
                   help="gap between frames within a window (must match training setup)")
    p.add_argument("--ckpt-attention", type=Path, default=Path(DEFAULT_CKPTS["attention"]))
    p.add_argument("--ckpt-mamba", type=Path, default=Path(DEFAULT_CKPTS["mamba"]))
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).resolve().parents[2] / "viz/output/phase2_video_eval")
    return p.parse_args()


def load_model(ckpt_path: Path, weights_root: Path, aggregator: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    # config might be from Phase 1 (no aggregator key) or Phase 2 (has it)
    cfg_agg = cfg.get("aggregator", "attention")
    if cfg_agg != aggregator:
        raise ValueError(f"ckpt {ckpt_path} has aggregator={cfg_agg}, expected {aggregator}")
    model = build_mini3r(
        cfg["encoder"], str(weights_root),
        aggregator_name=aggregator,
        n_xfm_layers=cfg["model"]["n_xfm_layers"],
        d_state=cfg.get("d_state", 128),
    )
    msg = model.load_state_dict(ckpt["model"], strict=False)
    # Encoder backbone keys are intentionally not in ckpt (they're frozen pretrained
    # weights loaded fresh by build_mini3r); filter those before reporting.
    non_enc_missing = [k for k in msg.missing_keys if not k.startswith("encoder.")]
    print(f"[load:{aggregator}] step={ckpt['step']} "
          f"missing(non-encoder)={len(non_enc_missing)} unexpected={len(msg.unexpected_keys)}")
    return model.cuda().eval()


def load_window(ds: TUMRGBDDataset, n: int, start: int, frame_stride: int = 10) -> dict:
    """Build a single N-frame window starting at `start`, with intra-window stride."""
    seq_name, recs = ds.sequences[0]
    last = start + (n - 1) * frame_stride
    assert last < len(recs), f"need frames up to {last}, have {len(recs)}"
    window = [recs[start + k * frame_stride] for k in range(n)]

    rgbs, depths, poses = [], [], []
    for r in window:
        rgbs.append(ds._load_rgb(r.rgb_path))
        depths.append(ds._load_depth(r.depth_path))
        poses.append(r.pose_w_c.astype(np.float32))
    rgb = np.stack(rgbs)
    depth = np.stack(depths)
    poses_w_c = np.stack(poses)
    valid = (depth > 0.01) & (depth < ds.depth_max_m)

    from vggt_mamba.data.tum_rgbd import intrinsics_for
    fx, fy, cx, cy = intrinsics_for(seq_name)
    sx = ds.img_size / 640.0
    sy = ds.img_size / 480.0
    K = np.array([[fx * sx, 0, cx * sx],
                  [0, fy * sy, cy * sy],
                  [0, 0, 1]], dtype=np.float32)

    return {
        "rgb": torch.from_numpy(rgb).permute(0, 3, 1, 2).unsqueeze(0).contiguous(),
        "depth": torch.from_numpy(depth).unsqueeze(0),
        "valid": torch.from_numpy(valid).unsqueeze(0),
        "poses_w_c": torch.from_numpy(poses_w_c).unsqueeze(0),
        "K": torch.from_numpy(K).unsqueeze(0),
        "seq": seq_name,
        "start": start,
    }


def depth_abs_rel_per_frame(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-frame |pred - gt| / gt on valid pixels. pred/gt/valid: (T, H, W)."""
    out = np.empty(pred.shape[0])
    for t in range(pred.shape[0]):
        m = valid[t]
        if m.sum() == 0:
            out[t] = np.nan
            continue
        p = np.maximum(pred[t][m], 1e-6)
        g = np.maximum(gt[t][m], 1e-6)
        out[t] = float(np.mean(np.abs(p - g) / g))
    return out


def depth_delta_per_frame(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray,
                          thresh: float = 1.25) -> np.ndarray:
    out = np.empty(pred.shape[0])
    for t in range(pred.shape[0]):
        m = valid[t]
        if m.sum() == 0:
            out[t] = np.nan
            continue
        p = np.maximum(pred[t][m], 1e-6)
        g = np.maximum(gt[t][m], 1e-6)
        ratio = np.maximum(p / g, g / p)
        out[t] = float(np.mean(ratio < thresh))
    return out


def run_one(model, batch, label: str) -> dict:
    """One forward pass + metrics on a prepared window."""
    rgb = batch["rgb"].cuda(non_blocking=True)
    valid = batch["valid"].cuda(non_blocking=True)
    poses = batch["poses_w_c"].cuda(non_blocking=True)
    K = batch["K"].cuda(non_blocking=True)
    depth_gt = batch["depth"].cuda(non_blocking=True)
    n = rgb.shape[1]

    # warmup once for accurate timing
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        _ = model(rgb)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred_pmap = model(rgb)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1e6

    # MVC operates in fp32 (the GT poses are fp32; mixing with bf16 from autocast
    # would fail in the rotation matmul).
    pred_pmap_fp32 = pred_pmap.float()
    pred_depth = pred_pmap_fp32[0, :, 2]                  # (T, H, W) — Z channel
    gt_depth = depth_gt[0]
    valid0 = valid[0]

    abs_rel = depth_abs_rel_per_frame(
        pred_depth.cpu().numpy(), gt_depth.cpu().numpy(), valid0.cpu().numpy()
    )
    delta = depth_delta_per_frame(
        pred_depth.cpu().numpy(), gt_depth.cpu().numpy(), valid0.cpu().numpy()
    )
    mvc = float(multi_view_consistency(pred_pmap_fp32, valid, poses, n_samples=2048))

    return {
        "label": label, "n": n,
        "abs_rel_per_frame": abs_rel.tolist(),
        "delta_per_frame": delta.tolist(),
        "abs_rel_mean": float(np.nanmean(abs_rel)),
        "delta_mean": float(np.nanmean(delta)),
        "mvc": mvc,
        "time_ms": elapsed * 1000,
        "peak_vram_mb": peak_vram,
        "pred_depth": pred_depth.cpu().numpy(),
    }


def dump_strip(rgb_t, gt_depth, valid, pred_depth, title: str, out_path: Path):
    """One PNG: row 0 RGB, row 1 GT depth, row 2 pred depth. Columns = frames."""
    rgb_np = rgb_t[0].permute(0, 2, 3, 1).cpu().numpy()
    gt = gt_depth[0].cpu().numpy()
    v = valid[0].cpu().numpy()
    t = rgb_np.shape[0]

    # Stride down to at most 8 columns so the strip stays readable.
    if t > 8:
        idx = np.linspace(0, t - 1, 8).round().astype(int)
    else:
        idx = np.arange(t)

    vmax = float(np.nanmax(np.where(v, gt, np.nan)))
    pred_clip = np.clip(pred_depth, 0, vmax)

    cols = len(idx)
    fig, axes = plt.subplots(3, cols, figsize=(2.2 * cols, 6.5))
    if cols == 1:
        axes = axes[:, None]
    for k, i in enumerate(idx):
        axes[0, k].imshow(rgb_np[i]); axes[0, k].axis("off")
        axes[0, k].set_title(f"frame {i}", fontsize=8)
        masked = np.where(v[i], gt[i], np.nan)
        axes[1, k].imshow(masked, cmap="viridis", vmin=0, vmax=vmax); axes[1, k].axis("off")
        if k == 0:
            axes[1, k].set_ylabel("GT depth", fontsize=8)
        axes[2, k].imshow(pred_clip[i], cmap="viridis", vmin=0, vmax=vmax); axes[2, k].axis("off")
        if k == 0:
            axes[2, k].set_ylabel("pred", fontsize=8)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "strips").mkdir(parents=True, exist_ok=True)

    print(f"[video-eval] sequence: {args.seq}")
    print(f"[video-eval] N values: {args.n_list}")

    # DINOv3 model expects 512x512.
    ds = TUMRGBDDataset(args.data_root, split=[args.seq], n_frames=4, stride=1, img_size=512)

    summary: dict[str, list[dict]] = {"attention": [], "mamba": []}

    # Run each model across N values. Load/free per model to keep VRAM lean.
    for agg, ckpt_path in [("attention", args.ckpt_attention), ("mamba", args.ckpt_mamba)]:
        print(f"\n[video-eval] === {agg} ({ckpt_path}) ===")
        model = load_model(ckpt_path, args.weights_root, agg)
        for n in args.n_list:
            # Extend frame_embed if N exceeds its training-time max (was 64).
            # New positions get zero — for time-complexity measurement only;
            # quality numbers beyond training T are not meaningful.
            cur_max_t = model.frame_embed.shape[1]
            if n > cur_max_t:
                extra = n - cur_max_t
                pad = torch.zeros(1, extra, 1, model.frame_embed.shape[-1],
                                  device=model.frame_embed.device,
                                  dtype=model.frame_embed.dtype)
                model.frame_embed = torch.nn.Parameter(
                    torch.cat([model.frame_embed.data, pad], dim=1)
                )
                print(f"  [extend frame_embed {cur_max_t} -> {n}]")

            batch = load_window(ds, n=n, start=args.start_frame,
                                frame_stride=args.frame_stride)
            r = run_one(model, batch, label=agg)
            print(f"[{agg}]  N={n:3d}  abs_rel={r['abs_rel_mean']:.4f}  "
                  f"δ<1.25={r['delta_mean']:.3f}  mvc={r['mvc']:.4f}  "
                  f"time={r['time_ms']:7.1f} ms  vram={r['peak_vram_mb']:6.0f} MB")
            dump_strip(
                batch["rgb"], batch["depth"], batch["valid"], r["pred_depth"],
                title=f"{agg} · N={n}  abs_rel={r['abs_rel_mean']:.4f}  "
                      f"δ<1.25={r['delta_mean']:.3f}  mvc={r['mvc']:.4f}",
                out_path=args.out_dir / "strips" / f"{agg}_N{n:03d}.png",
            )
            # Remove the bulky pred_depth array before serialising summary.
            r2 = {k: v for k, v in r.items() if k != "pred_depth"}
            summary[agg].append(r2)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Curves plot.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"attention": "tab:orange", "mamba": "tab:blue"}
    for agg, rows in summary.items():
        ns = [r["n"] for r in rows]
        axes[0, 0].plot(ns, [r["abs_rel_mean"] for r in rows], "o-",
                        label=agg, color=colors[agg], linewidth=2, markersize=8)
        axes[0, 1].plot(ns, [r["delta_mean"] for r in rows], "o-",
                        label=agg, color=colors[agg], linewidth=2, markersize=8)
        axes[1, 0].plot(ns, [r["mvc"] for r in rows], "o-",
                        label=agg, color=colors[agg], linewidth=2, markersize=8)
        axes[1, 1].plot(ns, [r["time_ms"] for r in rows], "o-",
                        label=agg, color=colors[agg], linewidth=2, markersize=8)

    axes[0, 0].set_title("depth Abs-Rel (lower better)")
    axes[0, 1].set_title("depth δ<1.25 inlier ratio (higher better)")
    axes[1, 0].set_title("multi-view consistency Chamfer (lower better)")
    axes[1, 1].set_title("forward time (ms)"); axes[1, 1].set_yscale("log")
    for ax in axes.flat:
        ax.set_xlabel("frames N")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"Phase 2.5 — real video eval · seq={args.seq} · start frame {args.start_frame}",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out_dir / "curves.png", dpi=120, bbox_inches="tight")

    (args.out_dir / "results.json").write_text(json.dumps(
        {"args": {"seq": args.seq, "start_frame": args.start_frame, "n_list": args.n_list},
         "summary": summary},
        indent=2,
    ))
    print(f"\n[video-eval] curves → {args.out_dir / 'curves.png'}")
    print(f"[video-eval] strips → {args.out_dir / 'strips'}")
    print(f"[video-eval] json   → {args.out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
