"""Streaming benchmark — does GeoMamba actually maintain constant memory
as N grows?

Loads the causal-only GeoMamba checkpoint, plays back a long real video one
frame at a time via streaming_forward, and records per-frame:
  - peak VRAM since the previous frame
  - wall time
  - depth Abs-Rel (when GT is available)

Plots:
  viz/output/phase3_streaming_bench/memory_per_frame.png  ← THE money plot
  viz/output/phase3_streaming_bench/quality_per_frame.png
  viz/output/phase3_streaming_bench/latency_per_frame.png

Reference baseline curve: simulated KV-cache memory of an attention-based
streaming method (StreamVGGT-shape).  We can't actually run StreamVGGT on
the 5070 Ti at 5000 frames, but the per-frame KV-cache cost is well-defined:
  KV ≈ frames * tokens_per_frame * D * 2 (K and V) * bf16_bytes
We plot that line clearly labelled "simulated KV cache (StreamVGGT-shape)"
so reviewers can verify the math.
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

from vggt_mamba.data.tum_rgbd import sync_sequence, intrinsics_for           # noqa: E402
from vggt_mamba.models.geomamba import build_geomamba                         # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--ckpt", type=Path,
                   default=Path("experiments/phase3_streaming/dinov3_mamba/ckpt_002000.pt"))
    p.add_argument("--seq", default="rgbd_dataset_freiburg3_long_office_household",
                   help="Long TUM sequence (~2500 frames)")
    p.add_argument("--max-frames", type=int, default=2000,
                   help="cap total frames processed")
    p.add_argument("--log-every", type=int, default=25,
                   help="record metrics every N frames")
    p.add_argument("--depth-eval-every", type=int, default=100,
                   help="compute depth Abs-Rel every N frames")
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).resolve().parents[2] / "viz/output/phase3_streaming_bench")
    p.add_argument("--use-cuda-graphs", action="store_true",
                   help="route streaming_forward through GraphedStreamingScan (Speed-B)")
    return p.parse_args()


def load_streaming_model(ckpt_path: Path, weights_root: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    assert cfg.get("aggregator", "mamba") == "mamba" and not cfg["model"]["bidirectional"], \
        "streaming_bench requires a causal-only Mamba checkpoint"
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
    msg = model.load_state_dict(ckpt["model"], strict=False)
    non_enc = [k for k in msg.missing_keys if not k.startswith("encoder.")]
    print(f"[stream] ckpt step={ckpt['step']}  missing(non-encoder)={len(non_enc)}")
    return model.cuda().eval(), cfg


def load_frame(rec, img_size: int):
    """Return (rgb_tensor, depth_np, valid_np)."""
    from PIL import Image
    img = Image.open(rec.rgb_path).convert("RGB").resize((img_size, img_size))
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous()
    d = np.asarray(
        Image.open(rec.depth_path).resize((img_size, img_size), Image.NEAREST),
        dtype=np.float32,
    ) / 5000.0
    return rgb_t, d, (d > 0.01) & (d < 8.0)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[stream] loading sequence {args.seq}")
    recs = sync_sequence(args.data_root / args.seq)
    print(f"[stream]   {len(recs)} synced frames")
    recs = recs[:args.max_frames]

    model, cfg = load_streaming_model(args.ckpt, args.weights_root)
    img_size = {"vjepa": 384, "dinov2": 518, "dinov3": 512}[cfg["encoder"]]
    state = model.init_streaming_state(batch_size=1, dtype=torch.bfloat16, device="cuda",
                                       use_cuda_graphs=args.use_cuda_graphs)
    if args.use_cuda_graphs:
        print(f"[stream] using CUDA graphs (Speed-B)")

    # Sizes of the state tensors (kept constant throughout).
    from vggt_mamba.models.aggregators import GraphedStreamingScan
    state_list = state.state if isinstance(state, GraphedStreamingScan) else state
    state_bytes = sum(
        s["conv"].element_size() * s["conv"].numel() + s["ssm"].element_size() * s["ssm"].numel()
        for s in state_list
    )
    print(f"[stream] Mamba state total: {state_bytes/1024:.1f} KB across {len(state_list)} layers")

    # Warmup: 3 frames to let allocator settle.
    for i in range(min(3, len(recs))):
        rgb, _, _ = load_frame(recs[i], img_size)
        rgb = rgb.cuda(non_blocking=True)
        _, state = model.streaming_forward(rgb, state, frame_idx=i)
    # Reset state for the measurement run. With graphs, reset_state() zeros the
    # captured buffers in place (preserving graph validity). Without graphs, just
    # rebuild the list.
    if isinstance(state, GraphedStreamingScan):
        state.reset_state()
    else:
        state = model.init_streaming_state(batch_size=1, dtype=torch.bfloat16, device="cuda")

    log: list[dict] = []
    abs_rel_log: list[dict] = []
    print(f"[stream] streaming {len(recs)} frames...")
    t_overall0 = time.perf_counter()
    for i, rec in enumerate(recs):
        rgb, depth_gt, valid = load_frame(rec, img_size)
        rgb = rgb.cuda(non_blocking=True)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        preds, state = model.streaming_forward(rgb, state, frame_idx=i)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1e6     # MB
        alloc = torch.cuda.memory_allocated() / 1e6        # MB resident after the frame

        if i % args.log_every == 0 or i == len(recs) - 1:
            log.append({"frame": i, "time_ms": dt * 1000,
                        "peak_vram_mb": peak, "resident_mb": alloc})

        if i % args.depth_eval_every == 0 or i == len(recs) - 1:
            pred_d = preds["pointmap"][0, 0, 2].float().cpu().numpy()
            v = valid
            if v.sum() > 0:
                p = np.maximum(pred_d[v], 1e-6)
                g = np.maximum(depth_gt[v], 1e-6)
                abs_rel = float(np.mean(np.abs(p - g) / g))
            else:
                abs_rel = float("nan")
            abs_rel_log.append({"frame": i, "abs_rel": abs_rel})

        if (i + 1) % 100 == 0:
            wall = time.perf_counter() - t_overall0
            fps = (i + 1) / wall
            last = log[-1] if log else {}
            print(f"  frame {i+1:5d}/{len(recs)}  fps={fps:5.1f}  "
                  f"peak={last.get('peak_vram_mb',0):6.0f}MB  "
                  f"resident={last.get('resident_mb',0):6.0f}MB  "
                  f"abs_rel={abs_rel_log[-1]['abs_rel'] if abs_rel_log else float('nan'):.3f}")

    (args.out_dir / "log.json").write_text(json.dumps({
        "log": log, "abs_rel_log": abs_rel_log,
        "state_bytes": state_bytes,
        "n_frames": len(recs),
    }, indent=2))

    # ---- Plot 1: memory per frame ----
    fig, ax = plt.subplots(figsize=(10, 6))
    frames = [r["frame"] for r in log]
    peaks = [r["peak_vram_mb"] for r in log]
    resi = [r["resident_mb"] for r in log]
    ax.plot(frames, peaks, "-", color="tab:blue", linewidth=2,
            label="GeoMamba streaming · peak VRAM per frame")
    ax.plot(frames, resi, "--", color="tab:blue", linewidth=1.5, alpha=0.7,
            label="GeoMamba streaming · resident VRAM after frame")

    # Simulated KV-cache baselines (memory cost only, well-defined math).
    # StreamVGGT-shape: per-frame KV at the model dim. Their actual config is
    # different from ours; this just illustrates the linear-growth shape.
    D = 1024
    KV_per_frame_kb = 2 * D * 2 / 1024   # K + V, bf16, one token-per-frame head; toy
    kv_curve = [(f * KV_per_frame_kb / 1024) for f in frames]  # MB
    # That's a lower bound — real StreamVGGT keeps full patch KVs.
    P_PER_FRAME = 1024  # DINOv3 patch count
    kv_curve_full = [f * P_PER_FRAME * 2 * D * 2 / 1024 / 1024 for f in frames]
    ax.plot(frames, kv_curve_full, ":", color="tab:orange", linewidth=2,
            label="simulated KV cache (P=1024 tokens/frame, dim=1024, bf16)")

    ax.set_xlabel("frames processed")
    ax.set_ylabel("VRAM (MB)")
    ax.set_title(f"GeoMamba streaming memory profile · {args.seq}\n"
                 f"Mamba state size = {state_bytes/1024:.1f} KB · constant in N")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_dir / "memory_per_frame.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- Plot 2: depth quality per frame ----
    fig, ax = plt.subplots(figsize=(10, 5))
    fs = [r["frame"] for r in abs_rel_log]
    ar = [r["abs_rel"] for r in abs_rel_log]
    ax.plot(fs, ar, "o-", color="tab:blue", markersize=4)
    ax.set_xlabel("frame")
    ax.set_ylabel("Abs-Rel depth error")
    ax.set_title("Streaming depth quality vs frame index — does state hold up?")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_dir / "quality_per_frame.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- Plot 3: latency per frame ----
    fig, ax = plt.subplots(figsize=(10, 5))
    fs = [r["frame"] for r in log]
    times = [r["time_ms"] for r in log]
    ax.plot(fs, times, "-", color="tab:blue")
    mean_t = float(np.mean(times))
    ax.axhline(mean_t, color="red", linestyle="--",
               label=f"mean = {mean_t:.0f} ms ({1000/mean_t:.1f} FPS)")
    ax.set_xlabel("frame")
    ax.set_ylabel("per-frame latency (ms)")
    ax.set_title("Streaming per-frame latency — should be flat in N")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_dir / "latency_per_frame.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    total_wall = time.perf_counter() - t_overall0
    print(f"\n[stream] processed {len(recs)} frames in {total_wall:.1f} s "
          f"({len(recs)/total_wall:.1f} FPS)")
    print(f"[stream] saved 3 plots to {args.out_dir}")


if __name__ == "__main__":
    main()
