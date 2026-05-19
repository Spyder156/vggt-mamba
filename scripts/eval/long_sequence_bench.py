"""Memory and time vs N — for the aggregator only.

Compares CrossFrameTransformer vs CrossFrameMamba directly on synthetic
patch tokens of shape (1, T*N, D). The encoder and DPT head are excluded
because:
  - Encoder is frozen and runs per-frame, O(N) for both aggregators.
  - DPT head is per-frame conv stack, O(N) for both aggregators; in
    practice it hits a PyTorch INT_MAX limit on the upsample around N=32.
This benchmark isolates the *aggregator* scaling — the actual variable.

Output: viz/output/phase2_long_seq/aggregator_memory_vs_n.png + json.

Usage:
    ./docker/run.sh python scripts/eval/long_sequence_bench.py
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt_mamba.models.aggregators import CrossFrameMamba, CrossFrameTransformer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=1024,
                   help="patch token dim (DINOv3 ViT-L = 1024)")
    p.add_argument("--patches-per-frame", type=int, default=1024,
                   help="DINOv3 ViT-L @ 512x512 = 32x32 = 1024")
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-list", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).resolve().parents[2] / "viz/output/phase2_long_seq")
    return p.parse_args()


def measure(agg: torch.nn.Module, n_frames: int, patches_per_frame: int, dim: int) -> dict:
    """One aggregator forward pass at sequence length n_frames * patches_per_frame."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    seq_len = n_frames * patches_per_frame
    try:
        x = torch.rand(1, seq_len, dim, device="cuda", dtype=torch.bfloat16)
        # warmup
        with torch.inference_mode():
            _ = agg(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        with torch.inference_mode():
            y = agg(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        del x, y
        return {
            "n": n_frames, "seq_len": seq_len, "ok": True,
            "time_s": elapsed, "peak_vram_mb": peak_mb,
        }
    except torch.cuda.OutOfMemoryError as e:
        gc.collect()
        torch.cuda.empty_cache()
        return {"n": n_frames, "seq_len": seq_len, "ok": False,
                "error": "OOM", "msg": str(e)[:120]}
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        traceback.print_exc()
        return {"n": n_frames, "seq_len": seq_len, "ok": False,
                "error": type(e).__name__, "msg": str(e)[:120]}


def bench_one(name: str, build_fn, n_list, patches_per_frame, dim):
    print(f"\n[bench] === aggregator={name} ===")
    agg = build_fn().cuda().to(torch.bfloat16).eval()
    n_params = sum(p.numel() for p in agg.parameters())
    print(f"[bench] params: {n_params/1e6:.2f}M")
    rows = []
    oom_seen = False
    for n in n_list:
        if oom_seen:
            rows.append({"n": n, "seq_len": n * patches_per_frame,
                         "ok": False, "error": "SKIP (prev OOM)"})
            print(f"[bench]   N={n:4d} (seq={n*patches_per_frame:6d})  SKIP")
            continue
        r = measure(agg, n, patches_per_frame, dim)
        rows.append(r)
        if r["ok"]:
            print(f"[bench]   N={n:4d} (seq={r['seq_len']:6d})  "
                  f"time={r['time_s']*1000:8.1f} ms   vram={r['peak_vram_mb']:7.0f} MB")
        else:
            print(f"[bench]   N={n:4d} (seq={r['seq_len']:6d})  {r['error']}")
            if r["error"] == "OOM":
                oom_seen = True
    del agg
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "attention": bench_one(
            "attention",
            lambda: CrossFrameTransformer(dim=args.dim, n_layers=args.n_layers),
            args.n_list, args.patches_per_frame, args.dim,
        ),
        "mamba": bench_one(
            "mamba",
            lambda: CrossFrameMamba(dim=args.dim, n_layers=args.n_layers, d_state=128),
            args.n_list, args.patches_per_frame, args.dim,
        ),
    }

    (args.out_dir / "aggregator_bench.json").write_text(json.dumps({
        "config": {"dim": args.dim, "patches_per_frame": args.patches_per_frame,
                   "n_layers": args.n_layers},
        "results": results,
    }, indent=2))

    fig, (ax_mem, ax_time) = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"attention": "tab:orange", "mamba": "tab:blue"}
    for agg_name, rows in results.items():
        ok = [r for r in rows if r["ok"]]
        ns = [r["n"] for r in ok]
        mems = [r["peak_vram_mb"] for r in ok]
        ts = [r["time_s"] * 1000 for r in ok]
        ax_mem.plot(ns, mems, "o-", label=agg_name, color=colors[agg_name],
                    linewidth=2, markersize=8)
        ax_time.plot(ns, ts, "o-", label=agg_name, color=colors[agg_name],
                     linewidth=2, markersize=8)
        # OOM points: dashed line up to GPU cap (16 GB on 5070 Ti)
        for r in rows:
            if not r["ok"] and r["error"] == "OOM":
                ax_mem.axvline(r["n"], color=colors[agg_name], linestyle="--", alpha=0.5)
                ax_mem.text(r["n"], 100, f"{agg_name}\nOOM",
                            color=colors[agg_name], fontsize=9, ha="center")

    # Reference scaling lines.
    if results["mamba"][0]["ok"]:
        m0 = results["mamba"][0]["peak_vram_mb"]
        n0 = results["mamba"][0]["n"]
        ns = [r["n"] for r in results["mamba"] if r["ok"]]
        ax_mem.plot(ns, [m0 * (n / n0) for n in ns], "--", color="gray", alpha=0.4, label="O(N) ref")
        ax_mem.plot(ns, [m0 * (n / n0) ** 2 for n in ns], ":", color="gray", alpha=0.4,
                    label="O(N²) ref")

    for ax, title, ylabel in [
        (ax_mem, "Aggregator peak VRAM vs frame count", "MB"),
        (ax_time, "Aggregator forward time vs frame count", "ms"),
    ]:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("frames N")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"Cross-frame aggregator scaling · dim={args.dim} · "
        f"{args.patches_per_frame} tokens/frame · {args.n_layers} layers · "
        f"{torch.cuda.get_device_name(0)}",
        fontsize=11,
    )
    plt.tight_layout()
    out = args.out_dir / "aggregator_memory_vs_n.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\n[bench] saved {out}")


if __name__ == "__main__":
    main()
