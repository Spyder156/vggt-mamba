"""GeoMamba scaling benchmark — does the time/memory wedge survive
the summary-token architecture?

Phase 2's headline plot was Mini-3R's aggregator at T*P tokens — that's
where attention's O(N²) wrecks it. GeoMamba's aggregator sees only T*K
(=4N) tokens, so the aggregator's compute share is tiny. The dominant
costs shift to the per-frame components.

This benchmark measures three model variants on synthetic input at
increasing N:
  - GeoMamba + Mamba (causal)
  - GeoMamba + Mamba (bidirectional)
  - GeoMamba + Attention

Per-component timing breakdown so we can see which terms actually grow.

Outputs:
  viz/output/phase3_scaling/total_time_vs_n.png
  viz/output/phase3_scaling/peak_vram_vs_n.png
  viz/output/phase3_scaling/component_breakdown.png
  viz/output/phase3_scaling/raw.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt_mamba.models.geomamba import build_geomamba  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="dinov3")
    p.add_argument("--weights-root", type=Path,
                   default=Path(os.environ.get("VGGT_MAMBA_DATA_ROOT", "/workspace/datasets")) / "weights")
    p.add_argument("--n-list", type=int, nargs="+",
                   default=[4, 8, 16, 32, 64, 96, 128])
    p.add_argument("--n-intra", type=int, default=4)
    p.add_argument("--n-xfm", type=int, default=6)
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).resolve().parents[2] / "viz/output/phase3_scaling")
    return p.parse_args()


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_components(model, rgb: torch.Tensor) -> dict[str, float]:
    """Manually walk GeoMamba's forward and time each block."""
    b, t, _, h, w = rgb.shape
    ts: dict[str, float] = {}

    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        # 1. Encoder
        flat_rgb = rgb.reshape(b * t, 3, h, w)
        _sync(); t0 = time.perf_counter()
        with torch.no_grad():
            enc_out = model.encoder(flat_rgb)
        _sync(); ts["encoder"] = (time.perf_counter() - t0) * 1000
        patches = enc_out.patches.reshape(b, t, -1, model.dim)

        # 2. IntraFrame
        _sync(); t0 = time.perf_counter()
        refined = model.intraframe(patches)
        _sync(); ts["intraframe"] = (time.perf_counter() - t0) * 1000

        # 3. Summary pool
        _sync(); t0 = time.perf_counter()
        summaries = model.summary_pool(refined)
        _sync(); ts["summary_pool"] = (time.perf_counter() - t0) * 1000

        # 4. Cross-frame aggregator
        summaries = summaries + model.frame_embed[:, :t]
        seq = summaries.reshape(b, t * model.n_summary, model.dim)
        _sync(); t0 = time.perf_counter()
        state_seq = model.cross_frame(seq)
        _sync(); ts["aggregator"] = (time.perf_counter() - t0) * 1000
        state_per_frame = state_seq.reshape(b, t, model.n_summary, model.dim)

        # 5. Camera head
        _sync(); t0 = time.perf_counter()
        _ = model.camera_head(state_per_frame)
        _sync(); ts["camera_head"] = (time.perf_counter() - t0) * 1000

        # 6. Dense readout + DPT
        _sync(); t0 = time.perf_counter()
        dense_in = model.dense_readout(refined, state_per_frame)
        grid = dense_in.reshape(b * t, -1, model.dim).transpose(1, 2)
        grid = grid.reshape(b * t, model.dim, model.grid_h, model.grid_w)
        chunk = 8
        pmap_chunks = [model.dpt(grid[i:i + chunk]) for i in range(0, b * t, chunk)]
        _ = torch.cat(pmap_chunks, dim=0)
        _sync(); ts["dense+dpt"] = (time.perf_counter() - t0) * 1000

    ts["total"] = sum(ts.values())
    return ts


def measure_at_n(model, n: int, img_size: int) -> dict:
    """One forward at sequence length n; return per-stage time + peak VRAM."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    rgb = torch.rand(1, n, 3, img_size, img_size, device="cuda")
    try:
        # warmup
        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _ = model(rgb)
        _sync()
        torch.cuda.reset_peak_memory_stats()
        comp = time_components(model, rgb)
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        del rgb
        return {"n": n, "ok": True, **comp, "peak_vram_mb": peak_mb}
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        return {"n": n, "ok": False, "error": "OOM"}


def extend_frame_embed(model: torch.nn.Module, t: int) -> None:
    cur = model.frame_embed.shape[1]
    if t <= cur:
        return
    extra = t - cur
    pad = torch.zeros(
        1, extra, 1, model.frame_embed.shape[-1],
        device=model.frame_embed.device, dtype=model.frame_embed.dtype,
    )
    model.frame_embed = torch.nn.Parameter(torch.cat([model.frame_embed.data, pad], dim=1))


def bench_variant(variant_name: str, n_list, args, builder) -> list[dict]:
    print(f"\n[bench] === {variant_name} ===")
    model = builder().cuda().eval()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[bench] img={model.img_size}  trainable={n_train/1e6:.2f}M")
    rows = []
    oom_seen = False
    for n in n_list:
        extend_frame_embed(model, n)
        if oom_seen:
            rows.append({"n": n, "ok": False, "error": "SKIP (prev OOM)"})
            print(f"  N={n:4d}  SKIP")
            continue
        r = measure_at_n(model, n, model.img_size)
        rows.append(r)
        if r["ok"]:
            order = ["encoder", "intraframe", "summary_pool",
                     "aggregator", "camera_head", "dense+dpt"]
            comp_str = "  ".join(f"{k}={r[k]:6.1f}" for k in order)
            print(f"  N={n:4d}  total={r['total']:7.1f}ms  vram={r['peak_vram_mb']:6.0f}MB  | "
                  f"{comp_str}")
        else:
            print(f"  N={n:4d}  {r['error']}")
            if r["error"] == "OOM":
                oom_seen = True
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "GeoMamba + Mamba causal": lambda: build_geomamba(
            args.encoder, str(args.weights_root),
            n_intraframe_layers=args.n_intra, n_xfm_layers=args.n_xfm,
            d_state=128, bidirectional=False, aggregator_name="mamba",
            track_enabled=False, max_frames=max(args.n_list) + 16,
        ),
        "GeoMamba + Mamba bidirectional": lambda: build_geomamba(
            args.encoder, str(args.weights_root),
            n_intraframe_layers=args.n_intra, n_xfm_layers=args.n_xfm,
            d_state=128, bidirectional=True, aggregator_name="mamba",
            track_enabled=False, max_frames=max(args.n_list) + 16,
        ),
        "GeoMamba + Attention": lambda: build_geomamba(
            args.encoder, str(args.weights_root),
            n_intraframe_layers=args.n_intra, n_xfm_layers=args.n_xfm,
            bidirectional=False, aggregator_name="attention",
            track_enabled=False, max_frames=max(args.n_list) + 16,
        ),
    }

    results: dict[str, list[dict]] = {}
    for name, builder in variants.items():
        results[name] = bench_variant(name, args.n_list, args, builder)

    (args.out_dir / "raw.json").write_text(json.dumps(results, indent=2))

    # ---------- Plot 1: total wall time vs N ----------
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"GeoMamba + Mamba causal": "tab:blue",
              "GeoMamba + Mamba bidirectional": "tab:green",
              "GeoMamba + Attention": "tab:orange"}
    for name, rows in results.items():
        ok = [r for r in rows if r["ok"]]
        if not ok:
            continue
        ns = [r["n"] for r in ok]
        ts = [r["total"] for r in ok]
        ax.plot(ns, ts, "o-", label=name, color=colors[name], linewidth=2, markersize=8)
        for r in rows:
            if not r["ok"] and r["error"] == "OOM":
                ax.axvline(r["n"], color=colors[name], linestyle="--", alpha=0.5)

    # Reference lines from the smallest measured point on each curve.
    if results["GeoMamba + Mamba causal"][0]["ok"]:
        r0 = results["GeoMamba + Mamba causal"][0]
        n0, t0 = r0["n"], r0["total"]
        ns = [r["n"] for r in results["GeoMamba + Mamba causal"] if r["ok"]]
        ax.plot(ns, [t0 * (n / n0) for n in ns], "--", color="gray", alpha=0.4,
                label="O(N) reference")
        ax.plot(ns, [t0 * (n / n0) ** 2 for n in ns], ":", color="gray", alpha=0.4,
                label="O(N²) reference")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("frames N"); ax.set_ylabel("forward time (ms)")
    ax.set_title("GeoMamba total forward time vs sequence length\n"
                 f"DINOv3 frozen · {args.n_intra} intraframe layers · "
                 f"{args.n_xfm} aggregator layers · 5070 Ti")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_dir / "total_time_vs_n.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---------- Plot 2: peak VRAM vs N ----------
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, rows in results.items():
        ok = [r for r in rows if r["ok"]]
        if not ok:
            continue
        ns = [r["n"] for r in ok]
        mems = [r["peak_vram_mb"] for r in ok]
        ax.plot(ns, mems, "o-", label=name, color=colors[name], linewidth=2, markersize=8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("frames N"); ax.set_ylabel("peak VRAM (MB)")
    ax.set_title("GeoMamba peak VRAM vs sequence length")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_dir / "peak_vram_vs_n.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---------- Plot 3: stacked component breakdown ----------
    # Pick the largest common N across the three variants.
    common_ns = sorted({r["n"] for r in results["GeoMamba + Mamba causal"] if r["ok"]} &
                      {r["n"] for r in results["GeoMamba + Attention"] if r["ok"]})
    pick_ns = []
    if common_ns:
        # show low / mid / high if available
        pick_ns = [common_ns[0]]
        if len(common_ns) >= 3:
            pick_ns.append(common_ns[len(common_ns) // 2])
        pick_ns.append(common_ns[-1])
        pick_ns = sorted(set(pick_ns))

    fig, axes = plt.subplots(1, max(1, len(pick_ns)), figsize=(5 * max(1, len(pick_ns)), 6),
                             sharey=True)
    if len(pick_ns) == 1:
        axes = [axes]
    components = ["encoder", "intraframe", "summary_pool",
                  "aggregator", "camera_head", "dense+dpt"]
    comp_colors = plt.cm.tab10(np.linspace(0, 0.8, len(components)))
    variant_short = {"GeoMamba + Mamba causal": "Mamba\ncausal",
                     "GeoMamba + Mamba bidirectional": "Mamba\nbidir",
                     "GeoMamba + Attention": "Attn"}
    for col, n in enumerate(pick_ns):
        ax = axes[col]
        labels = list(variant_short.values())
        bottoms = np.zeros(len(labels))
        for ci, comp in enumerate(components):
            heights = []
            for name in variants:
                r = next((rr for rr in results[name] if rr["n"] == n and rr["ok"]), None)
                heights.append(r[comp] if r else 0)
            ax.bar(labels, heights, bottom=bottoms, color=comp_colors[ci],
                   label=comp if col == 0 else None, edgecolor="white", linewidth=0.5)
            bottoms += np.array(heights)
        ax.set_title(f"N = {n}")
        ax.set_ylabel("time (ms)" if col == 0 else "")
        ax.tick_params(axis="x", labelsize=10)
    if pick_ns:
        axes[0].legend(loc="upper left", fontsize=9, title="component")
    fig.suptitle("Per-component time breakdown across N\n"
                 "→ shows which terms actually grow with sequence length", fontsize=11)
    plt.tight_layout()
    plt.savefig(args.out_dir / "component_breakdown.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---------- Plot 4: aggregator-only time vs N ----------
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, rows in results.items():
        ok = [r for r in rows if r["ok"]]
        if not ok:
            continue
        ns = [r["n"] for r in ok]
        aggs = [r["aggregator"] for r in ok]
        ax.plot(ns, aggs, "o-", label=name, color=colors[name], linewidth=2, markersize=8)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("frames N"); ax.set_ylabel("aggregator time (ms)")
    ax.set_title("Aggregator-only forward time vs N\n"
                 "(this is the part that depends on aggregator choice)")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_dir / "aggregator_only_time_vs_n.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[bench] saved 4 plots to {args.out_dir}")
    for f in sorted(args.out_dir.glob("*.png")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
