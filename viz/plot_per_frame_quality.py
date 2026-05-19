"""Per-frame depth quality vs frame position inside the window.

Reads results.json from video_sequence_eval.py (which already records
abs_rel_per_frame and delta_per_frame for every N tested), and plots:

  - Abs-Rel(t) vs frame-position t, one line per model, one panel per N
  - δ<1.25(t) vs t, same layout

This isolates the "state accumulates scene info" hypothesis:
  - Mamba is causal: patches in frame t see frames 0..t via the state.
    If the state is doing world-model work, abs_rel(t) should *decrease* as
    t grows.
  - Attention is bidirectional: every frame sees the whole window.
    Quality should be roughly flat across t.

A clear negative slope for Mamba + flat line for attention = direct
evidence that the state holds useful scene context.

Output: viz/output/phase2_video_eval/per_frame_quality.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path,
                   default=Path(__file__).resolve().parents[1] / "viz/output/phase2_video_eval/results.json")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parents[1] / "viz/output/phase2_video_eval/per_frame_quality.png")
    return p.parse_args()


def fit_slope(y):
    """Linear fit slope of y vs x=0..len(y)-1. Returns (slope, intercept)."""
    y = np.asarray(y, dtype=float)
    valid = ~np.isnan(y)
    if valid.sum() < 2:
        return float("nan"), float("nan")
    x = np.arange(len(y))[valid]
    return np.polyfit(x, y[valid], 1)


def main() -> None:
    args = parse_args()
    data = json.loads(args.results.read_text())
    summary = data["summary"]

    # gather all unique N values across both models
    n_values = sorted({r["n"] for r in summary["attention"]} & {r["n"] for r in summary["mamba"]})
    n_panels = len(n_values)

    fig, axes = plt.subplots(2, n_panels, figsize=(4.0 * n_panels, 8))
    if n_panels == 1:
        axes = axes.reshape(2, 1)
    colors = {"attention": "tab:orange", "mamba": "tab:blue"}

    for col, n in enumerate(n_values):
        for agg in ("attention", "mamba"):
            row = next(r for r in summary[agg] if r["n"] == n)
            abs_rel = row["abs_rel_per_frame"]
            delta = row["delta_per_frame"]
            t = np.arange(len(abs_rel))

            slope_ar, intercept_ar = fit_slope(abs_rel)
            slope_d, intercept_d = fit_slope(delta)
            trendy_ar = intercept_ar + slope_ar * t
            trendy_d = intercept_d + slope_d * t

            axes[0, col].plot(t, abs_rel, "o-", color=colors[agg], alpha=0.7,
                              label=f"{agg}   slope={slope_ar:+.2e}",
                              markersize=4, linewidth=1)
            axes[0, col].plot(t, trendy_ar, "--", color=colors[agg], alpha=0.5, linewidth=1)

            axes[1, col].plot(t, delta, "o-", color=colors[agg], alpha=0.7,
                              label=f"{agg}   slope={slope_d:+.2e}",
                              markersize=4, linewidth=1)
            axes[1, col].plot(t, trendy_d, "--", color=colors[agg], alpha=0.5, linewidth=1)

        axes[0, col].set_title(f"N={n}")
        axes[0, col].set_ylabel("Abs-Rel  (lower better)")
        axes[1, col].set_ylabel("δ<1.25   (higher better)")
        for ax in axes[:, col]:
            ax.set_xlabel("frame position in window")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

    fig.suptitle(
        "Per-frame quality vs position in window — testing 'state as world model'.\n"
        "Mamba causal: improvement with t  →  state accumulating scene info.\n"
        "Attention bidirectional: ~flat  →  symmetric access to whole window.",
        fontsize=10,
    )
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"[per-frame] saved {args.out}")

    # Print numerical slopes for direct interpretation.
    print("\nLinear slope of Abs-Rel vs frame-position (negative = improving with t):")
    print(f"  {'N':>4}  {'attention':>14}  {'mamba':>14}")
    for n in n_values:
        a_row = next(r for r in summary["attention"] if r["n"] == n)
        m_row = next(r for r in summary["mamba"] if r["n"] == n)
        s_a, _ = fit_slope(a_row["abs_rel_per_frame"])
        s_m, _ = fit_slope(m_row["abs_rel_per_frame"])
        print(f"  {n:>4}  {s_a:>+14.2e}  {s_m:>+14.2e}")


if __name__ == "__main__":
    main()
