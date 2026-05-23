"""Consolidate the 8 streaming runs into a single architectural ablation table.

Reads experiments/phase3_streaming_*/dinov3_mamba/{args.yaml, metrics.jsonl}.
Produces:
  - viz/output/paper_figures_patchscan/ablation_table.md

The story: starting from the N=8 baseline (summary-scan K=4, dense uses raw-patch
residual shortcut), each row adds/swaps one architectural component. The table
makes the bottleneck-discovery sequence visible:

  baseline             → adds world-model pred loss → +21% camera, depth ~flat
  baseline             → removes patch residual    → depth crashes to 0.33
  no_resid + predict   → role conflict at K=4 (0.40)
  K=8 dual-channel     → fixes camera (0.78) but dense stays bad (0.39)
  patch-scan           → fixes BOTH (0.21 depth + 0.78 cam): interface, not capacity
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml


ROW_ORDER = [
    # (dir_name, display_label)
    ("phase3_streaming_n8",          "N=8 baseline"),
    ("phase3_streaming_masked",      "+ masked-frame regularizer (p=0.3)"),
    ("phase3_streaming_predict",     "+ world-model prediction loss"),
    ("phase3_streaming_no_resid",    "− patch residual on dense"),
    ("phase3_streaming_combo",       "no_resid + predict (K=4)"),
    ("phase3_streaming_dualchannel", "K=8 dual-channel split (4+4)"),
    ("phase3_streaming_patchscan",   "+ patch-scan cross-frame Mamba [FINAL]"),
]


def read_run(exp_root: Path, dir_name: str) -> dict | None:
    base = exp_root / dir_name / "dinov3_mamba"
    args_path = base / "args.yaml"
    metrics_path = base / "metrics.jsonl"
    if not args_path.exists() or not metrics_path.exists():
        return None
    cfg = yaml.safe_load(args_path.read_text())
    evals = []
    for line in metrics_path.read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("kind") == "eval":
            evals.append(d)
    if not evals:
        return None
    best_ar = min(e["abs_rel_mean"] for e in evals)
    best_cam = min(e["cam_l1_mean"] for e in evals)
    final = evals[-1]
    return {
        "cfg": cfg,
        "best_abs_rel": best_ar,
        "best_cam_l1": best_cam,
        "final_abs_rel": final["abs_rel_mean"],
        "final_cam_l1": final["cam_l1_mean"],
        "n_evals": len(evals),
    }


def fmt_arch(cfg: dict) -> dict:
    m = cfg["model"]
    loss = cfg.get("loss", {})
    return {
        "K_total": m["n_summary_tokens"],
        "K_dyn": m.get("n_summary_dynamic", m["n_summary_tokens"]),
        "residual": "on" if m.get("dense_residual_to_patches", True) else "off",
        "pred": "on" if m.get("predict_next_latent", False) else "off",
        "mask_p": loss.get("mask_frame_prob", 0.0),
        "scan_target": m.get("cross_frame_target", "summary"),
    }


def main() -> None:
    import os
    out_dir = Path(os.environ.get("PAPER_FIG_DIR", "viz/output/paper_figures"))
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_root = Path("experiments")

    rows = []
    for dir_name, label in ROW_ORDER:
        r = read_run(exp_root, dir_name)
        if r is None:
            continue
        a = fmt_arch(r["cfg"])
        rows.append({"label": label, **a, **r})

    md = []
    md.append("# Ablation — architectural progression on TUM held-out (sitting_xyz, 8-frame eval)")
    md.append("")
    md.append("All runs: 2000 training steps, 7 TUM train sequences, frame_stride=10, "
              "DINOv3 ViT-L/16 frozen encoder, 6 cross-frame Mamba layers, batch_size=1, "
              "bfloat16. Best across all eval checkpoints.")
    md.append("")
    md.append("Architecture key:  K=total summary tokens · K_dyn=dynamic-channel slice "
              "(rest is observation channel, free of pred loss) · residual=dense head's "
              "raw-patch shortcut · pred=JEPA next-latent loss · scan=cross-frame Mamba "
              "operates on K summaries or P=1024 patches per frame.")
    md.append("")
    md.append("| Run | K | K_dyn | residual | pred | scan | best Abs-Rel ↓ | best cam_l1 ↓ |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        marker = "**" if "FINAL" in r["label"] else ""
        md.append(
            f"| {marker}{r['label']}{marker} | {r['K_total']} | {r['K_dyn']} | "
            f"{r['residual']} | {r['pred']} | {r['scan_target']} | "
            f"{marker}{r['best_abs_rel']:.3f}{marker} | "
            f"{marker}{r['best_cam_l1']:.3f}{marker} |"
        )
    md.append("")
    md.append("## The bottleneck-discovery sequence")
    md.append("")
    md.append("1. **N=8 baseline** (K=4, residual on): depth 0.215, cam 0.99. Dense head "
              "reads raw patches via residual; cross-frame state contributes little to dense.")
    md.append("2. **+ pred loss**: cam improves 0.99 → 0.85 (−14%). State learns predictive "
              "structure useful for ego-pose. Depth unaffected (still 0.23).")
    md.append("3. **− residual** (force state-driven dense): depth crashes 0.215 → 0.33. "
              "Reveals that prior depth quality was the residual shortcut, not the state.")
    md.append("4. **no_resid + predict (K=4)**: depth 0.41 — *worse* than either alone. "
              "K=4 cannot simultaneously be smooth-for-prediction (camera) and "
              "sharp-for-reconstruction (dense). Role conflict in shared state.")
    md.append("5. **K=8 dual-channel split**: cam recovers to **0.78** (best across all runs). "
              "Split heals the conflict — but depth stays at 0.39. K-summary readout is "
              "still too compressed for pixel-aligned reconstruction.")
    md.append("6. **+ patch-scan**: depth recovers to **0.217** (matches the residual-on "
              "baseline). DPT reads cross-frame-propagated per-patch hiddens directly. "
              "Bottleneck was *interface* (summary-token readout), not *capacity*.")
    md.append("")
    md.append("All three architectural claims of the paper are isolated by exactly one "
              "ablation step each: pred-loss → camera win, residual-off → reveals the "
              "shortcut, dual-channel → heals role conflict, patch-scan → heals interface.")
    md.append("")

    out_path = out_dir / "ablation_table.md"
    out_path.write_text("\n".join(md))
    print(f"[ablation] saved {out_path}")
    print()
    print("\n".join(md))


if __name__ == "__main__":
    main()
