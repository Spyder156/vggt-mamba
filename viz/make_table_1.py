"""Make Table 1 — quality + memory comparison vs published streaming methods.

Published numbers extracted from the referenced papers (TUM-dynamics).
Ours are measured on our trained checkpoint (Phase 3 streaming model).

Honest framing: our trainable surface is 109M (consumer-scale) vs StreamVGGT's
950M (datacenter-scale). Absolute quality is expected to lag; the memory
property is the architectural contribution.
"""

from __future__ import annotations

import json
from pathlib import Path


COMPETITORS = [
    # (name, params_M, abs_rel_tum_dynamics, ATE_tum_dynamics, memory_property)
    # Abs-Rel from each paper's Table 5 (video depth) on TUM-dynamics.
    # ATE from each paper's camera-pose table on TUM-dynamics.
    # Memory: O(N), O(N^2), or O(1) based on architecture.
    ("VGGT (offline)",         "1200", "0.057",  "0.012", "O(N²), OOMs at N≈200"),
    ("Spann3R",                "—",    "0.144",  "0.056", "O(N), bounded spatial memory"),
    ("CUT3R",                  "—",    "0.078",  "0.046", "O(N), growing RNN state"),
    ("Stream3R",               "—",    "0.075",  "0.213", "O(N), growing KV cache"),
    ("StreamVGGT",             "950",  "0.059",  "0.048", "O(N), growing KV cache"),
    ("Point3R",                "—",    "—",      "0.075", "O(N), growing pointer set"),
    ("ray-aware Point3R",      "—",    "—",      "0.049", "O(N), retain-or-replace"),
    # Our row computed live from the streaming bench log.
]


def main() -> None:
    import os
    out_dir = Path(os.environ.get("PAPER_FIG_DIR", "viz/output/paper_figures"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read our streaming bench stats from log.json. Default to the patch-scan
    # benches (current production architecture).
    log_files = [
        ("freiburg3_long_office_household (TRAIN seq)",
         "viz/output/phase3_streaming_bench_patchscan/log.json"),
        ("freiburg3_sitting_xyz (EVAL seq, held-out)",
         "viz/output/phase3_streaming_bench_patchscan_heldout/log.json"),
    ]
    ate_report_path = Path("viz/output/paper_figures_patchscan/ate_report.json")
    ate_data = json.loads(ate_report_path.read_text()) if ate_report_path.exists() else None
    our_rows = []
    for tag, path in log_files:
        p = Path(path)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        ar = [r["abs_rel"] for r in d["abs_rel_log"]]
        ar_mean = sum(ar) / len(ar) if ar else float("nan")
        peak = max(r["peak_vram_mb"] for r in d["log"]) / 1024
        fps_n = d["n_frames"]
        # FPS from total time = sum of per-frame times... use avg
        times = [r["time_ms"] for r in d["log"]]
        fps = 1000 / (sum(times) / len(times)) if times else float("nan")
        our_rows.append({
            "tag": tag, "abs_rel": ar_mean, "n": fps_n,
            "fps": fps, "peak_gb": peak,
            "state_kb": d["state_bytes"] / 1024,
        })

    # ---- Build markdown ----
    md = []
    md.append("# Table 1 — quality and memory comparison on TUM-dynamics")
    md.append("")
    md.append("Published numbers are from each method's paper (TUM-dynamics, "
              "video-depth + camera-pose tables). Ours are measured on a "
              "single consumer GPU (RTX 5070 Ti, 16 GB).")
    md.append("")
    md.append("## Quality and memory architecture")
    md.append("")
    md.append("| Method | Trainable params | Depth Abs-Rel ↓ | ATE ↓ | Memory scaling |")
    md.append("|---|---|---|---|---|")
    for name, params, abs_rel, ate, mem in COMPETITORS:
        md.append(f"| {name} | {params}M | {abs_rel} | {ate} | {mem} |")
    # Our row
    if our_rows:
        eval_row = next((r for r in our_rows if "EVAL" in r["tag"]), our_rows[0])
        ate_cell = "(pending)"
        ate_footnote = None
        if ate_data is not None:
            agg = ate_data["aggregate_across_sequences"]
            ate_cell = f"{agg['ate_sim3_rmse_m_mean']:.3f}¹"
            ate_footnote = (f"¹ Mean ATE Sim(3)-RMSE across {agg['n_sequences']} held-out "
                            "TUM-dynamics sequences (sitting_xyz, sitting_static, walking_xyz, "
                            "walking_halfsphere, walking_static). Static-scene subset: "
                            f"{0.5 * (ate_data['per_sequence'][1]['ate_sim3']['ate_rmse_m'] + ate_data['per_sequence'][4]['ate_sim3']['ate_rmse_m']):.3f} m. "
                            "Per-sequence breakdown below.")
        md.append(f"| **TerraWM (ours, this work)** | **110M** | "
                  f"**{eval_row['abs_rel']:.3f}** | **{ate_cell}** | "
                  f"**O(1) — {eval_row['state_kb']:.0f} KB constant state** |")
        md.append("")
        if ate_footnote:
            md.append(ate_footnote)
    md.append("")
    md.append("## Honest framing")
    md.append("")
    md.append("- **Absolute quality is mediocre** at our current training scale "
              "(110M trainable params, 2000 training steps, 7 TUM sequences). "
              "Published competitors are trained on the full 13-dataset VGGT mix "
              "(Co3D, BlendedMVS, MegaDepth, ScanNet, HyperSim, MVS-Synth, "
              "OmniObject3D, PointOdyssey, ARKitScenes, WildRGB, KITTI, Spring, "
              "Waymo) at datacenter scale.")
    md.append("- **The architectural contribution is the memory property**, not the "
              "absolute quality at this scale. With matched training "
              "compute + data, our setup is expected to land in the same Abs-Rel "
              "range as StreamVGGT (~0.06) while keeping the constant-memory advantage.")
    md.append("- **Hardware**: results above measured on RTX 5070 Ti (consumer, 16 GB). "
              "Competitors typically benchmarked on A100/H100 (datacenter, 40–80 GB). "
              "Streaming FPS reported below per regime.")
    md.append("")
    md.append("## Long-sequence streaming — our setup vs competitors at N=2000")
    md.append("")
    md.append("| Method | Memory at N=2000 | Wall time @ 30 FPS |")
    md.append("|---|---|---|")
    md.append("| StreamVGGT (extrapolated from Table 7) | ~330 GB | — (OOMs) |")
    md.append("| VGGT (offline) | OOM at N≈200 | — |")
    md.append("| CUT3R / Stream3R / Point3R | linear in N, multi-GB | depends |")
    if our_rows:
        train_row = next((r for r in our_rows if "TRAIN" in r["tag"]), our_rows[0])
        md.append(f"| **TerraWM (ours)** | **{train_row['peak_gb']:.2f} GB flat** | "
                  f"**{train_row['fps']:.1f} FPS sustained on RTX 5070 Ti** |")
    md.append("")

    md.append("## Our per-sequence streaming numbers")
    md.append("")
    md.append("| Sequence | Held-out? | Frames | FPS | Peak VRAM | Mean Abs-Rel |")
    md.append("|---|---|---|---|---|---|")
    for r in our_rows:
        held = "yes" if "EVAL" in r["tag"] else "no (TRAIN seq leak)"
        md.append(f"| {r['tag'].split(' ')[0]} | {held} | {r['n']} | {r['fps']:.1f} | "
                  f"{r['peak_gb']:.2f} GB | {r['abs_rel']:.3f} |")
    md.append("")

    if ate_data is not None:
        md.append("## Per-sequence camera trajectory error (held-out TUM-dynamics)")
        md.append("")
        md.append("Sim(3) Umeyama-aligned ATE and per-frame RPE on each of the 5 held-out "
                  "sequences. Note the static/dynamic split: static-camera scenes are "
                  "competitive; large-motion XYZ/halfsphere sequences drag the mean.")
        md.append("")
        md.append("| Sequence | Frames | ATE-Sim3 RMSE ↓ | RPE δ=1 trans ↓ | RPE δ=1 rot ↓ |")
        md.append("|---|---|---|---|---|")
        for s in ate_data["per_sequence"]:
            name = s["seq"].replace("rgbd_dataset_freiburg3_", "f3_")
            md.append(f"| {name} | {s['n_frames']} | "
                      f"{s['ate_sim3']['ate_rmse_m']:.3f} m | "
                      f"{s['rpe_delta1']['rpe_trans_rmse_m']:.3f} m | "
                      f"{s['rpe_delta1']['rpe_rot_rmse_deg']:.2f}° |")
        agg = ate_data["aggregate_across_sequences"]
        md.append(f"| **mean** | — | **{agg['ate_sim3_rmse_m_mean']:.3f} m** | "
                  f"**{agg['rpe_trans_rmse_m_mean']:.3f} m** | "
                  f"**{agg['rpe_rot_rmse_deg_mean']:.2f}°** |")
        md.append("")

    out_path = out_dir / "table_1_comparison.md"
    out_path.write_text("\n".join(md))
    print(f"[table1] saved {out_path}")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
