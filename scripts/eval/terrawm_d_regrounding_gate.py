"""Re-grounding gate test — primary verdict + co-guards.

Compares the BASELINE streaming run (no re-grounding applied) to the
RE-GROUNDING run on the SAME sequence, with the SAME trigger frames.

PRIMARY METRIC (corrected design): paired baseline-vs-RG ΔATE at matched
post-fire frames. For each fire at frame t:
    ΔATE_t = mean(ATE_withRG[t+1..t+W]) − mean(ATE_baseline[t+1..t+W])
Where W = post-fire window (default 20). Both runs collapse the same way up
to the fire; the divergence after is attributable to re-grounding.

(The within-RG before/after design conflated correction with whatever the
trajectory was doing anyway — explicitly NOT used here.)

CO-GUARDS:
  1. Geometric reasonableness: ≥70% of fires must have
       dist(refined, GT) < dist(initial, GT).
  2. Photometric mismatch drops at fire frame: ≥90% of fires must have
       mm_rel(after) < mm_rel(before). If <90%, the optimizer is broken.
  3. Scene-state ablation co-guard (separate run, not this script):
       post-reset Δt ≥ 0.10m on the streaming-with-re-grounding path.

Verdict buckets (LOCKED):
  STRONG_POSITIVE: mean ΔATE ≤ -0.5m, paired-t p<0.05
  PARTIAL:         -0.5m < mean ΔATE ≤ -0.1m
  NULL:            -0.1m < mean ΔATE < +0.1m  (or p>0.10)
  HARMFUL:         mean ΔATE ≥ +0.1m

CLEAN-GRID DISAMBIGUATION (only run on PARTIAL/NULL):
  A NULL/PARTIAL result is ambiguous: weak correction OR good correction
  capped by polluted grid. Run --clean-grid-disambig to re-stream the held-out
  sequence with the grid built from GT POSES (clean), then run re-grounding
  at the same fire frames against this clean grid. If ΔATE jumps to
  STRONG_POSITIVE on the clean grid, the bottleneck is grid pollution
  (needs un-write/re-write follow-up), not the correction itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as scistats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir", type=Path, required=True,
                   help="Output dir of terrawm_d_regrounding_stream.py --baseline")
    p.add_argument("--regrounding-dir", type=Path, required=True,
                   help="Output dir of terrawm_d_regrounding_stream.py (no --baseline)")
    p.add_argument("--window", type=int, default=20,
                   help="Post-fire ΔATE window (frames).")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    base_pf = np.load(args.baseline_dir / "per_frame.npz")
    rg_pf = np.load(args.regrounding_dir / "per_frame.npz")
    base_disp = base_pf["displacement"]
    rg_disp = rg_pf["displacement"]
    base_fires = json.loads((args.baseline_dir / "fires.json").read_text())
    rg_fires = json.loads((args.regrounding_dir / "fires.json").read_text())
    n = min(len(base_disp), len(rg_disp))

    # Pair fires by frame: the trigger logic is deterministic given the
    # mismatch threshold + cooldown. Both runs see the same cooldown timing
    # for frames before the first applied refinement; after that, the RG run's
    # subsequent fires can shift relative to the baseline because the trajectories
    # diverge. For the matched-frame ΔATE, we use the BASELINE's fire times
    # (since the baseline trajectory is the unperturbed reference).
    base_fire_frames = [f["frame"] for f in base_fires]
    rg_fire_frames = [f["frame"] for f in rg_fires]
    rg_applied_frames = [f["frame"] for f in rg_fires if f.get("applied")]
    print(f"[gate] baseline fires: {len(base_fires)}  "
          f"RG fires: {len(rg_fires)}  applied: {len(rg_applied_frames)}")

    # Use the BASELINE fire frames as the matched anchor points.
    # For each, compute post-fire-window ATE in both runs.
    paired_deltas = []
    per_fire = []
    for t in base_fire_frames:
        if t + args.window >= n:
            continue
        win_base = base_disp[t + 1 : t + 1 + args.window]
        win_rg = rg_disp[t + 1 : t + 1 + args.window]
        delta = float(win_rg.mean() - win_base.mean())                 # negative = RG helps
        paired_deltas.append(delta)
        per_fire.append({
            "frame": int(t),
            "delta_ATE_m": delta,
            "base_window_mean": float(win_base.mean()),
            "rg_window_mean": float(win_rg.mean()),
        })

    paired_deltas = np.array(paired_deltas)
    if len(paired_deltas) > 0:
        mean_delta = float(paired_deltas.mean())
        median_delta = float(np.median(paired_deltas))
        # Paired t-test against 0 (the null: re-grounding has no effect)
        t_stat, p_val = scistats.ttest_1samp(paired_deltas, 0.0)
        # Wilcoxon signed-rank for non-Gaussian robustness
        try:
            w_stat, w_p = scistats.wilcoxon(paired_deltas)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
    else:
        mean_delta = median_delta = t_stat = p_val = w_stat = w_p = float("nan")

    print(f"\n[gate] === PRIMARY METRIC (matched post-fire ΔATE) ===")
    print(f"  Fires paired: {len(paired_deltas)} / {len(base_fire_frames)} baseline fires")
    print(f"  Mean ΔATE per fire:   {mean_delta:+.4f} m  "
           f"(negative = re-grounding reduces drift)")
    print(f"  Median ΔATE per fire: {median_delta:+.4f} m")
    print(f"  Paired t-test:        t={t_stat:+.3f}  p={p_val:.4f}")
    print(f"  Wilcoxon signed-rank: W={w_stat}  p={w_p:.4f}" if w_p == w_p else "")

    # === CO-GUARDS ===
    print(f"\n[gate] === CO-GUARDS ===")
    # Co-guard 1: ≥70% of applied fires have dist(refined, GT) < dist(initial, GT).
    applied = [f for f in rg_fires if f.get("applied")]
    if applied:
        toward_gt = [
            f for f in applied
            if np.linalg.norm(np.array(f["final_pose_t"]) - np.array(f["gt_pos"]))
            < np.linalg.norm(np.array(f["initial_pose_t"]) - np.array(f["gt_pos"]))
        ]
        cg1 = len(toward_gt) / len(applied)
    else:
        cg1 = float("nan")
    cg1_pass = cg1 >= 0.70 if not np.isnan(cg1) else False
    print(f"  Co-guard 1 (geometric reasonableness): "
           f"{len(toward_gt) if applied else 0}/{len(applied)} = {cg1:.2%}  "
           f"({'PASS ≥70%' if cg1_pass else 'FAIL <70%'})")

    # Co-guard 2: ≥90% of applied fires have mm_rel_after < mm_rel_before.
    if applied:
        mm_drops = sum(1 for f in applied if f["mm_rel_after"] < f["mm_rel_before"])
        cg2 = mm_drops / len(applied)
    else:
        cg2 = float("nan")
    cg2_pass = cg2 >= 0.90 if not np.isnan(cg2) else False
    print(f"  Co-guard 2 (photo mismatch drops):      "
           f"{mm_drops if applied else 0}/{len(applied)} = {cg2:.2%}  "
           f"({'PASS ≥90%' if cg2_pass else 'FAIL <90%'})")
    print(f"  Co-guard 3 (ablation): RUN SEPARATELY — terrawm_d_regrounding_ablation.py")

    # === VERDICT ===
    coguards_ok = cg1_pass and cg2_pass
    if not coguards_ok:
        verdict = (f"CO-GUARD FAIL — verdict NOT trustworthy. "
                    f"cg1={cg1:.2%} (need ≥70%), cg2={cg2:.2%} (need ≥90%). "
                    f"Don't read the primary metric until co-guards pass.")
        bucket = "CO_GUARD_FAIL"
    elif np.isnan(mean_delta) or len(paired_deltas) < 3:
        verdict = f"INSUFFICIENT_DATA — only {len(paired_deltas)} paired fires."
        bucket = "INSUFFICIENT"
    elif mean_delta <= -0.5 and p_val < 0.05:
        verdict = (f"STRONG_POSITIVE — re-grounding reduces drift reliably "
                    f"(mean ΔATE {mean_delta:+.3f}m, p={p_val:.4f}). "
                    f"Ship; plan a bundled retrain with re-grounding ON in training.")
        bucket = "STRONG_POSITIVE"
    elif mean_delta <= -0.1:
        verdict = (f"PARTIAL — helps but not enough to ship alone "
                    f"(mean ΔATE {mean_delta:+.3f}m, p={p_val:.4f}). "
                    f"Run clean-grid disambiguation: is the correction capped by polluted grid?")
        bucket = "PARTIAL"
    elif mean_delta >= 0.1:
        verdict = (f"HARMFUL — re-grounding INCREASES drift "
                    f"(mean ΔATE {mean_delta:+.3f}m, p={p_val:.4f}). "
                    f"Revert immediately; do NOT enable in any downstream path.")
        bucket = "HARMFUL"
    else:
        verdict = (f"NULL — trigger fires don't move the needle "
                    f"(mean ΔATE {mean_delta:+.3f}m, p={p_val:.4f}). "
                    f"Run clean-grid disambiguation: weak correction or polluted grid?")
        bucket = "NULL"

    print(f"\n[gate] === VERDICT ===\n  {verdict}\n  [bucket: {bucket}]")
    args.out.write_text(json.dumps({
        "n_paired_fires": len(paired_deltas),
        "mean_delta_ATE_m": mean_delta,
        "median_delta_ATE_m": median_delta,
        "t_stat": float(t_stat) if not np.isnan(t_stat) else None,
        "p_value": float(p_val) if not np.isnan(p_val) else None,
        "wilcoxon_p": float(w_p) if w_p == w_p else None,
        "coguard_1_geom_pass_frac": cg1,
        "coguard_2_mm_drop_frac": cg2,
        "coguards_ok": coguards_ok,
        "verdict": verdict,
        "bucket": bucket,
        "per_fire": per_fire,
    }, indent=2))
    print(f"\n[gate] saved {args.out}")


if __name__ == "__main__":
    main()
