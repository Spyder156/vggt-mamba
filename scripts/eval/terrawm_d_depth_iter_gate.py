"""Gate for the A1/A2 depth-iteration tests.

Compares a baseline run (no iteration) against an iteration run (A1 or A2)
at MATCHED FRAMES. The matched-frame design avoids the within-run collapse-
confound that bit the prior re-grounding gate: both runs collapse the same
way through the un-iterated portions of the trajectory; the divergence is
attributable to iteration.

PRIMARY METRIC: paired baseline-vs-iteration ATE at matched event frames.
For each event frame t in the iteration run, compute:
    ΔATE_t = mean(disp_iter[t+1..t+W]) − mean(disp_base[t+1..t+W])
W = post-event window (default 20). Negative = iteration helps.

CO-GUARD 1 (THE LOAD-BEARING CHECK): for each event, did refinement move the
pose TOWARD GT? Across all events:
    cg1 = #events where disp_after < disp_before / total events
Photometric re-grounding got cg1 = 41% (CO_GUARD_FAIL). Depth-iteration must
clear ≥60% (ideally ≥70%) to count as a valid corrective signal.

CO-GUARD 2 (sanity, not gating): does refinement reduce the depth-mismatch
proxy? Implicitly satisfied if refinement converges; not separately reported.

Verdict buckets (LOCKED):
  STRONG: cg1 ≥ 70% AND mean ΔATE ≤ -0.5m, paired p<0.05
  PARTIAL: cg1 in [60%, 70%) AND mean ΔATE ≤ -0.1m
  WEAK: cg1 60-70% but ΔATE not improving — iteration moves toward GT but not enough
  FAIL: cg1 < 60% OR mean ΔATE ≥ 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as scistats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--iter-dir", type=Path, required=True,
                    help="A1 or A2 output directory")
    p.add_argument("--label", required=True, help="A1 or A2 label for the report")
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    base_disp = np.load(args.baseline_dir / "per_frame.npz")["displacement"]
    iter_disp = np.load(args.iter_dir / "per_frame.npz")["displacement"]
    # Events.json: A1 has one per frame; A2 has one per fire.
    events = json.loads((args.iter_dir / "events.json").read_text())
    n = min(len(base_disp), len(iter_disp))
    print(f"[gate-{args.label}] baseline frames: {len(base_disp)}  "
           f"iter frames: {len(iter_disp)}  events: {len(events)}")

    # === CO-GUARD 1: did refinement move toward GT? ===
    applied_events = [e for e in events if e.get("status") == "applied"]
    toward_gt = sum(1 for e in applied_events if e.get("toward_gt"))
    cg1 = toward_gt / max(len(applied_events), 1)
    print(f"\n[gate-{args.label}] === CO-GUARD 1 (load-bearing) ===")
    print(f"  Applied events: {len(applied_events)} / {len(events)}")
    print(f"  Toward-GT: {toward_gt}/{len(applied_events)} = {cg1:.2%}")
    print(f"  Reference: photometric re-grounding cg1 = 41% (FAILED)")
    print(f"  Pass criteria: cg1 ≥ 60% (PARTIAL) or ≥ 70% (STRONG)")

    # === PRIMARY METRIC: matched-frame paired ΔATE ===
    # Sample event frames evenly to avoid over-weighting hot zones in A1
    # (where every frame is an event). Use up to ~50 event frames for the
    # paired test to keep statistics manageable.
    if len(applied_events) > 100:
        idx = np.linspace(0, len(applied_events) - 1, 100).astype(int)
        sample_events = [applied_events[i] for i in idx]
    else:
        sample_events = applied_events
    print(f"\n[gate-{args.label}] === PRIMARY METRIC (matched post-event ΔATE, W={args.window}) ===")
    print(f"  Sampling {len(sample_events)} events for the paired test")
    paired_deltas = []
    for e in sample_events:
        t = e["frame"]
        if t + args.window >= n:
            continue
        win_base = base_disp[t + 1 : t + 1 + args.window]
        win_iter = iter_disp[t + 1 : t + 1 + args.window]
        paired_deltas.append(float(win_iter.mean() - win_base.mean()))
    paired_deltas = np.array(paired_deltas)
    if len(paired_deltas) >= 3:
        mean_delta = float(paired_deltas.mean())
        median_delta = float(np.median(paired_deltas))
        t_stat, p_val = scistats.ttest_1samp(paired_deltas, 0.0)
        try:
            w_stat, w_p = scistats.wilcoxon(paired_deltas)
            wilcoxon_p = float(w_p)
        except ValueError:
            wilcoxon_p = float("nan")
        print(f"  Paired n: {len(paired_deltas)}")
        print(f"  Mean ΔATE per event: {mean_delta:+.4f} m   (negative = iter helps)")
        print(f"  Median ΔATE per event: {median_delta:+.4f} m")
        print(f"  Paired t-test: t={t_stat:+.3f}  p={p_val:.4f}")
        print(f"  Wilcoxon: p={wilcoxon_p:.4f}")
    else:
        mean_delta = median_delta = t_stat = p_val = wilcoxon_p = float("nan")
        print(f"  Insufficient paired events ({len(paired_deltas)}).")

    # === VERDICT ===
    print(f"\n[gate-{args.label}] === VERDICT ===")
    if cg1 < 0.60:
        bucket = "FAIL"
        verdict = (f"FAIL — cg1 {cg1:.0%} < 60%. Depth iteration doesn't move toward GT "
                    f"reliably enough. Same family failure as photometric (41%); deeper "
                    f"geometric-query rethink earned.")
    elif np.isnan(mean_delta) or mean_delta >= 0:
        bucket = "WEAK"
        verdict = (f"WEAK — cg1 {cg1:.0%} passes but ΔATE = {mean_delta:+.4f}m doesn't "
                    f"improve. Refinement moves toward GT but not enough to beat baseline "
                    f"at matched frames. Possibly stable behavior on a collapsing trajectory.")
    elif cg1 >= 0.70 and mean_delta <= -0.5 and p_val < 0.05:
        bucket = "STRONG"
        verdict = (f"STRONG — cg1 {cg1:.0%} AND ΔATE {mean_delta:+.3f}m (p={p_val:.4f}). "
                    f"Depth iteration is the corrective mechanism. Build the training-time "
                    f"integration: pose head trained with this iteration loop on top.")
    else:
        bucket = "PARTIAL"
        verdict = (f"PARTIAL — cg1 {cg1:.0%}, ΔATE {mean_delta:+.3f}m (p={p_val:.4f}). "
                    f"Depth iteration helps but doesn't break the ceiling. Likely useful as "
                    f"a re-grounding correction (A2 case) but not enough to drive the long-"
                    f"horizon collapse.")
    print(f"  {verdict}")
    print(f"  [bucket: {bucket}]")

    args.out.write_text(json.dumps({
        "label": args.label, "n_events": len(events),
        "n_applied": len(applied_events),
        "n_paired": len(paired_deltas),
        "cg1_toward_gt_frac": cg1,
        "mean_delta_ate_m": mean_delta,
        "median_delta_ate_m": median_delta,
        "t_stat": float(t_stat) if not np.isnan(t_stat) else None,
        "p_value": float(p_val) if not np.isnan(p_val) else None,
        "wilcoxon_p": float(wilcoxon_p) if not np.isnan(wilcoxon_p) else None,
        "verdict": verdict, "bucket": bucket,
    }, indent=2))
    print(f"\n[gate-{args.label}] saved {args.out}")


if __name__ == "__main__":
    main()
