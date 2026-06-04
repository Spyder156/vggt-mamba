"""Photometric-mismatch trigger separability — the decisive instrument for the
"build re-grounding now or strengthen photometric first" decision.

The drift-blind check confirmed photometric mismatch is 2.48× higher under
drift than no-drift at matched coverage (strict cov>0.99, p<0.0001). That's
direction-correct and moderately strong. But for re-grounding, what matters
isn't the ratio of means — it's whether the DISTRIBUTIONS separate enough
that a threshold can flag "drifted" frames without firing constantly on
non-drifted noise.

This script:
  1. Loads per_frame.npz from terrawm_d_drift_blind_check_photo/.
  2. Restricts to matched-coverage frames (cov > 0.99).
  3. Bins: HIGH-drift (≥1m), LOW-drift (<0.5m), MEDIUM-drift (boundary).
  4. Plots overlaid histograms of mismatch_rel by bin.
  5. Reports ROC AUC for distinguishing HIGH vs LOW drift via mismatch_rel.
  6. Sweeps thresholds; reports TPR/FPR at the geometric-mean-optimal cut.
  7. Recommends: BUILD (separable enough to be a usable trigger) or
     STRENGTHEN (overlap too large; need stronger signal before re-grounding).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=Path,
                    default=Path("viz/output/terrawm_d_drift_blind_check_photo/per_frame.npz"))
    p.add_argument("--out-dir", type=Path,
                    default=Path("viz/output/terrawm_d_photo_trigger_separability"))
    p.add_argument("--high-drift-m", type=float, default=1.0)
    p.add_argument("--low-drift-m", type=float, default=0.5)
    p.add_argument("--coverage-floor", type=float, default=0.99)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(args.npz)
    disp = data["displacement"]
    cov = data["coverage"]
    mm_rel = data["mismatch_rel"]
    print(f"[separ] loaded {len(disp)} frames from {args.npz}")
    print(f"[separ] coverage histogram (lo/mid/hi):")
    print(f"  cov < 0.5:  {int((cov < 0.5).sum())}")
    print(f"  0.5–0.99:   {int(((cov >= 0.5) & (cov < 0.99)).sum())}")
    print(f"  cov ≥ 0.99: {int((cov >= 0.99).sum())}")

    cov_ok = cov > args.coverage_floor
    high = cov_ok & (disp >= args.high_drift_m)
    low = cov_ok & (disp < args.low_drift_m)
    mid = cov_ok & (disp >= args.low_drift_m) & (disp < args.high_drift_m)
    print(f"\n[separ] at cov > {args.coverage_floor}:")
    print(f"  HIGH-drift (≥{args.high_drift_m}m): n={int(high.sum())}")
    print(f"  MED-drift  ({args.low_drift_m}–{args.high_drift_m}m): n={int(mid.sum())}")
    print(f"  LOW-drift  (<{args.low_drift_m}m): n={int(low.sum())}")

    mm_high = mm_rel[high]
    mm_low = mm_rel[low]
    mm_mid = mm_rel[mid]

    print(f"\n[separ] mismatch_rel distribution by bin:")
    print(f"  HIGH: mean={mm_high.mean():.3f}  std={mm_high.std():.3f}  "
           f"q25={np.quantile(mm_high, 0.25):.3f}  q50={np.median(mm_high):.3f}  "
           f"q75={np.quantile(mm_high, 0.75):.3f}")
    print(f"  LOW:  mean={mm_low.mean():.3f}  std={mm_low.std():.3f}  "
           f"q25={np.quantile(mm_low, 0.25):.3f}  q50={np.median(mm_low):.3f}  "
           f"q75={np.quantile(mm_low, 0.75):.3f}")
    if len(mm_mid) > 0:
        print(f"  MED:  mean={mm_mid.mean():.3f}  std={mm_mid.std():.3f}  "
               f"q50={np.median(mm_mid):.3f}")

    # === ROC AUC: discriminating HIGH (positive) vs LOW (negative) using mismatch_rel ===
    y_true = np.concatenate([np.ones(len(mm_high)), np.zeros(len(mm_low))])
    y_score = np.concatenate([mm_high, mm_low])
    # Sort thresholds descending.
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    n_pos = float(y_true.sum())
    n_neg = float(len(y_true) - n_pos)
    tpr = tps / max(n_pos, 1)
    fpr = fps / max(n_neg, 1)
    # AUC via trapezoidal.
    auc = float(np.trapz(tpr, fpr))
    # Optimal threshold by Youden's J = TPR - FPR.
    j = tpr - fpr
    j_idx = int(np.argmax(j))
    best_thr = float(y_score[order[j_idx]])
    best_tpr = float(tpr[j_idx])
    best_fpr = float(fpr[j_idx])
    print(f"\n[separ] ROC AUC (HIGH vs LOW mismatch_rel as discriminator): {auc:.4f}")
    print(f"[separ] Youden-optimal threshold: {best_thr:.4f}")
    print(f"  TPR (HIGH-drift correctly flagged): {best_tpr:.3f}")
    print(f"  FPR (LOW-drift false-flagged):       {best_fpr:.3f}")
    print(f"  Precision at this threshold: {tps[j_idx]/(tps[j_idx]+fps[j_idx]):.3f}")

    # === Two practical operating points ===
    # 1. FPR ≤ 5% — "rare false flags" (safer for triggered recovery)
    safe_idx = np.where(fpr <= 0.05)[0]
    if len(safe_idx) > 0:
        si = safe_idx[-1]
        print(f"\n[separ] @ FPR ≤ 5% (cautious trigger):  thr={float(y_score[order[si]]):.4f}  "
               f"TPR={float(tpr[si]):.3f}  FPR={float(fpr[si]):.3f}")
    # 2. TPR ≥ 80% — "catch most drift" (aggressive trigger)
    aggr_idx = np.where(tpr >= 0.80)[0]
    if len(aggr_idx) > 0:
        ai = aggr_idx[0]
        print(f"[separ] @ TPR ≥ 80% (aggressive trigger): thr={float(y_score[order[ai]]):.4f}  "
               f"TPR={float(tpr[ai]):.3f}  FPR={float(fpr[ai]):.3f}")

    # === Histogram plot ===
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    # Determine x range from joint distribution (5th/95th).
    joint = np.concatenate([mm_high, mm_low, mm_mid])
    x_lo = float(np.quantile(joint, 0.01))
    x_hi = float(np.quantile(joint, 0.99))
    bins = np.linspace(x_lo, x_hi, 50)
    axes[0].hist(mm_low, bins=bins, alpha=0.5, label=f"LOW drift (<{args.low_drift_m}m), n={len(mm_low)}",
                  color="tab:blue", density=True)
    axes[0].hist(mm_high, bins=bins, alpha=0.5, label=f"HIGH drift (≥{args.high_drift_m}m), n={len(mm_high)}",
                  color="tab:red", density=True)
    if len(mm_mid) > 0:
        axes[0].hist(mm_mid, bins=bins, alpha=0.3, label=f"MED ({args.low_drift_m}–{args.high_drift_m}m), n={len(mm_mid)}",
                      color="tab:gray", density=True)
    axes[0].axvline(best_thr, color="black", linestyle="--", alpha=0.6,
                     label=f"Youden-optimal thr = {best_thr:.3f}")
    axes[0].set_xlabel("photometric mismatch_rel")
    axes[0].set_ylabel("density")
    axes[0].set_title(f"Photometric mismatch_rel by drift bin (cov > {args.coverage_floor})\n"
                       f"AUC = {auc:.3f}   2.48× ratio of means (strict-cov sanity)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    # ROC curve.
    axes[1].plot(fpr, tpr, color="tab:red", linewidth=2)
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.5, label="chance")
    axes[1].scatter([best_fpr], [best_tpr], color="black", s=60, zorder=5,
                     label=f"Youden: TPR={best_tpr:.2f}, FPR={best_fpr:.2f}")
    axes[1].set_xlabel("false-positive rate (LOW-drift falsely flagged)")
    axes[1].set_ylabel("true-positive rate (HIGH-drift correctly flagged)")
    axes[1].set_title(f"ROC — photometric mismatch_rel as drift-trigger\nAUC = {auc:.3f}")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(args.out_dir / "trigger_separability.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # === Recommendation ===
    print(f"\n[separ] === RECOMMENDATION ===")
    if auc >= 0.80 and best_tpr >= 0.7 and best_fpr <= 0.2:
        rec = ("BUILD RE-GROUNDING (b) — signal is clearly separable. AUC ≥ 0.80, "
                f"Youden-optimal achieves TPR {best_tpr:.2f} / FPR {best_fpr:.2f}. "
                f"Use threshold ~{best_thr:.3f} for the trigger; calibrate on a validation "
                f"sequence before deployment.")
    elif auc >= 0.70:
        rec = ("BUILD RE-GROUNDING WITH CAUTION (b) — signal separates moderately. "
                f"AUC {auc:.3f}, Youden TPR {best_tpr:.2f} / FPR {best_fpr:.2f}. Set the "
                f"trigger threshold high (~FPR≤5%) for cautious operation. Re-evaluate after "
                f"deployment whether to push photometric harder later.")
    elif auc >= 0.60:
        rec = ("BORDERLINE — AUC {auc:.3f}. Signal exists but separation is weak. "
                f"Push photometric harder (a) or fuse with another signal (e.g. coverage * "
                f"mismatch) before building re-grounding.")
    else:
        rec = (f"STRENGTHEN FIRST (a) — AUC {auc:.3f}. Distributions overlap too much to "
                f"build a usable trigger. Photometric needs another iteration "
                f"(longer training / multi-scale color / different feature) "
                f"before re-grounding is buildable.")
    print(f"  {rec}")

    import json
    (args.out_dir / "summary.json").write_text(json.dumps({
        "n_high": int(high.sum()), "n_low": int(low.sum()), "n_mid": int(mid.sum()),
        "auc": auc,
        "youden_optimal_threshold": best_thr,
        "youden_tpr": best_tpr,
        "youden_fpr": best_fpr,
        "high_stats": {"mean": float(mm_high.mean()), "std": float(mm_high.std()),
                         "q25": float(np.quantile(mm_high, 0.25)),
                         "q50": float(np.median(mm_high)),
                         "q75": float(np.quantile(mm_high, 0.75))},
        "low_stats":  {"mean": float(mm_low.mean()), "std": float(mm_low.std()),
                         "q25": float(np.quantile(mm_low, 0.25)),
                         "q50": float(np.median(mm_low)),
                         "q75": float(np.quantile(mm_low, 0.75))},
        "recommendation": rec,
    }, indent=2))
    print(f"\n[separ] saved {args.out_dir}/")


if __name__ == "__main__":
    main()
