"""Extract the cure-monitor trajectory from the photometric retrain's TB logs."""
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator(
    "experiments/phase4_terrawm_d_seed0_photometric/dinov3_terrawm_d/tb",
    size_guidance={event_accumulator.SCALARS: 0},
)
ea.Reload()
tags = ea.Tags().get("scalars", [])
t4 = [t for t in tags if t.startswith("tier4/")]
print(f"Tier-4 tags found: {len(t4)}")

targets = [
    "tier4/pose_err_corr_with_photo_mismatch_rel",
    "tier4/pose_err_corr_with_mismatch_rel",
    "tier4/pose_dt_corr_with_photo_mismatch_rel",
    "tier4/pose_dt_corr_with_mismatch_rel",
    "tier4/pose_photo_mismatch_rel_mean",
    "tier4/pose_mismatch_rel_mean",
    "tier4/pose_dt_mag_mean",
    "tier4/render_coverage",
    "tier4/grid_mass_total",
    "tier4/pose_err_corr_with_coverage",
]
for tag in targets:
    if tag not in tags:
        print(f"\n{tag}: NOT FOUND")
        continue
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    vals = [e.value for e in events]
    if not steps:
        continue
    n = len(steps)
    indices = sorted(set([0, n // 8, n // 4, n // 2, 3 * n // 4, 7 * n // 8, n - 1]))
    indices = [i for i in indices if i < n]
    print(f"\n{tag}:")
    for i in indices:
        v = vals[i]
        if v != v:
            print(f"  step {steps[i]:5d}: NaN")
        else:
            print(f"  step {steps[i]:5d}: {v:+.4f}")

# Locate first sign-cross of the cure monitor.
tag = "tier4/pose_err_corr_with_photo_mismatch_rel"
if tag in tags:
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    vals = [e.value for e in events]
    first_pos = None
    first_neg = None
    for s, v in zip(steps, vals):
        if v != v:
            continue
        if first_neg is None and v < 0:
            first_neg = (s, v)
        if first_pos is None and v > 0:
            first_pos = (s, v)
    print(f"\n[cure crossing] first negative: {first_neg}")
    print(f"[cure crossing] first positive: {first_pos}")
    last_window = [(s, v) for s, v in zip(steps, vals) if s >= 7000 and v == v]
    if last_window:
        avg = sum(v for _, v in last_window) / len(last_window)
        print(f"[cure crossing] mean step 7000..end (n={len(last_window)}): {avg:+.4f}")
