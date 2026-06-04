"""Temporal structure of the photometric trigger — does it fire reliably
DURING drift events even if per-frame AUC is only 0.66?"""
from pathlib import Path
import numpy as np

data = np.load("viz/output/terrawm_d_drift_blind_check_photo/per_frame.npz")
disp = data["displacement"]
cov = data["coverage"]
mm_rel = data["mismatch_rel"]

threshold = 0.334  # Youden-optimal from separability script

# Find sustained drift events: contiguous runs where displacement >= 1m.
high_drift = disp >= 1.0
runs = []
i = 0
n = len(disp)
while i < n:
    if high_drift[i]:
        start = i
        while i < n and high_drift[i]:
            i += 1
        runs.append((start, i))
    else:
        i += 1

print(f"Found {len(runs)} contiguous high-drift events (disp >= 1m)")
print(f"Total high-drift frames: {sum(e - s for s, e in runs)} / {n}")

triggered_events = 0
mean_lag_to_first_fire = []
events_silent = []
for s, e in runs:
    if e - s < 3:
        continue
    fires = np.where(mm_rel[s:e] >= threshold)[0]
    if len(fires) > 0:
        triggered_events += 1
        mean_lag_to_first_fire.append(int(fires[0]))
    else:
        events_silent.append((s, e, e - s))

print(f"\nWith threshold {threshold:.3f}:")
print(f"  Events triggered at least once: {triggered_events} / {len([r for r in runs if r[1]-r[0] >= 3])}")
if mean_lag_to_first_fire:
    print(f"  Mean lag to first fire within event (frames): {float(np.mean(mean_lag_to_first_fire)):.1f}")
    print(f"  Max lag: {int(max(mean_lag_to_first_fire))}")
print(f"  Silent events (never triggered): {len(events_silent)}")
for s, e, d in events_silent[:10]:
    avg_disp = float(disp[s:e].mean())
    avg_mm = float(mm_rel[s:e].mean())
    print(f"    f={s}..{e-1} ({d} frames)  mean_disp={avg_disp:.2f}m  mean_mm_rel={avg_mm:.3f}")

# Operational FPR: in low-drift contiguous runs, how often does the trigger spuriously fire?
low_drift = disp < 0.5
low_runs = []
i = 0
while i < n:
    if low_drift[i]:
        s = i
        while i < n and low_drift[i]:
            i += 1
        low_runs.append((s, i))
    else:
        i += 1

low_triggered = 0
for s, e in low_runs:
    if e - s < 3:
        continue
    if (mm_rel[s:e] >= threshold).any():
        low_triggered += 1
print(f"\nLow-drift (disp < 0.5m) events: {len([r for r in low_runs if r[1]-r[0] >= 3])}")
print(f"  Spurious triggers within them: {low_triggered}")
