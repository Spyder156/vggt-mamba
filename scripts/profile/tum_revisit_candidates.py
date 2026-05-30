"""Enumerate TUM sequences with actual return-to-region behavior.

For each sequence with a groundtruth.txt, compute a 'revisit signal':
  - For each frame index i, find the minimum distance from cam position(i)
    to cam position(j) for all j with j < i - 100 (i.e. at least 100 frames
    ago — recent neighbors don't count as revisits).
  - 'revisit_score' = number of frames with min-distance-to-old-frame < R_revisit
    (default 0.3 m, room-scale revisit).
  - Also report trajectory total length and bounding-box diagonal.

A useful revisit sequence has a substantial fraction of frames triggering
the revisit condition. A pure xyz-translation sequence (no looping) will
have ~zero revisits.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path,
                   default=Path("/home/raghav/workspace/CVPR/vggt-mamba/data/tum_rgbd"))
    p.add_argument("--r-revisit", type=float, default=0.3,
                   help="min distance to count as a revisit (m)")
    p.add_argument("--gap-frames", type=int, default=100,
                   help="minimum frame gap for a revisit to count")
    return p.parse_args()


def load_traj(gt_path: Path) -> np.ndarray:
    rows = []
    for line in gt_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        ts, tx, ty, tz = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
        rows.append([ts, tx, ty, tz])
    return np.array(rows)


def revisit_score(traj: np.ndarray, r_revisit: float, gap_frames: int) -> dict:
    """traj: (N, 4) [ts, tx, ty, tz]. Returns scalar stats."""
    T = traj.shape[0]
    if T < gap_frames + 10:
        return {"n_frames": T, "revisits": 0, "revisit_frac": 0.0,
                "min_revisit_dist_m": 0.0, "traj_length_m": 0.0,
                "bbox_diag_m": 0.0, "frame_rate_hz": 0.0}
    pos = traj[:, 1:4]                                          # (T, 3)
    n_revisits = 0
    min_dist_seen = float("inf")
    # Brute force, but trajectories are <10K frames so it's fine.
    for i in range(gap_frames + 1, T):
        old_pos = pos[: i - gap_frames]
        if old_pos.shape[0] == 0:
            continue
        d = np.linalg.norm(old_pos - pos[i], axis=-1)
        d_min = d.min()
        if d_min < r_revisit:
            n_revisits += 1
        if d_min < min_dist_seen:
            min_dist_seen = d_min
    # Trajectory length.
    diffs = np.diff(pos, axis=0)
    traj_len = float(np.linalg.norm(diffs, axis=-1).sum())
    # Bounding box.
    bbox = pos.max(axis=0) - pos.min(axis=0)
    bbox_diag = float(np.linalg.norm(bbox))
    # Frame rate.
    dts = np.diff(traj[:, 0])
    fps = float(1.0 / dts[dts > 0].mean()) if (dts > 0).any() else 0.0
    return {
        "n_frames": T,
        "revisits": int(n_revisits),
        "revisit_frac": float(n_revisits) / T,
        "min_revisit_dist_m": float(min_dist_seen),
        "traj_length_m": traj_len,
        "bbox_diag_m": bbox_diag,
        "frame_rate_hz": fps,
    }


def main():
    args = parse_args()
    seqs = sorted([d for d in args.data_root.iterdir() if d.is_dir()])
    print(f"[tum-revisit] r_revisit={args.r_revisit}m  gap_frames={args.gap_frames}")
    print(f"[tum-revisit] scanning {len(seqs)} sequences in {args.data_root}")
    print()
    print(f"{'sequence':50s} {'N':>6s} {'fps':>5s} {'len(m)':>7s} {'bbox(m)':>8s} "
          f"{'rvst':>5s} {'frac':>6s} {'min_d(m)':>9s}")
    print("-" * 105)
    results = []
    for seq_dir in seqs:
        gt_path = seq_dir / "groundtruth.txt"
        if not gt_path.exists():
            continue
        traj = load_traj(gt_path)
        if traj.shape[0] == 0:
            continue
        stats = revisit_score(traj, args.r_revisit, args.gap_frames)
        stats["name"] = seq_dir.name
        results.append(stats)
        print(f"{seq_dir.name:50s} {stats['n_frames']:6d} {stats['frame_rate_hz']:5.1f} "
              f"{stats['traj_length_m']:7.2f} {stats['bbox_diag_m']:8.2f} "
              f"{stats['revisits']:5d} {stats['revisit_frac']:6.3f} {stats['min_revisit_dist_m']:9.4f}")

    # Sort by revisit fraction.
    print()
    print("=== ranked by revisit fraction ===")
    for r in sorted(results, key=lambda r: -r["revisit_frac"])[:5]:
        print(f"  {r['name']:50s}  revisit_frac={r['revisit_frac']:.3f}  "
              f"min_dist={r['min_revisit_dist_m']:.4f}m  len={r['traj_length_m']:.2f}m")


if __name__ == "__main__":
    main()
