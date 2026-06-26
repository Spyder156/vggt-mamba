"""TerraWM-D drift measurement: ATE + RPE-vs-window vs GT.

Resolves the pose-chaos question SEPARATELY from the persistence question:
streams a sequence, integrates predicted deltas into an absolute trajectory
via streaming_forward, aligns to GT with Sim(3) Umeyama, reports:

  - ATE (RMSE / mean / median) on the full trajectory
  - RPE at multiple deltas (1, 5, 10, 50, 100, 200 frames)

If RPE-vs-delta scales sub-linearly (e.g. like sqrt(delta)) → unbiased
random-walk drift; longer-train fixes it. If it scales linearly → systematic
bias; need to fix the loss/grad path. If it explodes → chaotic feedback in
the grid→render→pose loop; structural fix needed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt_mamba.data.tum_rgbd import sync_sequence, intrinsics_for                 # noqa: E402
from vggt_mamba.eval.metrics import absolute_translation_error, relative_pose_error  # noqa: E402
from vggt_mamba.models.aggregators.anchor_pool import cam9_to_pose_w_c              # noqa: E402
from vggt_mamba.models.terrawm_d import build_terrawm_d                             # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--seqs", nargs="+", default=[
        "rgbd_dataset_freiburg3_sitting_xyz",   # in-distribution (in train)
        "rgbd_dataset_freiburg1_room",          # in-distribution (in train), bigger motion
        "rgbd_dataset_freiburg3_walking_xyz",   # OOD (in eval, never seen)
    ])
    p.add_argument("--n-frames", type=int, default=601)
    p.add_argument("--rpe-deltas", type=int, nargs="+", default=[1, 5, 10, 50, 100, 200])
    p.add_argument("--out-dir", type=Path, default=Path("viz/output/terrawm_d_ate_rpe"))
    return p.parse_args()


def load_model(ckpt_path, weights_root):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_terrawm_d(
        cfg["encoder"], str(weights_root),
        n_intraframe_layers=cfg["model"]["n_intraframe_layers"],
        voxel_bounds=tuple(cfg["model"]["voxel_bounds"]),
        voxel_resolution=tuple(cfg["model"]["voxel_resolution"]),
        voxel_feature_dim=cfg["model"]["voxel_feature_dim"],
        n_render_samples=cfg["model"]["n_render_samples"],
        render_near=cfg["model"]["render_near"],
        render_far=cfg["model"]["render_far"],
        bootstrap_hidden=cfg["model"]["bootstrap_hidden"],
        bootstrap_max_depth=cfg["model"]["bootstrap_max_depth"],
        pose_head_hidden=cfg["model"]["pose_head_hidden"],
        pose_max_dt=cfg["model"]["pose_max_dt"],
        pose_max_dq=cfg["model"]["pose_max_dq"],
        unwritten_mask_threshold=cfg["model"]["unwritten_mask_threshold"],
        use_write_confidence=cfg["model"].get("use_write_confidence", False),
        write_confidence_hidden=cfg["model"].get("write_confidence_hidden", 64),
        differentiable_write_geometry=cfg["model"].get("differentiable_write_geometry", False),
    )
    model.load_state_dict(ckpt["model"], strict=False)
    return model.cuda().eval(), cfg


def load_rgb(rec, img_size):
    from PIL import Image
    img = Image.open(rec.rgb_path).convert("RGB").resize((img_size, img_size))
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0
                            ).permute(2, 0, 1).unsqueeze(0).contiguous()


@torch.no_grad()
def stream_sequence(model, recs, K, fov) -> np.ndarray:
    """Stream the sequence through pose head; return (T, 4, 4) predicted abs poses."""
    img_size = model.img_size
    voxel_state = model.init_voxel_state(batch_size=1, device="cuda", dtype=torch.float32)
    # Frame 0 starts at identity in our convention.
    prev_pose_9 = torch.tensor([[0., 0., 0., 0., 0., 0., 1., 1.0, 1.0]],
                                device="cuda", dtype=torch.float32)
    pred_poses_4x4 = []
    for rec in recs:
        rgb = load_rgb(rec, img_size).cuda(non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out, new_abs_9 = model.streaming_forward(rgb, voxel_state, prev_pose_9, K, fov=fov)
        new_abs_T = cam9_to_pose_w_c(new_abs_9)
        pred_poses_4x4.append(new_abs_T[0].float().cpu().numpy())
        prev_pose_9 = new_abs_9.float()
    return np.stack(pred_poses_4x4)


def relativize_to_frame_0(gt_poses: np.ndarray) -> np.ndarray:
    P0_inv = np.linalg.inv(gt_poses[0])
    return np.einsum("ij,njk->nik", P0_inv, gt_poses)


def evaluate_one_seq(model, seq_dir: Path, seq_name: str, n_frames: int,
                      rpe_deltas: list[int]) -> dict:
    img_size = model.img_size
    fx, fy, cx, cy = intrinsics_for(seq_name)
    sx, sy = img_size / 640.0, img_size / 480.0
    K = torch.tensor([[[fx * sx, 0., cx * sx], [0., fy * sy, cy * sy], [0., 0., 1.]]],
                     device="cuda")
    fov = torch.tensor([[1.0, 1.0]], device="cuda")
    recs = sync_sequence(seq_dir)[:n_frames]

    t0 = time.perf_counter()
    pred = stream_sequence(model, recs, K, fov)                                  # (T, 4, 4)
    dt = time.perf_counter() - t0
    gt = np.stack([r.pose_w_c for r in recs])                                    # (T, 4, 4)
    gt_rel = relativize_to_frame_0(gt)                                            # (T, 4, 4) frame-0 at identity

    # ATE: Sim(3)-aligned.
    ate = absolute_translation_error(pred, gt_rel, align="sim3")
    # RPE at each delta.
    rpe = {d: relative_pose_error(pred, gt_rel, delta=d) for d in rpe_deltas}

    pred_t = pred[:, :3, 3]
    gt_t = gt_rel[:, :3, 3]
    return {
        "seq": seq_name,
        "n_frames": int(len(recs)),
        "stream_time_s": float(dt),
        "pred_traj_length_m": float(np.linalg.norm(np.diff(pred_t, axis=0), axis=-1).sum()),
        "gt_traj_length_m": float(np.linalg.norm(np.diff(gt_t, axis=0), axis=-1).sum()),
        "pred_bbox_diag_m": float(np.linalg.norm(pred_t.max(0) - pred_t.min(0))),
        "gt_bbox_diag_m": float(np.linalg.norm(gt_t.max(0) - gt_t.min(0))),
        "ate": ate,
        "rpe": rpe,
        "pred_t": pred_t.tolist(),
        "gt_t": gt_t.tolist(),
    }


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model, cfg = load_model(args.ckpt, args.weights_root)
    print(f"[d-drift] ckpt: {args.ckpt}")
    print(f"[d-drift] seqs: {args.seqs}")
    print(f"[d-drift] RPE deltas: {args.rpe_deltas}")
    print(f"[d-drift] n_frames per seq: {args.n_frames}")

    all_results = {}
    for seq in args.seqs:
        seq_dir = args.data_root / seq
        if not seq_dir.exists():
            print(f"[d-drift] skip {seq} (not found)")
            continue
        print(f"\n[d-drift] === {seq} ===")
        r = evaluate_one_seq(model, seq_dir, seq, args.n_frames, args.rpe_deltas)
        all_results[seq] = r
        print(f"[d-drift]   GT bbox diag: {r['gt_bbox_diag_m']:.3f} m   "
              f"GT traj len: {r['gt_traj_length_m']:.3f} m   ")
        print(f"[d-drift]   Pred bbox diag: {r['pred_bbox_diag_m']:.3f} m   "
              f"Pred traj len: {r['pred_traj_length_m']:.3f} m   ")
        print(f"[d-drift]   ATE (Sim3): RMSE={r['ate']['ate_rmse_m']:.4f} m   "
              f"mean={r['ate']['ate_mean_m']:.4f} m   "
              f"median={r['ate']['ate_median_m']:.4f} m")
        print(f"[d-drift]   RPE (per delta):")
        for d in args.rpe_deltas:
            if d in r["rpe"]:
                rd = r["rpe"][d]
                print(f"[d-drift]     Δ={d:4d}: trans_rmse={rd['rpe_trans_rmse_m']:.4f}m  "
                      f"rot_rmse={rd['rpe_rot_rmse_deg']:.3f}°  n_pairs={rd['n_pairs']}")

    # === Visualizations ===
    n_seq = len(all_results)
    if n_seq == 0:
        print("[d-drift] no seqs evaluated, exiting")
        return

    # 1. RPE growth curve per sequence: trans RMSE vs delta on log-log scale.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for seq, r in all_results.items():
        deltas = list(r["rpe"].keys())
        trans = [r["rpe"][d]["rpe_trans_rmse_m"] for d in deltas]
        rot = [r["rpe"][d]["rpe_rot_rmse_deg"] for d in deltas]
        axes[0].plot(deltas, trans, "o-", label=seq.replace("rgbd_dataset_", ""))
        axes[1].plot(deltas, rot, "o-", label=seq.replace("rgbd_dataset_", ""))
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("delta (frames)"); axes[0].set_ylabel("RPE translation RMSE (m)")
    axes[0].set_title("RPE translation vs delta (log-log)\n"
                       "sqrt(delta) slope = unbiased random walk;\n"
                       "linear slope = systematic bias;\n"
                       "super-linear = chaotic feedback")
    axes[0].grid(alpha=0.3, which="both"); axes[0].legend()
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("delta (frames)"); axes[1].set_ylabel("RPE rotation RMSE (deg)")
    axes[1].set_title("RPE rotation vs delta (log-log)")
    axes[1].grid(alpha=0.3, which="both"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "rpe_growth.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[d-drift]   saved rpe_growth.png")

    # 2. Trajectory plots per seq: xz plane, GT vs predicted (Sim3-aligned).
    fig, axes = plt.subplots(1, n_seq, figsize=(5 * n_seq, 5))
    if n_seq == 1:
        axes = [axes]
    for ax, (seq, r) in zip(axes, all_results.items()):
        pred = np.asarray(r["pred_t"])
        gt = np.asarray(r["gt_t"])
        # Apply Sim3 alignment (recompute, cheap).
        from vggt_mamba.eval.metrics import umeyama_sim3
        s, R, t = umeyama_sim3(pred, gt)
        pred_aligned = s * pred @ R.T + t
        ax.plot(gt[:, 0], gt[:, 2], "o-", color="green", label="GT", markersize=3)
        ax.plot(pred_aligned[:, 0], pred_aligned[:, 2], "o-", color="red",
                label=f"pred (sim3 ATE={r['ate']['ate_rmse_m']:.3f}m)", markersize=3)
        ax.scatter(gt[0, 0], gt[0, 2], color="black", s=80, zorder=5, label="start")
        ax.set_title(seq.replace("rgbd_dataset_", ""))
        ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)"); ax.legend(); ax.grid(alpha=0.3)
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(args.out_dir / "trajectories_xz.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[d-drift]   saved trajectories_xz.png")

    # === Summary JSON ===
    # Strip the non-JSON-serializable aligned trajectory from ate.
    def _clean(r):
        out = {k: v for k, v in r.items() if k not in ("pred_t", "gt_t")}
        out["ate"] = {k: v for k, v in out["ate"].items() if k != "aligned_trajectory_m"}
        return out
    summary = {
        "ckpt": str(args.ckpt),
        "n_frames": args.n_frames,
        "rpe_deltas": args.rpe_deltas,
        "results": {seq: _clean(r) for seq, r in all_results.items()},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[d-drift] saved {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
