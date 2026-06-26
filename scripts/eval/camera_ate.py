"""Camera ATE / RPE on TUM held-out sequences via streaming inference.

Streams the model frame-by-frame, collects predicted [tx,ty,tz,qx,qy,qz,qw]
camera per frame, converts to 4x4 world-from-camera, and reports:
  - ATE (Sim(3) Umeyama alignment, RMSE in meters)
  - RPE (relative pose error over delta=1 frame, translation RMSE in m,
    rotation RMSE in degrees)

Run:
    ./docker/run.sh python scripts/eval/camera_ate.py \\
        --ckpt experiments/phase3_streaming_patchscan/dinov3_mamba/ckpt_002000.pt \\
        --seqs rgbd_dataset_freiburg3_sitting_xyz rgbd_dataset_freiburg3_walking_xyz \\
        --out viz/output/paper_figures_patchscan/ate_report.json
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

from vggt_mamba.data.tum_rgbd import sync_sequence, _quat_to_rot, intrinsics_for  # noqa: E402
from vggt_mamba.eval.metrics import absolute_translation_error, relative_pose_error  # noqa: E402
from vggt_mamba.models.terrawm import build_terrawm                # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--seqs", nargs="+", required=True,
                   help="TUM sequence directory names to evaluate (held-out only)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="cap per sequence (default: all)")
    p.add_argument("--out", type=Path, default=Path("viz/output/ate_report.json"))
    p.add_argument("--use-cuda-graphs", action="store_true",
                   help="route streaming_forward through GraphedStreamingScan (Speed-B)")
    return p.parse_args()


def load_model(ckpt_path: Path, weights_root: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_terrawm(
        cfg["encoder"], str(weights_root),
        n_intraframe_layers=cfg["model"]["n_intraframe_layers"],
        n_summary_tokens=cfg["model"]["n_summary_tokens"],
        n_summary_dynamic=cfg["model"].get("n_summary_dynamic"),
        n_xfm_layers=cfg["model"]["n_xfm_layers"],
        d_state=cfg["model"]["d_state"],
        bidirectional=False,
        aggregator_name="mamba",
        track_enabled=cfg["model"]["track_enabled"],
        max_frames=ckpt["model"]["frame_embed"].shape[1],
        dense_residual_to_patches=cfg["model"].get("dense_residual_to_patches", True),
        predict_next_latent=cfg["model"].get("predict_next_latent", False),
        ema_momentum=cfg["model"].get("ema_momentum", 0.99),
        cross_frame_target=cfg["model"].get("cross_frame_target", "summary"),
        use_anchor_pool=cfg["model"].get("use_anchor_pool", False),
        n_anchors=cfg["model"].get("n_anchors", 32),
        n_anchor_writes=cfg["model"].get("n_anchor_writes", 4),
        anchor_match_threshold=cfg["model"].get("anchor_match_threshold", 0.5),
        delta_pose=cfg["model"].get("delta_pose", cfg["model"].get("terrawm", False)),
        motion_enc_freqs=cfg["model"].get("motion_enc_freqs", cfg["model"].get("terrawm_motion_freqs", 64)),
    )
    msg = model.load_state_dict(ckpt["model"], strict=False)
    non_enc = [k for k in msg.missing_keys if not k.startswith("encoder.")]
    print(f"[ate] ckpt step={ckpt['step']}  missing(non-encoder)={len(non_enc)}")
    return model.cuda().eval(), cfg


def load_rgb(rec, img_size: int) -> torch.Tensor:
    from PIL import Image
    img = Image.open(rec.rgb_path).convert("RGB").resize((img_size, img_size))
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous()


def cam9_to_pose_w_c(cam9: np.ndarray) -> np.ndarray:
    """[tx,ty,tz,qx,qy,qz,qw,fovx,fovy] -> 4x4 world-from-camera."""
    t = cam9[:3]
    q = cam9[3:7]
    q = q / max(np.linalg.norm(q), 1e-12)
    R = _quat_to_rot(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def eval_one_sequence(model, cfg, data_root: Path, seq: str,
                      max_frames: int | None, use_cuda_graphs: bool = False) -> dict:
    from vggt_mamba.models.aggregators import GraphedStreamingScan
    recs = sync_sequence(data_root / seq)
    if max_frames is not None:
        recs = recs[:max_frames]
    img_size = {"vjepa": 384, "dinov2": 518, "dinov3": 512}[cfg["encoder"]]
    state = model.init_streaming_state(batch_size=1, dtype=torch.bfloat16, device="cuda",
                                       use_cuda_graphs=use_cuda_graphs)
    # Anchor pool init + per-sequence intrinsics (only needed if model uses anchor pool).
    anchor_state = None
    K_intrinsics = None
    if getattr(model, "use_anchor_pool", False):
        anchor_state = model.init_anchor_state(batch_size=1, dtype=torch.bfloat16, device="cuda")
        fx, fy, cx, cy = intrinsics_for(seq)
        sx, sy = img_size / 640.0, img_size / 480.0
        K_intrinsics = torch.tensor(
            [[[fx * sx, 0., cx * sx], [0., fy * sy, cy * sy], [0., 0., 1.]]],
            device="cuda",
        )

    pred_poses = []
    gt_poses = []
    raw_cam9 = []                  # for TerraWM, the raw delta predictions before integration
    terrawm = getattr(model, "delta_pose", False)
    t0 = time.perf_counter()
    for i, rec in enumerate(recs):
        rgb = load_rgb(rec, img_size).cuda(non_blocking=True)
        preds, state = model.streaming_forward(
            rgb, state, frame_idx=i,
            anchor_state=anchor_state, K_intrinsics=K_intrinsics,
        )
        cam9 = preds["camera"][0, 0].float().cpu().numpy()
        if terrawm:
            raw_cam9.append(cam9)            # delta — integrate after the streaming loop
        else:
            pred_poses.append(cam9_to_pose_w_c(cam9))
        gt_poses.append(rec.pose_w_c)
        if (i + 1) % 200 == 0:
            print(f"  {seq}  frame {i+1}/{len(recs)}")
    dt = time.perf_counter() - t0

    if terrawm:
        # External delta integrator: start from GT[0] (so global frame matches
        # GT for free; Sim3 alignment would absorb any starting-pose offset
        # anyway but starting at GT[0] makes the per-frame error story cleaner).
        from vggt_mamba.models.pose_utils import integrate_deltas_to_absolute
        deltas_t = torch.tensor(np.stack(raw_cam9), dtype=torch.float32)     # (T, 9)
        initial = torch.tensor(recs[0].pose_w_c, dtype=torch.float32)
        abs_T = integrate_deltas_to_absolute(deltas_t[:, :7], initial)        # (T, 4, 4)
        pred_poses = abs_T.numpy()
    else:
        pred_poses = np.stack(pred_poses)
    gt_poses = np.stack(gt_poses)

    ate_sim3 = absolute_translation_error(pred_poses, gt_poses, align="sim3")
    ate_se3 = absolute_translation_error(pred_poses, gt_poses, align="se3")
    rpe_1 = relative_pose_error(pred_poses, gt_poses, delta=1)
    rpe_10 = relative_pose_error(pred_poses, gt_poses, delta=10)

    result = {
        "seq": seq,
        "n_frames": len(recs),
        "wall_s": dt,
        "fps": len(recs) / dt,
        "ate_sim3": {k: v for k, v in ate_sim3.items() if k != "aligned_trajectory_m"},
        "ate_se3": {k: v for k, v in ate_se3.items() if k != "aligned_trajectory_m"},
        "rpe_delta1": rpe_1,
        "rpe_delta10": rpe_10,
    }
    return result


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(args.ckpt, args.weights_root)

    if args.use_cuda_graphs:
        print("[ate] using CUDA graphs (Speed-B)")
    all_results = []
    for seq in args.seqs:
        print(f"\n[ate] {seq}")
        result = eval_one_sequence(model, cfg, args.data_root, seq, args.max_frames,
                                   use_cuda_graphs=args.use_cuda_graphs)
        all_results.append(result)
        a = result["ate_sim3"]
        r = result["rpe_delta1"]
        print(f"  ATE (Sim3) RMSE: {a['ate_rmse_m']:.4f} m   median: {a['ate_median_m']:.4f} m")
        print(f"  RPE δ=1   trans: {r['rpe_trans_rmse_m']:.4f} m   rot: {r['rpe_rot_rmse_deg']:.3f} deg")

    # Aggregate mean across sequences (matches how SLAM benchmarks report).
    if all_results:
        agg = {
            "ate_sim3_rmse_m_mean": float(np.mean([r["ate_sim3"]["ate_rmse_m"] for r in all_results])),
            "ate_sim3_rmse_m_median": float(np.median([r["ate_sim3"]["ate_rmse_m"] for r in all_results])),
            "rpe_trans_rmse_m_mean": float(np.mean([r["rpe_delta1"]["rpe_trans_rmse_m"] for r in all_results])),
            "rpe_rot_rmse_deg_mean": float(np.mean([r["rpe_delta1"]["rpe_rot_rmse_deg"] for r in all_results])),
            "n_sequences": len(all_results),
        }
    else:
        agg = {}

    payload = {
        "ckpt": str(args.ckpt),
        "per_sequence": all_results,
        "aggregate_across_sequences": agg,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\n[ate] saved {args.out}")
    if agg:
        print(f"[ate] mean ATE (Sim3): {agg['ate_sim3_rmse_m_mean']:.4f} m across "
              f"{agg['n_sequences']} held-out sequences")


if __name__ == "__main__":
    main()
