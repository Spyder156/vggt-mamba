"""Anchor pool inference-time diagnostics on a trained ckpt.

Three numbers per frame, aggregated across the stream:
  1. MATCH RATE: fraction of (patch, anchor) pairs with score > threshold,
     and absolute count above threshold per frame. Tells us if reads are
     firing at all.
  2. CORRECTION-MLP OUTPUT MAGNITUDE: ||corrected_pose - coarse_pose||₂
     for translation (m) and quaternion (unitless). Tells us if the MLP is
     producing any non-trivial output at inference.
  3. CONSISTENCY LOSS (inference): same formula as training, reprojection
     error of stored anchors through the corrected pose, weighted by score.
     Tells us if the anchor pool's stored geometry is even self-consistent
     with what the model is currently predicting.

This is the read-path measurement we should have done before exp 2b.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt_mamba.data.tum_rgbd import sync_sequence, _quat_to_rot, intrinsics_for  # noqa: E402
from vggt_mamba.eval.metrics import project_points_to_pixels                       # noqa: E402
from vggt_mamba.models.aggregators.anchor_pool import cam9_to_pose_w_c             # noqa: E402
from vggt_mamba.models.geomamba import build_geomamba                              # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--seq", default="rgbd_dataset_freiburg3_sitting_xyz")
    p.add_argument("--max-frames", type=int, default=1216)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="match-rate counting threshold (matches training default)")
    p.add_argument("--out", type=Path, default=Path("viz/output/anchor_diag.json"))
    return p.parse_args()


def load_model(ckpt_path, weights_root):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_geomamba(
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
    )
    model.load_state_dict(ckpt["model"], strict=False)
    return model.cuda().eval(), cfg


def load_rgb(rec, img_size):
    from PIL import Image
    img = Image.open(rec.rgb_path).convert("RGB").resize((img_size, img_size))
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0
                            ).permute(2, 0, 1).unsqueeze(0).contiguous()


@torch.no_grad()
def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model, cfg = load_model(args.ckpt, args.weights_root)
    assert getattr(model, "use_anchor_pool", False), "ckpt is not an anchor-pool model"

    recs = sync_sequence(args.data_root / args.seq)[:args.max_frames]
    img_size = {"vjepa": 384, "dinov2": 518, "dinov3": 512}[cfg["encoder"]]
    fx, fy, cx, cy = intrinsics_for(args.seq)
    sx, sy = img_size / 640.0, img_size / 480.0
    K = torch.tensor([[[fx * sx, 0., cx * sx], [0., fy * sy, cy * sy], [0., 0., 1.]]],
                     device="cuda")

    state = model.init_streaming_state(use_cuda_graphs=False)  # graphs would skip diag outputs
    anchor_state = model.init_anchor_state(batch_size=1, dtype=torch.bfloat16, device="cuda")

    per_frame = []
    print(f"[diag] streaming {len(recs)} frames of {args.seq}")
    print(f"[diag] threshold={args.threshold}  K_a={model.anchor_pool.K_a}  "
          f"P={model.grid_h * model.grid_w}")

    for i, rec in enumerate(recs):
        rgb = load_rgb(rec, img_size).cuda(non_blocking=True)
        out, state = model.streaming_forward(
            rgb, state, frame_idx=i, anchor_state=anchor_state, K_intrinsics=K,
        )
        if "scores" not in out:
            continue
        # ---- 1. MATCH RATE ----
        scores = out["scores"][0].float().cpu().numpy()       # (P, K_a)
        valid = out["anchor_valid_pre_write"][0].cpu().numpy() # (K_a,)
        # Mask invalid anchors out of the count (they always have score=0).
        scores_valid = scores[:, valid]                        # (P, n_valid)
        n_valid_anchors = int(valid.sum())
        if n_valid_anchors == 0:
            n_matches = 0
            match_rate = 0.0
            mean_score_active = 0.0
            max_score_active = 0.0
        else:
            above = scores_valid > args.threshold
            n_matches = int(above.sum())
            match_rate = float(above.mean())                    # fraction of (P × n_valid) above
            mean_score_active = float(scores_valid.mean())
            max_score_active = float(scores_valid.max())

        # ---- 2. CORRECTION MAGNITUDE ----
        coarse = out["camera_coarse"][0].float().cpu().numpy()  # (9,)
        corrected = out["camera"][0, 0].float().cpu().numpy()   # (9,)
        dt_norm = float(np.linalg.norm(corrected[:3] - coarse[:3]))
        dq_norm = float(np.linalg.norm(corrected[3:7] - coarse[3:7]))

        # ---- 3. INFERENCE-TIME CONSISTENCY LOSS ----
        if n_valid_anchors > 0:
            corrected_t = torch.from_numpy(corrected).unsqueeze(0).to("cuda")  # (1, 9)
            pose_w_c = cam9_to_pose_w_c(corrected_t)                            # (1, 4, 4)
            anchor_pos = out["anchor_positions_pre_write"][0].float()           # (K_a, 3)
            patch_pixel = out["patch_pixel"][0].float()                         # (P, 2)
            valid_mask = out["anchor_valid_pre_write"][0]                       # (K_a,)
            ap_valid = anchor_pos[valid_mask]                                   # (n_valid, 3)
            anchor_pixels, in_front = project_points_to_pixels(
                ap_valid.unsqueeze(0), pose_w_c, K
            )
            anchor_pixels = anchor_pixels[0]                                    # (n_valid, 2)
            in_front = in_front[0]                                              # (n_valid,)
            scores_t = out["scores"][0, :, valid_mask].float()                  # (P, n_valid)
            # diff (P, n_valid, 2) = patch_pixel (P, 1, 2) - anchor_pixels (1, n_valid, 2)
            diff = patch_pixel.unsqueeze(1) - anchor_pixels.unsqueeze(0)
            pix_err_sq = (diff ** 2).sum(dim=-1)                                # (P, n_valid)
            pix_err_norm = pix_err_sq / (img_size ** 2)
            score_mask = (scores_t > args.threshold).float()
            in_front_mask = in_front.float().unsqueeze(0)                       # (1, n_valid)
            mask = score_mask * in_front_mask
            weighted = scores_t * pix_err_norm * mask
            n_v = mask.sum().clamp_min(1.0)
            consistency = float((weighted.sum() / n_v).cpu())
            mean_pix_err_above_thresh = float(
                (pix_err_sq * mask).sum() / mask.sum().clamp_min(1.0)
            ) if mask.sum() > 0 else 0.0
            mean_pix_err_above_thresh = float(np.sqrt(mean_pix_err_above_thresh))
        else:
            consistency = 0.0
            mean_pix_err_above_thresh = 0.0

        per_frame.append({
            "frame": i,
            "n_valid_anchors": n_valid_anchors,
            "n_matches_above_thresh": n_matches,
            "match_rate_among_valid": match_rate,
            "mean_score_among_valid": mean_score_active,
            "max_score_among_valid": max_score_active,
            "dt_correction_m": dt_norm,
            "dq_correction": dq_norm,
            "consistency_loss_norm": consistency,
            "mean_pix_err_above_thresh_px": mean_pix_err_above_thresh,
        })

        if (i + 1) % 200 == 0 or i < 5:
            r = per_frame[-1]
            print(f"  frame {i+1:4d}/{len(recs)}  "
                  f"valid_anc={r['n_valid_anchors']:3d}  "
                  f"matches={r['n_matches_above_thresh']:7d}  "
                  f"rate={r['match_rate_among_valid']:.4f}  "
                  f"max_s={r['max_score_among_valid']:.3f}  "
                  f"|dt|={r['dt_correction_m']:.5f}m  "
                  f"|dq|={r['dq_correction']:.5f}  "
                  f"cons={r['consistency_loss_norm']:.6f}  "
                  f"pix_err={r['mean_pix_err_above_thresh_px']:.1f}px")

    # Aggregate over frames where the pool is non-empty.
    fired = [r for r in per_frame if r["n_valid_anchors"] > 0]

    def stats(field):
        v = np.array([r[field] for r in fired])
        return {
            "mean": float(v.mean()),
            "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)),
            "min": float(v.min()),
            "max": float(v.max()),
        }

    summary = {
        "seq": args.seq,
        "n_frames_with_anchors": len(fired),
        "n_frames_total": len(per_frame),
        "K_a": int(model.anchor_pool.K_a),
        "P": int(model.grid_h * model.grid_w),
        "threshold": args.threshold,
        "match_rate_among_valid": stats("match_rate_among_valid"),
        "n_matches_above_thresh": stats("n_matches_above_thresh"),
        "max_score_among_valid": stats("max_score_among_valid"),
        "mean_score_among_valid": stats("mean_score_among_valid"),
        "dt_correction_m": stats("dt_correction_m"),
        "dq_correction": stats("dq_correction"),
        "consistency_loss_norm": stats("consistency_loss_norm"),
        "mean_pix_err_above_thresh_px": stats("mean_pix_err_above_thresh_px"),
    }

    print("\n[diag] === AGGREGATE OVER STREAM ===")
    print(f"  frames with non-empty pool: {summary['n_frames_with_anchors']}/{summary['n_frames_total']}")
    print(f"  match rate among valid anchors (target: not ≈0):")
    print(f"    median: {summary['match_rate_among_valid']['median']:.4f}")
    print(f"    p10 / p90: {summary['match_rate_among_valid']['p10']:.4f} / {summary['match_rate_among_valid']['p90']:.4f}")
    print(f"  max score among valid (target: not ≈0):")
    print(f"    median: {summary['max_score_among_valid']['median']:.4f}")
    print(f"  correction magnitude:")
    print(f"    |Δt|  median: {summary['dt_correction_m']['median']:.5f} m  (target: non-zero if matches firing)")
    print(f"    |Δt|  p90:    {summary['dt_correction_m']['p90']:.5f} m")
    print(f"    |Δq|  median: {summary['dq_correction']['median']:.5f}")
    print(f"  consistency loss (inference, normalized):")
    print(f"    median: {summary['consistency_loss_norm']['median']:.6f}")
    print(f"    p90:    {summary['consistency_loss_norm']['p90']:.6f}")
    print(f"  mean pixel error on matched pairs:")
    print(f"    median: {summary['mean_pix_err_above_thresh_px']['median']:.2f} px")
    print(f"    p90:    {summary['mean_pix_err_above_thresh_px']['p90']:.2f} px")

    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\n[diag] saved -> {args.out}")


if __name__ == "__main__":
    main()
