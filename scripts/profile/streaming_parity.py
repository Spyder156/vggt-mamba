"""Speed-B parity test: loop streaming_step vs GraphedStreamingScan.

Verifies bit-perfect equivalence (B should be the same kernels in the same
order — only the launch mechanism differs). Per the equivalence-test
discipline saved in memory, checks all three:
  (a) output equivalence per-frame
  (b) carried-state equivalence (conv + ssm) per-frame
  (c) multi-step divergence over 50 frames — does any drift grow?

Tolerance: strict allclose (atol=0). Any deviation past floating-point register
noise indicates a bug, not a tolerance to widen. If strict fails, falls back to
register-noise tolerances and FLAGS this prominently — that becomes a correctness
issue to investigate before chunk-mode work begins.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt_mamba.models.aggregators import CrossFrameMamba, GraphedStreamingScan  # noqa: E402
from vggt_mamba.models.terrawm import build_terrawm                            # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path,
                   default=Path("experiments/phase3_streaming_patchscan/dinov3_mamba/ckpt_002000.pt"))
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--n-frames", type=int, default=50)
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
    )
    model.load_state_dict(ckpt["model"], strict=False)
    return model.cuda().eval(), cfg


def main() -> None:
    args = parse_args()
    model, cfg = load_model(args.ckpt, args.weights_root)
    cross_frame: CrossFrameMamba = model.cross_frame
    D = model.dim
    P = (model.img_size // 16) ** 2  # patches per frame for DINOv3 ViT-L/16
    print(f"[parity] D={D}  P={P}  n_layers={cross_frame.n_layers}  n_frames={args.n_frames}")

    # Synthesize a deterministic per-frame input.
    rng = torch.Generator(device="cuda").manual_seed(0)
    inputs = [torch.randn(1, P, D, generator=rng, device="cuda", dtype=torch.bfloat16)
              for _ in range(args.n_frames)]

    # --- Path A: loop streaming_step ---
    state_a = cross_frame.init_streaming_state(batch_size=1, dtype=torch.bfloat16, device="cuda")
    outs_a, conv_a, ssm_a = [], [], []
    for x in inputs:
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            o, state_a = cross_frame.streaming_step(x, state_a)
        outs_a.append(o.clone())
        # snapshot state per layer 0 (the most-mutated representative)
        conv_a.append(state_a[0]["conv"].clone())
        ssm_a.append(state_a[0]["ssm"].clone())

    # --- Path B: GraphedStreamingScan ---
    gs = GraphedStreamingScan(cross_frame, batch_size=1, K=P, dim=D, dtype=torch.bfloat16)
    gs.capture()
    outs_b, conv_b, ssm_b = [], [], []
    for x in inputs:
        o, state_b = gs.step(x)
        outs_b.append(o.clone())
        conv_b.append(state_b[0]["conv"].clone())
        ssm_b.append(state_b[0]["ssm"].clone())

    # --- Compare ---
    def diff_stats(a, b, label):
        d = (a.float() - b.float()).abs()
        return f"{label}: max={d.max().item():.3e}  mean={d.mean().item():.3e}"

    print("\n[parity] per-frame divergence (output | conv-state | ssm-state):")
    any_drift = False
    max_out_diff = 0.0
    max_state_diff = 0.0
    for i in range(args.n_frames):
        out_d = (outs_a[i].float() - outs_b[i].float()).abs().max().item()
        conv_d = (conv_a[i].float() - conv_b[i].float()).abs().max().item()
        ssm_d = (ssm_a[i].float() - ssm_b[i].float()).abs().max().item()
        max_out_diff = max(max_out_diff, out_d)
        max_state_diff = max(max_state_diff, conv_d, ssm_d)
        if i < 5 or i % 10 == 0 or i == args.n_frames - 1:
            print(f"  frame {i:3d}: out_max={out_d:.3e}  conv_max={conv_d:.3e}  ssm_max={ssm_d:.3e}")

    # Drift detection: does the per-frame divergence grow over time?
    out_diffs = [(outs_a[i].float() - outs_b[i].float()).abs().max().item() for i in range(args.n_frames)]
    # Simple linear-fit slope of out_diff vs frame index.
    import numpy as np
    x = np.arange(args.n_frames, dtype=float)
    y = np.asarray(out_diffs)
    if y.std() > 0:
        slope = np.polyfit(x, y, 1)[0]
    else:
        slope = 0.0
    drift_growth_per_frame = slope

    print()
    print(f"[parity] === verdict ===")
    print(f"  max output divergence over {args.n_frames} frames: {max_out_diff:.3e}")
    print(f"  max state  divergence over {args.n_frames} frames: {max_state_diff:.3e}")
    print(f"  drift growth (slope of out_diff over frames):     {drift_growth_per_frame:+.3e}/frame")

    # Decision: strict equivalence vs register-noise vs failure.
    STRICT = 0.0
    BF16_REGISTER_NOISE = 1e-3
    if max_out_diff == STRICT and max_state_diff == STRICT:
        print(f"  STRICT bit-perfect parity achieved.  Graph captured the same compute.")
        verdict = "STRICT_PASS"
    elif max_out_diff < BF16_REGISTER_NOISE and max_state_diff < BF16_REGISTER_NOISE:
        print(f"  Within bf16 register noise (< {BF16_REGISTER_NOISE:.0e}). Acceptable.")
        verdict = "REGISTER_NOISE_PASS"
    else:
        print(f"  *** FAILURE *** divergence exceeds register noise.")
        print(f"      Investigate before trusting Speed-B as the chunk-mode baseline.")
        verdict = "FAIL"

    if drift_growth_per_frame > 1e-6:
        print(f"  *** WARNING *** drift is growing per frame ({drift_growth_per_frame:.3e}/frame).")
        print(f"      Over 2000 frames that's ~{drift_growth_per_frame * 2000:.3e} cumulative.")
        verdict += "_WITH_DRIFT"

    return verdict


if __name__ == "__main__":
    print(main())
