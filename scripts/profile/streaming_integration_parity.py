"""End-to-end Speed-B integration parity test.

Runs TerraWM.streaming_forward both ways (use_cuda_graphs=False vs True) on
the same N-frame input sequence and compares per-frame outputs (camera +
pointmap). Stronger than the layer-level parity because it catches any
integration bug introduced by the streaming_forward dispatch logic.

Verdict: STRICT_PASS if all per-frame diffs are exactly 0.0 (output AND state).
Falls back to register-noise (bf16: atol=1e-3) only on real divergence and
flags it as a correctness issue to investigate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt_mamba.models.terrawm import build_terrawm  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path,
                   default=Path("experiments/phase3_streaming_patchscan/dinov3_mamba/ckpt_002000.pt"))
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--n-frames", type=int, default=20)
    return p.parse_args()


def load_model(ckpt_path, weights_root):
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


def main():
    args = parse_args()
    model, _ = load_model(args.ckpt, args.weights_root)
    H = W = model.img_size

    rng = torch.Generator(device="cuda").manual_seed(0)
    frames = [torch.rand(1, 3, H, W, generator=rng, device="cuda") for _ in range(args.n_frames)]

    # Path A: loop streaming_forward
    state_a = model.init_streaming_state(use_cuda_graphs=False)
    cam_a, pmap_a = [], []
    for i, rgb in enumerate(frames):
        out, state_a = model.streaming_forward(rgb, state_a, frame_idx=i)
        cam_a.append(out["camera"].clone())
        pmap_a.append(out["pointmap"].clone())

    # Path B: graphed streaming_forward
    state_b = model.init_streaming_state(use_cuda_graphs=True)
    cam_b, pmap_b = [], []
    for i, rgb in enumerate(frames):
        out, state_b = model.streaming_forward(rgb, state_b, frame_idx=i)
        cam_b.append(out["camera"].clone())
        pmap_b.append(out["pointmap"].clone())

    print(f"[integration-parity] n_frames={args.n_frames}")
    print(f"[integration-parity] per-frame max divergence (camera | pointmap):")
    max_cam_diff = 0.0
    max_pmap_diff = 0.0
    for i in range(args.n_frames):
        c = (cam_a[i].float() - cam_b[i].float()).abs().max().item()
        p = (pmap_a[i].float() - pmap_b[i].float()).abs().max().item()
        max_cam_diff = max(max_cam_diff, c)
        max_pmap_diff = max(max_pmap_diff, p)
        if i < 3 or i % 5 == 0 or i == args.n_frames - 1:
            print(f"  frame {i:3d}: cam_max={c:.3e}  pmap_max={p:.3e}")

    # Also check state at the end. Graphed path returns the wrapper; loop path
    # returns the list of dicts. Unwrap to compare like-for-like.
    from vggt_mamba.models.aggregators import GraphedStreamingScan
    b_list = state_b.state if isinstance(state_b, GraphedStreamingScan) else state_b
    state_a_conv = state_a[0]["conv"]
    state_b_conv = b_list[0]["conv"]
    state_a_ssm = state_a[0]["ssm"]
    state_b_ssm = b_list[0]["ssm"]
    conv_diff = (state_a_conv.float() - state_b_conv.float()).abs().max().item()
    ssm_diff = (state_a_ssm.float() - state_b_ssm.float()).abs().max().item()

    print()
    print(f"[integration-parity] === verdict ===")
    print(f"  max camera divergence over {args.n_frames} frames:    {max_cam_diff:.3e}")
    print(f"  max pointmap divergence over {args.n_frames} frames:  {max_pmap_diff:.3e}")
    print(f"  final-state conv divergence:                          {conv_diff:.3e}")
    print(f"  final-state ssm  divergence:                          {ssm_diff:.3e}")

    REGISTER_NOISE = 1e-3
    worst = max(max_cam_diff, max_pmap_diff, conv_diff, ssm_diff)
    if worst == 0.0:
        print(f"  STRICT bit-perfect integration parity.")
        return "STRICT_PASS"
    elif worst < REGISTER_NOISE:
        print(f"  Within bf16 register noise (< {REGISTER_NOISE:.0e}). Acceptable.")
        return "REGISTER_NOISE_PASS"
    else:
        print(f"  *** FAILURE *** divergence exceeds register noise.")
        return "FAIL"


if __name__ == "__main__":
    print(main())
