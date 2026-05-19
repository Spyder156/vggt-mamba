"""Saveable-state demo — the most paper-memorable property.

Three runs over the same 1000-frame video:

  A. CONTINUOUS  — single streaming pass frames 0..999. Reference outputs.
  B. SPLIT       — stream frames 0..499, save Mamba state to disk, exit.
                   Then load state on a fresh process/model, stream 500..999.
                   Outputs after frame 500 should ~match A's outputs after frame 500.
  C. STATELESS   — fresh model on frames 500..999 with zero state (no history).
                   Outputs degrade because the model never saw frames 0..499.

The diff (B - A) should be near zero (modulo bf16). The diff (C - A) should be
larger — proves the saved state actually carries scene info.

Output:
  viz/output/phase3_state_demo/state_dump.pt              ← the 3 MB scene state
  viz/output/phase3_state_demo/state_demo.png             ← side-by-side abs-rel curves
  viz/output/phase3_state_demo/state_demo.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt_mamba.data.tum_rgbd import sync_sequence              # noqa: E402
from vggt_mamba.models.geomamba import build_geomamba            # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("/workspace/datasets/tum_rgbd"))
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--ckpt", type=Path,
                   default=Path("experiments/phase3_streaming/dinov3_mamba/ckpt_002000.pt"))
    p.add_argument("--seq", default="rgbd_dataset_freiburg3_sitting_xyz")
    p.add_argument("--n-frames", type=int, default=1000)
    p.add_argument("--split-at", type=int, default=500)
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).resolve().parents[2] / "viz/output/phase3_state_demo")
    return p.parse_args()


def load_model(ckpt_path: Path, weights_root: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_geomamba(
        cfg["encoder"], str(weights_root),
        n_intraframe_layers=cfg["model"]["n_intraframe_layers"],
        n_summary_tokens=cfg["model"]["n_summary_tokens"],
        n_xfm_layers=cfg["model"]["n_xfm_layers"],
        d_state=cfg["model"]["d_state"],
        bidirectional=False,
        aggregator_name="mamba",
        track_enabled=cfg["model"]["track_enabled"],
        max_frames=ckpt["model"]["frame_embed"].shape[1],
    )
    model.load_state_dict(ckpt["model"], strict=False)
    return model.cuda().eval(), cfg


def save_state(state: list[dict], path: Path) -> int:
    """Persist state to disk via torch.save. Returns bytes written."""
    payload = {"state": [{k: v.cpu() for k, v in s.items()} for s in state]}
    torch.save(payload, path)
    return path.stat().st_size


def load_state(path: Path, device: str = "cuda") -> list[dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    return [{k: v.to(device) for k, v in s.items()} for s in payload["state"]]


def load_frame(rec, img_size: int):
    from PIL import Image
    img = Image.open(rec.rgb_path).convert("RGB").resize((img_size, img_size))
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).contiguous()
    d = np.asarray(
        Image.open(rec.depth_path).resize((img_size, img_size), Image.NEAREST),
        dtype=np.float32,
    ) / 5000.0
    return rgb_t, d, (d > 0.01) & (d < 8.0)


def abs_rel(pred_z: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    if valid.sum() == 0:
        return float("nan")
    p = np.maximum(pred_z[valid], 1e-6)
    g = np.maximum(gt[valid], 1e-6)
    return float(np.mean(np.abs(p - g) / g))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    recs = sync_sequence(args.data_root / args.seq)[:args.n_frames]
    print(f"[demo] loaded {len(recs)} frames from {args.seq}")

    model, cfg = load_model(args.ckpt, args.weights_root)
    img_size = {"vjepa": 384, "dinov2": 518, "dinov3": 512}[cfg["encoder"]]

    # ---- Run A: continuous reference ----
    print("\n[demo] A. continuous pass 0..N-1 (reference)")
    state_a = model.init_streaming_state(batch_size=1, dtype=torch.bfloat16, device="cuda")
    abs_rel_a = []
    pred_z_a = []
    for i, r in enumerate(recs):
        rgb, gt_d, valid = load_frame(r, img_size)
        rgb = rgb.cuda(non_blocking=True)
        preds, state_a = model.streaming_forward(rgb, state_a, frame_idx=i)
        pz = preds["pointmap"][0, 0, 2].float().cpu().numpy()
        abs_rel_a.append(abs_rel(pz, gt_d, valid))
        pred_z_a.append(pz)
        if (i + 1) % 200 == 0:
            print(f"  A frame {i+1}/{len(recs)}  abs_rel={abs_rel_a[-1]:.3f}")

    # ---- Run B: split + save/load ----
    print(f"\n[demo] B. stream 0..{args.split_at-1}, save state, reload, stream {args.split_at}..N-1")
    state_b = model.init_streaming_state(batch_size=1, dtype=torch.bfloat16, device="cuda")
    for i in range(args.split_at):
        rgb, _, _ = load_frame(recs[i], img_size)
        rgb = rgb.cuda(non_blocking=True)
        _, state_b = model.streaming_forward(rgb, state_b, frame_idx=i)

    state_path = args.out_dir / "state_dump.pt"
    state_bytes = save_state(state_b, state_path)
    print(f"[demo]   saved state to {state_path}  ({state_bytes/1024:.1f} KB)")

    # Wipe + reload on a fresh model instance to prove it's truly transportable.
    del model, state_b
    torch.cuda.empty_cache()
    model_b, _ = load_model(args.ckpt, args.weights_root)
    state_b = load_state(state_path)
    print(f"[demo]   reloaded into fresh model instance")

    abs_rel_b = [float("nan")] * args.split_at
    pred_z_b = list(pred_z_a[:args.split_at])  # placeholders; we only care 500..999
    for i in range(args.split_at, len(recs)):
        rgb, gt_d, valid = load_frame(recs[i], img_size)
        rgb = rgb.cuda(non_blocking=True)
        preds, state_b = model_b.streaming_forward(rgb, state_b, frame_idx=i)
        pz = preds["pointmap"][0, 0, 2].float().cpu().numpy()
        abs_rel_b.append(abs_rel(pz, gt_d, valid))
        pred_z_b.append(pz)
        if (i + 1) % 200 == 0:
            print(f"  B frame {i+1}/{len(recs)}  abs_rel={abs_rel_b[-1]:.3f}")

    # ---- Run C: stateless control (zero state, no history) ----
    print(f"\n[demo] C. stateless control — zero state, stream {args.split_at}..N-1")
    del model_b
    torch.cuda.empty_cache()
    model_c, _ = load_model(args.ckpt, args.weights_root)
    state_c = model_c.init_streaming_state(batch_size=1, dtype=torch.bfloat16, device="cuda")
    abs_rel_c = [float("nan")] * args.split_at
    pred_z_c = list(pred_z_a[:args.split_at])
    for i in range(args.split_at, len(recs)):
        rgb, gt_d, valid = load_frame(recs[i], img_size)
        rgb = rgb.cuda(non_blocking=True)
        # Frame idx STARTS from 0 because the model has no prior context.
        preds, state_c = model_c.streaming_forward(rgb, state_c,
                                                   frame_idx=i - args.split_at)
        pz = preds["pointmap"][0, 0, 2].float().cpu().numpy()
        abs_rel_c.append(abs_rel(pz, gt_d, valid))
        pred_z_c.append(pz)

    # ---- Numerical diff: B vs A (saved/loaded state) and C vs A (no state) ----
    diff_BA = []
    diff_CA = []
    for i in range(args.split_at, len(recs)):
        diff_BA.append(float(np.mean(np.abs(pred_z_b[i] - pred_z_a[i]))))
        diff_CA.append(float(np.mean(np.abs(pred_z_c[i] - pred_z_a[i]))))

    summary = {
        "abs_rel_continuous_mean": float(np.nanmean(abs_rel_a[args.split_at:])),
        "abs_rel_saved_state_mean": float(np.nanmean(abs_rel_b[args.split_at:])),
        "abs_rel_stateless_mean": float(np.nanmean(abs_rel_c[args.split_at:])),
        "depth_diff_BA_mean_m": float(np.mean(diff_BA)),
        "depth_diff_CA_mean_m": float(np.mean(diff_CA)),
        "state_size_bytes": state_bytes,
    }
    (args.out_dir / "state_demo.json").write_text(json.dumps({
        "summary": summary,
        "abs_rel_a": abs_rel_a, "abs_rel_b": abs_rel_b, "abs_rel_c": abs_rel_c,
    }, indent=2))

    print("\n[demo] === results ===")
    print(f"  state size on disk: {state_bytes/1024:.1f} KB")
    print(f"  Abs-Rel mean over frames {args.split_at}..{len(recs)-1}:")
    print(f"    A continuous:     {summary['abs_rel_continuous_mean']:.4f}")
    print(f"    B saved+loaded:   {summary['abs_rel_saved_state_mean']:.4f}")
    print(f"    C stateless ctrl: {summary['abs_rel_stateless_mean']:.4f}")
    print(f"  Mean per-pixel depth diff to A (lower = state preserved info):")
    print(f"    B (saved/loaded): {summary['depth_diff_BA_mean_m']*1000:.2f} mm")
    print(f"    C (stateless):    {summary['depth_diff_CA_mean_m']*1000:.2f} mm")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    frames = np.arange(len(recs))
    axes[0].plot(frames, abs_rel_a, "-", color="tab:blue", linewidth=2, label="A: continuous")
    axes[0].plot(frames[args.split_at:], abs_rel_b[args.split_at:], "--",
                 color="tab:green", linewidth=2, label="B: saved + reloaded state")
    axes[0].plot(frames[args.split_at:], abs_rel_c[args.split_at:], ":",
                 color="tab:red", linewidth=2,
                 label="C: stateless control (no history)")
    axes[0].axvline(args.split_at, color="black", linestyle=":", alpha=0.4,
                    label=f"save/load at frame {args.split_at}")
    axes[0].set_ylabel("Abs-Rel")
    axes[0].set_title(
        f"Saveable scene state demo · {args.seq} · {len(recs)} frames\n"
        f"State size on disk: {state_bytes/1024:.1f} KB · scene encoded as a 3 MB tensor",
        fontsize=11,
    )
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(frames[args.split_at:], np.array(diff_BA) * 1000, "-",
                 color="tab:green", linewidth=2, label="B vs A (saved/loaded ≈ continuous)")
    axes[1].plot(frames[args.split_at:], np.array(diff_CA) * 1000, "-",
                 color="tab:red", linewidth=2, label="C vs A (no history)")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("mean per-pixel depth Δ to A (mm)")
    axes[1].set_title("Per-frame depth difference relative to the continuous reference")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out_dir / "state_demo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[demo] saved viz → {args.out_dir}")


if __name__ == "__main__":
    main()
