"""Profile the patch-scan streaming forward to confirm launch-bound signature.

Tests in order:
  1. Per-component synchronized timing (encoder, intraframe, scan, summary, head, dpt).
     Tells us which component(s) dominate the 1210 ms/frame budget.
  2. Inside cross_frame.streaming_step: pure-Python-loop overhead by replacing
     layer.step with a no-op. Bounds the floor of launch overhead.
  3. CUDA-graph A/B: capture streaming_step in a CUDA graph, time again.
     Speedup ratio = launch-cost fraction. If capture fails, report that
     separately — capture failure is not "not launch-bound."
  4. Synthetic-input only (no PIL, no disk) — isolates model compute from I/O.

Out: viz/output/profile_patchscan/profile_report.json (+ printed summary)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vggt_mamba.models.geomamba import build_geomamba  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path,
                   default=Path("experiments/phase3_streaming_patchscan/dinov3_mamba/ckpt_002000.pt"))
    p.add_argument("--weights-root", type=Path, default=Path("/workspace/datasets/weights"))
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--out", type=Path, default=Path("viz/output/profile_patchscan/profile_report.json"))
    return p.parse_args()


def load_model(ckpt_path: Path, weights_root: Path):
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
    )
    model.load_state_dict(ckpt["model"], strict=False)
    return model.cuda().eval(), cfg


def time_op(fn, warmup: int, iters: int) -> tuple[float, float]:
    """Returns (mean_ms, std_ms) over `iters` runs after `warmup`."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    import statistics
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(args.ckpt, args.weights_root)
    H = W = model.img_size
    rng = torch.Generator(device="cuda").manual_seed(0)
    rgb = torch.rand(1, 3, H, W, generator=rng, device="cuda")
    state = model.init_streaming_state(batch_size=1, dtype=torch.bfloat16, device="cuda")

    report: dict = {"config": {"img_size": H, "n_xfm_layers": cfg["model"]["n_xfm_layers"]}}

    # ===== 1. Whole-frame streaming_forward =====
    print("[profile] timing whole-frame streaming_forward...")
    def whole():
        nonlocal state
        with torch.no_grad():
            _, state = model.streaming_forward(rgb, state, frame_idx=0)
    m, s = time_op(whole, args.warmup, args.iters)
    report["whole_frame_ms"] = {"mean": m, "std": s}
    print(f"  whole frame: {m:.1f} ± {s:.1f} ms  ({1000/m:.2f} FPS)")

    # ===== 2. Per-component breakdown (build mid-stage tensors first) =====
    print("\n[profile] per-component breakdown...")
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        # encoder
        def enc_only():
            with torch.no_grad():
                _ = model.encoder(rgb)
        m_enc, _ = time_op(enc_only, args.warmup, args.iters)

        enc_out = model.encoder(rgb)
        patches = enc_out.patches.unsqueeze(0).to(torch.bfloat16)  # (1, 1, P, D)
        P = patches.shape[2]

        def intra_only():
            _ = model.intraframe(patches)
        m_intra, _ = time_op(intra_only, args.warmup, args.iters)

        refined = model.intraframe(patches)
        patch_in = refined.reshape(1, P, model.dim).to(torch.bfloat16)
        # fresh state for the scan timing (don't accumulate from prior calls)
        state_scan = model.init_streaming_state(batch_size=1, dtype=torch.bfloat16, device="cuda")

        def scan_only():
            # mutates state_scan in place via streaming_step; reset each iter.
            nonlocal state_scan
            state_scan = model.init_streaming_state(batch_size=1, dtype=torch.bfloat16, device="cuda")
            _, state_scan = model.cross_frame.streaming_step(patch_in, state_scan)
        m_scan, _ = time_op(scan_only, args.warmup, args.iters)

        state_scan = model.init_streaming_state(batch_size=1, dtype=torch.bfloat16, device="cuda")
        scanned, _ = model.cross_frame.streaming_step(patch_in, state_scan)
        scanned_4d = scanned.unsqueeze(1)  # (1, 1, P, D)

        def summary_only():
            _ = model.summary_pool(scanned_4d)
        m_sum, _ = time_op(summary_only, args.warmup, args.iters)

        state_per_frame = model.summary_pool(scanned_4d)

        def cam_only():
            _ = model.camera_head(state_per_frame[:, :, :model.n_dynamic])
        m_cam, _ = time_op(cam_only, args.warmup, args.iters)

        grid = scanned_4d.reshape(1, -1, model.dim).transpose(1, 2)
        grid = grid.reshape(1, model.dim, model.grid_h, model.grid_w)
        def dpt_only():
            _ = model.dpt(grid)
        m_dpt, _ = time_op(dpt_only, args.warmup, args.iters)

    breakdown = {
        "encoder_ms": m_enc,
        "intraframe_ms": m_intra,
        "cross_frame_streaming_step_ms": m_scan,
        "summary_pool_ms": m_sum,
        "camera_head_ms": m_cam,
        "dpt_ms": m_dpt,
        "sum_ms": m_enc + m_intra + m_scan + m_sum + m_cam + m_dpt,
    }
    report["per_component_ms"] = breakdown
    print(f"  encoder            : {m_enc:7.2f} ms")
    print(f"  intraframe         : {m_intra:7.2f} ms")
    print(f"  cross_frame.scan   : {m_scan:7.2f} ms  <-- per-token loop")
    print(f"  summary_pool       : {m_sum:7.2f} ms")
    print(f"  camera_head        : {m_cam:7.2f} ms")
    print(f"  dpt                : {m_dpt:7.2f} ms")
    print(f"  sum                : {breakdown['sum_ms']:7.2f} ms  (vs whole-frame {m:.2f})")
    print(f"  scan fraction      : {m_scan/m*100:5.1f}% of whole frame")

    # ===== 3. Inside the scan: bound the launch floor with no-op layers =====
    print("\n[profile] launch-overhead floor (replace layer.step with no-op)...")
    layers = model.cross_frame.layers
    original_steps = [layer.step for layer in layers]
    def noop_step(x, conv_s, ssm_s):
        return x, conv_s, ssm_s
    for layer in layers:
        layer.step = noop_step
    try:
        state_noop = model.init_streaming_state(batch_size=1, dtype=torch.bfloat16, device="cuda")
        def scan_noop():
            nonlocal state_noop
            state_noop = model.init_streaming_state(batch_size=1, dtype=torch.bfloat16, device="cuda")
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, state_noop = model.cross_frame.streaming_step(patch_in, state_noop)
        m_noop, _ = time_op(scan_noop, args.warmup, args.iters)
    finally:
        for layer, orig in zip(layers, original_steps):
            layer.step = orig
    report["scan_noop_loop_ms"] = m_noop
    pure_kernel_estimate = m_scan - m_noop
    print(f"  scan with no-op layers (pure Python loop): {m_noop:6.2f} ms")
    print(f"  scan with real layers                   : {m_scan:6.2f} ms")
    print(f"  estimated pure-kernel time              : {pure_kernel_estimate:6.2f} ms")
    print(f"  estimated launch-overhead fraction      : {(1 - pure_kernel_estimate/m_scan)*100:5.1f}%")

    # ===== 4. CUDA-graph A/B on the scan =====
    print("\n[profile] CUDA-graph capture A/B...")
    cuda_graph_result = {}
    try:
        # The scan needs the same input + state each iteration to be capturable.
        # We capture a "step the state once with patch_in" call.
        state_for_graph = model.init_streaming_state(batch_size=1, dtype=torch.bfloat16, device="cuda")
        static_in = patch_in.clone()
        # Warmup pass before capture (allocator + JIT)
        for _ in range(3):
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out, state_for_graph = model.cross_frame.streaming_step(static_in, state_for_graph)
        torch.cuda.synchronize()

        # Capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                static_out, state_for_graph = model.cross_frame.streaming_step(
                    static_in, state_for_graph
                )

        def graph_replay():
            g.replay()
        m_graph, _ = time_op(graph_replay, args.warmup, args.iters)
        speedup = m_scan / m_graph if m_graph > 0 else float("inf")
        cuda_graph_result = {
            "captured": True,
            "graph_replay_ms": m_graph,
            "speedup_vs_loop": speedup,
        }
        print(f"  CUDA-graph replay     : {m_graph:6.2f} ms")
        print(f"  speedup over loop     : {speedup:5.2f}x")
        print(f"  interpretation        : "
              f"{'launch-bound (graph eliminates overhead)' if speedup > 2 else 'NOT clearly launch-bound (graph only modest gain)'}")
    except Exception as e:
        cuda_graph_result = {"captured": False, "error": str(e)}
        print(f"  CUDA graph capture FAILED: {e}")
        print(f"  (capture failure is not 'not launch-bound' — fallback signature is the")
        print(f"   noop-loop floor above. Launch fraction estimate stands.)")
    report["cuda_graph"] = cuda_graph_result

    # ===== 5. Per-layer per-token cost — quick scan to verify linearity =====
    print("\n[profile] sanity: per-(P × n_layers) cost ratio...")
    expected_calls = P * len(layers)
    print(f"  P tokens                = {P}")
    print(f"  n cross-frame layers    = {len(layers)}")
    print(f"  layer.step() calls/frame = {expected_calls}")
    print(f"  scan time per call       = {m_scan*1000/expected_calls:.2f} us")
    report["scan_per_step_us"] = m_scan * 1000 / expected_calls
    report["P_tokens"] = P
    report["n_xfm_layers"] = len(layers)

    args.out.write_text(json.dumps(report, indent=2))
    print(f"\n[profile] report -> {args.out}")


if __name__ == "__main__":
    main()
