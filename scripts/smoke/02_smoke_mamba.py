"""Phase 0 smoke test: Mamba-2 forward + backward on toy input.

Instantiates a single Mamba-2 block, runs a forward + backward pass on a
synthetic batch, and prints kernel timing. No data required.

Pass = forward + backward complete without errors on the available GPU
(Blackwell sm_120 in our case), and timing is sane (< 100 ms for the
shapes below).

Usage:
    python scripts/smoke/02_smoke_mamba.py
"""

from __future__ import annotations

import time

import torch


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke-mamba] device={device}")
    if device == "cuda":
        print(f"[smoke-mamba] gpu={torch.cuda.get_device_name(0)} "
              f"cap={torch.cuda.get_device_capability(0)}")

    from mamba_ssm import Mamba2  # type: ignore

    B, T, D = 2, 128, 512
    print(f"[smoke-mamba] (B, T, D) = ({B}, {T}, {D})")

    # state_size=128 matches Mamba-2 SSD default; head_dim=64 is the larger
    # head dimension supported by SSD.
    block = Mamba2(d_model=D, d_state=128, headdim=64).to(device)
    x = torch.randn(B, T, D, device=device, requires_grad=True)

    # warmup
    for _ in range(3):
        y = block(x)
        y.sum().backward()
        block.zero_grad(set_to_none=True)
        x.grad = None

    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()
    y = block(x)
    loss = y.sum()
    loss.backward()
    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.perf_counter() - t0

    assert y.shape == x.shape, f"unexpected output shape {y.shape} vs {x.shape}"
    assert not torch.isnan(y).any(), "NaN in Mamba output"
    assert x.grad is not None and not torch.isnan(x.grad).any(), "NaN in gradient"

    n_params = sum(p.numel() for p in block.parameters())
    print(f"[smoke-mamba] forward+backward: {elapsed*1000:.1f} ms")
    print(f"[smoke-mamba] params: {n_params/1e6:.2f} M")
    print("[smoke-mamba] OK")


if __name__ == "__main__":
    main()
