"""Cross-frame Mamba-2 aggregator.

Pre-norm Mamba-2 SSD layer stack. Two inference modes:

  - forward(tokens):  parallel SSD over the full sequence. Used during
                      training and offline batch inference.
  - streaming_step(tokens, state): recurrent one-token-at-a-time mode using
                      Mamba's step API. Used for streaming inference where
                      total memory must stay constant in N.

Optional bidirectional mode runs the same layers twice (forward + reversed)
and merges via a learned gate. Causal-only (bidirectional=False) is the
streaming variant.

State per layer: conv_state ≈ (B, 2D+ngroups*d_state, 4), ssm_state ≈
(B, n_heads*2, head_dim, d_state). For dim=1024, d_state=128, head_dim=64,
expand=2: ssm_state has 32 heads × 64 × 128 = 262k floats, conv_state has
1×2304×4 = 9k floats. About 540 KB bf16 per layer, constant in N.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mamba_ssm.modules.mamba2 import Mamba2


class Mamba2Block(nn.Module):
    """Pre-norm Mamba-2 residual block."""

    def __init__(self, dim: int, d_state: int = 128, headdim: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba2(d_model=dim, d_state=d_state, headdim=headdim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, conv_state: torch.Tensor,
             ssm_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One-token streaming step.

        x_t: (B, 1, D). Returns (block_out, new_conv_state, new_ssm_state).
        """
        normed = self.norm(x_t)
        out, conv_state, ssm_state = self.mamba.step(normed, conv_state, ssm_state)
        return x_t + out, conv_state, ssm_state


class CrossFrameMamba(nn.Module):
    """Stack of Mamba-2 blocks over the flattened token sequence."""

    def __init__(
        self,
        dim: int,
        n_layers: int = 4,
        d_state: int = 128,
        headdim: int = 64,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [Mamba2Block(dim, d_state=d_state, headdim=headdim) for _ in range(n_layers)]
        )
        self.norm_out = nn.LayerNorm(dim)
        self.dim = dim
        self.d_state = d_state
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        if bidirectional:
            self.merge = nn.Linear(dim * 2, dim)

    def _scan(self, tokens: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            tokens = layer(tokens)
        return self.norm_out(tokens)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        fwd = self._scan(tokens)
        if not self.bidirectional:
            return fwd
        bwd = self._scan(tokens.flip(dims=[1])).flip(dims=[1])
        return self.merge(torch.cat([fwd, bwd], dim=-1))

    # ---------- streaming inference path ----------

    @torch.no_grad()
    def init_streaming_state(self, batch_size: int = 1, dtype=torch.bfloat16,
                             device="cuda") -> list[dict]:
        """Zero-initialised per-layer state for streaming inference."""
        if self.bidirectional:
            raise RuntimeError("streaming_step only valid for bidirectional=False model")
        states = []
        for layer in self.layers:
            conv_state, ssm_state = layer.mamba.allocate_inference_cache(
                batch_size=batch_size, max_seqlen=1, dtype=dtype,
            )
            states.append({
                "conv": conv_state.zero_().to(device),
                "ssm": ssm_state.zero_().to(device),
            })
        return states

    @torch.no_grad()
    def streaming_step(
        self,
        tokens: torch.Tensor,
        state: list[dict],
    ) -> tuple[torch.Tensor, list[dict]]:
        """Run K new tokens through the stack, mutating state in-place semantics.

        tokens: (B, K, D) — the K new summary tokens for one frame.
        state: list of per-layer {conv, ssm} state dicts (from init_streaming_state).

        Returns:
            output: (B, K, D) — same shape, after all layers + norm_out.
            new_state: same structure as input state, advanced by K time-steps.
        """
        b, k, d = tokens.shape
        outs_per_token = []
        for ki in range(k):
            x = tokens[:, ki:ki + 1, :]            # (B, 1, D)
            for li, layer in enumerate(self.layers):
                x, conv_s, ssm_s = layer.step(x, state[li]["conv"], state[li]["ssm"])
                state[li]["conv"] = conv_s
                state[li]["ssm"] = ssm_s
            outs_per_token.append(x)
        stacked = torch.cat(outs_per_token, dim=1)  # (B, K, D)
        return self.norm_out(stacked), state


class GraphedStreamingScan:
    """CUDA-graph wrapper around CrossFrameMamba.streaming_step.

    Speed-fix Option B: same kernels, same order — graph capture eliminates
    per-kernel cudaLaunchKernel overhead. Mechanically bit-perfect by
    construction; verify empirically with the parity test before trusting.

    Usage:
        gs = GraphedStreamingScan(cross_frame, batch_size=1, K=1024,
                                  dim=1024, dtype=torch.bfloat16)
        gs.capture()
        # Per-frame:
        out, state = gs.step(tokens)    # state is gs.state (mutated in-place)
        # Reset between sequences:
        gs.reset_state()

    Assumptions:
      - Static K, batch_size, dtype per captured graph (re-capture if any change).
      - mamba.step writes conv_state / ssm_state in-place (it does in mamba_ssm
        v2.2.x — Mamba2.step holds tensor identity and writes via copy_/scatter).
      - Caller copies new input via gs.step(); graph reads from gs.static_in.
    """

    def __init__(self, cross_frame: CrossFrameMamba, batch_size: int, K: int,
                 dim: int, dtype: torch.dtype = torch.bfloat16, device: str = "cuda"):
        self.cross_frame = cross_frame
        self.B = batch_size
        self.K = K
        self.dim = dim
        self.dtype = dtype
        self.device = device
        self.static_in = torch.zeros(batch_size, K, dim, device=device, dtype=dtype)
        # Allocate state once and keep references — mamba.step writes in-place
        # to these tensors so the graph captures their addresses.
        self.state = cross_frame.init_streaming_state(
            batch_size=batch_size, dtype=dtype, device=device,
        )
        self.static_out: torch.Tensor | None = None
        self.graph: torch.cuda.CUDAGraph | None = None

    @torch.no_grad()
    def reset_state(self) -> None:
        """Zero the state buffers in-place (preserves addresses, so graph still valid)."""
        for s in self.state:
            s["conv"].zero_()
            s["ssm"].zero_()

    @torch.no_grad()
    def capture(self, warmup_iters: int = 3) -> None:
        # Warmup: alloc workspace, JIT, settle caching allocator.
        for _ in range(warmup_iters):
            with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                out, self.state = self.cross_frame.streaming_step(self.static_in, self.state)
        torch.cuda.synchronize()
        self.reset_state()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
                captured_out, self.state = self.cross_frame.streaming_step(
                    self.static_in, self.state
                )
        # static_out is the tensor produced inside the captured region — replay
        # writes to its memory, so subsequent .step() reads return updated values.
        self.static_out = captured_out
        self.reset_state()

    @torch.no_grad()
    def step(self, tokens: torch.Tensor) -> tuple[torch.Tensor, "GraphedStreamingScan"]:
        """Replay the captured graph on new input tokens.

        Returns (output_view, self) — returning `self` keeps the wrapper alive
        across frames in caller code like `out, state = state.step(tokens)`,
        so subsequent isinstance(state, GraphedStreamingScan) dispatch still
        picks the graphed path. Returning self.state (the underlying list)
        would silently fall back to the loop path on frame 2+.
        """
        assert self.graph is not None, "call capture() before step()"
        assert tokens.shape == self.static_in.shape, \
            f"shape mismatch: got {tuple(tokens.shape)}, captured for {tuple(self.static_in.shape)}"
        self.static_in.copy_(tokens.to(self.static_in.dtype))
        self.graph.replay()
        return self.static_out, self


if __name__ == "__main__":
    for bidir in (False, True):
        blk = CrossFrameMamba(dim=1024, n_layers=4, d_state=128,
                              bidirectional=bidir).cuda()
        x = torch.randn(1, 80, 1024, device="cuda", requires_grad=True)
        y = blk(x)
        y.sum().backward()
        print(f"[mamba bidir={bidir}] in {tuple(x.shape)} -> {tuple(y.shape)}  "
              f"params: {sum(p.numel() for p in blk.parameters())/1e6:.2f}M")

    # ---- streaming parity check: split a sequence into chunks, scan each chunk
    #      with streaming_step, compare to a single forward() call. ----
    print("\n[mamba streaming] parity check (causal-only)…")
    blk = CrossFrameMamba(dim=256, n_layers=2, d_state=64, bidirectional=False).cuda().eval()
    K = 4
    T = 5
    x = torch.randn(1, T * K, 256, device="cuda", dtype=torch.bfloat16)
    blk = blk.to(torch.bfloat16)

    with torch.inference_mode():
        ref = blk(x)                              # (1, T*K, 256)

        state = blk.init_streaming_state(batch_size=1, dtype=torch.bfloat16, device="cuda")
        chunks = []
        for t in range(T):
            chunk = x[:, t * K:(t + 1) * K, :]
            out, state = blk.streaming_step(chunk, state)
            chunks.append(out)
        stream = torch.cat(chunks, dim=1)         # (1, T*K, 256)

    diff = (ref.float() - stream.float()).abs()
    print(f"  ref  shape={tuple(ref.shape)}  mean={ref.float().mean().item():+.4f}")
    print(f"  diff max={diff.max().item():.3e}  mean={diff.mean().item():.3e}")
    print("  (note: norm_out is re-applied per chunk in streaming → drift expected)")
