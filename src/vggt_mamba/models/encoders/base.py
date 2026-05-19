"""Common interface for image encoders used in Phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class EncoderOutput:
    """Standardized encoder output.

    patches: (B, N, D) per-image patch tokens
    grid_h:  patch grid height
    grid_w:  patch grid width
    dim:     feature dimension D
    """
    patches: torch.Tensor
    grid_h: int
    grid_w: int
    dim: int


class FrameEncoder(Protocol):
    """Encoders consume (B, 3, H, W) and return patch tokens."""

    img_size: int
    dim: int

    def forward(self, x: torch.Tensor) -> EncoderOutput: ...
