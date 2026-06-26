"""Frozen image encoders. Currently: DINOv3 ViT-L/16."""

from .base import EncoderOutput, FrameEncoder
from .dinov3 import DINOv3Encoder

__all__ = ["EncoderOutput", "FrameEncoder", "DINOv3Encoder"]
