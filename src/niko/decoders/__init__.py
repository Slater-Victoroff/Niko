"""Decoders package."""

from .shared_conv import SharedConvDecoder
from .shared_heads import SharedTrunkFieldHeadsDecoder

__all__ = ["SharedConvDecoder", "SharedTrunkFieldHeadsDecoder"]