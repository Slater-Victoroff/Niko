import torch
import torch.nn as nn
from torch import Tensor

from typing import Literal, Optional

from core.modules import EncoderBase, validate_call
from core.states import LatentState, Params
from core.blocks import ConvNeXtBlock, FiLMConvNeXtBlock


class SequenceConvEncoder(EncoderBase):
    def __init__(
        self,
        in_channels: int,
        context_frames: int,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        cond_dim: Optional[int] = None,
        block_kernel_size: int = 7,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.context_frames = context_frames
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim

        total_in = in_channels * context_frames

        self.net = nn.Sequential(
            nn.Conv2d(total_in, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            ConvNeXtBlock(hidden_dim, kernel_size=block_kernel_size),
            nn.Conv2d(hidden_dim, latent_dim, kernel_size=1),
        )

    def forward(self, x: Tensor, cond: Tensor | None = None, params: Params | None = None) -> LatentState:
        reshaped_context = x.view(x.shape[0], self.context_frames * self.in_channels, x.shape[3], x.shape[4])
        z = self.net(reshaped_context)
        return LatentState(real_grid=z)
