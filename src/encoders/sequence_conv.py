import torch
import torch.nn as nn
from torch import Tensor

from typing import Literal, Optional

from core.modules import EncoderBase, validate_call
from core.states import LatentState, Params
from core.blocks import ConvNeXtBlock, FiLMConvNeXtBlock


class SequenceConvEncoder(EncoderBase):
    """
    Input:
      x_context: [B, T, C, H, W]

    Output:
      LatentState.grid: [B, latent_dim, H/2, W/2]

    Convention:
      Internal latent layout is NCHW.
    """

    def __init__(
        self,
        in_channels: int,
        context_frames: int,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.context_frames = context_frames
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim

        total_in = in_channels * context_frames

        if self.cond_dim is not None:
            self.conv1 = nn.Conv2d(total_in, hidden_dim, kernel_size=3, padding=1)
            self.gelu1 = nn.GELU()

            self.block1 = FiLMConvNeXtBlock(hidden_dim, cond_dim=self.cond_dim)

            self.down = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
            self.gelu2 = nn.GELU()

            self.block2 = FiLMConvNeXtBlock(hidden_dim, cond_dim=self.cond_dim)

            self.out_conv = nn.Conv2d(hidden_dim, latent_dim, kernel_size=1)
        else:
            self.net = nn.Sequential(
                nn.Conv2d(total_in, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),

                ConvNeXtBlock(hidden_dim),

                # Fixed k=2 downsample.
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                nn.GELU(),

                ConvNeXtBlock(hidden_dim),

                nn.Conv2d(hidden_dim, latent_dim, kernel_size=1),
            )

    def forward(self, x: Tensor, cond: Tensor | None = None, params: Params | None = None) -> LatentState:
        reshaped_context = x.view(x.shape[0], self.context_frames * self.in_channels, x.shape[3], x.shape[4])
        if cond is not None and self.cond_dim is not None:
            x = self.conv1(reshaped_context)
            x = self.gelu1(x)
            x = self.block1(x, cond)

            x = self.down(x)
            x = self.gelu2(x)
            x = self.block2(x, cond)

            z = self.out_conv(x)
        else:
            z = self.net(reshaped_context)

        return LatentState(grid=z)
