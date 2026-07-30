import torch.nn as nn
from torch import Tensor


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pw1 = nn.Conv2d(dim, expansion * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(expansion * dim, dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dw(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return residual + x


class FiLMConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int, expansion: int = 4):
        super().__init__()

        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # Okay, technically AdaGN, but it's really just FiLM with a GroupNorm layer.
        self.norm = nn.GroupNorm(1, dim)
        self.film = nn.Linear(cond_dim, 2 * dim)
        self.film.weight.data.zero_()
        self.film.bias.data.zero_()

        self.pw1 = nn.Conv2d(dim, expansion * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(expansion * dim, dim, kernel_size=1)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        residual = x

        x = self.dw(x)
        x = self.norm(x)

        # cond: [B, cond_dim]
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]

        # Identity-centered FiLM
        x = x * (1.0 + gamma) + beta

        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)

        return residual + x
