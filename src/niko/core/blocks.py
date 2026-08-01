import torch
import torch.nn as nn
from torch import Tensor


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, kernel_size: int = 7):
        super().__init__()
        # Depthwise conv cost scales as dim*K^2 (linear in channels), while the
        # pointwise convs below (the block's actual FLOP majority) scale as
        # dim^2 -- widening kernel_size is a cheap way to add depthwise params
        # without meaningfully changing the block's total compute.
        self.dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
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


class LocallyConnected1x1(nn.Module):
    """Per-spatial-position (unshared) 1x1 projection: mathematically the
    exact same multiply-add count as a shared nn.Conv2d(..., kernel_size=1)
    applied at every position -- the only difference is each grid position
    gets its own weight/bias instead of reusing one set globally. A way to
    add real parameters without adding FLOPs, when weight-sharing across
    positions isn't obviously the right inductive bias (e.g. across
    frequency bins of a spectrum, which aren't translation-invariant the
    way image pixels are).

    Unlike a conv, this requires the exact spatial grid size up front (the
    weight tensor is sized for it) -- not resolution-agnostic.
    """

    def __init__(self, in_channels: int, out_channels: int, grid_h: int, grid_w: int):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        n = grid_h * grid_w
        self.weight = nn.Parameter(torch.empty(n, out_channels, in_channels))
        self.bias = nn.Parameter(torch.zeros(n, out_channels))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)  # matches nn.Conv2d's default init

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        if h != self.grid_h or w != self.grid_w:
            raise ValueError(
                f"LocallyConnected1x1 sized for grid ({self.grid_h}, {self.grid_w}), got ({h}, {w})"
            )
        x_flat = x.reshape(b, c, h * w).permute(2, 0, 1)  # (N, B, Cin)
        out = torch.einsum("nbc,noc->nbo", x_flat, self.weight)  # (N, B, Cout)
        out = out + self.bias[:, None, :]
        return out.permute(1, 2, 0).reshape(b, -1, h, w)


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
