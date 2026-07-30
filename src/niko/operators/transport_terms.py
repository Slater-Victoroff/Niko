import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from core.blocks import ConvNeXtBlock, FiLMConvNeXtBlock
from core.states import LatentState


def dx_central(x: Tensor) -> Tensor:
    return 0.5 * (torch.roll(x, shifts=-1, dims=-1) - torch.roll(x, shifts=1, dims=-1))


def dy_central(x: Tensor) -> Tensor:
    return 0.5 * (torch.roll(x, shifts=-1, dims=-2) - torch.roll(x, shifts=1, dims=-2))


def laplacian(x: Tensor) -> Tensor:
    return (
        torch.roll(x, shifts=1, dims=-1) + torch.roll(x, shifts=-1, dims=-1)
        + torch.roll(x, shifts=1, dims=-2) + torch.roll(x, shifts=-1, dims=-2)
        - 4.0 * x
    )


def _broadcast_cond(cond: Tensor, x: Tensor) -> Tensor:
    """[B, cond_dim] -> [B, cond_dim, H, W], broadcast to x's spatial size."""
    b, _, h, w = x.shape
    cond_map = cond.to(device=x.device, dtype=x.dtype)[:, :, None, None]
    return cond_map.expand(b, cond.shape[1], h, w)


def _zero_init(conv: nn.Conv2d) -> nn.Conv2d:
    """Zero-init the output projection so a term starts as a no-op and fades in during training."""
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv


class _ConcatConvNet(nn.Module):
    """Concatenate a broadcast `cond` map onto `x`, then a small conv stack."""

    def __init__(self, latent_dim: int, hidden_dim: int, cond_dim: int, out_channels: int,
                 in_kernel_size: int, use_block: bool):
        super().__init__()
        pad = in_kernel_size // 2
        self.proj_in = nn.Conv2d(latent_dim + cond_dim, hidden_dim, kernel_size=in_kernel_size, padding=pad)
        self.body = ConvNeXtBlock(hidden_dim) if use_block else nn.GELU()
        self.out = _zero_init(nn.Conv2d(hidden_dim, out_channels, kernel_size=1))
        self.act = nn.Tanh()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h = self.proj_in(torch.cat([x, _broadcast_cond(cond, x)], dim=1))
        h = self.body(h)
        return self.act(self.out(h))


class _FiLMConvNet(nn.Module):
    """Project `x`, modulate with a FiLM-conditioned ConvNeXt block, project out."""

    def __init__(self, latent_dim: int, hidden_dim: int, cond_dim: int, out_channels: int,
                 in_kernel_size: int):
        super().__init__()
        pad = in_kernel_size // 2
        self.proj_in = nn.Conv2d(latent_dim, hidden_dim, kernel_size=in_kernel_size, padding=pad)
        self.body = FiLMConvNeXtBlock(hidden_dim, cond_dim)
        self.out = _zero_init(nn.Conv2d(hidden_dim, out_channels, kernel_size=1))

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h = self.body(self.proj_in(x), cond)
        return self.out(h)


def _conditioned_net(latent_dim: int, hidden_dim: int, cond_dim: int, out_channels: int,
                     in_kernel_size: int, use_block: bool, film: bool) -> nn.Module:
    if film:
        return _FiLMConvNet(latent_dim, hidden_dim, cond_dim, out_channels, in_kernel_size)
    return _ConcatConvNet(latent_dim, hidden_dim, cond_dim, out_channels, in_kernel_size, use_block)


# ---------------------------------------------------------------------------
# Transport terms
# ---------------------------------------------------------------------------

class AdvectionTerm(nn.Module):
    """
    a = net([x, cond])  ->  -(a_x · dx(x) + a_y · dy(x))

    Learns a 2-channel velocity field and returns the advective transport of
    `x` along it.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 64, cond_dim: int = 32, film: bool = False):
        super().__init__()
        self.net = _conditioned_net(latent_dim, hidden_dim, cond_dim, out_channels=2,
                                    in_kernel_size=3, use_block=True, film=film)

    def forward(self, z: LatentState, cond: Tensor) -> Tensor:
        a = self.net(z.real_grid, cond)
        return -(a[:, 0:1] * dx_central(z.real_grid) + a[:, 1:2] * dy_central(z.real_grid))


class DiffusionTerm(nn.Module):
    def __init__(self, log_nu_init: float = -7.0):
        super().__init__()
        self.log_nu = nn.Parameter(torch.tensor(log_nu_init))

    def forward(self, z: LatentState, cond: Optional[Tensor] = None) -> Tensor:
        nu = torch.exp(torch.clamp(self.log_nu, min=-12.0, max=math.log(0.25)))
        return nu * laplacian(z.real_grid)


class ForcingTerm(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 64, cond_dim: int = 32, film: bool = False):
        super().__init__()
        self.net = _conditioned_net(latent_dim, hidden_dim, cond_dim, out_channels=latent_dim,
                                    in_kernel_size=1, use_block=False, film=film)

    def forward(self, z: LatentState, cond: Tensor) -> Tensor:
        return self.net(z.real_grid, cond)


class SkewTerm(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.skew_raw = nn.Parameter(torch.zeros(latent_dim, latent_dim))

    def forward(self, z: LatentState, cond: Optional[Tensor] = None) -> Tensor:
        k = self.skew_raw - self.skew_raw.transpose(0, 1)
        return torch.einsum("ij,bjhw->bihw", k, z.real_grid)


class ComplexRotationTerm(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.net = _conditioned_net(
            2 * latent_dim,
            hidden_dim=hidden_dim,
            cond_dim=cond_dim,
            out_channels=latent_dim,
            in_kernel_size=1,
            use_block=False,
            film=True,
        )

    def forward(self, z: LatentState, cond: Tensor) -> Tensor:
        return self.net(z.channel_sgrid, cond)  # real [B, latent_dim, H, W] phase angles


class ComplexAmplitudeTerm(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.net = _conditioned_net(
            2 * latent_dim,
            hidden_dim=hidden_dim,
            cond_dim=cond_dim,
            out_channels=latent_dim,
            in_kernel_size=1,
            use_block=False,
            film=True,
        )

    def forward(self, z: LatentState, cond: Tensor) -> Tensor:
        return self.net(z.channel_sgrid, cond)  # real [B, latent_dim, H, W] log-amplitude
