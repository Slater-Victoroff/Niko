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
    """nu * laplacian(x), where nu = exp(log_nu(cond)) is predicted per-sample
    from the conditioning vector instead of being one global scalar shared
    across every Rayleigh/Prandtl regime -- diffusivity plausibly does
    depend on the regime, so a single fixed nu for the whole dataset was a
    real (if convenient) modeling assumption, not an obviously correct one.

    Zero-inited (weight only, bias keeps log_nu_init) so log_nu(cond) ==
    log_nu_init for every cond at the start of training -- matches this
    project's zero-init convention for newly-conditioned terms elsewhere
    (FiLMConvNeXtBlock's film layer, DirectFieldOperator's head,
    FFTSplitEncoderWide's complex_proj output): starts out behaving exactly
    like the old unconditioned DiffusionTerm, and only diverges per-regime
    as training finds it useful.
    """

    def __init__(self, cond_dim: int, log_nu_init: float = -7.0):
        super().__init__()
        self.log_nu_head = nn.Linear(cond_dim, 1)
        nn.init.zeros_(self.log_nu_head.weight)
        nn.init.constant_(self.log_nu_head.bias, log_nu_init)

    def forward(self, z: LatentState, cond: Tensor) -> Tensor:
        log_nu = self.log_nu_head(cond)  # [B, 1]
        log_nu = torch.clamp(log_nu, min=-12.0, max=math.log(0.25))
        nu = torch.exp(log_nu)[:, :, None, None]  # [B, 1, 1, 1], broadcasts over channels+spatial
        return nu * laplacian(z.real_grid)


class ForcingTerm(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 64, cond_dim: int = 32, film: bool = False):
        super().__init__()
        self.net = _conditioned_net(latent_dim, hidden_dim, cond_dim, out_channels=latent_dim,
                                    in_kernel_size=1, use_block=False, film=film)

    def forward(self, z: LatentState, cond: Tensor) -> Tensor:
        return self.net(z.real_grid, cond)


class SkewTerm(nn.Module):
    """Learned skew-symmetric channel-mixing matrix, predicted per-sample
    from the conditioning vector instead of one global matrix shared across
    every Rayleigh/Prandtl regime.

    Zero-inited (weight AND bias) so the predicted matrix is exactly zero
    for every cond at the start of training -- matches skew_raw's own
    original zero-init (torch.zeros(latent_dim, latent_dim)): this term
    still starts as a true no-op regardless of conditioning, and only grows
    a (now per-regime) mixing matrix as training finds it useful.
    """

    def __init__(self, latent_dim: int, cond_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.skew_head = nn.Linear(cond_dim, latent_dim * latent_dim)
        nn.init.zeros_(self.skew_head.weight)
        nn.init.zeros_(self.skew_head.bias)

    def forward(self, z: LatentState, cond: Tensor) -> Tensor:
        b = cond.shape[0]
        skew_raw = self.skew_head(cond).view(b, self.latent_dim, self.latent_dim)
        k = skew_raw - skew_raw.transpose(-1, -2)
        return torch.einsum("bij,bjhw->bihw", k, z.real_grid)
