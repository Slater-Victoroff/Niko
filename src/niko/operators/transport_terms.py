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
    """Project `x`, modulate with a FiLM-conditioned ConvNeXt block, project out.

    Tanh-bounded output, matching _ConcatConvNet's own act -- this was
    previously the one gap between the two conditioning paths: _ConcatConvNet
    always capped its output, _FiLMConvNet never did, so every FiLM-conditioned
    term (advection, forcing, and the complex rotation/amplitude terms all
    route through this) had no ceiling on its per-step contribution. Confirmed
    via direct trace (see EXPERIMENT_LOG.md) as the mechanism behind a real
    rollout blowup: an anomalous input frame produced a larger-than-normal
    advection velocity field, which combined with that same frame's larger
    gradient to produce a large first-step update, which produced an even
    larger velocity+gradient the next step, compounding to 10^13 scale over 6
    steps. Tanh is smooth and near-identity for the normal operating range (it
    barely touches typical-magnitude outputs), so this doesn't blunt normal
    training -- it only removes the *unbounded* tail that let one pathological
    input compound multiplicatively instead of just being unusually large once.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, cond_dim: int, out_channels: int,
                 in_kernel_size: int):
        super().__init__()
        pad = in_kernel_size // 2
        self.proj_in = nn.Conv2d(latent_dim, hidden_dim, kernel_size=in_kernel_size, padding=pad)
        self.body = FiLMConvNeXtBlock(hidden_dim, cond_dim)
        self.out = _zero_init(nn.Conv2d(hidden_dim, out_channels, kernel_size=1))
        self.act = nn.Tanh()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h = self.body(self.proj_in(x), cond)
        return self.act(self.out(h))


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


def _local_shifts(x: Tensor, window: int) -> Tensor:
    """Stack window*window periodically-shifted copies of `x` along a new
    dim (inserted at position 2): [B, C, H, W] -> [B, C, window*window, H, W].

    Uses torch.roll, matching the circular-boundary convention already used
    by dx_central/dy_central/laplacian above, rather than nn.Unfold's default
    zero-padding at the grid edges.
    """
    r = window // 2
    shifts = [
        torch.roll(x, shifts=(dy, dx), dims=(-2, -1))
        for dy in range(-r, r + 1)
        for dx in range(-r, r + 1)
    ]
    return torch.stack(shifts, dim=2)


class LocalAttentionTerm(nn.Module):
    """Learned local windowed self-attention, as a direct alternative to
    SkewTerm within HelmholtzTransportOperator's term set.

    At each grid position, query/key/value project the (cond-conditioned)
    latent, attend over the window x window periodic neighborhood (via
    `_local_shifts`), and project the result back to latent_dim. Unlike
    SkewTerm's fixed skew-symmetric channel mixing, this term isn't
    constrained to a norm-preserving rotation -- and unlike AdvectionTerm's
    single learned velocity field, each position can attend anisotropically
    to any neighbor. In principle it can express translation, isotropic
    blurring, and channel mixing all through one learned mechanism, at the
    cost of the physical structure (a·∇z, skew-symmetry) that made those
    other terms interpretable and cheap to fit.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 64, cond_dim: int = 32,
                 window: int = 3, n_heads: int = 4, film: bool = False):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})")
        self.window = window
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.film_cond = film

        if film:
            self.film = nn.Linear(cond_dim, 2 * latent_dim)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
            in_dim = latent_dim
        else:
            self.film = None
            in_dim = latent_dim + cond_dim

        self.q_proj = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
        self.out = _zero_init(nn.Conv2d(hidden_dim, latent_dim, kernel_size=1))

    def forward(self, z: LatentState, cond: Tensor) -> Tensor:
        x = z.real_grid
        b, _, h, w = x.shape

        if self.film_cond:
            gamma, beta = self.film(cond).chunk(2, dim=-1)
            x_in = x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        else:
            x_in = torch.cat([x, _broadcast_cond(cond, x)], dim=1)

        q = self.q_proj(x_in).view(b, self.n_heads, self.head_dim, 1, h, w)
        k = _local_shifts(self.k_proj(x_in), self.window).view(b, self.n_heads, self.head_dim, -1, h, w)
        v = _local_shifts(self.v_proj(x_in), self.window).view(b, self.n_heads, self.head_dim, -1, h, w)

        logits = (q * k).sum(dim=2) / (self.head_dim ** 0.5)  # [B, n_heads, window^2, H, W]
        attn = logits.softmax(dim=2)
        out = (attn.unsqueeze(2) * v).sum(dim=3).reshape(b, -1, h, w)  # [B, hidden_dim, H, W]
        return self.out(out)


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


class ComplexRotationTerm(nn.Module):
    """Predicts a per-pixel, per-channel rotation angle from the current
    spectral content (z.channel_sgrid) and cond -- purely a phase rotation,
    no relation to the real-space advection/diffusion/skew terms. Meant to
    be read by TransportOperator.forward's complex_term branch, not called
    standalone: it produces the angle, the operator combines it with
    ComplexAmplitudeTerm's log-amplitude into spectral_grid * exp(amp + i*angle).
    """

    def __init__(self, latent_dim: int, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.net = _conditioned_net(
            2 * latent_dim, hidden_dim=hidden_dim, cond_dim=cond_dim, out_channels=latent_dim,
            in_kernel_size=1, use_block=False, film=True,
        )

    def forward(self, z: LatentState, cond: Tensor) -> Tensor:
        return self.net(z.channel_sgrid, cond)  # real [B, latent_dim, H, W] phase angles


class ComplexAmplitudeTerm(nn.Module):
    """Predicts a per-pixel, per-channel log-amplitude from the current
    spectral content and cond -- paired with ComplexRotationTerm, see there.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.net = _conditioned_net(
            2 * latent_dim, hidden_dim=hidden_dim, cond_dim=cond_dim, out_channels=latent_dim,
            in_kernel_size=1, use_block=False, film=True,
        )

    def forward(self, z: LatentState, cond: Tensor) -> Tensor:
        return self.net(z.channel_sgrid, cond)  # real [B, latent_dim, H, W] log-amplitude
