import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from core.blocks import ConvNeXtBlock, FiLMConvNeXtBlock
from core.states import LatentState
from core.boundary import blended_pad


def _default_circular_weights(batch: int, device, dtype) -> Tensor:
    """[1,0,0,0]-style fallback when no bc_weights is supplied (e.g. direct/
    standalone use of these functions, or tests) -- exactly [0,1,0] (100%
    circular) so behavior matches this file's original torch.roll-only code."""
    w = torch.zeros(batch, 3, device=device, dtype=dtype)
    w[:, 1] = 1.0
    return w


def dx_central(x: Tensor, weights_x: Optional[Tensor] = None, weights_y: Optional[Tensor] = None) -> Tensor:
    """See core/boundary.py: blended zero/circular/replicate padding per-axis,
    not the unconditional torch.roll (always-circular) this used to be. Passing
    no weights reproduces the exact old torch.roll behavior (defaults to 100%
    circular on both axes)."""
    b = x.shape[0]
    if weights_x is None:
        weights_x = _default_circular_weights(b, x.device, x.dtype)
    if weights_y is None:
        weights_y = _default_circular_weights(b, x.device, x.dtype)
    padded = blended_pad(x, weights_x, weights_y, pad=1)
    return 0.5 * (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2])


def dy_central(x: Tensor, weights_x: Optional[Tensor] = None, weights_y: Optional[Tensor] = None) -> Tensor:
    b = x.shape[0]
    if weights_x is None:
        weights_x = _default_circular_weights(b, x.device, x.dtype)
    if weights_y is None:
        weights_y = _default_circular_weights(b, x.device, x.dtype)
    padded = blended_pad(x, weights_x, weights_y, pad=1)
    return 0.5 * (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1])


def laplacian(x: Tensor, weights_x: Optional[Tensor] = None, weights_y: Optional[Tensor] = None) -> Tensor:
    b = x.shape[0]
    if weights_x is None:
        weights_x = _default_circular_weights(b, x.device, x.dtype)
    if weights_y is None:
        weights_y = _default_circular_weights(b, x.device, x.dtype)
    padded = blended_pad(x, weights_x, weights_y, pad=1)
    return (
        padded[:, :, 1:-1, 2:] + padded[:, :, 1:-1, :-2]
        + padded[:, :, 2:, 1:-1] + padded[:, :, :-2, 1:-1]
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
                 in_kernel_size: int, use_block: bool, block_kernel_size: int = 7):
        super().__init__()
        pad = in_kernel_size // 2
        self.proj_in = nn.Conv2d(latent_dim + cond_dim, hidden_dim, kernel_size=in_kernel_size, padding=pad)
        self.body = ConvNeXtBlock(hidden_dim, kernel_size=block_kernel_size) if use_block else nn.GELU()
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
                 in_kernel_size: int, block_kernel_size: int = 7):
        super().__init__()
        pad = in_kernel_size // 2
        self.proj_in = nn.Conv2d(latent_dim, hidden_dim, kernel_size=in_kernel_size, padding=pad)
        self.body = FiLMConvNeXtBlock(hidden_dim, cond_dim, kernel_size=block_kernel_size)
        self.out = _zero_init(nn.Conv2d(hidden_dim, out_channels, kernel_size=1))
        self.act = nn.Tanh()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        h = self.body(self.proj_in(x), cond)
        return self.act(self.out(h))


def _conditioned_net(latent_dim: int, hidden_dim: int, cond_dim: int, out_channels: int,
                     in_kernel_size: int, use_block: bool, film: bool,
                     block_kernel_size: int = 7) -> nn.Module:
    if film:
        return _FiLMConvNet(latent_dim, hidden_dim, cond_dim, out_channels, in_kernel_size,
                            block_kernel_size=block_kernel_size)
    return _ConcatConvNet(latent_dim, hidden_dim, cond_dim, out_channels, in_kernel_size, use_block,
                          block_kernel_size=block_kernel_size)


# ---------------------------------------------------------------------------
# Transport terms
# ---------------------------------------------------------------------------

class AdvectionTerm(nn.Module):
    """
    a = net([x, cond])  ->  -(a_x · dx(x) + a_y · dy(x))

    Learns a 2-channel velocity field and returns the advective transport of
    `x` along it.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 64, cond_dim: int = 32, film: bool = False,
                 block_kernel_size: int = 7):
        super().__init__()
        self.net = _conditioned_net(latent_dim, hidden_dim, cond_dim, out_channels=2,
                                    in_kernel_size=3, use_block=True, film=film,
                                    block_kernel_size=block_kernel_size)

    def forward(self, z: LatentState, cond: Tensor, bc_weights: Optional[tuple] = None) -> Tensor:
        wx, wy = bc_weights if bc_weights is not None else (None, None)
        a = self.net(z.real_grid, cond)
        return -(a[:, 0:1] * dx_central(z.real_grid, wx, wy) + a[:, 1:2] * dy_central(z.real_grid, wx, wy))


class DiffusionTerm(nn.Module):
    """nu * laplacian(x), where nu = exp(log_nu(z, cond)) is predicted per-pixel
    from BOTH the conditioning vector AND the current field state (via the same
    _conditioned_net conv-net pattern AdvectionTerm/ForcingTerm already use), not
    one scalar per sample.

    2026-08-22: was one scalar per sample (nu = exp(log_nu_head(cond)), a plain
    Linear on cond only, spatially uniform and state-independent) -- an earlier
    revision's own justification for THAT step ("diffusivity plausibly depends on
    the regime, so a single fixed nu for the whole dataset was a real if
    convenient assumption") applies exactly as well one level further: real
    diffusivity (eddy diffusivity in turbulence closures, boundary-layer vs. bulk
    behavior, heterogeneous media) commonly varies in space AND with the local
    field state, not just with which regime/task a sample belongs to. Reading
    z.real_grid (not just cond) makes nu a genuine nonlinear function of the
    current field, closer to a real diffusivity closure -- worth being a little
    careful about exactly because state-dependent feedback is the kind of thing
    that can misbehave if unbounded, which is why the same tanh-then-clamp
    structure every other conditioned term here already uses is kept, not relaxed.

    self.net's output projection is zero-inited (via _conditioned_net's
    _zero_init, same convention as every other term here), so net(...) == 0
    everywhere for every input at the start of training; log_nu = 0 + log_nu_init
    reproduces the old scalar version's exact behavior (spatially uniform,
    log_nu_init everywhere) at init, and only grows spatial/state-dependent
    variation as training finds it useful -- verified via a direct correctness
    check against the old Linear-based version at init.

    out_channels=1 (not latent_dim): nu still broadcasts across every latent
    channel, same capacity profile as the old scalar version, just now spatially
    varying instead of a single number -- per-channel-and-per-pixel nu (out_
    channels=latent_dim) was considered and explicitly not chosen, to avoid also
    multiplying this term's parameter count by latent_dim in the same change.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 64, cond_dim: int = 32,
                 film: bool = False, log_nu_init: float = -7.0):
        super().__init__()
        self.log_nu_init = log_nu_init
        self.net = _conditioned_net(latent_dim, hidden_dim, cond_dim, out_channels=1,
                                    in_kernel_size=3, use_block=True, film=film)

    def forward(self, z: LatentState, cond: Tensor, bc_weights: Optional[tuple] = None) -> Tensor:
        wx, wy = bc_weights if bc_weights is not None else (None, None)
        # self.net ends in a zero-inited conv + tanh (see _conditioned_net) -- output
        # is exactly 0 everywhere at init regardless of input, so this reproduces the
        # old scalar version's log_nu_init-everywhere behavior exactly at start.
        log_nu = self.net(z.real_grid, cond) + self.log_nu_init  # [B, 1, H, W]
        log_nu = torch.clamp(log_nu, min=-12.0, max=math.log(0.25))
        nu = torch.exp(log_nu)  # [B, 1, H, W] -- broadcasts over channels, varies over space
        return nu * laplacian(z.real_grid, wx, wy)


class LocalDiffusionTerm(nn.Module):
    """DiffusionTerm, made spatially-varying and state-responsive instead of
    one scalar-per-sample nu -- the local instinct behind real turbulence
    closures (Smagorinsky-style eddy viscosity makes diffusivity a function
    of the local strain rate, not a global constant), applied here as a
    direct extension of the existing mechanism rather than a replacement.

    log_nu = log_nu_head(cond) + local_net(z.real_grid, cond)

    log_nu_head is *exactly* the original DiffusionTerm's scalar path,
    unchanged (same init, same clamp range) -- kept as the floor/baseline
    every sample still gets regardless of local state. local_net is a small
    FiLM-conditioned conv (see _conditioned_net) producing a per-pixel
    correction on top of that floor, zero-inited (via _FiLMConvNet's own
    zero-inited output conv + tanh) so it's an exact no-op at init: nu
    starts out identical to plain DiffusionTerm's for every sample and
    pixel, and only grows spatial/state structure as training finds it
    useful -- this codebase's established convention for every newly-added
    conditioned mechanism (see e.g. HelmholtzRotationTerm, ComplexAmplitudeTerm,
    FFTSplitEncoderWide's complex_proj in EXPERIMENT_LOG.md), followed here
    for the same reason: skipping it is what caused this project's actual
    historical blowup bugs.

    Broadcasts the single-channel nu field across every latent channel, same
    as the original (nu is a diffusivity for the *medium*, not learned
    per-channel) -- only its (H, W) dependence is new.
    """

    def __init__(self, latent_dim: int, cond_dim: int, hidden_dim: int = 32, log_nu_init: float = -7.0):
        super().__init__()
        self.log_nu_head = nn.Linear(cond_dim, 1)
        nn.init.zeros_(self.log_nu_head.weight)
        nn.init.constant_(self.log_nu_head.bias, log_nu_init)
        self.local_net = _conditioned_net(latent_dim, hidden_dim, cond_dim, out_channels=1,
                                          in_kernel_size=3, use_block=False, film=True)

    def forward(self, z: LatentState, cond: Tensor) -> Tensor:
        log_nu_base = self.log_nu_head(cond)[:, :, None, None]  # [B, 1, 1, 1]
        log_nu_local = self.local_net(z.real_grid, cond)  # [B, 1, H, W], ~0 at init
        log_nu = torch.clamp(log_nu_base + log_nu_local, min=-12.0, max=math.log(0.25))
        nu = torch.exp(log_nu)  # [B, 1, H, W], broadcasts over channels
        return nu * laplacian(z.real_grid)


class ForcingTerm(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 64, cond_dim: int = 32, film: bool = False,
                 block_kernel_size: int = 7):
        super().__init__()
        self.net = _conditioned_net(latent_dim, hidden_dim, cond_dim, out_channels=latent_dim,
                                    in_kernel_size=1, use_block=False, film=film,
                                    block_kernel_size=block_kernel_size)

    def forward(self, z: LatentState, cond: Tensor, bc_weights: Optional[tuple] = None) -> Tensor:
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

    def forward(self, z: LatentState, cond: Tensor, bc_weights: Optional[tuple] = None) -> Tensor:
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

    def forward(self, z: LatentState, cond: Tensor, bc_weights: Optional[tuple] = None) -> Tensor:
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

    def __init__(self, latent_dim: int, hidden_dim: int, cond_dim: int, block_kernel_size: int = 7):
        super().__init__()
        self.net = _conditioned_net(
            2 * latent_dim, hidden_dim=hidden_dim, cond_dim=cond_dim, out_channels=latent_dim,
            in_kernel_size=1, use_block=False, film=True, block_kernel_size=block_kernel_size,
        )

    def forward(self, z: LatentState, cond: Tensor, bc_weights: Optional[tuple] = None) -> Tensor:
        return self.net(z.channel_sgrid, cond)  # real [B, latent_dim, H, W] phase angles


class ComplexAmplitudeTerm(nn.Module):
    """Predicts a per-pixel, per-channel log-amplitude from the current
    spectral content and cond -- paired with ComplexRotationTerm, see there.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, cond_dim: int, block_kernel_size: int = 7):
        super().__init__()
        self.net = _conditioned_net(
            2 * latent_dim, hidden_dim=hidden_dim, cond_dim=cond_dim, out_channels=latent_dim,
            in_kernel_size=1, use_block=False, film=True, block_kernel_size=block_kernel_size,
        )

    def forward(self, z: LatentState, cond: Tensor, bc_weights: Optional[tuple] = None) -> Tensor:
        return self.net(z.channel_sgrid, cond)  # real [B, latent_dim, H, W] log-amplitude


# ---------------------------------------------------------------------------
# Helmholtz-split phase rotation
#
# HelmholtzRotationTerm rotates a designated 2-channel real_grid vector-pair
# (vector_channels, default the first 2 latent channels) in Fourier space,
# recomputed fresh from the CURRENT real_grid every rollout step -- unlike
# FoundationModel's complex_proj/spectral_grid (a separate complex latent,
# initialized once from the t=0 context frame, then evolved purely by its
# own multiplicative rule), it reads and writes real_grid directly, so every
# other term (advection/diffusion/forcing/skew) still fully acts on the same
# 2 channels every step -- nothing here is reserved or frozen, this is a
# correction layered on top of the normal dynamics, not a replacement for
# them. Returns a delta on just those 2 channels (zero elsewhere), so it
# fits the same additive terms-sum contract as every other term and needs no
# changes to TransportOperator itself -- just one REAL_TERM_BUILDERS entry.
# ---------------------------------------------------------------------------

def _rfft_khat(h: int, w: int, device) -> tuple:
    """Unit wavevector (k_hat_y, k_hat_x) for an rfft2(..., norm="ortho") grid
    of a [.., h, w] real signal -- dim=-2 is y (ky = fftfreq(h), signed), dim=-1
    is x (kx = rfftfreq(w), non-negative), matching this file's existing
    dx_central/dy_central/laplacian convention. Zero at the DC bin (k=0, where
    a direction is undefined) -- k_hat=0 there sends the whole DC component to
    the irrotational branch for both terms below.
    """
    ky = torch.fft.fftfreq(h, device=device)[:, None]     # [H, 1]
    kx = torch.fft.rfftfreq(w, device=device)[None, :]    # [1, Wf]
    k2 = kx * kx + ky * ky
    safe_k2 = torch.where(k2 == 0, torch.ones_like(k2), k2)
    k_hat_y = torch.where(k2 == 0, torch.zeros_like(ky.expand_as(k2)), ky / safe_k2.sqrt())
    k_hat_x = torch.where(k2 == 0, torch.zeros_like(kx.expand_as(k2)), kx / safe_k2.sqrt())
    return k_hat_y, k_hat_x  # each [H, Wf], real


def _rfft_rotatable_mask(h: int, w: int, device) -> Tensor:
    """True at every rfft2(..., norm="ortho") bin that's safe to multiply by
    an arbitrary per-sample phase without corrupting the round-trip; False
    at the kx=0 column and (w even) the kx=w/2 Nyquist column -- the *only*
    columns rfft2's reduced last axis stores a conjugate-linked pair within
    (X[ky,0] and X[h-ky,0] must satisfy X[h-ky,0]=conj(X[ky,0]) for a
    genuinely real signal; same for kx=w/2 if present). For every OTHER kx
    in (0, w/2) exclusive, rfft2 stores only the positive-frequency bin --
    its conjugate partner at w-kx is never stored at all, so that bin is
    completely free to be set to anything with no consistency constraint.

    Confirmed empirically (not just derived): rotating every bin by the same
    phase then round-tripping through irfft2 -> rfft2 mismatches the
    original at literally every ky in the kx=0 and kx=w/2 columns, and
    nowhere else (see EXPERIMENT_LOG.md) -- this is what produced the ~2.7x
    "drift" test_helmholtz_rotation.py first caught, since a single global
    per-sample phase (not a matched +-theta pair across ky<->h-ky) breaks
    that column's conjugate-pairing whenever theta != 0, pi. Masking those
    two columns out entirely (phase=1, left alone) is simpler and more
    robustly correct than trying to pair-rotate them, at the cost of ~2/Wf
    columns never rotating -- the vast majority of bins stay free.
    """
    kx_idx = torch.arange(w // 2 + 1, device=device)
    kx_constrained = (kx_idx == 0) | ((w % 2 == 0) & (kx_idx == w // 2))
    return (~kx_constrained)[None, :].expand(h, -1)  # [H, Wf]


def _pad_channels(delta_pair: Tensor, lo: int, hi: int, total: int) -> Tensor:
    before = torch.zeros_like(delta_pair[:, :1]).expand(-1, lo, -1, -1) if lo > 0 else delta_pair[:, :0]
    after = torch.zeros_like(delta_pair[:, :1]).expand(-1, total - hi, -1, -1) if hi < total else delta_pair[:, :0]
    return torch.cat([before, delta_pair, after], dim=1)


class HelmholtzRotationTerm(nn.Module):
    """Splits a designated 2-channel real_grid vector-pair into curl-free
    (irrotational) and divergence-free (solenoidal) parts via k_hat(k_hat.V)
    in Fourier space -- the literal Helmholtz decomposition, recomputed
    fresh from the current real_grid every rollout step -- then applies a
    learned, per-frequency-bin phase rotation to ONLY the solenoidal part,
    leaving the irrotational part untouched by this term (it still evolves
    normally via every other term in the stack, same as any other channel).

    A pure phase multiply exactly preserves |S(k)| at every frequency bin
    it's applied to -- all but the kx=0 and (w even) kx=w/2 columns, left
    unrotated by _rfft_rotatable_mask to keep the irfft2 round-trip exactly
    consistent (see its docstring) -- physically the right place for a
    conservative rotation (divergence-free = vortical structure, Kelvin's-
    circulation-theorem territory in the inviscid limit). Note this isn't a
    hard whole-operator invariant: advection/diffusion/forcing/skew still
    act on these same 2 channels every step (deliberately -- see module
    docstring), so it's a structural bias layered on full dynamics, not a
    guarantee.

    theta (the rotation angle) is predicted per frequency bin, not as a
    single global scalar -- a full FiLM-conditioned _conditioned_net (the
    same architecture ComplexRotationTerm already uses elsewhere in this
    file, operating on the (ky,kx) grid rather than spatial (y,x)) consuming
    both `cond` and the current V's real/imag content. Confirmed via
    ablation this matters: a single scalar phase measurably underperformed
    this per-bin version -- see EXPERIMENT_LOG.md.
    """

    def __init__(self, cond_dim: int, latent_dim: int, vector_channels: tuple = (0, 2), hidden_dim: int = 32):
        super().__init__()
        lo, hi = vector_channels
        assert hi - lo == 2, "vector_channels must span exactly 2 channels"
        self.lo, self.hi, self.latent_dim = lo, hi, latent_dim
        # in_channels=4: V's [Re(Vy), Im(Vy), Re(Vx), Im(Vx)], same real/imag-concat
        # convention as LatentState.channel_sgrid elsewhere in this codebase.
        self.phase_net = _conditioned_net(4, hidden_dim, cond_dim, out_channels=1,
                                          in_kernel_size=1, use_block=False, film=True)

    def forward(self, z: LatentState, cond: Tensor, dt: float = 1.0) -> Tensor:
        v = z.real_grid[:, self.lo:self.hi]  # [B, 2, H, W]
        h, w = v.shape[-2:]
        V = torch.fft.rfft2(v, norm="ortho")  # [B, 2, H, Wf] complex
        k_hat_y, k_hat_x = _rfft_khat(h, w, v.device)  # each [H, Wf], real

        Vy, Vx = V[:, 0], V[:, 1]  # [B, H, Wf] complex, dim order matches real_grid's (y-like, x-like) pair
        k_dot_V = k_hat_x * Vx + k_hat_y * Vy  # [B, H, Wf] complex (real k_hat broadcasts fine against complex V)
        Ix, Iy = k_hat_x * k_dot_V, k_hat_y * k_dot_V  # irrotational part
        Sx, Sy = Vx - Ix, Vy - Iy  # solenoidal part

        V_ri = torch.stack([Vy.real, Vy.imag, Vx.real, Vx.imag], dim=1)  # [B, 4, H, Wf]
        # dt: interprets phase_net's raw output as an angular VELOCITY, not a fixed
        # per-frame rotation -- theta*dt is the angle actually swept this call, so
        # calling forward() N times at dt=1/N sweeps the same total angle as one
        # dt=1 call would in the small-angle limit, but via a genuinely different
        # (and hopefully better-conditioned) discretization the phase net was never
        # trained under. This is what makes sub-step integration meaningful here at
        # all -- without it, every substep would apply a fresh full-magnitude
        # rotation instead of a fraction of one. See EXPERIMENT_LOG.md's DISCO-motivated
        # substepping experiment.
        theta = self.phase_net(V_ri, cond).squeeze(1) * dt  # [B, H, Wf]
        phase = torch.exp(1j * theta)
        rotatable = _rfft_rotatable_mask(h, w, v.device)[None, :, :]  # see its docstring: kx=0/Nyquist excluded
        Sx_rot = torch.where(rotatable, Sx * phase, Sx)
        Sy_rot = torch.where(rotatable, Sy * phase, Sy)

        V_new = torch.stack([Iy + Sy_rot, Ix + Sx_rot], dim=1)  # [B, 2, H, Wf], back to (y,x) channel order
        v_new = torch.fft.irfft2(V_new, s=(h, w), norm="ortho")
        return _pad_channels(v_new - v, self.lo, self.hi, self.latent_dim)
