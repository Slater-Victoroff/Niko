from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from core.modules import OperatorBase
from core.states import LatentState
from operators.transport_terms import (
    AdvectionTerm, DiffusionTerm, ForcingTerm, SkewTerm, LocalAttentionTerm,
    ComplexRotationTerm, ComplexAmplitudeTerm,
)


class TransportOperator(OperatorBase):
    """Sums a configurable set of transport-term mixins onto the latent grid each step."""

    REAL_TERM_BUILDERS = {
        "advection": lambda latent_dim, hidden_dim, cond_dim, film:
            AdvectionTerm(latent_dim, hidden_dim, cond_dim, film=film),
        "diffusion": lambda latent_dim, hidden_dim, cond_dim, film:
            DiffusionTerm(cond_dim),
        "forcing": lambda latent_dim, hidden_dim, cond_dim, film:
            ForcingTerm(latent_dim, hidden_dim, cond_dim, film=film),
        "skew": lambda latent_dim, hidden_dim, cond_dim, film:
            SkewTerm(latent_dim, cond_dim),
        "local_attention": lambda latent_dim, hidden_dim, cond_dim, film:
            LocalAttentionTerm(latent_dim, hidden_dim, cond_dim, film=film),
    }

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        cond_dim: int = 32,
        terms: Sequence[str] = ("advection", "diffusion", "forcing"),
        film: bool = False,
        complex_term: bool = False,
        amplitude_scale: float = 1.0,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.complex_term = complex_term
        self.amplitude_scale = amplitude_scale

        unknown = [t for t in terms if t not in self.REAL_TERM_BUILDERS]
        if unknown:
            raise ValueError(
                f"Unknown transport term(s) {unknown}; available: {sorted(list(self.REAL_TERM_BUILDERS.keys()))}"
            )

        self.real_terms = nn.ModuleDict({
            name: self.REAL_TERM_BUILDERS[name](latent_dim, hidden_dim, cond_dim, film)
            for name in terms if name in self.REAL_TERM_BUILDERS
        })

        # Built once here (not per-forward!) so they're registered submodules --
        # model.to(device) and the optimizer only see parameters that exist at
        # construction time.
        if complex_term:
            self.complex_amplitude = ComplexAmplitudeTerm(latent_dim, hidden_dim, cond_dim)
            self.complex_rotation = ComplexRotationTerm(latent_dim, hidden_dim, cond_dim)

    def forward(self, z: LatentState, cond: Optional[Tensor] = None) -> LatentState:
        if cond is None:
            raise ValueError(f"{type(self).__name__} requires cond shaped [B, cond_dim].")

        real_out = z.real_grid
        for term in self.real_terms.values():
            real_out = real_out + term(z, cond)

        spectral_out = z.spectral_grid
        if self.complex_term:
            amplitude = self.complex_amplitude(z, cond)
            # Zero-mean spatially then tanh: zero-mean prevents net spectral energy
            # drift over K rollout steps; tanh bounds the per-step factor to
            # exp(+-amplitude_scale) with smooth gradients (no dead zones at a hard
            # clamp). amplitude_scale=1.0 allows exp(+-1)~=2.72x per step, ~=405x
            # over a 6-step rollout in the worst case -- lower to tighten that
            # per-step ceiling (e.g. 0.5 -> exp(+-0.5)~=1.65x/step, ~=20x over 6 steps).
            amplitude = amplitude - amplitude.mean(dim=(-2, -1), keepdim=True)
            amplitude = self.amplitude_scale * torch.tanh(amplitude)
            rotation = self.complex_rotation(z, cond)
            spectral_out = z.spectral_grid * torch.exp(amplitude + 1j * rotation)

        return z.replace_state(real_grid=real_out, spectral_grid=spectral_out)


# ---------------------------------------------------------------------------
# Named presets -- the term combinations the existing configs reference by
# name. Each just fixes `terms` / `film`; add a new preset the same way.
# ---------------------------------------------------------------------------

class AdvectionDiffusionOperator(TransportOperator):
    """z_next = z − a·∇z + ν·Δz + f, concat-conditioned."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64, cond_dim: int = 32):
        super().__init__(latent_dim, hidden_dim, cond_dim,
                         terms=("advection", "diffusion", "forcing"), film=False)


class HelmholtzTransportOperator(TransportOperator):
    """AdvectionDiffusionOperator plus a learned skew-symmetric channel-mixing term."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64, cond_dim: int = 32):
        super().__init__(latent_dim, hidden_dim, cond_dim,
                         terms=("advection", "diffusion", "skew", "forcing"),
                         film=False)


class LocalAttentionTransportOperator(TransportOperator):
    """AdvectionDiffusionOperator plus a learned local windowed self-attention
    term, in place of HelmholtzTransportOperator's skew-symmetric channel
    mixing -- a direct alternative to compare against it.

    `window` is a real hyperparameter here (unlike the other terms' fixed
    stencils) since it sets the term's receptive field per rollout step;
    default 5 sits between DiffusionTerm's 3x3 Laplacian stencil and
    AdvectionTerm's kernel_size=7 velocity-field conv.

    `attn_hidden_dim` sizes the attention term independently of the shared
    `hidden_dim` (which still sizes advection/forcing) -- window doesn't add
    parameters at all (it only changes how many neighbors get gathered via
    torch.roll), so widening the term's actual capacity means widening this
    instead. Defaults to `hidden_dim` when unset, matching prior behavior.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 64, cond_dim: int = 32,
                 window: int = 5, n_heads: int = 4, attn_hidden_dim: Optional[int] = None):
        super().__init__(latent_dim, hidden_dim, cond_dim,
                         terms=("advection", "diffusion", "forcing"), film=False)
        self.real_terms["local_attention"] = LocalAttentionTerm(
            latent_dim, attn_hidden_dim or hidden_dim, cond_dim, window=window, n_heads=n_heads, film=False)


class FiLMAdvectionDiffusionOperator(TransportOperator):
    """AdvectionDiffusionOperator, FiLM-conditioned instead of concat-conditioned."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64, cond_dim: int = 32):
        super().__init__(latent_dim, hidden_dim, cond_dim,
                         terms=("advection", "diffusion", "forcing"), film=True)


class FiLMHelmholtzTransportOperator(TransportOperator):
    """HelmholtzTransportOperator, FiLM-conditioned instead of concat-conditioned."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64, cond_dim: int = 32):
        super().__init__(latent_dim, hidden_dim, cond_dim,
                         terms=("advection", "diffusion", "skew", "forcing"),
                         film=True)


class FiLMLocalAttentionTransportOperator(TransportOperator):
    """LocalAttentionTransportOperator, FiLM-conditioned instead of concat-conditioned."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64, cond_dim: int = 32,
                 window: int = 5, n_heads: int = 4, attn_hidden_dim: Optional[int] = None):
        super().__init__(latent_dim, hidden_dim, cond_dim,
                         terms=("advection", "diffusion", "forcing"), film=True)
        self.real_terms["local_attention"] = LocalAttentionTerm(
            latent_dim, attn_hidden_dim or hidden_dim, cond_dim, window=window, n_heads=n_heads, film=True)
