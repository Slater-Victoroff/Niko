from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from core.modules import OperatorBase
from core.states import LatentState
from operators.transport_terms import (
    AdvectionTerm, DiffusionTerm, LocalDiffusionTerm, ForcingTerm, SkewTerm, LocalAttentionTerm,
    ComplexRotationTerm, ComplexAmplitudeTerm, HelmholtzRotationTerm,
)


class TransportOperator(OperatorBase):
    """Sums a configurable set of transport-term mixins onto the latent grid each step."""

    REAL_TERM_BUILDERS = {
        "advection": lambda latent_dim, hidden_dim, cond_dim, film, block_kernel_size:
            AdvectionTerm(latent_dim, hidden_dim, cond_dim, film=film, block_kernel_size=block_kernel_size),
        # 2026-08-22: DiffusionTerm itself became spatially-/state-adaptive (reads
        # z.real_grid, not just cond) -- see its docstring in transport_terms.py.
        # LocalDiffusionTerm (below) was an independent, differently-shaped attempt
        # at the same idea built in parallel; kept as a separate selectable term
        # rather than deleted, since the two aren't identical and it's cheap to
        # leave available for ablation.
        "diffusion": lambda latent_dim, hidden_dim, cond_dim, film, block_kernel_size:
            DiffusionTerm(latent_dim, hidden_dim, cond_dim, film=film),
        "local_diffusion": lambda latent_dim, hidden_dim, cond_dim, film, block_kernel_size:
            LocalDiffusionTerm(latent_dim, cond_dim, hidden_dim=hidden_dim),
        "forcing": lambda latent_dim, hidden_dim, cond_dim, film, block_kernel_size:
            ForcingTerm(latent_dim, hidden_dim, cond_dim, film=film, block_kernel_size=block_kernel_size),
        "skew": lambda latent_dim, hidden_dim, cond_dim, film, block_kernel_size:
            SkewTerm(latent_dim, cond_dim),
        "local_attention": lambda latent_dim, hidden_dim, cond_dim, film, block_kernel_size:
            LocalAttentionTerm(latent_dim, hidden_dim, cond_dim, film=film),
        "helmholtz_rotation": lambda latent_dim, hidden_dim, cond_dim, film, block_kernel_size:
            HelmholtzRotationTerm(cond_dim, latent_dim, hidden_dim=hidden_dim),
    }

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        cond_dim: int = 32,
        terms: Sequence[str] = ("advection", "diffusion", "forcing"),
        film: bool = False,
        complex_term: bool = False,
        block_kernel_size: int = 7,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.complex_term = complex_term

        unknown = [t for t in terms if t not in self.REAL_TERM_BUILDERS]
        if unknown:
            raise ValueError(
                f"Unknown transport term(s) {unknown}; available: {sorted(list(self.REAL_TERM_BUILDERS.keys()))}"
            )

        self.real_terms = nn.ModuleDict({
            name: self.REAL_TERM_BUILDERS[name](latent_dim, hidden_dim, cond_dim, film, block_kernel_size)
            for name in terms if name in self.REAL_TERM_BUILDERS
        })

        # Built once here (not per-forward!) so they're registered submodules --
        # model.to(device) and the optimizer only see parameters that exist at
        # construction time.
        if complex_term:
            self.complex_amplitude = ComplexAmplitudeTerm(latent_dim, hidden_dim, cond_dim,
                                                            block_kernel_size=block_kernel_size)
            self.complex_rotation = ComplexRotationTerm(latent_dim, hidden_dim, cond_dim,
                                                          block_kernel_size=block_kernel_size)

    def forward(self, z: LatentState, cond: Optional[Tensor] = None,
                bc_weights: Optional[tuple] = None, dt: float = 1.0) -> LatentState:
        if cond is None:
            raise ValueError(f"{type(self).__name__} requires cond shaped [B, cond_dim].")

        # dt: sub-step scale for one operator call, default 1.0 = exactly the
        # original single-full-step behavior (unchanged for every existing
        # config/checkpoint). Calling forward() N times at dt=1/N instead of once
        # at dt=1 is a DISCO-inspired experiment (see EXPERIMENT_LOG.md) in giving
        # the operator smaller, more numerically-stable sub-steps per output frame
        # -- motivated by helmholtz_rotation's long-horizon blowup (§18), a classic
        # signature of a step size that's too large relative to the fastest/most
        # oscillatory term in the operator.
        #
        # advection/diffusion/forcing/skew are already structured as rate-like PDE
        # terms (a*grad(z), nu*Laplacian(z), etc.) -- consistent with treating them
        # as dz/dt and scaling their contribution by dt directly, standard Euler
        # sub-stepping. helmholtz_rotation is different: it returns a full finite
        # rotation's delta, not a rate, so it needs its own dt-aware forward (scales
        # the rotation ANGLE by dt, not the resulting delta -- see its docstring)
        # rather than being scaled after the fact like the others.
        real_out = z.real_grid
        for name, term in self.real_terms.items():
            if name == "helmholtz_rotation":
                # No bc_weights: FFT-based, periodicity is baked into the transform
                # itself (same reason complex_term is scoped out of BC-geometry
                # below) -- and it returns a full finite delta already, so it isn't
                # dt-scaled after the fact like the others (dt scales its rotation
                # ANGLE internally instead, see its forward's docstring comment).
                real_out = real_out + term(z, cond, dt=dt)
            else:
                real_out = real_out + dt * term(z, cond, bc_weights=bc_weights)

        spectral_out = z.spectral_grid
        if self.complex_term:
            amplitude = self.complex_amplitude(z, cond)
            # Zero-mean spatially then tanh: zero-mean prevents net spectral energy
            # drift over K rollout steps; tanh bounds the per-step factor to
            # exp(+-1)~=2.72x, ~=405x over a 6-step rollout in the worst case, with
            # smooth gradients (no dead zones at a hard clamp). (This used to be
            # scaled by a separate `amplitude_scale` hyperparameter -- removed, it
            # was fixed at 1.0 everywhere it was ever actually used, so the scaling
            # was a pure no-op multiply, not a real tuning knob. If tightening this
            # bound ever turns out to matter, that's a fresh decision, not reviving
            # this.)
            amplitude = amplitude - amplitude.mean(dim=(-2, -1), keepdim=True)
            amplitude = torch.tanh(amplitude)
            rotation = self.complex_rotation(z, cond)
            # amplitude/rotation are also full-step quantities like helmholtz_rotation's
            # theta, not rates -- same dt-on-the-angle/log-amplitude treatment, not a
            # post-hoc scale of the exp(...) factor.
            spectral_out = z.spectral_grid * torch.exp(dt * amplitude + 1j * dt * rotation)

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


class HelmholtzRotationTransportOperator(TransportOperator):
    """HelmholtzTransportOperator (advection+diffusion+skew+forcing) plus
    HelmholtzRotationTerm: a 2-channel real_grid vector-pair (channels 0:2)
    is Helmholtz-split every step and only its solenoidal (divergence-free)
    part gets a conservative, per-frequency-bin phase rotation -- see
    HelmholtzRotationTerm's docstring for the physical motivation. The
    reference operator for this codebase's latent dynamics -- see
    EXPERIMENT_LOG.md for the ablation history behind this specific
    configuration (per-frequency-bin phase, single vector pair, no
    additional conserved-channel reservation or complex branch: all of
    those were tried and lost).
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 64, cond_dim: int = 32):
        super().__init__(latent_dim, hidden_dim, cond_dim,
                         terms=("advection", "diffusion", "skew", "forcing", "helmholtz_rotation"),
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


class FiLMHelmholtzRotationTransportOperator(TransportOperator):
    """HelmholtzRotationTransportOperator, FiLM-conditioned instead of
    concat-conditioned. This is the reference operator actually used by
    context_cond_helmholtz_rotation.yaml -- see HelmholtzRotationTransportOperator
    and EXPERIMENT_LOG.md.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 64, cond_dim: int = 32):
        super().__init__(latent_dim, hidden_dim, cond_dim,
                         terms=("advection", "diffusion", "skew", "forcing", "helmholtz_rotation"),
                         film=True)


class FiLMLocalAttentionTransportOperator(TransportOperator):
    """LocalAttentionTransportOperator, FiLM-conditioned instead of concat-conditioned."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64, cond_dim: int = 32,
                 window: int = 5, n_heads: int = 4, attn_hidden_dim: Optional[int] = None):
        super().__init__(latent_dim, hidden_dim, cond_dim,
                         terms=("advection", "diffusion", "forcing"), film=True)
        self.real_terms["local_attention"] = LocalAttentionTerm(
            latent_dim, attn_hidden_dim or hidden_dim, cond_dim, window=window, n_heads=n_heads, film=True)
