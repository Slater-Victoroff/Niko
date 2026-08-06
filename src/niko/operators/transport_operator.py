from typing import Optional, Sequence

import torch.nn as nn
from torch import Tensor

from core.modules import OperatorBase
from core.states import LatentState
from operators.transport_terms import AdvectionTerm, DiffusionTerm, ForcingTerm, SkewTerm


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
    }

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        cond_dim: int = 32,
        terms: Sequence[str] = ("advection", "diffusion", "forcing"),
        film: bool = False,
    ):
        super().__init__()
        self.cond_dim = cond_dim

        unknown = [t for t in terms if t not in self.REAL_TERM_BUILDERS]
        if unknown:
            raise ValueError(
                f"Unknown transport term(s) {unknown}; available: {sorted(list(self.REAL_TERM_BUILDERS.keys()))}"
            )

        self.real_terms = nn.ModuleDict({
            name: self.REAL_TERM_BUILDERS[name](latent_dim, hidden_dim, cond_dim, film)
            for name in terms if name in self.REAL_TERM_BUILDERS
        })

    def forward(self, z: LatentState, cond: Optional[Tensor] = None) -> LatentState:
        if cond is None:
            raise ValueError(f"{type(self).__name__} requires cond shaped [B, cond_dim].")

        real_out = z.real_grid
        for term in self.real_terms.values():
            real_out = real_out + term(z, cond)

        return z.replace_state(real_grid=real_out)


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
