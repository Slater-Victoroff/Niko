import torch
import torch.nn as nn
from torch import Tensor

from core.modules import OperatorBase
from core.states import LatentState, Params
from core.blocks import ConvNeXtBlock
from operators.advection_diffusion import dx_central, dy_central, laplacian
import logging

logger = logging.getLogger(__name__)


def _summ(t: Tensor) -> str:
    try:
        return f"shape={tuple(t.shape)} min={float(t.min()):.6g} max={float(t.max()):.6g} mean={float(t.mean()):.6g} has_nan={torch.isnan(t).any().item()} has_inf={torch.isinf(t).any().item()}"
    except Exception as e:
        return f"(could not summarize: {e})"


class HelmholtzTransportOperator(OperatorBase):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        use_potential: bool = True,
        use_stream: bool = True,
        use_residual: bool = True,
        use_diffusion: bool = True,
        use_forcing: bool = True,
        zero_init_last: bool = True,
    ):
        super().__init__()

        self.use_potential = use_potential
        self.use_stream = use_stream
        self.use_residual = use_residual
        self.use_diffusion = use_diffusion
        self.use_forcing = use_forcing

        out_dim = 0
        if use_potential:
            out_dim += 1  # phi
        if use_stream:
            out_dim += 1  # psi
        if use_residual:
            out_dim += 2  # h_x, h_y

        self.potential_net = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_dim, 3, padding=1),
            ConvNeXtBlock(hidden_dim),
            nn.Conv2d(hidden_dim, out_dim, 1),
        )

        if zero_init_last:
            nn.init.zeros_(self.potential_net[-1].weight)
            nn.init.zeros_(self.potential_net[-1].bias)

        if use_diffusion:
            self.log_nu = nn.Parameter(torch.tensor(-4.0))

        if use_forcing:
            self.forcing = nn.Sequential(
                nn.Conv2d(latent_dim, hidden_dim, 1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, latent_dim, 1),
            )
            if zero_init_last:
                nn.init.zeros_(self.forcing[-1].weight)
                nn.init.zeros_(self.forcing[-1].bias)

    def build_transport(self, x: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        raw = self.potential_net(x)

        logger.debug("potential_net raw: %s", _summ(raw))

        cursor = 0
        b, _, h, w = x.shape

        ax = torch.zeros(b, 1, h, w, device=x.device, dtype=x.dtype)
        ay = torch.zeros_like(ax)

        extras = {}

        if self.use_potential:
            phi = raw[:, cursor:cursor + 1]
            cursor += 1

            ax = ax + dx_central(phi)
            ay = ay + dy_central(phi)

            extras["phi"] = phi

        if self.use_stream:
            psi = raw[:, cursor:cursor + 1]
            cursor += 1

            # curl_perp psi = [dpsi/dy, -dpsi/dx]
            ax = ax + dy_central(psi)
            ay = ay - dx_central(psi)

            extras["psi"] = psi

        if self.use_residual:
            hvec = raw[:, cursor:cursor + 2]
            cursor += 2

            ax = ax + hvec[:, 0:1]
            ay = ay + hvec[:, 1:2]

            extras["h"] = hvec

        a = torch.cat([ax, ay], dim=1)
        extras["transport"] = a

        logger.debug("built transport a: %s", _summ(a))

        return a, extras

    def forward(
        self,
        z: LatentState,
        cond: Tensor | None = None,
    ) -> LatentState:
        x = z.primary()

        a, extras = self.build_transport(x)
        ax = a[:, 0:1]
        ay = a[:, 1:2]

        zx = dx_central(x)
        zy = dy_central(x)

        adv = -(ax * zx + ay * zy)

        rhs = adv

        if self.use_diffusion:
            nu = torch.exp(self.log_nu)
            rhs = rhs + nu * laplacian(x)

        if self.use_forcing:
            rhs = rhs + self.forcing(x)

        z_next = x + rhs

        meta = dict(z.meta or {})
        meta.update({f"operator/{k}": v for k, v in extras.items()})

        return LatentState(
            grid=z_next,
            real_tucker=z.real_tucker,
            complex_tucker=z.complex_tucker,
            spectral_grid=z.spectral_grid,
            aux=z.aux,
            meta=meta,
        )
