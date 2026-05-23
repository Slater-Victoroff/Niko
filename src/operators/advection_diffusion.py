import math
import torch
import torch.nn as nn
from torch import Tensor

from core.modules import OperatorBase
from core.states import LatentState
from core.blocks import ConvNeXtBlock


def dx_central(x: Tensor) -> Tensor:
    return 0.5 * (
        torch.roll(x, shifts=-1, dims=-1)
        - torch.roll(x, shifts=1, dims=-1)
    )


def dy_central(x: Tensor) -> Tensor:
    return 0.5 * (
        torch.roll(x, shifts=-1, dims=-2)
        - torch.roll(x, shifts=1, dims=-2)
    )


def laplacian(x: Tensor) -> Tensor:
    return (
        torch.roll(x, shifts=1, dims=-1)
        + torch.roll(x, shifts=-1, dims=-1)
        + torch.roll(x, shifts=1, dims=-2)
        + torch.roll(x, shifts=-1, dims=-2)
        - 4.0 * x
    )


class AdvectionDiffusionOperator(OperatorBase):
    """
    Single-path latent transport operator.

        a = transport_net([z, cond])
        f = forcing_net([z, cond])

        z_next = z - a · grad(z) + nu * laplacian(z) + f

    `cond` is expected to be shaped [B, cond_dim].
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        cond_dim: int = 32,
    ):
        super().__init__()

        self.cond_dim = cond_dim

        in_dim = latent_dim + cond_dim

        self.transport_net = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1),
            ConvNeXtBlock(hidden_dim),
            nn.Conv2d(hidden_dim, 2, kernel_size=1),
            nn.Tanh(),
        )

        self.forcing_net = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, latent_dim, kernel_size=1),
            nn.Tanh(),
        )

        self.log_nu = nn.Parameter(torch.tensor(-7.0))

        nn.init.zeros_(self.transport_net[-2].weight)
        nn.init.zeros_(self.transport_net[-2].bias)
        nn.init.zeros_(self.forcing_net[-2].weight)
        nn.init.zeros_(self.forcing_net[-2].bias)

    def _conditioned_input(self, x: Tensor, cond: Tensor) -> Tensor:
        if cond.ndim != 2:
            raise ValueError(
                f"Expected cond shape [B, {self.cond_dim}], got {tuple(cond.shape)}"
            )

        b, _, h, w = x.shape

        if cond.shape[0] != b:
            raise ValueError(
                f"Batch mismatch: x has B={b}, cond has B={cond.shape[0]}"
            )

        if cond.shape[1] != self.cond_dim:
            raise ValueError(
                f"Expected cond_dim={self.cond_dim}, got cond.shape[1]={cond.shape[1]}"
            )

        cond_map = cond.to(device=x.device, dtype=x.dtype)[:, :, None, None]
        cond_map = cond_map.expand(b, self.cond_dim, h, w)

        return torch.cat([x, cond_map], dim=1)

    def forward(
        self,
        z: LatentState,
        cond: Tensor | None = None,
    ) -> LatentState:
        if cond is None:
            raise ValueError(
                "AdvectionDiffusionOperator requires cond shaped [B, cond_dim]."
            )

        x = z.primary()
        xc = self._conditioned_input(x, cond)

        a = self.transport_net(xc)
        ax = a[:, 0:1]
        ay = a[:, 1:2]

        adv = -(ax * dx_central(x) + ay * dy_central(x))

        nu = torch.exp(torch.clamp(self.log_nu, min=-12.0, max=math.log(0.25)))
        diff = nu * laplacian(x)

        forcing = self.forcing_net(xc)

        out = x + adv + diff + forcing

        return z.replace_grid(out)
