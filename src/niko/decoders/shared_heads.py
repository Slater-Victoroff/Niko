import torch
import torch.nn as nn
from torch import Tensor

from core.modules import DecoderBase
from core.states import FieldState, LatentState
from core.blocks import ConvNeXtBlock


class SharedTrunkFieldHeadsDecoder(DecoderBase):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        upsample: int = 2,
        zero_mean_pressure: bool = True,
        use_streamfunction: bool = False,
    ):
        super().__init__()

        self.zero_mean_pressure = zero_mean_pressure
        self.use_streamfunction = use_streamfunction

        trunk = [
            nn.Conv2d(latent_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            ConvNeXtBlock(hidden_dim),
        ]

        if upsample == 2:
            trunk += [
                nn.Conv2d(hidden_dim, hidden_dim * 4, 3, padding=1),
                nn.PixelShuffle(2),
                ConvNeXtBlock(hidden_dim),
            ]

        self.trunk = nn.Sequential(*trunk)

        self.pressure_head = nn.Conv2d(hidden_dim, 1, 1)
        self.buoyancy_head = nn.Conv2d(hidden_dim, 1, 1)
        if self.use_streamfunction:
            self.stream_head = nn.Conv2d(hidden_dim, 1, 1)
        else:
            self.velocity_head = nn.Conv2d(hidden_dim, 2, 1)

    def forward(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        h = self.trunk(x)

        p = self.pressure_head(h)
        b = self.buoyancy_head(h)

        if self.use_streamfunction:
            psi = self.stream_head(h)
            field = FieldState.from_pressure_buoyancy_streamfunction(
                pressure=p,
                buoyancy=b,
                psi=psi,
            )
        else:
            uv = self.velocity_head(h)
            field = FieldState(
                pressure=p,
                buoyancy=b,
                velocity_x=uv[:, 0:1],
                velocity_y=uv[:, 1:2],
            )

        if self.zero_mean_pressure:
            field = field.zero_mean_pressure()

        return field.to_tensor()
