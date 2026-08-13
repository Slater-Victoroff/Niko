from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from core.modules import DecoderBase
from core.states import FieldState, LatentState
from core.blocks import ConvNeXtBlock


class SharedTrunkFieldHeadsDecoder(DecoderBase):
    """scalar_field_names (optional, e.g. ["concentration"] for active_matter):
    when None (default), the original fixed RB path is used unchanged --
    pressure_head + buoyancy_head, routed through FieldState (unaffected by
    anything below; every existing config/checkpoint keeps its exact current
    behavior). When a list is given, that fixed pair is replaced by one plain
    Conv2d(hidden_dim, 1, kernel_size=1) head per name (bypassing FieldState's
    fixed 4-field schema, which doesn't fit e.g. active_matter's single
    concentration scalar), concatenated in [scalar_field_names...,
    velocity/streamfunction, tensor_field_names...] order -- match this order
    to field_spec's channel order for whatever dataset is being decoded.
    zero_mean_pressure still applies, but only if "pressure" is literally one
    of the given names (not assumed for an arbitrary scalar field like
    concentration, which generally does have a meaningful nonzero mean).

    tensor_field_names (optional, e.g. ["D", "E"] for active_matter): one
    extra head per name, each a plain Conv2d(hidden_dim, 4, kernel_size=1)
    off the same shared (already pixelshuffle-upsampled) trunk output as
    every other head here -- no dedicated architecture for the tensor case,
    same pattern as the scalar heads, just 4 output channels instead of 1.
    Kept as 4 flat channels (row-major: T_00,T_01,T_10,T_11) concatenated
    onto the end of the output, not reshaped to a (...,2,2) tensor here --
    everything downstream that consumes decoder output (rollout stacking,
    well_style_vrmse, etc.) already expects a flat [B,C,H,W]/[B,K,C,H,W]
    channel tensor, so keeping tensor fields in that same flat-channel
    convention means nothing else needs to special-case a 5D shape; reshape
    to (...,2,2) at whatever point actually needs the matrix structure (e.g.
    a future physics diagnostic), not baked in here.

    Both default to None/empty, changing nothing about existing configs.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        upsample: int = 2,
        zero_mean_pressure: bool = True,
        use_streamfunction: bool = False,
        tensor_field_names: Optional[list] = None,
        scalar_field_names: Optional[list] = None,
    ):
        super().__init__()

        self.zero_mean_pressure = zero_mean_pressure
        self.use_streamfunction = use_streamfunction
        self.tensor_field_names = list(tensor_field_names) if tensor_field_names else []
        self.scalar_field_names = list(scalar_field_names) if scalar_field_names is not None else None

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

        if self.scalar_field_names is None:
            self.pressure_head = nn.Conv2d(hidden_dim, 1, 1)
            self.buoyancy_head = nn.Conv2d(hidden_dim, 1, 1)
        else:
            self.scalar_heads = nn.ModuleDict({
                name: nn.Conv2d(hidden_dim, 1, 1) for name in self.scalar_field_names
            })

        if self.use_streamfunction:
            self.stream_head = nn.Conv2d(hidden_dim, 1, 1)
        else:
            self.velocity_head = nn.Conv2d(hidden_dim, 2, 1)

        self.tensor_heads = nn.ModuleDict({
            name: nn.Conv2d(hidden_dim, 4, kernel_size=1) for name in self.tensor_field_names
        })

    def forward(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        h = self.trunk(x)

        if self.scalar_field_names is None:
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

            out = field.to_tensor()
        else:
            scalar_outs = [self.scalar_heads[name](h) for name in self.scalar_field_names]
            if self.zero_mean_pressure and "pressure" in self.scalar_field_names:
                idx = self.scalar_field_names.index("pressure")
                scalar_outs[idx] = scalar_outs[idx] - scalar_outs[idx].mean(dim=(-2, -1), keepdim=True)

            if self.use_streamfunction:
                psi = self.stream_head(h)
                vx, vy = FieldState.velocity_from_streamfunction(psi)
                vel_out = torch.cat([vx, vy], dim=1)
            else:
                vel_out = self.velocity_head(h)

            out = torch.cat(scalar_outs + [vel_out], dim=1)

        if self.tensor_field_names:
            tensor_outs = [self.tensor_heads[name](h) for name in self.tensor_field_names]
            out = torch.cat([out] + tensor_outs, dim=1)

        return out
