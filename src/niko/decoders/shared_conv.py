import torch
import torch.nn as nn
from torch import Tensor

from core.modules import DecoderBase
from core.states import FieldState, LatentState
from core.blocks import ConvNeXtBlock


class SharedConvDecoder(DecoderBase):
    """Shared backbone + physical heads decoder.

    Supports:
    - streamfunction mode (predict psi, derive velocity via FieldState helper)
    - direct velocity mode
    - optional output scaling (global or cond-conditioned)
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        out_channels: int = 4,
        upsample: int = 2,
        zero_mean_pressure: bool = True,
        use_streamfunction: bool = True,
        cond_dim: int | None = None,
        scale_hidden_dim: int = 64,
        use_output_scales: bool = False,
        max_log_scale: float = 4.0,
    ):
        super().__init__()

        self.zero_mean_pressure = zero_mean_pressure
        self.use_streamfunction = use_streamfunction
        self.cond_dim = cond_dim
        self.use_output_scales = use_output_scales
        self.max_log_scale = max_log_scale

        if self.use_streamfunction:
            out_channels = 3

        self.backbone = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_dim, 3, padding=1),
            ConvNeXtBlock(hidden_dim, expansion=1),
            nn.Conv2d(hidden_dim, hidden_dim * upsample ** 2, 3, padding=1),
            nn.PixelShuffle(upsample),
        )

        self.backbone = nn.Sequential(*layers)

        self.pressure_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.buoyancy_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

        if self.use_streamfunction:
            self.psi_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
            self.scale_channels = 3
        else:
            self.velocity_head = nn.Conv2d(hidden_dim, 2, kernel_size=1)
            self.scale_channels = 4

        if not use_output_scales:
            self.scale_net = None
            self.log_scales = None
        elif cond_dim is not None:
            self.scale_net = nn.Sequential(
                nn.Linear(cond_dim, scale_hidden_dim),
                nn.GELU(),
                nn.Linear(scale_hidden_dim, self.scale_channels),
            )
            nn.init.zeros_(self.scale_net[-1].weight)
            nn.init.zeros_(self.scale_net[-1].bias)
            self.log_scales = None
        else:
            self.scale_net = None
            self.log_scales = nn.Parameter(torch.zeros(1, self.scale_channels, 1, 1))

    def _get_log_scales(self, h: Tensor, cond: Tensor | None) -> Tensor | None:
        if not self.use_output_scales:
            return None

        B = h.shape[0]
        if self.scale_net is not None:
            if cond is None:
                raise ValueError("SharedConvDecoder initialized with cond_dim, but cond=None was passed.")
            if cond.shape[0] != B:
                raise ValueError(
                    f"cond batch mismatch: cond.shape={tuple(cond.shape)}, h.shape={tuple(h.shape)}"
                )
            log_scales = self.scale_net(cond).view(B, self.scale_channels, 1, 1)
        else:
            log_scales = self.log_scales.expand(B, -1, -1, -1)

        if self.max_log_scale is not None:
            log_scales = self.max_log_scale * torch.tanh(log_scales / self.max_log_scale)
        return log_scales

    def forward(self, z: LatentState | Tensor, cond: Tensor | None = None) -> Tensor:
        z_grid = z.primary() if isinstance(z, LatentState) else z
        h = self.backbone(z_grid)
        log_scales = self._get_log_scales(h, cond)

        pressure = self.pressure_head(h)
        buoyancy = self.buoyancy_head(h)

        if self.use_streamfunction:
            psi = self.psi_head(h)
            if log_scales is not None:
                pressure = pressure * log_scales[:, 0:1].exp()
                buoyancy = buoyancy * log_scales[:, 1:2].exp()
                psi = psi * log_scales[:, 2:3].exp()

            field = FieldState.from_pressure_buoyancy_streamfunction(
                pressure=pressure,
                buoyancy=buoyancy,
                psi=psi,
            )
        else:
            velocity = self.velocity_head(h)
            velocity_x = velocity[:, 0:1]
            velocity_y = velocity[:, 1:2]
            if log_scales is not None:
                pressure = pressure * log_scales[:, 0:1].exp()
                buoyancy = buoyancy * log_scales[:, 1:2].exp()
                velocity_x = velocity_x * log_scales[:, 2:3].exp()
                velocity_y = velocity_y * log_scales[:, 3:4].exp()

            field = FieldState(
                pressure=pressure,
                buoyancy=buoyancy,
                velocity_x=velocity_x,
                velocity_y=velocity_y,
            )

        if self.zero_mean_pressure:
            field = field.zero_mean_pressure()

        return field.to_tensor()
