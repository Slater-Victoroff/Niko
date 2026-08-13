from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import torch
from torch import Tensor


@dataclass
class Params:
    """
    Physical / simulation parameters.

    For Rayleigh-Benard this might include Rayleigh number, Prandtl number,
    boundary condition id, timestep, etc.
    """
    values: Tensor  # [B, P]
    names: Optional[List[str]] = None


@dataclass
class LatentState:
    real_grid: Optional[Tensor] = None
    spectral_grid: Optional[Tensor] = None
    aux: Optional[Tensor] = None
    meta: Optional[Dict[str, Any]] = None

    @property
    def grid(self) -> Tensor:
        """Real_grid, fused with the complex branch (irfft2'd back to real
        space and added) when a spectral_grid is present -- the one place
        this fusion happens, so callers (decoder, rollout stacking) never
        need to know whether a complex branch exists at all."""
        if self.real_grid is None and self.spectral_grid is None:
            raise ValueError("LatentState has no grid data.")
        if self.real_grid is not None and self.spectral_grid is not None:
            return self.real_grid + torch.fft.irfft2(self.spectral_grid, s=self.real_grid.shape[-2:], norm="ortho")
        return self.real_grid

    @property
    def channel_sgrid(self) -> Tensor:
        """spectral_grid's real/imag parts concatenated along the channel
        dim -- a real-valued [B, 2*latent_dim, H, W] tensor, the input shape
        a normal conv net expects. For terms that read (not evolve) the
        current spectral content, e.g. predicting the next rotation angle."""
        if self.spectral_grid is None:
            raise ValueError("LatentState has no spectral_grid data.")
        return torch.cat([self.spectral_grid.real, self.spectral_grid.imag], dim=1)

    def replace_state(self, real_grid: Tensor, spectral_grid: Optional[Tensor] = None) -> "LatentState":
        return LatentState(
            real_grid=real_grid,
            spectral_grid=spectral_grid if spectral_grid is not None else self.spectral_grid,
            aux=self.aux,
            meta=self.meta,
        )


@dataclass
class FieldState:
    pressure: Tensor
    buoyancy: Tensor
    velocity_x: Tensor
    velocity_y: Tensor

    @classmethod
    def from_tensor(cls, x: Tensor) -> "FieldState":
        assert x.shape[-3] == 4, f"Expected 4 channels [p,b,u,v], got {x.shape}"
        return cls(
            pressure=x[..., 0:1, :, :],
            buoyancy=x[..., 1:2, :, :],
            velocity_x=x[..., 2:3, :, :],
            velocity_y=x[..., 3:4, :, :],
        )

    def to_tensor(self) -> Tensor:
        return torch.cat(
            [self.pressure, self.buoyancy, self.velocity_x, self.velocity_y],
            dim=-3,
        )

    def zero_mean_pressure(self) -> "FieldState":
        p = self.pressure - self.pressure.mean(dim=(-2, -1), keepdim=True)
        return FieldState(p, self.buoyancy, self.velocity_x, self.velocity_y)

    @staticmethod
    def velocity_from_streamfunction(psi: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute velocity components from streamfunction using central differences.

        velocity_x = dpsi/dy, velocity_y = -dpsi/dx
        """
        dpsi_dy = 0.5 * (
            torch.roll(psi, shifts=-1, dims=-2)
            - torch.roll(psi, shifts=1, dims=-2)
        )
        dpsi_dx = 0.5 * (
            torch.roll(psi, shifts=-1, dims=-1)
            - torch.roll(psi, shifts=1, dims=-1)
        )
        return dpsi_dy, -dpsi_dx

    @classmethod
    def from_pressure_buoyancy_streamfunction(
        cls,
        pressure: Tensor,
        buoyancy: Tensor,
        psi: Tensor,
    ) -> "FieldState":
        velocity_x, velocity_y = cls.velocity_from_streamfunction(psi)
        return cls(
            pressure=pressure,
            buoyancy=buoyancy,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
        )

    def channel_means(self):
        return {
            "pressure": self.pressure.mean(),
            "buoyancy": self.buoyancy.mean(),
            "velocity_x": self.velocity_x.mean(),
            "velocity_y": self.velocity_y.mean(),
        }
