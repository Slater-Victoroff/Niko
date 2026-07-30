import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from core.blocks import FiLMConvNeXtBlock, ConvNeXtBlock
from core.states import FieldState


def _wavenumber_grids(
    H: int, W: int, device: torch.device, dtype: torch.dtype
) -> Tuple[Tensor, Tensor]:
    """Returns (ky [H,1], k2 [H, W//2+1]) for rfft2 spectral layout."""
    ky = torch.fft.fftfreq(H, device=device).to(dtype) * (2 * math.pi)
    kx = torch.fft.rfftfreq(W, device=device).to(dtype) * (2 * math.pi)
    ky = ky.view(-1, 1)
    kx = kx.view(1, -1)
    k2 = ky ** 2 + kx ** 2
    return ky, k2


def streamfunction_from_velocity(u: Tensor, v: Tensor) -> Tensor:
    """Spectral Poisson solve: ψ̂ = ω̂/k², ψ̂₀₀ = 0 (zero mean).

    Convention: u = ∂ψ/∂y, v = -∂ψ/∂x, so ω = ∂v/∂x - ∂u/∂y = k²ψ̂.
    """
    H, W = u.shape[-2], u.shape[-1]
    du_dy = 0.5 * (torch.roll(u, -1, -2) - torch.roll(u, 1, -2))
    dv_dx = 0.5 * (torch.roll(v, -1, -1) - torch.roll(v, 1, -1))
    omega = dv_dx - du_dy

    omega_hat = torch.fft.rfft2(omega)
    _, k2 = _wavenumber_grids(H, W, u.device, u.dtype)
    k2 = k2[None, None]

    safe_k2 = k2.clone()
    safe_k2[..., 0, 0] = 1.0

    psi_hat = omega_hat / safe_k2
    psi_hat[..., 0, 0] = 0.0

    return torch.fft.irfft2(psi_hat, s=(H, W))


def pressure_from_buoyancy(b: Tensor) -> Tensor:
    """Spectral hydrostatic solve: ∇²p = ∂b/∂y → p̂ = -iky·b̂/k², p̂₀₀ = 0.

    Uses the linearized buoyancy-driven pressure, zeroing the mean.
    """
    H, W = b.shape[-2], b.shape[-1]
    b_hat = torch.fft.rfft2(b)
    ky, k2 = _wavenumber_grids(H, W, b.device, b.dtype)
    ky = ky[None, None]
    k2 = k2[None, None]

    safe_k2 = k2.clone()
    safe_k2[..., 0, 0] = 1.0

    # p̂ = -iky·b̂ / k²
    p_hat = (-1j * ky) * b_hat / safe_k2
    p_hat[..., 0, 0] = 0.0

    return torch.fft.irfft2(p_hat, s=(H, W))


class DirectFieldOperator(nn.Module):
    """Operator-only model in zero-mean (b, ψ) space.

    Maps [B, 4, H, W] → [B, 4, H, W] directly in physical space.
    Velocity is divergence-free by construction (derived from ψ via curl).
    Pressure is derived from b via spectral hydrostatic balance.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        depth: int = 8,
        cond_dim: int = 0,
    ):
        super().__init__()
        self.cond_dim = cond_dim

        self.stem = nn.Sequential(
            nn.Conv2d(2, hidden_dim, 3, padding=1),
            nn.GELU(),
        )

        if cond_dim > 0:
            self.blocks = nn.ModuleList(
                [FiLMConvNeXtBlock(hidden_dim, cond_dim) for _ in range(depth)]
            )
        else:
            self.blocks = nn.ModuleList(
                [ConvNeXtBlock(hidden_dim) for _ in range(depth)]
            )

        self.head = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _to_normed(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """[B, 4, H, W] → (b_zm, ψ_zm), each [B, 1, H, W] zero-meaned."""
        field = FieldState.from_tensor(x)
        psi = streamfunction_from_velocity(field.velocity_x, field.velocity_y)
        b_zm = field.buoyancy - field.buoyancy.mean(dim=(-2, -1), keepdim=True)
        psi_zm = psi - psi.mean(dim=(-2, -1), keepdim=True)
        return b_zm, psi_zm

    def _from_normed(self, b: Tensor, psi: Tensor) -> Tensor:
        """(b, ψ) → [B, 4, H, W] with u,v from curl(ψ) and p from hydrostatic."""
        u, v = FieldState.velocity_from_streamfunction(psi)
        p = pressure_from_buoyancy(b)
        return FieldState(pressure=p, buoyancy=b, velocity_x=u, velocity_y=v).to_tensor()

    def forward(self, x: Tensor, cond: Optional[Tensor] = None) -> Tensor:
        b, psi = self._to_normed(x)
        h = self.stem(torch.cat([b, psi], dim=1))

        for block in self.blocks:
            h = block(h, cond) if self.cond_dim > 0 else block(h)

        delta = self.head(h)
        b_new = b + delta[:, 0:1]
        b_new = b_new - b_new.mean(dim=(-2, -1), keepdim=True)
        psi_new = psi + delta[:, 1:2]
        psi_new = psi_new - psi_new.mean(dim=(-2, -1), keepdim=True)

        return self._from_normed(b_new, psi_new)
