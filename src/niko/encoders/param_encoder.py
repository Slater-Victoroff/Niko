import torch
import torch.nn as nn
from torch import Tensor
from typing import Literal

from core.states import Params


class ScalarParamHead(nn.Module):
    """
    Encodes one scalar parameter into an embedding.

    Each parameter owns its coordinate transform. This keeps parameters like
    Rayleigh number from being mixed into the model as giant spatial constants.
    """

    def __init__(
        self,
        target_dim: int,
        h_dim: int,
        transform: Literal["identity", "log10", "log", "signed_log10"] = "identity",
    ):
        super().__init__()

        self.transform = transform

        self.net = nn.Sequential(
            nn.Linear(1, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, target_dim),
        )

    def transform_input(self, x: Tensor) -> Tensor:
        if self.transform == "identity":
            return x

        if self.transform == "log10":
            return torch.log10(torch.clamp_min(x, 1e-30))

        if self.transform == "log":
            return torch.log(torch.clamp_min(x, 1e-30))

        if self.transform == "signed_log10":
            return torch.sign(x) * torch.log10(1.0 + torch.abs(x))

        raise ValueError(f"Unknown transform: {self.transform}")

    def forward(self, x: Tensor) -> Tensor:
        x = self.transform_input(x.float())
        return self.net(x)


class PerParamEncoder(nn.Module):
    """
    Encodes each parameter independently, then mixes the resulting embeddings.

    params.values: [B, P]
    output: [B, cond_dim]
    """

    def __init__(
        self,
        heads: list[ScalarParamHead],
        target_dim: int,
        h_dim: int,
    ):
        super().__init__()

        if len(heads) == 0:
            raise ValueError("PerParamEncoder requires at least one ScalarParamHead.")

        self.heads = nn.ModuleList(heads)
        input_dim = sum(head.net[-1].out_features for head in heads)

        self.mixer = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, target_dim),
        )

    def forward(self, params: Params) -> Tensor:
        p = params.values

        embeds = [
            head(p[:, i:i+1])
            for i, head in enumerate(self.heads)
        ]

        x = torch.cat(embeds, dim=-1)
        return self.mixer(x)


class RBParamEncoder(PerParamEncoder):
    def __init__(self, cond_dim: int, h_dim: int):
        head_hdim = h_dim // 2
        head_cond_dim = cond_dim // 2
        heads = [
            ScalarParamHead(
                target_dim=head_cond_dim,
                h_dim=head_hdim,
                transform="log10",
            ),
            ScalarParamHead(
                target_dim=head_cond_dim,
                h_dim=head_hdim,
                transform="log10",
            ),
        ]
        super().__init__(heads=heads, target_dim=cond_dim, h_dim=h_dim)


class MultiParamEncoder(PerParamEncoder):
    """Generalizes RBParamEncoder to an arbitrary number of scalar params,
    each with its own configurable transform -- e.g. active_matter's 3 params
    (L, zeta, alpha): alpha is negative (rules out plain log10, which NaNs on
    negative input), and L is constant across the whole dataset (harmless to
    include as just another param -- a per-param linear head fed a constant
    input reduces to a learned bias term, no special-casing needed to detect
    and exclude "doesn't actually vary" params).
    """

    def __init__(self, cond_dim: int, h_dim: int, transforms: list[str]):
        n = len(transforms)
        if n == 0:
            raise ValueError("MultiParamEncoder requires at least one transform.")
        head_hdim = max(1, h_dim // n)
        head_cond_dim = max(1, cond_dim // n)
        heads = [
            ScalarParamHead(target_dim=head_cond_dim, h_dim=head_hdim, transform=t)
            for t in transforms
        ]
        super().__init__(heads=heads, target_dim=cond_dim, h_dim=h_dim)