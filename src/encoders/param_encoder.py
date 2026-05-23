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