import torch.nn as nn
from torch import Tensor

from core.modules import OperatorBase
from core.states import LatentState


class LinearLocalOperator(OperatorBase):
    def __init__(
        self,
        latent_dim: int,
    ):
        super().__init__()

        self.pointwise = nn.Conv2d(latent_dim, latent_dim, kernel_size=1)
        self.spatial = nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1)

        for m in [self.pointwise, self.spatial]:
            if m is not None:
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        z: LatentState,
        cond: Tensor | None = None,
    ) -> LatentState:
        x = z.grid

        out = x + self.pointwise(x) + self.spatial(x)
        return z.replace_state(real_grid=out)
