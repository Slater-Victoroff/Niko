from abc import ABC, abstractmethod
from typing import Optional
from torch import Tensor
from .states import LatentState, Params


class Encoder(ABC):
    @abstractmethod
    def forward(self, x_context: Tensor, params: Optional[Params] = None) -> LatentState:
        """
        x_context: [B, T_context, C, H, W] or [B, T_context*C, H, W]
        returns: latent state z_t
        """
        raise NotImplementedError


class LatentOperator(ABC):
    @abstractmethod
    def forward(
        self,
        z: LatentState,
        params: Optional[Params] = None,
        dt: float | Tensor = 1.0,
    ) -> LatentState:
        """
        z_t -> z_{t+1}
        """
        raise NotImplementedError


class Decoder(ABC):
    @abstractmethod
    def forward(self, z: LatentState, params: Optional[Params] = None) -> Tensor:
        """
        z_t -> decoded physical tensor x_t [B, C, H, W]
        """
        raise NotImplementedError
